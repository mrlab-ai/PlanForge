use super::{ConfigCall, ConfigValue, HeuristicSpec, parse_call};
use std::fmt;

// =============================================================================
// HeuristicSpec + SearchConfig
// =============================================================================

#[derive(Debug, Clone, PartialEq)]
pub enum SearchConfig {
    AStar {
        heuristic: HeuristicSpec,
        lazy: Option<HeuristicSpec>,
        mpd: bool,
    },
    Gbfs {
        heuristic: HeuristicSpec,
    },
}

impl SearchConfig {
    /// Every heuristic the engine will build, in configuration order.
    pub fn heuristics(&self) -> impl Iterator<Item = &HeuristicSpec> {
        let (first, second) = match self {
            Self::AStar {
                heuristic, lazy, ..
            } => (heuristic, lazy.as_ref()),
            Self::Gbfs { heuristic } => (heuristic, None),
        };
        std::iter::once(first).chain(second)
    }

    pub fn contains_call(&self, name: &str) -> bool {
        self.heuristics()
            .any(|heuristic| heuristic.contains_call(name))
    }
}

// =============================================================================
// Display — used to round-trip the spec back into `--search SPEC` form
// =============================================================================

impl fmt::Display for SearchConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AStar {
                heuristic,
                lazy,
                mpd,
            } => {
                write!(f, "astar({heuristic}")?;
                if let Some(lazy) = lazy {
                    write!(f, ", lazy={lazy}")?;
                }
                if *mpd {
                    write!(f, ", mpd=true")?;
                }
                write!(f, ")")
            }
            Self::Gbfs { heuristic } => write!(f, "gbfs({heuristic})"),
        }
    }
}

// =============================================================================
// Parser entry
// =============================================================================

pub fn parse_search_spec(raw: &str) -> Result<SearchConfig, String> {
    let spec = build_search_spec(&parse_call(raw)?)?;
    for heuristic in spec.heuristics() {
        crate::heuristic_factory::validate_heuristic_spec(heuristic)?;
    }
    Ok(spec)
}

// =============================================================================
// Search-engine dispatch
// =============================================================================

fn build_search_spec(call: &ConfigCall) -> Result<SearchConfig, String> {
    if call.name == "search" {
        if call.args.len() != 1 {
            return Err("`search(...)` expects exactly one search engine".to_string());
        }
        let nested = call_from_value(&call.args[0].value)?;
        return build_search_spec(&nested);
    }

    match call.name.as_str() {
        "astar" => extract_astar_options(call),
        "gbfs" => Ok(SearchConfig::Gbfs {
            heuristic: extract_heuristic_for_search(call)?,
        }),
        "astar_fs" => {
            let mut fast = None;
            let mut slow = None;
            for arg in &call.args {
                let key = arg.key.as_deref().ok_or_else(|| {
                    "`astar_fs(...)` expects named `fast=...` and `slow=...` arguments".to_string()
                })?;
                match key {
                    "fast" => fast = Some(HeuristicSpec::from_value(&arg.value)?),
                    "slow" => slow = Some(HeuristicSpec::from_value(&arg.value)?),
                    other => return Err(format!("unknown option `{other}` for `astar_fs`")),
                }
            }
            let fast = fast.ok_or_else(|| "`astar_fs(...)` requires `fast=...`".to_string())?;
            let slow = slow.ok_or_else(|| "`astar_fs(...)` requires `slow=...`".to_string())?;
            Ok(SearchConfig::AStar {
                heuristic: fast,
                lazy: Some(slow),
                mpd: false,
            })
        }
        other => Err(format!(
            "unknown search engine `{other}`; expected one of: astar, gbfs, astar_fs"
        )),
    }
}

fn extract_astar_options(call: &ConfigCall) -> Result<SearchConfig, String> {
    let mut heuristic = None;
    let mut mpd = false;
    let mut saw_mpd = false;
    let mut lazy = None;
    let mut saw_lazy = false;
    for arg in &call.args {
        match arg.key.as_deref() {
            None | Some("heuristic") => {
                if heuristic.is_some() {
                    return Err("`astar(...)` received more than one heuristic".to_string());
                }
                heuristic = Some(HeuristicSpec::from_value(&arg.value)?);
            }
            Some("lazy") => {
                if saw_lazy {
                    return Err("duplicate option `lazy` for `astar`".to_string());
                }
                saw_lazy = true;
                lazy = match &arg.value {
                    ConfigValue::Atom(value) if value == "none" => None,
                    value => Some(HeuristicSpec::from_value(value)?),
                };
            }
            Some("mpd") => {
                if saw_mpd {
                    return Err("duplicate option `mpd` for `astar`".to_string());
                }
                saw_mpd = true;
                mpd = match &arg.value {
                    ConfigValue::Atom(value) if value == "true" => true,
                    ConfigValue::Atom(value) if value == "false" => false,
                    _ => return Err("`astar` option `mpd` expects true or false".to_string()),
                };
            }
            Some(other) => return Err(format!("unknown option `{other}` for `astar`")),
        }
    }
    Ok(SearchConfig::AStar {
        heuristic: heuristic.unwrap_or_else(HeuristicSpec::blind),
        lazy,
        mpd,
    })
}

fn extract_heuristic_for_search(call: &ConfigCall) -> Result<HeuristicSpec, String> {
    if call.args.is_empty() {
        return Ok(HeuristicSpec::blind());
    }
    if call.args.len() != 1 {
        return Err(format!(
            "`{}(...)` expects a single heuristic argument",
            call.name
        ));
    }
    let arg = &call.args[0];
    if let Some(key) = &arg.key
        && key != "heuristic"
    {
        return Err(format!(
            "`{}(...)` expects `heuristic=...`, got `{key}=...`",
            call.name
        ));
    }
    HeuristicSpec::from_value(&arg.value)
}

fn call_from_value(value: &ConfigValue) -> Result<ConfigCall, String> {
    match value {
        ConfigValue::Call(call) => Ok(call.clone()),
        ConfigValue::Atom(name) => Ok(ConfigCall {
            name: name.clone(),
            args: Vec::new(),
        }),
        ConfigValue::List(_) => Err("expected call, got a list".to_string()),
    }
}
