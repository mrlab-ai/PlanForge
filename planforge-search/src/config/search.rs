use super::{ConfigCall, ConfigValue, FromOptionValue, HeuristicConfig, parse_call};
use std::fmt;

// =============================================================================
// HeuristicConfig + SearchConfig
// =============================================================================

#[derive(Debug, Clone, PartialEq)]
pub enum SearchConfig {
    AStar {
        heuristic: HeuristicConfig,
        lazy: Option<HeuristicConfig>,
        mpd: bool,
    },
    Gbfs {
        heuristic: HeuristicConfig,
    },
}

impl SearchConfig {
    /// Every heuristic the engine will build, in configuration order.
    pub fn heuristics(&self) -> impl Iterator<Item = &HeuristicConfig> {
        let (first, second) = match self {
            Self::AStar {
                heuristic, lazy, ..
            } => (heuristic, lazy.as_ref()),
            Self::Gbfs { heuristic } => (heuristic, None),
        };
        std::iter::once(first).chain(second)
    }

    pub fn requirements(&self) -> crate::heuristic_factory::TaskRequirements {
        self.heuristics().fold(
            crate::heuristic_factory::TaskRequirements::ANY,
            |requirements, heuristic| requirements.union(heuristic.requirements()),
        )
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
    Ok(spec)
}

// =============================================================================
// Search-engine dispatch
// =============================================================================

fn build_search_spec(call: &ConfigCall) -> Result<SearchConfig, String> {
    if call.name.eq_ignore_ascii_case("search") {
        if call.args.len() != 1 {
            return Err("`search(...)` expects exactly one search engine".to_string());
        }
        let nested = call_from_value(&call.args[0].value)?;
        return build_search_spec(&nested);
    }

    match call.name.to_ascii_lowercase().as_str() {
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
                    "fast" => fast = Some(super::HeuristicSpec::from_value(&arg.value)?),
                    "slow" => slow = Some(super::HeuristicSpec::from_value(&arg.value)?),
                    other => return Err(format!("unknown option `{other}` for `astar_fs`")),
                }
            }
            let fast = fast.ok_or_else(|| "`astar_fs(...)` requires `fast=...`".to_string())?;
            let slow = slow.ok_or_else(|| "`astar_fs(...)` requires `slow=...`".to_string())?;
            Ok(SearchConfig::AStar {
                heuristic: HeuristicConfig::from_spec(&fast)
                    .map_err(|error| error.at_option("fast").at("astar_fs"))?,
                lazy: Some(
                    HeuristicConfig::from_spec(&slow)
                        .map_err(|error| error.at_option("slow").at("astar_fs"))?,
                ),
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
                heuristic = Some(
                    HeuristicConfig::from_value(&arg.value)
                        .map_err(|error| error.at_option("heuristic").at("astar"))?,
                );
            }
            Some("lazy") => {
                if saw_lazy {
                    return Err("duplicate option `lazy` for `astar`".to_string());
                }
                saw_lazy = true;
                lazy = Option::<HeuristicConfig>::from_option_value(&arg.value)
                    .map_err(|error| error.at_option("lazy").at("astar"))?;
            }
            Some("mpd") => {
                if saw_mpd {
                    return Err("duplicate option `mpd` for `astar`".to_string());
                }
                saw_mpd = true;
                mpd = bool::from_option_value(&arg.value)
                    .map_err(|_| "`astar` option `mpd` expects true or false".to_string())?;
            }
            Some(other) => return Err(format!("unknown option `{other}` for `astar`")),
        }
    }
    Ok(SearchConfig::AStar {
        heuristic: heuristic.unwrap_or_else(HeuristicConfig::blind),
        lazy,
        mpd,
    })
}

fn extract_heuristic_for_search(call: &ConfigCall) -> Result<HeuristicConfig, String> {
    if call.args.is_empty() {
        return Ok(HeuristicConfig::blind());
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
    Ok(HeuristicConfig::from_value(&arg.value)
        .map_err(|error| error.at_option("heuristic").at(&call.name))?)
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
