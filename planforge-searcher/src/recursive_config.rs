use std::fmt;

// The AST nodes, the option traits and `HeuristicSpec` all live in
// `planforge-search`, next to the typed configs they populate; this module only
// produces them.
use planforge_search::config::parse_call;
pub use planforge_search::config::{
    ApplyOptions, ConfigArg, ConfigCall, ConfigValue, FromOptionValue, HeuristicSpec, atom,
    for_each_option, parse_heuristic_spec,
};
use planforge_search::search::{SearchOptionKind, search_algorithm, search_algorithm_names};

#[cfg(test)]
mod tests;

// =============================================================================
// HeuristicSpec + SearchSpec
// =============================================================================

#[derive(Debug, Clone, PartialEq)]
pub enum SearchSpec {
    /// A* with an optional monotonically-dynamic pop-time re-evaluation.
    Astar(HeuristicSpec, bool),
    Gbfs(HeuristicSpec),
    /// A* with two admissible heuristics: a *fast* one for initial open-
    /// list ordering and a *slow* but possibly tighter one evaluated
    /// lazily when a state is about to be expanded.
    AstarFs(HeuristicSpec, HeuristicSpec),
    /// Search-free plan synthesis by gradient descent. Not a search engine at
    /// all; it shares `--search` only to reuse the translation, resource-limit
    /// and plan-writing plumbing.
    ///
    /// The arguments are kept as raw `ConfigArg`s rather than a `HeuristicSpec`
    /// because a `HeuristicSpec` prints its own name, which would round-trip as
    /// `sgd(sgd(...))` and break the self-re-exec.
    Sgd(Vec<ConfigArg>),
}

impl SearchSpec {
    /// Every heuristic the engine will build, in configuration order. The `sgd`
    /// engine yields none: it is not allowed a heuristic.
    pub fn heuristics(&self) -> impl Iterator<Item = &HeuristicSpec> {
        let (first, second) = match self {
            Self::Astar(heuristic, _) | Self::Gbfs(heuristic) => (Some(heuristic), None),
            Self::AstarFs(fast, slow) => (Some(fast), Some(slow)),
            Self::Sgd(_) => (None, None),
        };
        first.into_iter().chain(second)
    }

    pub fn contains_call(&self, name: &str) -> bool {
        self.heuristics()
            .any(|heuristic| heuristic.contains_call(name))
    }
}

// =============================================================================
// Display — used to round-trip the spec back into `--search SPEC` form
// =============================================================================

impl fmt::Display for SearchSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Astar(h, false) => write!(f, "astar({h})"),
            Self::Astar(h, true) => write!(f, "astar({h}, mpd=true)"),
            Self::Gbfs(h) => write!(f, "gbfs({h})"),
            Self::AstarFs(fast, slow) => write!(f, "astar_fs(fast={fast}, slow={slow})"),
            Self::Sgd(args) => {
                if args.is_empty() {
                    return write!(f, "sgd()");
                }
                let parts: Vec<String> = args
                    .iter()
                    .map(|arg| match arg.key() {
                        Some(key) => format!("{key}={}", arg.value()),
                        None => arg.value().to_string(),
                    })
                    .collect();
                write!(f, "sgd({})", parts.join(", "))
            }
        }
    }
}

// =============================================================================
// Parser entry
// =============================================================================

pub fn parse_search_spec(raw: &str) -> Result<SearchSpec, String> {
    let spec = build_search_spec(&parse_call(raw)?)?;
    for heuristic in spec.heuristics() {
        planforge_search::heuristic_factory::validate_heuristic_spec(heuristic)?;
    }
    Ok(spec)
}

// =============================================================================
// Search-engine dispatch
// =============================================================================

fn build_search_spec(call: &ConfigCall) -> Result<SearchSpec, String> {
    if call.name == "search" {
        if call.args.len() != 1 {
            return Err("`search(...)` expects exactly one search engine".to_string());
        }
        let nested = call_from_value(&call.args[0].value)?;
        return build_search_spec(&nested);
    }

    if call.name == "sgd" {
        return Ok(SearchSpec::Sgd(call.args.clone()));
    }

    let plugin = search_algorithm(&call.name).ok_or_else(|| {
        format!(
            "unknown search engine `{}`; expected one of: {}",
            call.name,
            search_algorithm_names().collect::<Vec<_>>().join(", ")
        )
    })?;
    match plugin.options {
        SearchOptionKind::AStar => {
            let (heuristic, mpd) = extract_astar_options(call)?;
            Ok(SearchSpec::Astar(heuristic, mpd))
        }
        SearchOptionKind::GreedyBestFirst => {
            Ok(SearchSpec::Gbfs(extract_heuristic_for_search(call)?))
        }
        SearchOptionKind::FastSlow => {
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
            Ok(SearchSpec::AstarFs(fast, slow))
        }
    }
}

fn extract_astar_options(call: &ConfigCall) -> Result<(HeuristicSpec, bool), String> {
    let mut heuristic = None;
    let mut mpd = false;
    let mut saw_mpd = false;
    for arg in &call.args {
        match arg.key.as_deref() {
            None | Some("heuristic") => {
                if heuristic.is_some() {
                    return Err("`astar(...)` received more than one heuristic".to_string());
                }
                heuristic = Some(HeuristicSpec::from_value(&arg.value)?);
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
    Ok((heuristic.unwrap_or_else(HeuristicSpec::blind), mpd))
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
