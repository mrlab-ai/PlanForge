use super::{ConfigCall, HeuristicConfig, configure, parse_call};
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

    /// Fail before translation if a heuristic needs a solver backend this
    /// build does not have.
    pub fn preflight(&self) -> std::io::Result<()> {
        self.heuristics().try_for_each(HeuristicConfig::preflight)
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

/// The `--search` help text: the engines and every registered heuristic.
pub fn search_algorithm_help() -> String {
    format!(
        "Search specification. Available algorithms: astar([heuristic], mpd=false, lazy=none), gbfs([heuristic])\n\n{}",
        crate::heuristic_factory::heuristic_help()
    )
}

pub fn parse_search_spec(raw: &str) -> Result<SearchConfig, String> {
    let spec = build_search_spec(&parse_call(raw)?)?;
    Ok(spec)
}

// =============================================================================
// Search-engine dispatch
// =============================================================================

/// `astar(heuristic, lazy=none, mpd=false)`; the heuristic defaults to blind.
#[derive(Default, super::ApplyOptions)]
struct AStarOptions {
    #[option(positional)]
    heuristic: HeuristicConfig,
    lazy: Option<HeuristicConfig>,
    mpd: bool,
}

/// `gbfs(heuristic)`; the heuristic defaults to blind.
#[derive(Default, super::ApplyOptions)]
struct GbfsOptions {
    #[option(positional)]
    heuristic: HeuristicConfig,
}

fn build_search_spec(call: &ConfigCall) -> Result<SearchConfig, String> {
    Ok(match call.name.as_str() {
        "astar" => {
            let AStarOptions {
                heuristic,
                lazy,
                mpd,
            } = configure("astar", AStarOptions::default(), &call.args)?;
            SearchConfig::AStar {
                heuristic,
                lazy,
                mpd,
            }
        }
        "gbfs" => {
            let GbfsOptions { heuristic } = configure("gbfs", GbfsOptions::default(), &call.args)?;
            SearchConfig::Gbfs { heuristic }
        }
        other => {
            return Err(format!(
                "unknown search engine `{other}`; expected one of: astar, gbfs"
            ));
        }
    })
}

#[cfg(test)]
mod tests;
