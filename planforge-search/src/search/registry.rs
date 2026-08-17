use super::{AStar, FastSlow, GreedyBestFirst, SearchAlgorithm, SearchDriver};
use crate::evaluation::Heuristic;
use anyhow::{Result, bail};
use planforge_sas::numeric_task::AbstractNumericTask;
use planforge_sas::state_registry::StateRegistry;
use std::time::Duration;

/// Parser-visible identity of a built-in search algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchOptionKind {
    AStar,
    GreedyBestFirst,
    FastSlow,
}

/// Everything required to construct one search driver.
pub struct SearchBuildContext<'a> {
    pub task: &'a dyn AbstractNumericTask,
    pub state_registry: StateRegistry<'a>,
    pub primary_heuristic: Option<Box<dyn Heuristic + 'a>>,
    pub secondary_heuristic: Option<Box<dyn Heuristic + 'a>>,
    pub time_limit: Option<Duration>,
    pub max_memory_bytes: Option<u64>,
    pub mpd: bool,
}

pub type SearchBuilder = for<'a> fn(SearchBuildContext<'a>) -> Result<Box<dyn SearchDriver + 'a>>;

/// One declarative search-algorithm registration.
///
/// External crates can use this descriptor with [`crate::plugin_registry!`]
/// to publish an additional name without modifying the best-first driver.
pub struct SearchAlgorithmPlugin {
    pub options: SearchOptionKind,
    pub option_schema: &'static str,
    pub build: SearchBuilder,
}

fn build_astar<'a>(ctx: SearchBuildContext<'a>) -> Result<Box<dyn SearchDriver + 'a>> {
    AStar.build(ctx)
}

fn build_gbfs<'a>(ctx: SearchBuildContext<'a>) -> Result<Box<dyn SearchDriver + 'a>> {
    if ctx.secondary_heuristic.is_some() || ctx.mpd {
        bail!("GBFS accepts one heuristic and does not support mpd")
    }
    GreedyBestFirst.build(ctx)
}

fn build_fast_slow<'a>(mut ctx: SearchBuildContext<'a>) -> Result<Box<dyn SearchDriver + 'a>> {
    let fast = ctx
        .primary_heuristic
        .take()
        .ok_or_else(|| anyhow::anyhow!("fast/slow A* requires a fast heuristic"))?;
    let slow = ctx
        .secondary_heuristic
        .take()
        .ok_or_else(|| anyhow::anyhow!("fast/slow A* requires a slow heuristic"))?;
    if ctx.mpd {
        bail!("fast/slow A* does not support mpd")
    }
    ctx.primary_heuristic = Some(fast);
    ctx.secondary_heuristic = None;
    FastSlow { slow }.build(ctx)
}

crate::plugin_registry! {
    pub static SEARCH_ALGORITHMS: SearchAlgorithmPlugin;
    pub fn search_algorithm;
    entries {
        "astar" => SearchAlgorithmPlugin {
            options: SearchOptionKind::AStar,
            option_schema: "astar([heuristic], mpd=false)",
            build: build_astar,
        },
        "gbfs" => SearchAlgorithmPlugin {
            options: SearchOptionKind::GreedyBestFirst,
            option_schema: "gbfs([heuristic])",
            build: build_gbfs,
        },
        "astar_fs" => SearchAlgorithmPlugin {
            options: SearchOptionKind::FastSlow,
            option_schema: "astar_fs(fast=HEURISTIC, slow=HEURISTIC)",
            build: build_fast_slow,
        },
    }
}

pub fn search_algorithm_names() -> impl Iterator<Item = &'static str> {
    SEARCH_ALGORITHMS.iter().map(|(name, _)| *name)
}

pub fn search_algorithm_help() -> String {
    let schemas = SEARCH_ALGORITHMS
        .iter()
        .map(|(_, plugin)| plugin.option_schema)
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "Search specification. Available algorithms: {schemas}\n\n{}",
        crate::heuristic_factory::HEURISTIC_HELP
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn registered_search_names_are_unique() {
        let names = search_algorithm_names().collect::<Vec<_>>();
        assert_eq!(
            names.len(),
            names.iter().copied().collect::<HashSet<_>>().len()
        );
        assert!(names.contains(&"astar"));
        assert!(names.contains(&"gbfs"));
        assert!(names.contains(&"astar_fs"));
    }
}
