use std::time::Duration;

use planforge_sas::numeric_task::TaskRef;
use planforge_sas::state_registry::StateRegistry;
use tracing::info;

use super::{AStarSearch, GreedyBestFirstSearch, SearchEngine, SearchParams, SearchResult};
use crate::config::SearchConfig;
use crate::heuristic_factory::{BuildContext, HeuristicBuildError};

/// Searches `task` as `spec` configures it, within the limits.
pub fn solve_task(
    task: TaskRef<'_>,
    spec: &SearchConfig,
    time_limit: Option<Duration>,
    memory_limit: Option<u64>,
) -> std::io::Result<SearchResult> {
    let state_registry = StateRegistry::for_task(task.clone());
    let context = BuildContext::new(&task);
    let mut engine: Box<dyn SearchEngine + '_> = match spec {
        SearchConfig::AStar {
            heuristic,
            lazy,
            mpd,
        } => {
            let heuristic = heuristic
                .build(&context)
                .map_err(HeuristicBuildError::into_io_error)?;
            let lazy = lazy
                .as_ref()
                .map(|spec| {
                    spec.build(&context)
                        .map_err(HeuristicBuildError::into_io_error)
                })
                .transpose()?;
            info!("Starting A* search, mpd={mpd}, lazy={}", lazy.is_some());
            Box::new(AStarSearch::new(
                &*task,
                state_registry,
                SearchParams {
                    heuristic,
                    lazy,
                    reevaluate_dynamic: *mpd,
                    time_limit,
                    max_memory_bytes: memory_limit,
                },
            ))
        }
        SearchConfig::Gbfs { heuristic } => {
            let heuristic = heuristic
                .build(&context)
                .map_err(HeuristicBuildError::into_io_error)?;
            info!("Starting GBFS search...");
            Box::new(GreedyBestFirstSearch::new(
                &*task,
                state_registry,
                SearchParams {
                    time_limit,
                    max_memory_bytes: memory_limit,
                    ..SearchParams::new(heuristic)
                },
            ))
        }
    };
    engine
        .search()
        .map_err(|error| std::io::Error::other(format!("search failed: {error:#}")))
}
