use std::time::Duration;

use planforge_sas::numeric_task::TaskRef;
use tracing::info;

use super::{AStarSearch, GreedyBestFirstSearch, SearchConfig, SearchParams, SearchResult};
use crate::heuristic_factory::{BuildContext, HeuristicBuildError};

/// Searches `task` as `spec` configures it, within the limits.
pub fn solve_task(
    task: TaskRef<'_>,
    spec: &SearchConfig,
    time_limit: Option<Duration>,
    memory_limit: Option<u64>,
) -> std::io::Result<SearchResult> {
    let context = BuildContext::new(&task);
    let build = |config: &crate::heuristic_factory::HeuristicConfig| {
        config
            .build(&context)
            .map_err(HeuristicBuildError::into_io_error)
    };
    let result = match spec {
        SearchConfig::AStar {
            heuristic,
            lazy,
            mpd,
        } => {
            let params = SearchParams {
                heuristic: build(heuristic)?,
                lazy: lazy.as_ref().map(build).transpose()?,
                mpd: *mpd,
                time_limit,
                max_memory_bytes: memory_limit,
            };
            info!(
                "Starting A* search, mpd={mpd}, lazy={}",
                params.lazy.is_some()
            );
            AStarSearch::new(task.clone(), params).run()
        }
        SearchConfig::Gbfs { heuristic } => {
            let params = SearchParams {
                time_limit,
                max_memory_bytes: memory_limit,
                ..SearchParams::new(build(heuristic)?)
            };
            info!("Starting GBFS search...");
            GreedyBestFirstSearch::new(task.clone(), params).run()
        }
    };
    result.map_err(|error| std::io::Error::other(format!("search failed: {error:#}")))
}
