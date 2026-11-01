//! A search algorithm of your own, named by the `--search` grammar: uniform-cost
//! search, best-first search ordered by `g` alone.

use anyhow::Result;
use planforge_sas::numeric_task::Task;
use planforge_search::evaluation::heuristic::BlindHeuristic;
use planforge_search::heuristic_factory::{EmptyConfig, HeuristicConfig};
use planforge_search::search::best_first::{TieBreakingOpenList, UniformCost};
use planforge_search::search::{
    BestFirstSearch, SEARCH_ALGORITHMS, SearchFeature, SearchLimits, SearchOutcome, SearchParams,
    SearchResult, entry, parse_search_spec,
};
use std::sync::Arc;
use std::time::Duration;

/// `ucs()`: uniform-cost search. It evaluates no heuristic.
struct UniformCostFeature;

impl SearchFeature for UniformCostFeature {
    const NAME: &'static str = "ucs";
    type Config = EmptyConfig;

    fn heuristics(_: &EmptyConfig) -> Vec<&HeuristicConfig> {
        Vec::new()
    }

    fn run(
        _: &EmptyConfig,
        task: Arc<Task>,
        limits: SearchLimits,
    ) -> std::io::Result<SearchResult> {
        let params = SearchParams {
            time_limit: limits.time,
            max_memory_bytes: limits.memory_bytes,
            ..SearchParams::new(Box::new(BlindHeuristic::for_task(&task)))
        };
        BestFirstSearch::<UniformCost, TieBreakingOpenList>::new(task, params)
            .run()
            .map_err(|error| std::io::Error::other(format!("{error:#}")))
    }
}

fn main() -> Result<()> {
    // A binary registers its search algorithms once, before it parses a
    // configuration; `planforge::run_with` does so for the real CLI.
    SEARCH_ALGORITHMS
        .register(vec![entry::<UniformCostFeature>()])
        .map_err(anyhow::Error::msg)?;
    let search = parse_search_spec("ucs()").map_err(anyhow::Error::msg)?;

    let task: Arc<Task> = Arc::new(Task::from_file("tests/assets/numeric_sas/example2.sas"));
    let result = search.solve(
        task,
        SearchLimits {
            time: Some(Duration::from_millis(50)),
            memory_bytes: None,
        },
    )?;
    assert!(matches!(
        result.outcome,
        SearchOutcome::Solved { .. } | SearchOutcome::Timeout
    ));
    assert!(result.statistics.nodes.expanded > 0);
    println!(
        "{search} ended with {:?} after {} expansions",
        result.outcome, result.statistics.nodes.expanded
    );
    Ok(())
}
