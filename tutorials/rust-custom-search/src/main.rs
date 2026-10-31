use anyhow::Result;
use planforge_sas::numeric_task::{NumericRootTask, TaskRef};
use planforge_search::evaluation::heuristic::BlindHeuristic;
use planforge_search::search::best_first::{TieBreakingOpenList, UniformCost};
use planforge_search::search::{BestFirstSearch, SearchOutcome, SearchParams};
use std::sync::Arc;
use std::time::Duration;

fn main() -> Result<()> {
    let task: TaskRef<'static> = Arc::new(NumericRootTask::from_file(
        "tests/assets/numeric_sas/example2.sas",
    ));
    // Uniform-cost search: best-first search ordered by `g` alone.
    let params = SearchParams {
        time_limit: Some(Duration::from_millis(50)),
        ..SearchParams::new(Box::new(BlindHeuristic::for_task(&*task)))
    };
    let result = BestFirstSearch::<UniformCost, TieBreakingOpenList>::new(task, params).run()?;
    assert!(matches!(
        result.outcome,
        SearchOutcome::Solved { .. } | SearchOutcome::Timeout
    ));
    assert!(result.statistics.nodes.expanded > 0);
    println!(
        "uniform_cost ended with {:?} after {} expansions",
        result.outcome, result.statistics.nodes.expanded
    );
    Ok(())
}
