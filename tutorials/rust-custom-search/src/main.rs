use anyhow::Result;
use planforge_sas::numeric_task::{NumericRootTask, TaskRef};
use planforge_sas::state_registry::StateRegistry;
use planforge_search::evaluation::heuristic::BlindHeuristic;
use planforge_search::search::{
    BestFirstSearch, G, SearchEngine, SearchParams, SearchStatus, TieBreakingOpenList,
};
use std::sync::Arc;
use std::time::Duration;

fn main() -> Result<()> {
    let task: TaskRef<'static> = Arc::new(NumericRootTask::from_file(
        "tests/assets/numeric_sas/example2.sas",
    ));
    let registry = StateRegistry::for_task(task.clone());
    let mut driver = BestFirstSearch::<G, TieBreakingOpenList>::new(
        &*task,
        registry,
        SearchParams {
            heuristic: Box::new(BlindHeuristic::for_task(&*task)),
            lazy: None,
            time_limit: Some(Duration::from_millis(50)),
            max_memory_bytes: None,
            reevaluate_dynamic: false,
        },
    );
    let result = driver.search()?;
    assert!(matches!(
        result.status,
        SearchStatus::Solved(_) | SearchStatus::Timeout
    ));
    assert!(result.nodes_expanded > 0);
    println!(
        "uniform_cost ended with {:?}, cost {:?}, after {} expansions",
        result.status, result.solution_cost, result.nodes_expanded
    );
    Ok(())
}
