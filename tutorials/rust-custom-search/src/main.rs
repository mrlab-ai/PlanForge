use anyhow::Result;
use planforge_sas::numeric_task::{NumericRootTask, TaskRef};
use planforge_sas::state_registry::StateRegistry;
use planforge_search::evaluation::heuristic::BlindHeuristic;
use planforge_search::plugin_registry;
use planforge_search::search::{
    SearchAlgorithm, SearchAlgorithmPlugin, SearchBuildContext, SearchDriver, SearchOptionKind,
    SearchStatus,
};
use std::sync::Arc;
use std::time::Duration;

/// Uniform-cost search is a shallow extension: only its priority key differs.
struct UniformCost;

impl SearchAlgorithm for UniformCost {
    #[inline]
    fn priority(&self, g_value: f64, _h_value: f64) -> f64 {
        g_value
    }
}

fn build_uniform_cost<'a>(ctx: SearchBuildContext<'a>) -> Result<Box<dyn SearchDriver + 'a>> {
    UniformCost.build(ctx)
}

// An external crate registers one name, its option schema, and its builder.
// Leaving out any field is a compile error.
plugin_registry! {
    static TUTORIAL_SEARCH_ALGORITHMS: SearchAlgorithmPlugin;
    fn tutorial_search_algorithm;
    entries {
        "uniform_cost" => SearchAlgorithmPlugin {
            options: SearchOptionKind::GreedyBestFirst,
            option_schema: "uniform_cost()",
            build: build_uniform_cost,
        },
    }
}

fn main() -> Result<()> {
    let task: TaskRef<'static> = Arc::new(NumericRootTask::from_file(
        "tests/assets/numeric_sas/example2.sas",
    ));
    let registry = StateRegistry::for_task(task.clone());
    let plugin = tutorial_search_algorithm("uniform_cost").expect("registered above");
    let mut driver = (plugin.build)(SearchBuildContext {
        task: &*task,
        state_registry: registry,
        primary_heuristic: Some(Box::new(BlindHeuristic::new(None))),
        secondary_heuristic: None,
        // The committed SAS examples are deliberately substantial. A short
        // limit keeps this tutorial executable while still exercising the
        // external algorithm in the real expansion loop.
        time_limit: Some(Duration::from_millis(50)),
        max_memory_bytes: None,
        mpd: false,
    })?;
    let result = driver.run()?;
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
