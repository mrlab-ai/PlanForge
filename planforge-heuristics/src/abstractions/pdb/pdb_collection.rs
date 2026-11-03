//! The pattern databases of a pattern collection.

#[cfg(test)]
mod tests;

use super::pattern_collection::PatternCollection;
use super::pattern_database::{PatternDatabase, PdbBuildConfig};
use planforge_sas::numeric_task::Task;
use planforge_search::heuristic_features::HeuristicBuildError;
use planforge_search::task_transformations::projection::{Pattern, ProjectedTask};
use planforge_search::task_transformations::restriction::validate_restricted_task;
use tracing::info;

/// The pattern database of every pattern in `patterns`, in order.
pub fn build_pattern_databases(
    task: &Task,
    patterns: &PatternCollection,
    config: PdbBuildConfig,
) -> Result<Vec<PatternDatabase>, HeuristicBuildError> {
    validate_restricted_task(task)?;
    patterns
        .iter()
        .map(|pattern| {
            let projected_task = ProjectedTask::new(task, pattern)?;
            print_projection_summary(task, pattern, &projected_task);
            PatternDatabase::build(projected_task, config)
        })
        .collect()
}

/// Logs the sizes of the projection of `base` to `pattern`.
fn print_projection_summary(base: &Task, pattern: &Pattern, projected_task: &ProjectedTask) {
    info!("=== NUMERIC PDB PROJECTION ===");
    info!(
        "  propositional vars: base={} pattern={} projected={}",
        base.variables().len(),
        pattern.regular.len(),
        projected_task.task().variables().len()
    );
    info!(
        "  numeric vars:       base={} pattern={} projected={}",
        base.numeric_variables().len(),
        pattern.numeric.len(),
        projected_task.task().numeric_variables().len()
    );
    info!(
        "  goals/operators:    goals={} operators={}",
        projected_task.task().goals().len(),
        projected_task.task().operators().len()
    );
}
