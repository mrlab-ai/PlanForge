#[cfg(test)]
mod tests;

use super::pattern_collection::PatternCollection;
use super::pattern_database::{PatternDatabase, PdbBuildConfig};
use super::utils;
use crate::task_transformations::projection::ProjectedTask;
use crate::task_transformations::restriction::validate_restricted_task;
use planforge_sas::numeric_task::AbstractNumericTask;

/// The pattern database of every pattern in `patterns`, in order.
pub fn build_pattern_databases<'task>(
    task: &'task dyn AbstractNumericTask,
    patterns: &PatternCollection,
    config: PdbBuildConfig,
) -> Result<Vec<PatternDatabase<'task>>, String> {
    validate_restricted_task(task)?;
    patterns
        .iter()
        .map(|pattern| {
            let projected_task =
                ProjectedTask::new(task, pattern).map_err(|err| err.to_string())?;
            utils::print_projection_summary(task, pattern, &projected_task);
            PatternDatabase::build(projected_task, config)
        })
        .collect()
}
