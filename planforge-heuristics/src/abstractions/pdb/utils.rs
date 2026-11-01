use planforge_sas::numeric_task::Task;
use tracing::info;

use planforge_search::task_transformations::projection::{Pattern, ProjectedTask};

pub(crate) fn print_projection_summary(
    base: &Task,
    pattern: &Pattern,
    projected_task: &ProjectedTask,
) {
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
