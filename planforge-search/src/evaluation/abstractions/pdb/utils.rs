use planforge_sas::numeric_task::AbstractNumericTask;
use tracing::info;

use crate::task_transformations::projection::{Pattern, ProjectedTask};

pub(crate) fn print_projection_summary(
    base: &dyn AbstractNumericTask,
    pattern: &Pattern,
    projected_task: &ProjectedTask<'_>,
) {
    info!("=== NUMERIC PDB PROJECTION ===");
    info!(
        "  propositional vars: base={} pattern={} projected={}",
        base.variables().len(),
        pattern.regular.len(),
        projected_task.variables().len()
    );
    info!(
        "  numeric vars:       base={} pattern={} projected={}",
        base.numeric_variables().len(),
        pattern.numeric.len(),
        projected_task.numeric_variables().len()
    );
    info!(
        "  goals/operators:    goals={} operators={}",
        projected_task.goals().len(),
        projected_task.operators().len()
    );
}
