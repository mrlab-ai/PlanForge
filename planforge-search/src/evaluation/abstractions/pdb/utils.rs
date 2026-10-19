use planforge_sas::numeric_task::AbstractNumericTask;
use tracing::info;

use super::projected_task::{Pattern, ProjectedTask};

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
        projected_task.get_num_goals(),
        projected_task.get_operators().len()
    );
}
