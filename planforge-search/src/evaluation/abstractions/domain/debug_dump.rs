//! Debug output of domain-abstraction CEGAR.

use planforge_sas::numeric_task::AbstractNumericTask;
use planforge_sas::numeric_task::VariableIndex;
use tracing::info;

use super::factory::DomainAbstractionFactory;

/// The partitions a finished CEGAR run left, at `info` level.
pub(crate) fn log_final_abstraction(
    task: &dyn AbstractNumericTask,
    factory: &DomainAbstractionFactory,
) {
    info!("domain abstraction final domains:");
    for (numeric_var_id, &size) in factory
        .numeric_domain_sizes()
        .iter()
        .enumerate()
        .filter(|(_, size)| **size > 1)
    {
        let name = task.numeric_variables()[numeric_var_id].name();
        let parts = factory
            .partitions()
            .partitions(VariableIndex::from_usize(numeric_var_id));
        let intervals = parts
            .iter()
            .enumerate()
            .map(|(part_id, interval)| format!("p{part_id}:{interval:?}"))
            .collect::<Vec<_>>()
            .join(" ");
        info!("  n{numeric_var_id}={name}, size={size}, {intervals}");
    }
    for (var_id, &size) in factory
        .domain_sizes()
        .iter()
        .enumerate()
        .filter(|(_, size)| **size > 1)
    {
        let name = task
            .get_variable_name(VariableIndex::from_usize(var_id))
            .expect("every propositional variable has a name");
        info!("  p{var_id}={name}, size={size}");
    }
}
