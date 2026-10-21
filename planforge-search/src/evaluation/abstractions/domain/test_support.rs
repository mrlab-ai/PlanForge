//! Domain abstractions the tests build by hand.

use planforge_sas::numeric_task::{AbstractNumericTask, ExplicitValueIndex, VariableIndex};

use super::abstract_operators::DomainMapping;

/// The abstraction that keeps every propositional value apart, except on
/// condition variables, which are derived from the numeric partitions.
pub(crate) fn identity_domain_mapping_and_sizes(
    task: &dyn AbstractNumericTask,
) -> (DomainMapping, Vec<usize>) {
    let num_vars = task.get_num_variables();
    let mut domain_mapping: DomainMapping = Vec::with_capacity(num_vars);
    let mut domain_sizes: Vec<usize> = Vec::with_capacity(num_vars);
    for var_id in (0..num_vars).map(VariableIndex::from_usize) {
        let size = task.variables()[var_id.index()].domain_size();
        if task.numeric_conditions().is_condition_var(var_id) {
            domain_mapping.push(vec![ExplicitValueIndex::new(0); size]);
            domain_sizes.push(1);
        } else {
            domain_mapping.push((0..size).map(ExplicitValueIndex::from_usize).collect());
            domain_sizes.push(size);
        }
    }

    (domain_mapping, domain_sizes)
}
