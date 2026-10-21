//! The domain mappings CEGAR starts a domain abstraction from.

use anyhow::{Context, Result, ensure};
use planforge_sas::numeric_task::{AbstractNumericTask, ExplicitValueIndex, VariableIndex};

use super::abstract_operators::DomainMapping;

pub(super) fn trivial_domain_mapping_and_sizes(
    task: &dyn AbstractNumericTask,
) -> Result<(DomainMapping, Vec<usize>)> {
    let num_vars = task.get_num_variables();

    let domain_sizes: Vec<usize> = vec![1; num_vars];
    let mut domain_mapping: DomainMapping = Vec::with_capacity(num_vars);

    for var in 0..num_vars {
        let size = task
            .get_variable_domain_size(VariableIndex::from_usize(var))
            .map_err(|e| anyhow::anyhow!(e.to_string()))
            .with_context(|| format!("get_variable_domain_size({var}) failed"))?;
        ensure!(size > 0, "non-positive domain size for var {var}: {size}");
        domain_mapping.push(vec![ExplicitValueIndex::new(0); size]);
    }

    Ok((domain_mapping, domain_sizes))
}

/// The abstraction that keeps every propositional value apart, except on
/// condition variables, which are derived from the numeric partitions.
#[cfg(test)]
pub(crate) fn identity_domain_mapping_and_sizes(
    task: &dyn AbstractNumericTask,
) -> Result<(DomainMapping, Vec<usize>)> {
    let num_vars = task.get_num_variables();
    let mut domain_mapping: DomainMapping = Vec::with_capacity(num_vars);
    let mut domain_sizes: Vec<usize> = Vec::with_capacity(num_vars);
    for var_id in (0..num_vars).map(VariableIndex::from_usize) {
        let size = task
            .get_variable_domain_size(var_id)
            .map_err(|e| anyhow::anyhow!(e.to_string()))
            .with_context(|| format!("failed to get domain size for variable {var_id}"))?;
        if task.numeric_conditions().is_condition_var(var_id) {
            domain_mapping.push(vec![ExplicitValueIndex::new(0); size]);
            domain_sizes.push(1);
        } else {
            domain_mapping.push((0..size).map(ExplicitValueIndex::from_usize).collect());
            domain_sizes.push(size);
        }
    }

    Ok((domain_mapping, domain_sizes))
}
