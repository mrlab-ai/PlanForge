use anyhow::{Context, Result, anyhow};
use std::collections::{BTreeSet, HashMap};
use std::fmt::Write as _;

use planforge_sas::axioms::AxiomEvaluator;
use planforge_sas::numeric_task::{
    AbstractNumericTask, ExplicitValueIndex, NumericValue, VariableIndex,
};
use planforge_sas::state_registry::ConcreteStateView;
use planforge_sas::utils::state_packer::StatePacker;
use tracing::debug;

use super::cegar::flaw_search::Flaw;
use super::factory::{DomainAbstractionFactory, WildcardPlanResult};
use super::numeric_partitions::NumericPartitions;
use crate::evaluation::abstractions::common::cegar::progress_concrete_state;
use crate::evaluation::abstractions::common::transition_system::AbstractDistanceTable;
use crate::evaluation::abstractions::domain::abstract_operator_generator::DomainMapping;
use crate::evaluation::abstractions::domain::cegar::flaw_search::SplitDirection;
use crate::evaluation::abstractions::domain::cegar::flaw_search::progression::{
    NumericTransitionStates, PartitionedTask, get_progression_numeric_deviation_flaws,
    get_progression_precondition_flaws,
};
mod debug_dump;
mod partitioning;
pub(crate) use debug_dump::*;
pub(crate) use partitioning::*;

pub(crate) fn compute_abstraction_size_u128(
    domain_sizes: &[usize],
    numeric_domain_sizes: &[usize],
) -> Option<u128> {
    let mut size: u128 = 1;
    for &d in domain_sizes.iter() {
        let du = u128::try_from(d).ok()?;
        if du == 0 {
            return Some(0);
        }
        size = size.checked_mul(du)?;
    }
    for &p in numeric_domain_sizes.iter() {
        let pu = u128::try_from(p).ok()?;
        if pu == 0 {
            return Some(0);
        }
        size = size.checked_mul(pu)?;
    }
    Some(size)
}

#[allow(unused)]
pub(crate) fn identity_domain_mapping_and_sizes(
    task: &dyn AbstractNumericTask,
) -> Result<(DomainMapping, Vec<usize>)> {
    let num_vars = task.get_num_variables();
    let mut domain_mapping: DomainMapping = Vec::with_capacity(num_vars);
    let mut domain_sizes: Vec<usize> = Vec::with_capacity(num_vars);
    for var_id in 0..num_vars {
        let size = task
            .get_variable_domain_size(VariableIndex::from_usize(var_id))
            .map_err(|e| anyhow!(e.to_string()))
            .with_context(|| format!("failed to get domain size for variable {var_id}"))?;
        domain_mapping.push((0..size).map(ExplicitValueIndex::from_usize).collect());
        domain_sizes.push(size);
    }

    Ok((domain_mapping, domain_sizes))
}
