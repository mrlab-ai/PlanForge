use anyhow::{Context, Result, anyhow};
use std::collections::{BTreeSet, HashMap};
use std::fmt::Write as _;

use planforge_sas::axioms::AxiomEvaluator;
use planforge_sas::numeric_task::AbstractNumericTask;
use planforge_sas::state_registry::ConcreteStateView;
use planforge_sas::utils::float_tolerance;
use planforge_sas::utils::int_packer::IntDoublePacker;
use tracing::debug;

use super::cegar::flaw_search::Flaw;
use super::domain_abstraction::NumericPartitions;
use super::domain_abstraction_factory::{
    AbstractDistanceTable, DomainAbstractionFactory, WildcardPlanResult,
};
use crate::evaluation::cegar::progress_concrete_state;
use crate::evaluation::domain_abstractions::abstract_operator_generator::DomainMapping;
use crate::evaluation::domain_abstractions::cegar::flaw_search::SplitDirection;
use crate::evaluation::domain_abstractions::cegar::flaw_search::progression::{
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
            .get_variable_domain_size(var_id)
            .map_err(|e| anyhow!(e.to_string()))
            .with_context(|| format!("failed to get domain size for variable {var_id}"))?;
        domain_mapping.push((0..size).collect());
        domain_sizes.push(size);
    }

    Ok((domain_mapping, domain_sizes))
}

pub(crate) fn make_prop_state_packer(task: &dyn AbstractNumericTask) -> IntDoublePacker {
    let mut domain_sizes: Vec<u64> = Vec::with_capacity(task.variables().len());
    for var in task.variables().iter() {
        domain_sizes.push(var.domain_size() as u64);
    }
    IntDoublePacker::new(&domain_sizes)
}

pub(crate) fn set_initial_prop_values(
    task: &dyn AbstractNumericTask,
    packer: &IntDoublePacker,
    buffer: &mut [u64],
) {
    let init = task.get_initial_propositional_state_values();
    for (var_id, &val) in init.iter().enumerate() {
        packer.set(buffer, var_id, val as u64);
    }
}

pub(crate) fn get_initial_state(
    task: &dyn AbstractNumericTask,
    state_packer: &IntDoublePacker,
    axiom_evaluator: &AxiomEvaluator,
) -> Result<(Vec<u64>, Vec<f64>)> {
    let mut buffer = vec![0u64; state_packer.num_bins()];
    set_initial_prop_values(task, state_packer, &mut buffer);
    let mut numeric_state: Vec<f64> = task.get_initial_numeric_state_values().to_vec();
    for value in &mut numeric_state {
        *value = float_tolerance::canonicalize(*value);
    }

    axiom_evaluator
        .evaluate(&mut buffer, &mut numeric_state)
        .map_err(|e| anyhow::anyhow!("failed to evaluate axioms for initial state: {e:?}"))?;

    Ok((buffer, numeric_state))
}
