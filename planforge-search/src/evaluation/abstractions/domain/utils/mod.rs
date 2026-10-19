use anyhow::{Result, anyhow};
use std::collections::{BTreeSet, HashMap};
use std::fmt::Write as _;

use planforge_sas::axioms::AxiomEvaluator;
use planforge_sas::numeric_task::{AbstractNumericTask, NumericValue};
use planforge_sas::state_registry::ConcreteStateView;
use planforge_sas::utils::state_packer::StatePacker;
use tracing::debug;

use super::factory::{DomainAbstractionFactory, WildcardPlanResult};
use super::flaws::Flaw;
use super::partitions::NumericPartitions;
use crate::evaluation::abstractions::common::cegar::progress_concrete_state;
use crate::evaluation::abstractions::common::transition_system::AbstractDistanceTable;
use crate::evaluation::abstractions::domain::flaws::SplitDirection;
use crate::evaluation::abstractions::domain::flaws::progression::{
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
