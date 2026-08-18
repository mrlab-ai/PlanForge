#[cfg(test)]
mod tests;

use std::cmp::Reverse;
#[cfg(any(test, debug_assertions))]
use std::collections::HashSet;
use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail, ensure};
use ordered_float::NotNan;
use rand::seq::SliceRandom;
use rand::{SeedableRng, rngs::SmallRng};
use tracing::debug;

use planforge_sas::numeric_task::{
    AbstractNumericTask, AssignmentOperation, ExplicitFact, NumericType, Operator,
    metric_operator_cost_from_initial_values,
};
use planforge_sas::utils::float_tolerance;

use super::abstract_operator_generator::{
    AbstractOperator, AbstractOperatorGenerator, DomainMapping,
};
use super::abstraction_numeric_var;
use super::additive_numeric_views::AdditiveNumericViews;
use super::domain_abstraction::NumericPartitions;
use super::numeric_context::{
    AbstractStateHash, prepare_comparison_tree_inputs_from_abstract_state,
    prepare_comparison_tree_inputs_from_abstract_state_into,
};
use super::utils;

mod distances;
mod footprints;
mod plan_extraction;
mod saturation;
mod state_encoding;
mod transition_system;

use crate::evaluation::abstraction_collections::cost_partitioning::{
    AbstractOperatorCostFunction, AbstractOperatorFootprint, AbstractTransition,
    AbstractTransitionCostFunction, AbstractTransitionSystem, ConcreteOperatorFootprint,
    RegionalCostAllocation, RegionalCostAllocationEntry, StateRegion, TransitionResidualCosts,
    saturated_abstract_operator_costs, saturation_need, state_region_intersection,
};
pub use distances::*;
use footprints::*;
pub use plan_extraction::*;
use planforge_sas::numeric_conditions::{ConditionValue, NumericConditions};
use planforge_sas::utils::interval::Interval;
pub use saturation::*;
use state_encoding::*;
pub use transition_system::*;

fn ensure_online_scp_deadline(deadline: Option<Instant>) -> Result<()> {
    crate::resource_limits::ensure_before_deadline(deadline, "online SCP")
}

#[derive(Debug, Clone)]
pub struct DomainAbstractionFactory {
    domain_mapping: DomainMapping,
    domain_sizes: Vec<usize>,
    partitions: NumericPartitions,
    numeric_domain_sizes: Vec<usize>,
    /// Shared with the task the factory was built from: the conditions are
    /// task data, not abstraction data, but the factory outlives the borrow.
    numeric_conditions: Arc<NumericConditions>,
    additive_numeric_views: AdditiveNumericViews,
    /// Per-concrete-operator metric cost, evaluated once over the initial
    /// numeric state. The cost is task-deterministic, so caching here (and
    /// sharing the `Arc` into every per-iteration `AbstractOperatorGenerator`)
    /// avoids the `task.get_operators() × assignment_effects` scan that
    /// `metric_operator_cost_from_initial_values` does on every call.
    cached_operator_costs: Arc<[f64]>,
}

impl DomainAbstractionFactory {
    pub fn new(
        task: &dyn AbstractNumericTask,
        domain_mapping: DomainMapping,
        domain_sizes: Vec<usize>,
        partitions: NumericPartitions,
        numeric_domain_sizes: Vec<usize>,
    ) -> Result<Self> {
        ensure!(
            domain_mapping.len() == domain_sizes.len(),
            "domain_mapping/domain_sizes length mismatch"
        );
        for (var, &abs_size) in domain_sizes.iter().enumerate() {
            ensure!(
                abs_size > 0,
                "non-positive abstract domain size for var {var}: {abs_size}"
            );

            let concrete_size = task
                .get_variable_domain_size(var)
                .map_err(|e| anyhow!(e.to_string()))
                .with_context(|| format!("get_variable_domain_size({var}) failed"))?;
            ensure!(
                concrete_size > 0,
                "non-positive concrete domain size for var {var}: {concrete_size}"
            );
            ensure!(
                abs_size <= concrete_size,
                "abstract domain size for var {var} exceeds concrete size ({abs_size} > {concrete_size})"
            );

            ensure!(
                domain_mapping[var].len() == concrete_size,
                "domain_mapping[{var}] has len {}, expected concrete size {concrete_size}",
                domain_mapping[var].len()
            );

            for (val, &mapped) in domain_mapping[var].iter().enumerate() {
                ensure!(
                    mapped < abs_size,
                    "domain_mapping[{var}][{val}]={mapped} out of range for abstract size {abs_size}"
                );
            }
        }
        for (n, &parts) in numeric_domain_sizes.iter().enumerate() {
            ensure!(parts > 0, "numeric_domain_sizes[{n}] must be > 0");
            let actual = partitions.partitions(n).map(|p| p.len()).unwrap_or(0);
            ensure!(
                actual == parts,
                "numeric_domain_sizes[{n}]={parts} does not match partitions len {actual}"
            );
        }

        let cached_operator_costs: Arc<[f64]> = task
            .get_operators()
            .iter()
            .map(|op| metric_operator_cost_from_initial_values(task, op))
            .collect();
        let additive_numeric_views =
            AdditiveNumericViews::for_active_dimensions(task, &numeric_domain_sizes)?;
        Ok(Self {
            domain_mapping,
            domain_sizes,
            partitions,
            numeric_domain_sizes,
            numeric_conditions: Arc::clone(task.numeric_conditions()),
            additive_numeric_views,
            cached_operator_costs,
        })
    }

    pub fn partitions(&self) -> &NumericPartitions {
        &self.partitions
    }

    pub fn domain_mapping(&self) -> &DomainMapping {
        &self.domain_mapping
    }

    pub fn domain_sizes(&self) -> &[usize] {
        &self.domain_sizes
    }

    pub fn numeric_domain_sizes(&self) -> &[usize] {
        &self.numeric_domain_sizes
    }

    /// The coordinated representation pieces changed by one CEGAR refinement.
    ///
    /// Keeping this crate-private and returning all four at once makes the
    /// factory the only mutation boundary. Read-only consumers use the named
    /// accessors above.
    pub(super) fn refinement_parts(
        &mut self,
    ) -> (
        &mut DomainMapping,
        &mut [usize],
        &mut NumericPartitions,
        &mut [usize],
    ) {
        (
            &mut self.domain_mapping,
            &mut self.domain_sizes,
            &mut self.partitions,
            &mut self.numeric_domain_sizes,
        )
    }

    pub fn numeric_conditions(&self) -> &NumericConditions {
        &self.numeric_conditions
    }

    pub fn make_operator_generator(
        &self,
        task: &dyn AbstractNumericTask,
        combine_labels: bool,
    ) -> Result<AbstractOperatorGenerator> {
        AbstractOperatorGenerator::new_with_cached_costs(
            task,
            self.domain_mapping.clone(),
            self.domain_sizes.clone(),
            self.partitions.clone(),
            self.numeric_domain_sizes.clone(),
            combine_labels,
            Arc::clone(&self.cached_operator_costs),
        )
    }
}
