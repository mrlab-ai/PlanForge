#[cfg(test)]
mod tests;

#[cfg(any(test, debug_assertions))]
use std::collections::HashSet;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, ensure};

use planforge_sas::numeric_task::{
    AbstractNumericTask, AssignmentOperation, ExplicitFact, ExplicitValueIndex, NumericTaskExt,
    NumericType, NumericValue, Operator, OperatorIndex, VariableIndex,
};
use planforge_sas::utils::float_tolerance;

use super::abstract_operators::{AbstractOperator, AbstractOperatorGenerator, DomainMapping};
use super::abstraction::abstract_state_count;
use super::abstraction_numeric_var;
use super::comparison_inputs::{
    AbstractStateHash, prepare_comparison_tree_inputs_from_abstract_state,
};
use super::debug_dump::dump_distances;
use super::partitions::NumericPartitions;
use super::partitions::partition_for_value;
use crate::evaluation::abstractions::common::cegar::split::Split;
use crate::evaluation::abstractions::common::coordinates::AdditiveNumericViews;

mod distances;
mod saturation;
mod source_sets;
mod state_encoding;
mod transition_system;

use crate::evaluation::abstractions::common::cartesian_set::{
    CartesianSet, cartesian_set_intersection,
};
use crate::evaluation::abstractions::common::transition_system::{
    AbstractDistanceTable, AbstractOperatorSourceSets, AbstractTransition,
    AbstractTransitionSystem, BackwardGraph, CostedTransitionSystem, LabelSourceSet,
    backward_goal_distances,
};
use crate::evaluation::abstractions::cost_partitioning::costs::{
    AbstractTransitionCostFunction, RemainingCosts, SourceSetCost, SourceSetCostAllocation,
    saturation_need,
};
use crate::resource_limits::Deadline;
pub use distances::*;
use planforge_sas::numeric_conditions::{ConditionValue, NumericConditions};
use planforge_sas::utils::interval::Interval;
use saturation::*;
use source_sets::*;
use state_encoding::*;
pub use transition_system::*;

/// The operation an expired online SCP deadline names.
const ONLINE_SCP: &str = "online SCP";

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
    /// Per-concrete-operator cost, read once through the task's
    /// [`NumericTaskExt::abstract_operator_cost`] hook and shared (via `Arc`)
    /// into every per-iteration `AbstractOperatorGenerator`. Task views that
    /// move the initial state keep these costs pinned to the base task.
    cached_operator_costs: Arc<[NumericValue]>,
    /// Propositional dimensions that a single abstract state can constrain,
    /// ascending: exactly the variables this abstraction split into more than
    /// one class. A variable with one class has that class covering its whole
    /// concrete domain, so every state region admits every value there.
    ///
    /// This depends only on the abstraction, never on which state is being
    /// materialized, so it is computed once and shared into every region rather
    /// than rebuilt per state -- state regions are materialized once per
    /// transition, millions of times per cost-partitioning build.
    state_constrained_props: Arc<[u32]>,
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
            ensure!(
                abs_size == 1
                    || !task
                        .numeric_conditions()
                        .is_condition_var(VariableIndex::from_usize(var)),
                "condition variable {var} has {abs_size} abstract values; conditions are derived from the numeric partitions"
            );

            let concrete_size = task
                .get_variable_domain_size(VariableIndex::from_usize(var))
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
                    mapped.index() < abs_size,
                    "domain_mapping[{var}][{val}]={mapped} out of range for abstract size {abs_size}"
                );
            }
        }
        for (n, &parts) in numeric_domain_sizes.iter().enumerate() {
            ensure!(parts > 0, "numeric_domain_sizes[{n}] must be > 0");
            let actual = partitions
                .partitions(VariableIndex::from_usize(n))
                .with_context(|| format!("numeric variable {n} has no partitions"))?
                .len();
            ensure!(
                actual == parts,
                "numeric_domain_sizes[{n}]={parts} does not match partitions len {actual}"
            );
        }

        let cached_operator_costs: Arc<[NumericValue]> = (0..task.get_operators().len())
            .map(|operator_id| task.abstract_operator_cost(OperatorIndex::from_usize(operator_id)))
            .collect();
        let additive_numeric_views =
            AdditiveNumericViews::for_active_dimensions(task, &numeric_domain_sizes)?;
        let state_constrained_props = Self::derive_state_constrained_props(&domain_sizes);

        Ok(Self {
            domain_mapping,
            domain_sizes,
            partitions,
            numeric_domain_sizes,
            numeric_conditions: Arc::clone(task.numeric_conditions()),
            additive_numeric_views,
            cached_operator_costs,
            state_constrained_props,
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

    /// Refine the abstraction in place. This is the factory's only mutation
    /// boundary; read-only consumers use the accessors above.
    ///
    /// Takes a closure rather than handing out the four `&mut` parts so that
    /// [`Self::state_constrained_props`] is always rederived afterwards. That
    /// list is derived from `domain_sizes`, and a refinement that splits a
    /// variable changes which variables a state region constrains; leaving it
    /// stale makes disjoint regions compare as overlapping, which silently lets
    /// one operator's cost be claimed twice. A caller cannot forget to refresh
    /// it because a caller never gets the chance.
    pub(super) fn refine<R>(
        &mut self,
        refinement: impl FnOnce(
            &mut DomainMapping,
            &mut [usize],
            &mut NumericPartitions,
            &mut [usize],
        ) -> R,
    ) -> R {
        let result = refinement(
            &mut self.domain_mapping,
            &mut self.domain_sizes,
            &mut self.partitions,
            &mut self.numeric_domain_sizes,
        );
        self.state_constrained_props = Self::derive_state_constrained_props(&self.domain_sizes);
        result
    }

    /// Refine with `split` in every abstract state: a propositional split
    /// gives the wanted values of the witness's abstract value a new abstract
    /// value, a numeric split cuts the partition the boundary lies in.
    pub(super) fn apply_split(
        &mut self,
        task: &dyn AbstractNumericTask,
        split: &Split,
    ) -> Result<()> {
        self.refine(|domain_mapping, domain_sizes, partitions, numeric_domain_sizes| {
            match split {
                Split::Propositional {
                    var_id,
                    wanted,
                    witness_value,
                    ..
                } => {
                    ensure!(
                        !task.numeric_conditions().is_condition_var(*var_id),
                        "split of condition variable {var_id}; conditions are derived from the numeric partitions"
                    );
                    let mapping = &mut domain_mapping[var_id.index()];
                    let class = mapping[witness_value.index()];
                    let new_class = ExplicitValueIndex::from_usize(domain_sizes[var_id.index()]);
                    let mut separated = 0usize;
                    let mut kept = 0usize;
                    for (value, abstract_value) in mapping.iter_mut().enumerate() {
                        if *abstract_value != class {
                            continue;
                        }
                        if wanted.binary_search(&ExplicitValueIndex::from_usize(value)).is_ok() {
                            *abstract_value = new_class;
                            separated += 1;
                        } else {
                            kept += 1;
                        }
                    }
                    ensure!(
                        separated > 0 && kept > 0,
                        "non-strict propositional split of variable {var_id}: wanted={wanted:?}"
                    );
                    domain_sizes[var_id.index()] += 1;
                }
                Split::Numeric {
                    var_id,
                    boundary,
                    lower_includes_boundary,
                    integer_lattice,
                    ..
                } => {
                    ensure!(
                        partitions.split_at(
                            *var_id,
                            *boundary,
                            *lower_includes_boundary,
                            *integer_lattice,
                        ),
                        "non-strict numeric split of variable {var_id} at {boundary}"
                    );
                    numeric_domain_sizes[var_id.index()] = partitions
                        .partitions(*var_id)
                        .expect("a split numeric variable has partitions")
                        .len();
                }
            }
            Ok(())
        })
    }

    /// Propositional variables an abstract state can constrain: exactly those
    /// this abstraction split into more than one class. A variable with a single
    /// class has that class covering its whole concrete domain.
    fn derive_state_constrained_props(domain_sizes: &[usize]) -> Arc<[u32]> {
        domain_sizes
            .iter()
            .enumerate()
            .filter(|(_, abstract_size)| **abstract_size > 1)
            .map(|(var, _)| u32::try_from(var).expect("propositional var id exceeds u32"))
            .collect::<Vec<_>>()
            .into()
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
