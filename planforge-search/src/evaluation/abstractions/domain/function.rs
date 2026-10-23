#[cfg(test)]
mod tests;

use std::sync::Arc;

use anyhow::{Result, ensure};
use planforge_sas::numeric_conditions::NumericConditions;
use planforge_sas::numeric_task::{
    AbstractNumericTask, ExplicitValueIndex, OperatorIndex, VariableIndex,
};

use super::abstract_operators::{AbstractOperator, DomainMapping, build_abstract_operators};
use super::abstraction_numeric_var;
use super::layout::StateLayout;
use super::partitions::NumericPartitions;
use crate::evaluation::abstractions::common::cegar::semantics::CegarSemantics;
use crate::evaluation::abstractions::common::cegar::split::Split;
use crate::evaluation::abstractions::common::coordinates::AdditiveNumericView;
use crate::resource_limits::Deadline;

mod distances;
mod source_sets;
mod state_encoding;

pub(crate) use distances::DomainTransitions;
pub(crate) use source_sets::source_set_within;

/// The operations expired deadlines name.
const DISTANCE_TABLE: &str = "abstract distance table";
const TRANSITION_SYSTEM: &str = "abstract transition system";

#[derive(Debug, Clone)]
pub struct DomainAbstractionFunction {
    domain_mapping: DomainMapping,
    partitions: NumericPartitions,
    /// Derived from the abstract domain sizes of `domain_mapping` and the
    /// partitions; shared with the match trees built over it.
    layout: Arc<StateLayout>,
    /// Shared with the task the function was built from: the conditions are
    /// task data, not abstraction data, but the function outlives the borrow.
    numeric_conditions: Arc<NumericConditions>,
    /// The additive view of every derived coordinate.
    additive_numeric_views: Arc<[Option<AdditiveNumericView>]>,
    /// The validated cost of every concrete operator, shared with the CEGAR
    /// semantics. Task views that move the initial state keep these costs
    /// pinned to the base task.
    operator_costs: Arc<[f64]>,
    /// Propositional dimensions that a single abstract state can constrain,
    /// ascending: exactly the variables this abstraction split into more than
    /// one class. A variable with one class has that class covering its whole
    /// concrete domain, so every state region admits every value there.
    ///
    /// This depends only on the abstraction, never on which state is being
    /// materialized, so it is computed once and shared into every region rather
    /// than rebuilt per state -- state regions are materialized once per
    /// transition, millions of times per cost-partitioning build. It is
    /// rederived with the layout: leaving it stale makes disjoint regions
    /// compare as overlapping, which lets one operator's cost be claimed twice.
    state_constrained_props: Arc<[u32]>,
}

impl DomainAbstractionFunction {
    /// The trivial domain abstraction of the task of `semantics`: one
    /// abstract state, with every constant at its value.
    pub(crate) fn trivial(semantics: &CegarSemantics<'_>) -> Result<Self> {
        let task = semantics.task();
        let domain_mapping = task
            .variables()
            .iter()
            .map(|variable| vec![ExplicitValueIndex::new(0); variable.domain_size()])
            .collect();
        Self::new(
            semantics,
            domain_mapping,
            vec![1; task.get_num_variables()],
            NumericPartitions::trivial(task),
        )
    }

    /// The domain abstraction of the task of `semantics` that maps
    /// propositional values by `domain_mapping` into `domain_sizes` abstract
    /// values per variable, and numeric values by `partitions`. Operator
    /// costs and the additive views of derived coordinates are those of
    /// `semantics`.
    pub(crate) fn new(
        semantics: &CegarSemantics<'_>,
        domain_mapping: DomainMapping,
        domain_sizes: Vec<usize>,
        partitions: NumericPartitions,
    ) -> Result<Self> {
        let task = semantics.task();
        ensure!(
            domain_mapping.len() == domain_sizes.len()
                && domain_sizes.len() == task.get_num_variables(),
            "the domain mapping covers {} and the domain sizes {} of {} variables",
            domain_mapping.len(),
            domain_sizes.len(),
            task.get_num_variables()
        );
        for (var, &abs_size) in domain_sizes.iter().enumerate() {
            ensure!(
                abs_size == 1
                    || !task
                        .numeric_conditions()
                        .is_condition_var(VariableIndex::from_usize(var)),
                "condition variable {var} has {abs_size} abstract values; conditions are derived from the numeric partitions"
            );
            let concrete_size = task.variables()[var].domain_size();
            ensure!(
                abs_size <= concrete_size,
                "abstract domain size for var {var} exceeds concrete size ({abs_size} > {concrete_size})"
            );
            ensure!(
                domain_mapping[var].len() == concrete_size,
                "domain_mapping[{var}] has len {}, expected concrete size {concrete_size}",
                domain_mapping[var].len()
            );
            let mut mapped_onto = vec![false; abs_size];
            for (val, &mapped) in domain_mapping[var].iter().enumerate() {
                ensure!(
                    mapped.index() < abs_size,
                    "domain_mapping[{var}][{val}]={mapped} out of range for abstract size {abs_size}"
                );
                mapped_onto[mapped.index()] = true;
            }
            ensure!(
                mapped_onto.iter().all(|&mapped| mapped),
                "an abstract value of var {var} has no concrete value"
            );
        }
        let numeric_domain_sizes = partitions.numeric_domain_sizes();
        ensure!(
            numeric_domain_sizes.len() == task.numeric_variables().len(),
            "the partitions cover {} of {} numeric variables",
            numeric_domain_sizes.len(),
            task.numeric_variables().len()
        );
        let layout = StateLayout::new(&domain_sizes, &numeric_domain_sizes)?;
        let additive_numeric_views = semantics.coordinates().shared_views();
        for (numeric_var_id, &domain_size) in numeric_domain_sizes.iter().enumerate() {
            assert!(
                domain_size == 1
                    || semantics
                        .coordinates()
                        .is_coordinate(VariableIndex::from_usize(numeric_var_id)),
                "numeric variable {numeric_var_id} is partitioned but is no coordinate"
            );
        }
        let state_constrained_props = Self::derive_state_constrained_props(layout.domain_sizes());

        Ok(Self {
            domain_mapping,
            partitions,
            layout: Arc::new(layout),
            numeric_conditions: Arc::clone(task.numeric_conditions()),
            additive_numeric_views,
            operator_costs: semantics.shared_operator_costs(),
            state_constrained_props,
        })
    }

    pub fn partitions(&self) -> &NumericPartitions {
        &self.partitions
    }

    pub fn domain_mapping(&self) -> &DomainMapping {
        &self.domain_mapping
    }

    pub fn layout(&self) -> &StateLayout {
        &self.layout
    }

    pub fn domain_sizes(&self) -> &[usize] {
        self.layout.domain_sizes()
    }

    pub fn numeric_domain_sizes(&self) -> &[usize] {
        self.layout.numeric_domain_sizes()
    }

    pub(super) fn additive_numeric_views(&self) -> &[Option<AdditiveNumericView>] {
        &self.additive_numeric_views
    }

    pub(super) fn operator_cost(&self, op_id: OperatorIndex) -> f64 {
        self.operator_costs[op_id.index()]
    }

    /// The number of abstract states after `split`, exactly: it splits one
    /// abstract value of a variable in every state.
    pub(super) fn states_after_split(&self, split: &Split) -> u128 {
        self.layout.states_after_split(match split {
            Split::Propositional { var_id, .. } => var_id.index(),
            Split::Numeric { var_id, .. } => {
                abstraction_numeric_var(self.layout.domain_sizes().len(), *var_id).index()
            }
        })
    }

    /// Refine with `split` in every abstract state: a propositional split
    /// gives the wanted values of the witness's abstract value a new abstract
    /// value, a numeric split divides the partition the boundary lies in.
    pub(super) fn apply_split(
        &mut self,
        task: &dyn AbstractNumericTask,
        split: &Split,
    ) -> Result<()> {
        let mut domain_sizes = self.layout.domain_sizes().to_vec();
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
                let mapping = &mut self.domain_mapping[var_id.index()];
                let class = mapping[witness_value.index()];
                let new_class = ExplicitValueIndex::from_usize(domain_sizes[var_id.index()]);
                let mut separated = 0usize;
                let mut kept = 0usize;
                for (value, abstract_value) in mapping.iter_mut().enumerate() {
                    if *abstract_value != class {
                        continue;
                    }
                    if wanted
                        .binary_search(&ExplicitValueIndex::from_usize(value))
                        .is_ok()
                    {
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
                lattice,
                ..
            } => {
                ensure!(
                    self.partitions.split_at(
                        *var_id,
                        *boundary,
                        *lower_includes_boundary,
                        *lattice,
                    ),
                    "non-strict numeric split of variable {var_id} at {boundary}"
                );
            }
        }
        self.layout = Arc::new(StateLayout::new(
            &domain_sizes,
            &self.partitions.numeric_domain_sizes(),
        )?);
        self.state_constrained_props = Self::derive_state_constrained_props(&domain_sizes);
        Ok(())
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

    /// The abstract operators of the abstraction, with labels combined if
    /// `combine_labels`.
    pub fn abstract_operators(
        &self,
        task: &dyn AbstractNumericTask,
        combine_labels: bool,
        deadline: Deadline,
    ) -> Result<Vec<AbstractOperator>> {
        build_abstract_operators(self, task, combine_labels, deadline)
    }
}
