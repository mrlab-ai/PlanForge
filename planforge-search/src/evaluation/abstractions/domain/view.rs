//! A domain abstraction with its goal distances, as CEGAR's flaw search sees
//! it.

use std::borrow::Cow;

use anyhow::{Context, Result};
use planforge_sas::numeric_task::{
    AbstractNumericTask, ExplicitValueIndex, NumericValue, OperatorIndex,
};

use super::abstract_operators::AbstractOperator;
use super::config::DomainAbstractionConfig;
use super::factory::DomainAbstractionFactory;
use crate::evaluation::abstractions::common::cartesian_set::CartesianSet;
use crate::evaluation::abstractions::common::cegar::refine::Refinable;
use crate::evaluation::abstractions::common::cegar::split::Split;
use crate::evaluation::abstractions::common::cegar::view::{
    AbstractionView, RefinementLimits, TransitionKey,
};
use crate::evaluation::abstractions::common::transition_system::AbstractDistanceTable;
use crate::resource_limits::Deadline;

pub(super) struct DomainView<'a> {
    pub(super) task: &'a dyn AbstractNumericTask,
    pub(super) factory: &'a DomainAbstractionFactory,
    pub(super) operators: &'a [AbstractOperator],
    pub(super) table: &'a AbstractDistanceTable,
    pub(super) limits: RefinementLimits<'a>,
    /// Plan steps execute any label of their abstract operator, not just
    /// the first.
    pub(super) wildcard_plans: bool,
}

impl AbstractionView for DomainView<'_> {
    /// The number of abstract states after the split.
    type Growth = u128;

    fn num_states(&self) -> usize {
        self.table.distances.len()
    }

    fn abstract_state(
        &self,
        propositions: &[ExplicitValueIndex],
        numeric: &[NumericValue],
    ) -> Result<usize> {
        self.factory.abstract_state(propositions, numeric)
    }

    fn region(&self, state: usize) -> Cow<'_, CartesianSet> {
        Cow::Owned(self.factory.state_region(state))
    }

    fn goal_distance(&self, state: usize) -> f64 {
        self.table.distances[state]
    }

    /// Evaluated on demand: the flaw search only asks about the states along
    /// the abstract plan.
    fn is_goal(&self, state: usize) -> Result<bool> {
        self.factory
            .is_goal_state(self.task, state, &self.table.goal_facts, &mut Vec::new())
    }

    fn optimal_transition(&self, state: usize) -> Result<TransitionKey> {
        let op_id = self.table.generating_op_ids[state].with_context(|| {
            format!(
                "non-goal abstract state {state} with finite distance has no generating operator"
            )
        })?;
        let op = &self.operators[op_id];
        let target = state
            .checked_add_signed(-op.regression_offset)
            .filter(|&target| target < self.table.distances.len())
            .with_context(|| {
                format!("abstract operator {op_id} leads from state {state} out of the abstraction")
            })?;
        let concrete_op_id = *op
            .concrete_op_ids
            .iter()
            .min()
            .with_context(|| format!("abstract operator {op_id} has no concrete operator"))?;
        Ok(TransitionKey {
            source: state,
            concrete_op_id,
            target,
        })
    }

    fn growth(&self, split: &Split) -> Result<u128> {
        Ok(self.states_after(split))
    }

    fn limits(&self) -> RefinementLimits<'_> {
        self.limits
    }

    /// Splitting a dimension of `values` values multiplies the abstract state
    /// count by `(values + 1) / values`.
    fn states_after(&self, split: &Split) -> u128 {
        self.factory.states_after_split(split)
    }

    fn step_operators(&self, transition: TransitionKey) -> Vec<OperatorIndex> {
        if !self.wildcard_plans {
            return vec![transition.concrete_op_id];
        }
        let op_id = self.table.generating_op_ids[transition.source]
            .expect("an optimal transition leaves through its generating operator");
        let mut labels = self.operators[op_id].concrete_op_ids.clone();
        labels.sort_unstable();
        labels.dedup();
        labels
    }
}

/// A domain abstraction under refinement. A split refines a variable in every
/// abstract state, so every iteration rebuilds the abstract operators and the
/// goal distances.
pub(super) struct DomainRefinement<'a> {
    pub(super) task: &'a dyn AbstractNumericTask,
    pub(super) factory: DomainAbstractionFactory,
    pub(super) limits: RefinementLimits<'a>,
    pub(super) combine_labels: bool,
    pub(super) wildcard_plans: bool,
    /// The operators and goal distances of the current abstraction, built by
    /// [`Refinable::view`].
    built: Option<(Vec<AbstractOperator>, AbstractDistanceTable)>,
}

impl<'a> DomainRefinement<'a> {
    pub(super) fn new(
        task: &'a dyn AbstractNumericTask,
        factory: DomainAbstractionFactory,
        limits: RefinementLimits<'a>,
        config: &DomainAbstractionConfig,
    ) -> Self {
        Self {
            task,
            factory,
            limits,
            combine_labels: config.cegar.combine_labels,
            wildcard_plans: config.use_wildcard_plans,
            built: None,
        }
    }

    /// The abstraction, with the operators and goal distances its last view
    /// built if no split followed.
    pub(super) fn into_parts(
        self,
    ) -> (
        DomainAbstractionFactory,
        Option<(Vec<AbstractOperator>, AbstractDistanceTable)>,
    ) {
        (self.factory, self.built)
    }
}

impl Refinable for DomainRefinement<'_> {
    type View<'b>
        = DomainView<'b>
    where
        Self: 'b;

    fn view(&mut self, deadline: Deadline) -> Result<DomainView<'_>> {
        let operators =
            self.factory
                .abstract_operators(self.task, self.combine_labels, deadline)?;
        let table = self
            .factory
            .build_distance_table(self.task, &operators, deadline)?;
        let (operators, table) = self.built.insert((operators, table));
        Ok(DomainView {
            task: self.task,
            factory: &self.factory,
            operators,
            table,
            limits: self.limits,
            wildcard_plans: self.wildcard_plans,
        })
    }

    fn apply_split(&mut self, split: Split) -> Result<()> {
        self.built = None;
        self.factory.apply_split(self.task, &split)
    }
}
