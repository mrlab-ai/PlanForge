//! A domain abstraction with its goal distances, as CEGAR's flaw search sees
//! it.

use std::borrow::Cow;

use anyhow::{Context, Result};
use planforge_sas::numeric_task::{
    AbstractNumericTask, ExplicitValueIndex, NumericValue, OperatorIndex,
};

use super::config::DomainAbstractionConfig;
use super::factory::{DomainAbstractionFactory, DomainTransitions};
use crate::evaluation::abstractions::common::cartesian_set::CartesianSet;
use crate::evaluation::abstractions::common::cegar::refine::Refinable;
use crate::evaluation::abstractions::common::cegar::split::Split;
use crate::evaluation::abstractions::common::cegar::view::{
    AbstractionView, RefinementLimits, TransitionKey,
};
use crate::resource_limits::Deadline;

pub(super) struct DomainView<'a> {
    pub(super) factory: &'a DomainAbstractionFactory,
    pub(super) built: &'a DomainBuild,
    pub(super) limits: RefinementLimits<'a>,
    /// Plan steps execute any label of their abstract operator, not just
    /// the first.
    pub(super) wildcard_plans: bool,
}

impl AbstractionView for DomainView<'_> {
    /// The number of abstract states after the split.
    type Growth = u128;

    fn num_states(&self) -> usize {
        self.built.distances.len()
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
        self.built.distances[state]
    }

    fn is_goal(&self, state: usize) -> Result<bool> {
        Ok(self.built.transitions.is_goal(state))
    }

    fn optimal_transition(&self, state: usize) -> Result<TransitionKey> {
        let op_id = self.built.generating_op_ids[state].with_context(|| {
            format!(
                "non-goal abstract state {state} with finite distance has no generating operator"
            )
        })?;
        let op = &self.built.transitions.operators[op_id];
        let target = state
            .checked_add_signed(-op.regression_offset)
            .filter(|&target| target < self.built.distances.len())
            .with_context(|| {
                format!("abstract operator {op_id} leads from state {state} out of the abstraction")
            })?;
        let concrete_op_id = *op
            .concrete_op_ids
            .first()
            .expect("every abstract operator has a label");
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

    /// With wildcard plans, every label of the transition's abstract
    /// operator, ascending.
    fn step_operators<'a>(&'a self, transition: &'a TransitionKey) -> &'a [OperatorIndex] {
        if !self.wildcard_plans {
            return std::slice::from_ref(&transition.concrete_op_id);
        }
        let op_id = self.built.generating_op_ids[transition.source]
            .expect("an optimal transition leaves through its generating operator");
        &self.built.transitions.operators[op_id].concrete_op_ids
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
    /// The transitions and goal distances of the current abstraction, built
    /// by [`Refinable::view`].
    built: Option<DomainBuild>,
}

/// The transitions of a domain abstraction and the goal distances under
/// them, with the abstract operator each state's cheapest goal path starts
/// with.
#[derive(Debug)]
pub(super) struct DomainBuild {
    pub(super) transitions: DomainTransitions,
    pub(super) distances: Vec<f64>,
    pub(super) generating_op_ids: Vec<Option<usize>>,
}

impl DomainBuild {
    pub(super) fn new(transitions: DomainTransitions, deadline: Deadline) -> Result<Self> {
        let (distances, generating_op_ids) = transitions.goal_distances(deadline)?;
        Ok(Self {
            transitions,
            distances,
            generating_op_ids,
        })
    }
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

    /// The abstraction, with the transitions and goal distances its last
    /// view built if no split followed.
    pub(super) fn into_parts(self) -> (DomainAbstractionFactory, Option<DomainBuild>) {
        (self.factory, self.built)
    }
}

impl Refinable for DomainRefinement<'_> {
    type View<'b>
        = DomainView<'b>
    where
        Self: 'b;

    fn view(&mut self, deadline: Deadline) -> Result<DomainView<'_>> {
        let transitions = self
            .factory
            .transitions(self.task, self.combine_labels, deadline)?;
        let built = self.built.insert(DomainBuild::new(transitions, deadline)?);
        Ok(DomainView {
            factory: &self.factory,
            built,
            limits: self.limits,
            wildcard_plans: self.wildcard_plans,
        })
    }

    fn apply_split(&mut self, split: Split) -> Result<()> {
        self.built = None;
        self.factory.apply_split(self.task, &split)
    }
}
