//! A domain abstraction with its goal distances, as CEGAR's flaw search sees
//! it.

use std::borrow::Cow;
use std::collections::HashSet;

use anyhow::{Context, Result, ensure};
use planforge_sas::numeric_task::{
    AbstractNumericTask, ExplicitValueIndex, NumericValue, OperatorIndex, VariableIndex,
};

use super::abstract_operators::AbstractOperator;
use super::factory::DomainAbstractionFactory;
use crate::evaluation::abstractions::common::cartesian_set::CartesianSet;
use crate::evaluation::abstractions::common::cegar::split::Split;
use crate::evaluation::abstractions::common::cegar::view::{AbstractionView, TransitionKey};
use crate::evaluation::abstractions::common::transition_system::AbstractDistanceTable;

pub(super) struct DomainView<'a> {
    pub(super) task: &'a dyn AbstractNumericTask,
    pub(super) factory: &'a DomainAbstractionFactory,
    pub(super) numeric_domain_sizes: &'a [usize],
    pub(super) hash_multipliers: &'a [usize],
    pub(super) operators: &'a [AbstractOperator],
    pub(super) table: &'a AbstractDistanceTable,
    pub(super) limits: RefinementLimits<'a>,
    /// Plan steps execute any label of their abstract operator, not just
    /// the first.
    pub(super) wildcard_plans: bool,
}

/// What a domain abstraction may still refine: a split must keep it within
/// `max_states` abstract states and leave the blacklisted variables alone.
#[derive(Clone, Copy)]
pub(super) struct RefinementLimits<'a> {
    pub(super) max_states: usize,
    pub(super) blacklisted_prop_var_ids: &'a HashSet<VariableIndex>,
    pub(super) blacklisted_numeric_var_ids: &'a HashSet<VariableIndex>,
}

impl DomainView<'_> {
    /// The abstract state count, which splitting a dimension of `values`
    /// values multiplies by `(values + 1) / values`.
    fn size_after_splitting_dimension_of(&self, values: usize) -> u128 {
        let states = self.table.distances.len() as u128;
        let values = values as u128;
        states / values * (values + 1)
    }
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
        self.factory.abstract_state_hash(
            propositions,
            numeric,
            self.numeric_domain_sizes,
            self.hash_multipliers,
        )
    }

    fn region(&self, state: usize) -> Cow<'_, CartesianSet> {
        Cow::Owned(
            self.factory
                .cartesian_set_from_hash(state, self.numeric_domain_sizes, self.hash_multipliers)
                .expect("every abstract state hash decodes to a region"),
        )
    }

    fn goal_distance(&self, state: usize) -> f64 {
        self.table.distances[state]
    }

    /// Evaluated on demand: the flaw search only asks about the states along
    /// the abstract plan.
    fn is_goal(&self, state: usize) -> Result<bool> {
        self.factory.is_goal_state(
            self.task,
            state,
            &self.table.goal_facts,
            self.numeric_domain_sizes,
            self.hash_multipliers,
        )
    }

    fn optimal_transition(&self, state: usize) -> Result<TransitionKey> {
        let op_id = self.table.generating_op_ids[state].with_context(|| {
            format!(
                "non-goal abstract state {state} with finite distance has no generating operator"
            )
        })?;
        let op = &self.operators[op_id];
        let target = (state as i64 - op.hash_effect as i64) as usize;
        ensure!(
            target < self.table.distances.len(),
            "abstract operator {op_id} leads from state {state} out of the abstraction"
        );
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
        let values = match split {
            Split::Propositional { var_id, .. } => self.factory.domain_sizes()[var_id.index()],
            Split::Numeric { var_id, .. } => self.numeric_domain_sizes[var_id.index()],
        };
        Ok(self.size_after_splitting_dimension_of(values))
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

    fn refinable(&self, split: &Split) -> bool {
        let blacklisted = match split {
            Split::Propositional { var_id, .. } => {
                self.limits.blacklisted_prop_var_ids.contains(var_id)
            }
            Split::Numeric { var_id, .. } => {
                self.limits.blacklisted_numeric_var_ids.contains(var_id)
            }
        };
        !blacklisted
            && self
                .growth(split)
                .is_ok_and(|states| states <= self.limits.max_states as u128)
    }
}
