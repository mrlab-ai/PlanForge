use anyhow::{Context, Result};
use planforge_sas::numeric_task::{AbstractNumericTask, OperatorIndex};

use super::state_encoding::MatchTree;
use super::{DISTANCE_TABLE, DomainAbstractionFunction, TRANSITION_SYSTEM};
use crate::evaluation::abstractions::common::transition_system::{
    AbstractTransition, AbstractTransitionSystem, BackwardGraph, backward_goal_distances,
};
use crate::evaluation::abstractions::cost_partitioning::costs::{SaturationGraph, cheapest_label};
use crate::evaluation::abstractions::domain::abstract_operators::{
    AbstractOperator, build_abstract_operators,
};
use crate::resource_limits::Deadline;

/// The abstract operators of a domain abstraction, with the match tree that
/// finds the operators regressing from a state, and the abstract goal states.
#[derive(Debug, Clone)]
pub(crate) struct DomainTransitions {
    pub(crate) operators: Vec<AbstractOperator>,
    match_tree: MatchTree,
    /// Ascending.
    goal_states: Vec<usize>,
    num_states: usize,
}

impl DomainAbstractionFunction {
    /// The abstract operators of the abstraction, with labels combined if
    /// `combine_labels`, and its goal states.
    pub(crate) fn transitions(
        &self,
        task: &dyn AbstractNumericTask,
        combine_labels: bool,
        deadline: Deadline,
    ) -> Result<DomainTransitions> {
        let operators = build_abstract_operators(self, task, combine_labels, deadline)?;
        deadline.ensure(DISTANCE_TABLE)?;
        let goal_states = self.goal_states(task, &self.compute_abstract_goals(task), deadline)?;
        Ok(DomainTransitions {
            match_tree: MatchTree::new(&self.layout, &operators),
            operators,
            goal_states,
            num_states: self.layout.num_states(),
        })
    }
}

impl DomainTransitions {
    pub(crate) fn graph(&self) -> MatchTreeGraph<'_> {
        MatchTreeGraph::new(
            &self.operators,
            &self.match_tree,
            self.num_states,
            &self.goal_states,
        )
    }

    pub(crate) fn is_goal(&self, state: usize) -> bool {
        self.goal_states.binary_search(&state).is_ok()
    }

    /// The goal distance of every abstract state under the operators' own
    /// costs, and the abstract operator each state's cheapest goal path
    /// starts with.
    pub(crate) fn goal_distances(
        &self,
        deadline: Deadline,
    ) -> Result<(Vec<f64>, Vec<Option<usize>>)> {
        let mut generating_op_ids = vec![None; self.num_states];
        let distances = backward_goal_distances(
            &mut self.graph(),
            &self.goal_states,
            || deadline.ensure(DISTANCE_TABLE),
            Some(&mut generating_op_ids),
        )?;
        Ok((distances, generating_op_ids))
    }

    /// The explicit transition system of the operators, without self-loops.
    pub(crate) fn explicit_system(&self, deadline: Deadline) -> Result<AbstractTransitionSystem> {
        deadline.ensure(TRANSITION_SYSTEM)?;
        let mut transitions = Vec::with_capacity(self.num_states);
        let mut backward = vec![Vec::new(); self.num_states];
        let mut graph = self.graph();
        let mut incoming = Vec::new();
        for (target_state, target_backward) in backward.iter_mut().enumerate() {
            if target_state.is_multiple_of(64) {
                deadline.ensure(TRANSITION_SYSTEM)?;
            }
            graph.incoming(target_state, &mut incoming)?;
            for &(source_state, abstract_op_id, _) in &incoming {
                if source_state == target_state {
                    continue;
                }
                let transition_id = transitions.len();
                transitions.push(AbstractTransition {
                    transition_id,
                    abstract_op_id,
                    concrete_op_ids: self.operators[abstract_op_id].concrete_op_ids.clone(),
                    source_state,
                    target_state,
                });
                target_backward.push(transition_id);
            }
        }
        Ok(AbstractTransitionSystem {
            transitions,
            backward,
            goal_states: self.goal_states.clone(),
        })
    }

    /// The concrete operators that label a transition between two distinct
    /// abstract states, ascending: the labels of every operator that changes
    /// the state. Every abstract operator applies in some state of the
    /// product state space.
    pub(crate) fn relevant_operator_ids(&self) -> Vec<OperatorIndex> {
        let mut relevant = self
            .operators
            .iter()
            .filter(|operator| operator.regression_offset != 0)
            .flat_map(|operator| operator.concrete_op_ids.iter().copied())
            .collect::<Vec<_>>();
        relevant.sort_unstable();
        relevant.dedup();
        relevant
    }
}

/// The implicit transition graph of a domain abstraction: every abstract
/// operator the match tree finds applicable in a state, indexed by its
/// effects, leads into it from the state its regression offset names.
pub(crate) struct MatchTreeGraph<'a> {
    operators: &'a [AbstractOperator],
    match_tree: &'a MatchTree,
    num_states: usize,
    goal_states: &'a [usize],
    /// The cost of every abstract operator.
    costs: Vec<f64>,
    applicable: Vec<usize>,
}

impl<'a> MatchTreeGraph<'a> {
    /// The graph of `operators`, indexed by `match_tree`, over `num_states`
    /// states with goal states `goal_states`, under the operators' own costs.
    pub(crate) fn new(
        operators: &'a [AbstractOperator],
        match_tree: &'a MatchTree,
        num_states: usize,
        goal_states: &'a [usize],
    ) -> Self {
        Self {
            operators,
            match_tree,
            num_states,
            goal_states,
            costs: operators.iter().map(|op| op.cost.value()).collect(),
            applicable: Vec::new(),
        }
    }
}

impl BackwardGraph for MatchTreeGraph<'_> {
    type Label = usize;

    fn num_states(&self) -> usize {
        self.num_states
    }

    fn incoming(&mut self, target: usize, incoming: &mut Vec<(usize, usize, f64)>) -> Result<()> {
        incoming.clear();
        self.match_tree
            .get_applicable_operator_ids(target, &mut self.applicable);
        for &op_id in &self.applicable {
            let op = &self.operators[op_id];
            let source = target
                .checked_add_signed(op.regression_offset)
                .filter(|&source| source < self.num_states)
                .with_context(|| {
                    format!(
                        "abstract operator {op_id} regresses state {target} by {} out of the {} states",
                        op.regression_offset, self.num_states
                    )
                })?;
            incoming.push((source, op_id, self.costs[op_id]));
        }
        Ok(())
    }
}

impl SaturationGraph for MatchTreeGraph<'_> {
    fn goal_states(&self) -> &[usize] {
        self.goal_states
    }

    fn set_operator_costs(&mut self, operator_costs: &[f64]) {
        for (cost, op) in self.costs.iter_mut().zip(self.operators) {
            *cost = cheapest_label(&op.concrete_op_ids, operator_costs);
        }
    }

    fn label_operators(&self, op_id: usize) -> &[OperatorIndex] {
        &self.operators[op_id].concrete_op_ids
    }
}
