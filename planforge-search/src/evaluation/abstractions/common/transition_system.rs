//! Explicit abstract transition systems and their goal distances.

use std::sync::Arc;

use planforge_sas::numeric_task::OperatorIndex;

use super::cartesian_set::CartesianSet;
use crate::graph::BackwardGraph;

#[derive(Debug, Clone, PartialEq)]
pub struct AbstractTransition {
    pub transition_id: usize,
    pub abstract_op_id: usize,
    pub concrete_op_ids: Vec<OperatorIndex>,
    pub source_state: usize,
    pub target_state: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AbstractOperatorSourceSets {
    pub labels: Vec<LabelSourceSet>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LabelSourceSet {
    pub concrete_op_id: OperatorIndex,
    pub source: Arc<CartesianSet>,
}

/// An explicit abstract transition system.
#[derive(Debug, Clone, PartialEq)]
pub struct AbstractTransitionSystem {
    pub transitions: Vec<AbstractTransition>,
    /// The transitions into each state.
    pub backward: Vec<Vec<usize>>,
    /// The abstract goal states, ascending.
    pub goal_states: Vec<usize>,
}

/// An explicit transition system under per-transition costs; labels are
/// abstract operator ids.
pub(crate) struct CostedTransitionSystem<'a> {
    pub(crate) system: &'a AbstractTransitionSystem,
    pub(crate) transition_costs: &'a [f64],
}

impl BackwardGraph for CostedTransitionSystem<'_> {
    type Label = usize;

    fn num_states(&self) -> usize {
        self.system.backward.len()
    }

    fn incoming(&mut self, target: usize, incoming: &mut Vec<(usize, usize, f64)>) {
        incoming.clear();
        for &transition_id in &self.system.backward[target] {
            let transition = &self.system.transitions[transition_id];
            debug_assert_eq!(transition.target_state, target);
            incoming.push((
                transition.source_state,
                transition.abstract_op_id,
                self.transition_costs[transition_id],
            ));
        }
    }
}
