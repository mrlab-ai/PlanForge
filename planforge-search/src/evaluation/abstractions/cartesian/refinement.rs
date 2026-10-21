use anyhow::Result;

use super::semantics::CartesianSemantics;
use super::working::{OperatorBitSet, WorkingAbstraction};
use crate::evaluation::abstractions::common::cartesian_set::CartesianSet;
use crate::evaluation::abstractions::common::cegar::split::{
    Split, SplitDimension, split_child_regions,
};
use crate::evaluation::abstractions::common::cegar::view::TransitionKey;

impl WorkingAbstraction {
    /// Splits a state by `split`: the old state keeps the child with the
    /// witness, the other child becomes a new state, whose id this returns.
    pub(super) fn apply_split(
        &mut self,
        semantics: &CartesianSemantics<'_>,
        split: Split,
    ) -> Result<usize> {
        let old_state_id = split.state_id();
        let split_dimension = split.dimension();
        let new_state_id = self.states.len();
        let children = split_child_regions(&self.states[old_state_id], &split)?;
        let (old_leaf, new_leaf) = self.hierarchy.split(
            self.leaf_node_ids[old_state_id],
            old_state_id,
            new_state_id,
            &split,
            children.witness_is_wanted_or_lower,
        )?;
        self.states[old_state_id] = children.witness;
        self.states.push(children.other);
        self.outgoing.push(Vec::new());
        self.incoming.push(Vec::new());
        self.leaf_node_ids[old_state_id] = old_leaf;
        self.leaf_node_ids.push(new_leaf);

        let operator_count = semantics.task().get_operators().len();
        let old_self_loops = std::mem::replace(
            &mut self.self_loop_operator_ids[old_state_id],
            OperatorBitSet::empty(operator_count),
        );
        let split_dependent_operators = semantics.split_dependent_operators(split_dimension);
        self.self_loop_operator_ids[old_state_id] =
            old_self_loops.clone_without(split_dependent_operators);
        self.self_loop_operator_ids
            .push(old_self_loops.clone_without(split_dependent_operators));

        let old_transitions = self.remove_incident_transitions(old_state_id);
        let replacements = replacement_transitions(
            semantics,
            SplitStates {
                split_state_id: old_state_id,
                new_state_id,
                children: [&self.states[old_state_id], &self.states[new_state_id]],
                split_dimension,
            },
            |state| &self.states[state],
            &old_transitions,
            &old_self_loops,
        )?;
        for transition in replacements {
            self.add_transition(
                transition.source,
                transition.concrete_op_id,
                transition.target,
            );
        }
        Ok(new_state_id)
    }
}

/// A state being split: its id, which its first child keeps, the id of its
/// second child, the two children's regions, and the split dimension.
#[derive(Clone, Copy)]
pub(super) struct SplitStates<'a> {
    pub(super) split_state_id: usize,
    pub(super) new_state_id: usize,
    pub(super) children: [&'a CartesianSet; 2],
    pub(super) split_dimension: SplitDimension,
}

/// The transitions that replace the transitions `incident` to the split state
/// and its `self_loops` on operators the split can affect, once the state is
/// split as `split` describes; `region` gives every other state's region.
/// The result may contain self loops and repeats.
pub(super) fn replacement_transitions<'a>(
    semantics: &CartesianSemantics<'_>,
    split: SplitStates<'a>,
    region: impl Fn(usize) -> &'a CartesianSet,
    incident: &[TransitionKey],
    self_loops: &OperatorBitSet,
) -> Result<Vec<TransitionKey>> {
    let SplitStates {
        split_state_id,
        new_state_id,
        children,
        split_dimension,
    } = split;
    let region = |state: usize| {
        if state == split_state_id {
            children[0]
        } else if state == new_state_id {
            children[1]
        } else {
            region(state)
        }
    };
    let both = [split_state_id, new_state_id];
    let mut replacements = Vec::new();
    for transition in incident {
        debug_assert!(
            transition.source != transition.target,
            "Cartesian non-loop storage contains a self loop"
        );
        let sources: &[usize] = if transition.source == split_state_id {
            &both
        } else {
            std::slice::from_ref(&transition.source)
        };
        let targets: &[usize] = if transition.target == split_state_id {
            &both
        } else {
            std::slice::from_ref(&transition.target)
        };
        for &source in sources {
            for &target in targets {
                let may_transition = if semantics
                    .operator_depends_on_split(transition.concrete_op_id, split_dimension)
                {
                    semantics.regions().may_transition(
                        region(source),
                        transition.concrete_op_id,
                        region(target),
                    )?
                } else {
                    semantics.may_transition_after_independent_split(
                        region(source),
                        transition.concrete_op_id,
                        region(target),
                        split_dimension,
                    )?
                };
                if may_transition {
                    replacements.push(TransitionKey {
                        source,
                        concrete_op_id: transition.concrete_op_id,
                        target,
                    });
                }
            }
        }
    }
    let split_dependent_operators = semantics.split_dependent_operators(split_dimension);
    for concrete_op_id in self_loops.intersection_iter(split_dependent_operators) {
        for source in both {
            let may_targets = semantics.parent_loop_source_to_split_children(
                region(source),
                concrete_op_id,
                children,
                split_dimension,
            )?;
            for (target, may_transition) in both.into_iter().zip(may_targets) {
                if may_transition {
                    replacements.push(TransitionKey {
                        source,
                        concrete_op_id,
                        target,
                    });
                }
            }
        }
    }
    Ok(replacements)
}
