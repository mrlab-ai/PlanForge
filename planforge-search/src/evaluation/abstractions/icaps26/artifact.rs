//! Split-selection policies from Schindler, Speck, and Helmert (ICAPS 2026).
//!
//! The artifact constructs one Cartesian abstraction for the original task,
//! replays the first flawed abstract plan, and selects a split randomly or by
//! the number of values excluded from the desired child. The generator owns
//! replay and refinement; this module keeps the artifact-specific policy
//! explicit so native collection generation does not silently inherit it.

use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Icaps26SplitSelection {
    Random,
    MinUnwanted,
    MaxUnwanted,
}

/// The artifact uses `std::mt19937`; preserving its stream makes RANDOM a
/// reproducible artifact policy rather than a new Rust-specific policy.
pub(super) struct ArtifactMt19937 {
    state: [u32; 624],
    index: usize,
}

impl ArtifactMt19937 {
    pub(super) fn new(seed: u32) -> Self {
        let mut state = [0; 624];
        state[0] = seed;
        for index in 1..state.len() {
            let previous = state[index - 1];
            state[index] = 1_812_433_253_u32
                .wrapping_mul(previous ^ (previous >> 30))
                .wrapping_add(index as u32);
        }
        Self { state, index: 624 }
    }

    fn next_u32(&mut self) -> u32 {
        if self.index == self.state.len() {
            for index in 0..self.state.len() {
                let joined = (self.state[index] & 0x8000_0000)
                    | (self.state[(index + 1) % self.state.len()] & 0x7fff_ffff);
                let mut twisted = joined >> 1;
                if joined & 1 != 0 {
                    twisted ^= 0x9908_b0df;
                }
                self.state[index] = self.state[(index + 397) % self.state.len()] ^ twisted;
            }
            self.index = 0;
        }
        let mut value = self.state[self.index];
        self.index += 1;
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c_5680;
        value ^= (value << 15) & 0xefc6_0000;
        value ^ (value >> 18)
    }

    pub(super) fn uniform_index(&mut self, bound: usize) -> usize {
        assert!(bound > 0, "cannot sample from an empty ICAPS split set");
        let range = u32::try_from(bound).expect("ICAPS split set exceeds the artifact RNG range");
        // This is libstdc++'s unbiased 32-bit uniform_int_distribution.
        let threshold = range.wrapping_neg() % range;
        loop {
            let product = u64::from(self.next_u32()) * u64::from(range);
            if product as u32 >= threshold {
                return (product >> 32) as usize;
            }
        }
    }
}

pub(super) fn icaps26_retire_arc(
    working: &mut WorkingAbstraction,
    transition_id: usize,
) -> WorkingTransition {
    assert!(
        working.transition_ids_by_key.is_none(),
        "ICAPS arc refinement requires unindexed transitions"
    );
    working.transitions[transition_id]
        .take()
        .expect("ICAPS adjacency references a removed transition")
}

pub(super) fn icaps26_release_arc_slot(working: &mut WorkingAbstraction, transition_id: usize) {
    assert!(
        working.transitions[transition_id].is_none(),
        "ICAPS retired transition slot is occupied"
    );
    working.free_transition_ids.push(transition_id);
}

pub(super) fn icaps26_swap_remove_arc(adjacency: &mut Vec<usize>, transition_id: usize) {
    let position = adjacency
        .iter()
        .position(|&candidate| candidate == transition_id)
        .expect("ICAPS counterpart adjacency is missing an arc");
    adjacency.swap_remove(position);
}

pub(super) fn add_icaps26_propositional_loop_replacements(
    working: &mut WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    op_id: OperatorIndex,
    var_id: VariableIndex,
    old_state_id: usize,
    new_state_id: usize,
) {
    let op = &semantics.task.get_operators()[op_id.index()];
    let pre: Option<ExplicitValueIndex> = op
        .preconditions()
        .iter()
        .find(|fact| fact.var_index() == var_id)
        .map(ExplicitFact::value_index);
    let effect = op
        .effects()
        .iter()
        .find(|effect| effect.var_id() == var_id)
        .map(|effect| effect.value());
    let post = effect.or(pre);
    let old_contains = |value: usize| {
        working.states[old_state_id].propositions()[var_id.index()]
            .binary_search(&ExplicitValueIndex::from_usize(value))
            .is_ok()
    };

    match (pre, post) {
        (None, None) => {
            working.add_transition(old_state_id, op_id, old_state_id);
            working.add_transition(new_state_id, op_id, new_state_id);
        }
        (None, Some(post)) if !old_contains(post.index()) => {
            working.add_transition(old_state_id, op_id, new_state_id);
            working.add_transition(new_state_id, op_id, new_state_id);
        }
        (None, Some(post)) => {
            assert!(old_contains(post.index()));
            working.add_transition(old_state_id, op_id, old_state_id);
            working.add_transition(new_state_id, op_id, old_state_id);
        }
        (Some(pre), Some(post)) if old_contains(pre.index()) => {
            if old_contains(post.index()) {
                working.add_transition(old_state_id, op_id, old_state_id);
            } else {
                working.add_transition(old_state_id, op_id, new_state_id);
            }
        }
        (Some(_), Some(post)) => {
            if old_contains(post.index()) {
                working.add_transition(new_state_id, op_id, old_state_id);
            } else {
                working.add_transition(new_state_id, op_id, new_state_id);
            }
        }
        (Some(_), None) => unreachable!("a prevail condition defines its own post value"),
    }
}

pub(super) fn apply_icaps26_transition_split(
    working: &mut WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    old_state_id: usize,
    new_state_id: usize,
    split_dimension: SplitDimension,
    old_loop_order: Vec<OperatorIndex>,
) -> Result<()> {
    let old_incoming = std::mem::take(&mut working.incoming[old_state_id]);
    for transition_id in old_incoming {
        let transition = icaps26_retire_arc(working, transition_id);
        debug_assert_eq!(transition.target, old_state_id);
        let source = transition.source;
        for target in [old_state_id, new_state_id] {
            if semantics.split_dimension_may_transition(
                &working.states[source],
                transition.concrete_op_id,
                &working.states[target],
                split_dimension,
            )? {
                working.add_transition(source, transition.concrete_op_id, target);
            }
        }
        icaps26_swap_remove_arc(&mut working.outgoing[source], transition_id);
        icaps26_release_arc_slot(working, transition_id);
    }

    let old_outgoing = std::mem::take(&mut working.outgoing[old_state_id]);
    for transition_id in old_outgoing {
        let transition = icaps26_retire_arc(working, transition_id);
        debug_assert_eq!(transition.source, old_state_id);
        let target = transition.target;
        for source in [old_state_id, new_state_id] {
            if semantics.operator_may_apply(&working.states[source], transition.concrete_op_id)?
                && semantics.split_dimension_may_transition(
                    &working.states[source],
                    transition.concrete_op_id,
                    &working.states[target],
                    split_dimension,
                )?
            {
                working.add_transition(source, transition.concrete_op_id, target);
            }
        }
        icaps26_swap_remove_arc(&mut working.incoming[target], transition_id);
        icaps26_release_arc_slot(working, transition_id);
    }

    for op_id in old_loop_order {
        match split_dimension {
            SplitDimension::Propositional(var_id) => {
                add_icaps26_propositional_loop_replacements(
                    working,
                    semantics,
                    op_id,
                    var_id,
                    old_state_id,
                    new_state_id,
                );
            }
            SplitDimension::Numeric(_) => {
                let old_targets = semantics.parent_loop_source_to_split_children(
                    &working.states[old_state_id],
                    op_id,
                    [&working.states[old_state_id], &working.states[new_state_id]],
                    split_dimension,
                )?;
                let new_targets = semantics.parent_loop_source_to_split_children(
                    &working.states[new_state_id],
                    op_id,
                    [&working.states[old_state_id], &working.states[new_state_id]],
                    split_dimension,
                )?;
                if old_targets[0] {
                    working.add_transition(old_state_id, op_id, old_state_id);
                }
                if new_targets[1] {
                    working.add_transition(new_state_id, op_id, new_state_id);
                }
                if old_targets[1] {
                    working.add_transition(old_state_id, op_id, new_state_id);
                }
                if new_targets[0] {
                    working.add_transition(new_state_id, op_id, old_state_id);
                }
            }
        }
    }
    Ok(())
}
