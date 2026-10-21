use std::collections::HashMap;
use std::sync::Arc;

use planforge_sas::numeric_task::OperatorIndex;

use super::abstraction::{CartesianAbstraction, CartesianTransitions};
use super::distances::ShortestPaths;
use super::semantics::CartesianSemantics;
use super::working::WorkingAbstraction;
use crate::evaluation::abstractions::common::abstraction::{AbstractionCore, AbstractionMetadata};
use crate::evaluation::abstractions::common::cegar::options::CegarOptions;
use crate::evaluation::abstractions::common::transition_system::{
    AbstractTransition, AbstractTransitionSystem,
};

/// The abstraction CEGAR refined into `working`, with the goal distances of
/// `shortest_paths`. Its abstract operators are the transitions between
/// distinct states, the labels of each source and target pair combined into
/// one if `options.combine_labels`; they are kept if
/// `options.keep_transition_system`.
pub(super) fn finalize_abstraction(
    working: WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    shortest_paths: &ShortestPaths,
    initial_state: usize,
    options: &CegarOptions,
    metadata: AbstractionMetadata,
) -> CartesianAbstraction {
    let num_states = working.states().len();
    assert_eq!(
        shortest_paths.distances().len(),
        num_states,
        "the shortest paths cover every Cartesian state"
    );
    // Self-loops need nothing on a shortest path nor of a saturated cost;
    // CEGAR keeps them only because a later split can turn one into a
    // transition between the children.
    let mut operators: Vec<(usize, usize, Vec<OperatorIndex>)> = if options.combine_labels {
        let mut grouped: HashMap<(usize, usize), Vec<OperatorIndex>> = HashMap::new();
        for transition_id in working.active_transition_ids() {
            let transition = working.transition(transition_id);
            grouped
                .entry((transition.source, transition.target))
                .or_default()
                .push(transition.concrete_op_id);
        }
        grouped
            .into_iter()
            .map(|((source, target), mut labels)| {
                labels.sort_unstable();
                labels.dedup();
                (source, target, labels)
            })
            .collect()
    } else {
        working
            .active_transition_ids()
            .map(|transition_id| {
                let transition = working.transition(transition_id);
                (
                    transition.source,
                    transition.target,
                    vec![transition.concrete_op_id],
                )
            })
            .collect()
    };
    operators.sort_unstable();
    let mut relevant_operator_ids = operators
        .iter()
        .flat_map(|(source, target, labels)| {
            assert_ne!(source, target, "CEGAR keeps no self-loop as a transition");
            labels.iter().copied()
        })
        .collect::<Vec<_>>();
    relevant_operator_ids.sort_unstable();
    relevant_operator_ids.dedup();
    let core = AbstractionCore {
        distances: shortest_paths.distances().to_vec(),
        initial_state,
        relevant_operator_ids,
        num_abstract_operators: operators.len(),
        metadata,
    };
    let transitions = options.keep_transition_system.then(|| {
        let mut backward = vec![Vec::new(); num_states];
        let transitions = operators
            .into_iter()
            .enumerate()
            .map(|(transition_id, (source, target, labels))| {
                backward[target].push(transition_id);
                AbstractTransition {
                    transition_id,
                    abstract_op_id: transition_id,
                    concrete_op_ids: labels,
                    source_state: source,
                    target_state: target,
                }
            })
            .collect();
        CartesianTransitions {
            system: AbstractTransitionSystem {
                transitions,
                backward,
                goal_states: shortest_paths.goal_states(),
            },
            regions: working.states.into_iter().map(Arc::new).collect(),
            source_sets: None,
        }
    });
    CartesianAbstraction::new(
        working.hierarchy,
        semantics.cegar().coordinates().shared_views(),
        core,
        transitions,
    )
}
