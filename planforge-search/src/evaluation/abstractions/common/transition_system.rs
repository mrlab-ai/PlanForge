//! Explicit abstract transition systems and their goal distances.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use ordered_float::NotNan;
use planforge_sas::numeric_task::{ExplicitFact, OperatorIndex};
use planforge_sas::utils::float_tolerance;

use super::cartesian_set::CartesianSet;

#[derive(Debug, Clone, PartialEq)]
pub struct AbstractTransition {
    pub transition_id: usize,
    pub abstract_op_id: usize,
    pub concrete_op_ids: Vec<OperatorIndex>,
    pub source_hash: usize,
    pub target_hash: usize,
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

#[derive(Debug, Clone, PartialEq)]
pub struct AbstractTransitionSystem {
    pub transitions: Vec<AbstractTransition>,
    pub duplicate_transition_attempts: usize,
    pub backward: Vec<Vec<usize>>,
    pub forward: Vec<Vec<usize>>,
    pub goal_facts: Vec<ExplicitFact>,
    pub goal_state_hashes: Vec<usize>,
    pub initial_state_hash: usize,
    pub hash_multipliers: Vec<usize>,
    pub numeric_domain_sizes: Vec<usize>,
    pub cartesian_sets: Vec<Arc<CartesianSet>>,
}

#[derive(Debug, Clone)]
pub struct AbstractDistanceTable {
    pub distances: Vec<f64>,
    // Per-state operator leading to a goal along a shortest path.
    pub generating_op_ids: Vec<Option<usize>>,
    pub initial_state_hash: usize,
    pub goal_facts: Vec<ExplicitFact>,
    pub hash_multipliers: Vec<usize>,
    pub numeric_domain_sizes: Vec<usize>,
}

pub(crate) trait BackwardDijkstraGraph {
    fn num_states(&self) -> usize;
    fn goal_state_ids(&self) -> &[usize];
    fn incoming_transition_ids(&self, target_id: usize) -> Option<&[usize]>;
    fn transition_endpoints(&self, transition_id: usize) -> Option<(usize, usize)>;
    fn transition_operator_id(&self, transition_id: usize) -> Option<usize>;
    fn num_transitions(&self) -> usize;
}

impl BackwardDijkstraGraph for AbstractTransitionSystem {
    fn num_states(&self) -> usize {
        self.backward.len()
    }

    fn goal_state_ids(&self) -> &[usize] {
        &self.goal_state_hashes
    }

    fn incoming_transition_ids(&self, target_id: usize) -> Option<&[usize]> {
        self.backward.get(target_id).map(Vec::as_slice)
    }

    fn transition_endpoints(&self, transition_id: usize) -> Option<(usize, usize)> {
        self.transitions
            .get(transition_id)
            .map(|transition| (transition.source_hash, transition.target_hash))
    }

    fn transition_operator_id(&self, transition_id: usize) -> Option<usize> {
        self.transitions
            .get(transition_id)
            .map(|transition| transition.abstract_op_id)
    }

    fn num_transitions(&self) -> usize {
        self.transitions.len()
    }
}

/// Backward Dijkstra shared by abstract transition systems and the concrete
/// reachable-state graph. The graph adapter owns the representation choice;
/// this routine owns cost validation, epsilon handling, and relaxation.
pub(crate) fn backward_goal_distances<G, F>(
    graph: &G,
    transition_costs: &[f64],
    mut check_limit: F,
    mut generating: Option<&mut [Option<usize>]>,
) -> Result<Vec<f64>>
where
    G: BackwardDijkstraGraph,
    F: FnMut() -> Result<()>,
{
    ensure!(
        graph.num_transitions() == transition_costs.len(),
        "transition system/cost vector size mismatch: {} vs {}",
        graph.num_transitions(),
        transition_costs.len()
    );
    let num_states = graph.num_states();
    if let Some(generating) = &generating {
        ensure!(
            generating.len() == num_states,
            "generating-operator table/state count mismatch: {} vs {num_states}",
            generating.len()
        );
    }
    let mut distances = vec![f64::INFINITY; num_states];
    let mut heap = BinaryHeap::new();
    for &goal_state_id in graph.goal_state_ids() {
        ensure!(
            goal_state_id < num_states,
            "goal state id {goal_state_id} out of bounds for {num_states} states"
        );
        distances[goal_state_id] = 0.0;
        heap.push((
            Reverse(NotNan::new(0.0).expect("zero is not NaN")),
            goal_state_id,
        ));
    }
    let mut expansions = 0usize;
    while let Some((Reverse(distance), target_id)) = heap.pop() {
        if expansions.is_multiple_of(1024) {
            check_limit()?;
        }
        expansions += 1;
        let distance = distance.into_inner();
        if distance > distances[target_id] + float_tolerance::SEARCH_EPSILON {
            continue;
        }
        let predecessors = graph
            .incoming_transition_ids(target_id)
            .with_context(|| format!("missing predecessor list for state {target_id}"))?;
        for &transition_id in predecessors {
            let (source_id, transition_target_id) = graph
                .transition_endpoints(transition_id)
                .with_context(|| format!("missing transition {transition_id}"))?;
            ensure!(
                transition_target_id == target_id,
                "backward transition {transition_id} targets {transition_target_id}, expected {target_id}",
            );
            let cost = transition_costs[transition_id];
            ensure!(
                cost.is_finite() && cost >= -float_tolerance::SEARCH_EPSILON,
                "transition {transition_id} has invalid cost {cost}"
            );
            let alternative = distance + cost.max(0.0);
            let source_distance = distances.get_mut(source_id).with_context(|| {
                format!(
                    "transition {transition_id} source {source_id} out of bounds for {num_states} states",
                )
            })?;
            if alternative + float_tolerance::SEARCH_EPSILON < *source_distance {
                *source_distance = alternative;
                if let Some(generating) = &mut generating {
                    generating[source_id] = Some(
                        graph
                            .transition_operator_id(transition_id)
                            .with_context(|| {
                                format!("transition {transition_id} has no operator id")
                            })?,
                    );
                }
                heap.push((
                    Reverse(NotNan::new(alternative).context("abstract distance is NaN")?),
                    source_id,
                ));
            }
        }
    }
    Ok(distances)
}
