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
    /// The transitions into each state.
    pub backward: Vec<Vec<usize>>,
    pub goal_facts: Vec<ExplicitFact>,
    pub goal_state_hashes: Vec<usize>,
    pub initial_state_hash: usize,
    /// The region of every state, or empty if the system was built without.
    pub cartesian_sets: Vec<Arc<CartesianSet>>,
}

#[derive(Debug, Clone)]
pub struct AbstractDistanceTable {
    pub distances: Vec<f64>,
    // Per-state operator leading to a goal along a shortest path.
    pub generating_op_ids: Vec<Option<usize>>,
    pub initial_state_hash: usize,
    pub goal_facts: Vec<ExplicitFact>,
}

/// A graph whose goal distances [`backward_goal_distances`] computes.
pub(crate) trait BackwardGraph {
    /// What a transition records in the generating table: the label of the
    /// transition a state's cheapest goal path starts with.
    type Label: Copy;

    fn num_states(&self) -> usize;

    /// Replaces `incoming` with the transitions into `target`, as
    /// `(source, label, cost)`. A transition of infinite cost is never on a
    /// path of finite cost.
    fn incoming(
        &mut self,
        target: usize,
        incoming: &mut Vec<(usize, Self::Label, f64)>,
    ) -> Result<()>;
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

    fn incoming(&mut self, target: usize, incoming: &mut Vec<(usize, usize, f64)>) -> Result<()> {
        incoming.clear();
        for &transition_id in &self.system.backward[target] {
            let transition = &self.system.transitions[transition_id];
            debug_assert_eq!(transition.target_hash, target);
            incoming.push((
                transition.source_hash,
                transition.abstract_op_id,
                self.transition_costs[transition_id],
            ));
        }
        Ok(())
    }
}

/// The goal distance of every state of `graph` by backward Dijkstra from
/// `goal_states`, infinite for states that reach no goal. Records in
/// `generating` the label each state's cheapest goal path starts with.
/// `check_limit` runs every 1024 expansions and stops the search with its
/// error.
pub(crate) fn backward_goal_distances<G: BackwardGraph>(
    graph: &mut G,
    goal_states: &[usize],
    mut check_limit: impl FnMut() -> Result<()>,
    mut generating: Option<&mut [Option<G::Label>]>,
) -> Result<Vec<f64>> {
    let num_states = graph.num_states();
    if let Some(generating) = &generating {
        assert_eq!(
            generating.len(),
            num_states,
            "the generating table needs one entry per state"
        );
    }
    let mut distances = vec![f64::INFINITY; num_states];
    let mut heap = BinaryHeap::new();
    for &goal_state in goal_states {
        distances[goal_state] = 0.0;
        heap.push((
            Reverse(NotNan::new(0.0).expect("zero is not NaN")),
            goal_state,
        ));
    }
    let mut incoming = Vec::new();
    let mut expansions = 0usize;
    while let Some((Reverse(distance), target)) = heap.pop() {
        if expansions.is_multiple_of(1024) {
            check_limit()?;
        }
        expansions += 1;
        let distance = distance.into_inner();
        if distance > distances[target] + float_tolerance::SEARCH_EPSILON {
            continue;
        }
        graph.incoming(target, &mut incoming)?;
        for &(source, label, cost) in &incoming {
            ensure!(
                cost >= -float_tolerance::SEARCH_EPSILON,
                "a transition into state {target} has invalid cost {cost}"
            );
            if cost == f64::INFINITY {
                continue;
            }
            let alternative = distance + cost.max(0.0);
            if alternative + float_tolerance::SEARCH_EPSILON < distances[source] {
                distances[source] = alternative;
                if let Some(generating) = &mut generating {
                    generating[source] = Some(label);
                }
                heap.push((
                    Reverse(NotNan::new(alternative).context("goal distance is NaN")?),
                    source,
                ));
            }
        }
    }
    Ok(distances)
}
