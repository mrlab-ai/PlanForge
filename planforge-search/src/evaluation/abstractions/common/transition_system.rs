//! Explicit abstract transition systems and their goal distances.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::convert::Infallible;
use std::sync::Arc;

use ordered_float::NotNan;
use planforge_sas::numeric_task::OperatorIndex;
use planforge_sas::utils::float_tolerance;

use super::cartesian_set::CartesianSet;

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

/// A graph whose goal distances [`backward_goal_distances`] computes.
pub(crate) trait BackwardGraph {
    /// What a transition records in the generating table: the label of the
    /// transition a state's cheapest goal path starts with.
    type Label: Copy;

    fn num_states(&self) -> usize;

    /// Replaces `incoming` with the transitions into `target`, as
    /// `(source, label, cost)`. Costs are non-negative; a transition of
    /// infinite cost is never on a path of finite cost.
    fn incoming(&mut self, target: usize, incoming: &mut Vec<(usize, Self::Label, f64)>);
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

/// The goal distance of every state of `graph` by backward Dijkstra from
/// `seeds`, `(state, distance)` pairs -- the goal states at distance zero,
/// and states whose distance an estimate bounds -- infinite for states that
/// reach no seed. Records in `generating` the label each state's cheapest
/// path starts with. `check_limit` runs every 1024 expansions and stops the
/// search with its error; an unlimited search passes one that cannot fail.
pub(crate) fn backward_goal_distances<G: BackwardGraph, E>(
    graph: &mut G,
    seeds: impl IntoIterator<Item = (usize, f64)>,
    mut check_limit: impl FnMut() -> Result<(), E>,
    mut generating: Option<&mut [Option<G::Label>]>,
) -> Result<Vec<f64>, E> {
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
    for (state, distance) in seeds {
        assert!(
            distance >= 0.0,
            "state {state} is seeded at the distance {distance}"
        );
        if distance + float_tolerance::SEARCH_EPSILON < distances[state] {
            distances[state] = distance;
            heap.push((Reverse(not_nan(distance)), state));
        }
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
        graph.incoming(target, &mut incoming);
        for &(source, label, cost) in &incoming {
            assert!(
                cost >= 0.0,
                "a transition into state {target} has the negative cost {cost}"
            );
            if cost == f64::INFINITY {
                continue;
            }
            let alternative = distance + cost;
            if alternative + float_tolerance::SEARCH_EPSILON < distances[source] {
                distances[source] = alternative;
                if let Some(generating) = &mut generating {
                    generating[source] = Some(label);
                }
                heap.push((Reverse(not_nan(alternative)), source));
            }
        }
    }
    Ok(distances)
}

/// The limit of a search that runs to completion.
pub(crate) fn no_limit() -> Result<(), Infallible> {
    Ok(())
}

fn not_nan(distance: f64) -> NotNan<f64> {
    NotNan::new(distance).expect("goal distances are sums of non-negative costs")
}
