//! Goal distances in an explicit graph: the backward Dijkstra search that the
//! exact state space and every abstraction run over their transitions.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::convert::Infallible;

use ordered_float::NotNan;
use planforge_sas::utils::float_tolerance;

/// A graph whose goal distances [`backward_goal_distances`] computes.
pub trait BackwardGraph {
    /// What a transition records in the generating table: the label of the
    /// transition a state's cheapest goal path starts with.
    type Label: Copy;

    fn num_states(&self) -> usize;

    /// Replaces `incoming` with the transitions into `target`, as
    /// `(source, label, cost)`. Costs are non-negative; a transition of
    /// infinite cost is never on a path of finite cost.
    fn incoming(&mut self, target: usize, incoming: &mut Vec<(usize, Self::Label, f64)>);
}

/// The goal distance of every state of `graph` by backward Dijkstra from
/// `seeds`, `(state, distance)` pairs -- the goal states at distance zero,
/// and states whose distance an estimate bounds -- infinite for states that
/// reach no seed. Records in `generating` the label each state's cheapest
/// path starts with. `check_limit` runs every 1024 expansions and stops the
/// search with its error; an unlimited search passes one that cannot fail.
pub fn backward_goal_distances<G: BackwardGraph, E>(
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
pub fn no_limit() -> Result<(), Infallible> {
    Ok(())
}

fn not_nan(distance: f64) -> NotNan<f64> {
    NotNan::new(distance).expect("goal distances are sums of non-negative costs")
}
