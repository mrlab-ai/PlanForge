//! Saturating one abstraction: its goal distances under the remaining costs,
//! and the part of those costs the distances need (Seipp, Keller and Helmert
//! 2020). Label saturation keeps one cost per operator; regional saturation
//! keeps one per source set of every transition, so costs an abstraction
//! does not need on part of the state space stay available there.

use anyhow::{Result, ensure};
use planforge_sas::numeric_task::OperatorIndex;
use planforge_sas::utils::float_tolerance;

use super::{RemainingCosts, SCP_TABLE_CONSTRUCTION};
use crate::evaluation::abstractions::common::transition_system::{
    AbstractOperatorSourceSets, AbstractTransition, AbstractTransitionSystem, BackwardGraph,
    LabelSourceSet, backward_goal_distances,
};
use crate::resource_limits::Deadline;

#[derive(Debug, Clone)]
pub struct SourceSetCostAllocation {
    entries: Vec<SourceSetCost>,
}

#[derive(Debug, Clone)]
pub struct SourceSetCost {
    pub source_set: LabelSourceSet,
    pub amount: f64,
}

impl SourceSetCostAllocation {
    pub fn new(entries: Vec<SourceSetCost>) -> Self {
        Self { entries }
    }

    pub fn entries(&self) -> &[SourceSetCost] {
        &self.entries
    }
}

/// A transition graph label saturation runs on. Every label names the
/// concrete operators that induce the transitions it labels.
pub(crate) trait SaturationGraph: BackwardGraph<Label = usize> {
    fn goal_states(&self) -> &[usize];

    /// Costs every transition the cheapest of its operators' `operator_costs`.
    fn set_operator_costs(&mut self, operator_costs: &[f64]);

    fn label_operators(&self, label: usize) -> &[OperatorIndex];
}

/// An explicit transition system under one cost per transition; labels are
/// transition ids.
pub(crate) struct ExplicitSaturationGraph<'a> {
    system: &'a AbstractTransitionSystem,
    transition_costs: Vec<f64>,
}

impl<'a> ExplicitSaturationGraph<'a> {
    pub(crate) fn new(system: &'a AbstractTransitionSystem) -> Self {
        Self {
            system,
            transition_costs: vec![0.0; system.transitions.len()],
        }
    }
}

impl BackwardGraph for ExplicitSaturationGraph<'_> {
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
                transition_id,
                self.transition_costs[transition_id],
            ));
        }
        Ok(())
    }
}

impl SaturationGraph for ExplicitSaturationGraph<'_> {
    fn goal_states(&self) -> &[usize] {
        &self.system.goal_state_hashes
    }

    fn set_operator_costs(&mut self, operator_costs: &[f64]) {
        for (cost, transition) in self
            .transition_costs
            .iter_mut()
            .zip(&self.system.transitions)
        {
            *cost = cheapest_label(&transition.concrete_op_ids, operator_costs);
        }
    }

    fn label_operators(&self, transition_id: usize) -> &[OperatorIndex] {
        &self.system.transitions[transition_id].concrete_op_ids
    }
}

/// The cheapest of `operators` under `operator_costs`.
pub(crate) fn cheapest_label(operators: &[OperatorIndex], operator_costs: &[f64]) -> f64 {
    assert!(!operators.is_empty(), "a transition has a label");
    operators
        .iter()
        .map(|operator| operator_costs[operator.index()])
        .fold(f64::INFINITY, f64::min)
}

/// The goal distances of `graph` under `operator_costs`, and the saturated
/// cost of every operator: the most any transition it labels needs,
/// `h(source) - h(target)`.
///
/// With a `perimeter_state`, saturation sees only the states no farther from
/// the goal than that state (the perimeter), and the returned distances are
/// those under the saturated costs, so they stay valid outside it.
pub(crate) fn saturate_labels<G: SaturationGraph>(
    graph: &mut G,
    operator_costs: &[f64],
    perimeter_state: Option<usize>,
    deadline: Deadline,
) -> Result<(Vec<f64>, Vec<f64>)> {
    graph.set_operator_costs(operator_costs);
    let distances = goal_distances(graph, deadline)?;
    let saturated = saturated_operator_costs(
        graph,
        operator_costs.len(),
        &perimeter(&distances, perimeter_state),
        deadline,
    )?;
    if perimeter_state.is_none() {
        return Ok((distances, saturated));
    }
    graph.set_operator_costs(&saturated);
    Ok((goal_distances(graph, deadline)?, saturated))
}

/// Regional saturation of the explicit transition system `system`, whose
/// transitions' labels have the source sets `source_sets` of their abstract
/// operators.
///
/// A transition's label costs what remains of the operator's cost on the
/// part of the source set the transition leaves from, which
/// `transition_source_set` narrows the abstract operator's source set to.
/// Every transition saturates on its own, so the returned allocation charges
/// each label's source set what its transition needs. With a
/// `perimeter_state` saturation is limited as in [`saturate_labels`].
pub(crate) fn saturate_regions(
    system: &AbstractTransitionSystem,
    source_sets: &[AbstractOperatorSourceSets],
    mut transition_source_set: impl FnMut(
        &AbstractTransition,
        &LabelSourceSet,
    ) -> Result<LabelSourceSet>,
    remaining: &RemainingCosts,
    perimeter_state: Option<usize>,
    deadline: Deadline,
) -> Result<(Vec<f64>, SourceSetCostAllocation)> {
    let mut graph = ExplicitSaturationGraph::new(system);
    for (transition_id, transition) in system.transitions.iter().enumerate() {
        if transition_id.is_multiple_of(1024) {
            deadline.ensure(SCP_TABLE_CONSTRUCTION)?;
        }
        let mut cost = f64::INFINITY;
        for source_set in label_source_sets(source_sets, transition) {
            let source_set = transition_source_set(transition, source_set)?;
            cost = cost.min(remaining.cost_for_source_set(&source_set));
        }
        graph.transition_costs[transition_id] = cost;
    }
    let distances = goal_distances(&mut graph, deadline)?;
    let saturation_distances = perimeter(&distances, perimeter_state);

    let mut saturated = vec![0.0; system.transitions.len()];
    let mut entries = Vec::new();
    for (transition_id, transition) in system.transitions.iter().enumerate() {
        if transition_id.is_multiple_of(1024) {
            deadline.ensure(SCP_TABLE_CONSTRUCTION)?;
        }
        let Some(need) = saturation_need(
            saturation_distances[transition.source_hash],
            saturation_distances[transition.target_hash],
            graph.transition_costs[transition_id],
            "saturated transition cost",
        )?
        else {
            continue;
        };
        saturated[transition_id] = need;
        if need <= float_tolerance::SEARCH_EPSILON {
            continue;
        }
        for source_set in label_source_sets(source_sets, transition) {
            if remaining.base_cost(source_set.concrete_op_id) <= float_tolerance::SEARCH_EPSILON {
                continue;
            }
            let source_set = transition_source_set(transition, source_set)?;
            let available = remaining.cost_for_source_set(&source_set);
            ensure!(
                need <= available + float_tolerance::AGREEMENT_EPSILON,
                "transition {transition_id} needs {need} of operator {}, which has only {available} left on its source set",
                source_set.concrete_op_id
            );
            entries.push(SourceSetCost {
                source_set,
                amount: need,
            });
        }
    }
    let allocation = SourceSetCostAllocation::new(entries);
    if perimeter_state.is_none() {
        return Ok((distances, allocation));
    }
    graph.transition_costs = saturated;
    Ok((goal_distances(&mut graph, deadline)?, allocation))
}

/// The source sets of `transition`'s labels.
fn label_source_sets<'a>(
    source_sets: &'a [AbstractOperatorSourceSets],
    transition: &'a AbstractTransition,
) -> impl Iterator<Item = &'a LabelSourceSet> {
    let operator_source_sets = &source_sets[transition.abstract_op_id].labels;
    transition
        .concrete_op_ids
        .iter()
        .map(move |&concrete_op_id| {
            operator_source_sets
                .iter()
                .find(|source_set| source_set.concrete_op_id == concrete_op_id)
                .expect("an abstract operator has a source set for each of its labels")
        })
}

fn goal_distances<G: SaturationGraph>(graph: &mut G, deadline: Deadline) -> Result<Vec<f64>> {
    let goal_states = graph.goal_states().to_vec();
    backward_goal_distances(
        graph,
        &goal_states,
        || deadline.ensure(SCP_TABLE_CONSTRUCTION),
        None,
    )
}

/// `distances` limited to the perimeter of `perimeter_state`: the states
/// farther from the goal than it, or unsolvable, are left out (`-inf`). An
/// unsolvable perimeter state limits nothing.
fn perimeter(distances: &[f64], perimeter_state: Option<usize>) -> Vec<f64> {
    let mut limited = distances.to_vec();
    if let Some(state) = perimeter_state {
        let limit = distances[state];
        if limit.is_finite() {
            for distance in &mut limited {
                if !distance.is_finite() || *distance > limit {
                    *distance = f64::NEG_INFINITY;
                }
            }
        }
    }
    limited
}

/// The saturated cost of every operator for `distances` of `graph`: the most
/// any transition it labels needs, over every transition into a state of
/// finite distance and not only the shortest-path ones.
fn saturated_operator_costs<G: SaturationGraph>(
    graph: &mut G,
    num_operators: usize,
    distances: &[f64],
    deadline: Deadline,
) -> Result<Vec<f64>> {
    let mut saturated = vec![0.0_f64; num_operators];
    let mut incoming = Vec::new();
    for (target, &target_distance) in distances.iter().enumerate() {
        if target.is_multiple_of(1024) {
            deadline.ensure(SCP_TABLE_CONSTRUCTION)?;
        }
        if !target_distance.is_finite() {
            continue;
        }
        graph.incoming(target, &mut incoming)?;
        for &(source, label, cost) in &incoming {
            let Some(need) = saturation_need(
                distances[source],
                target_distance,
                cost,
                "saturated operator cost",
            )?
            else {
                continue;
            };
            for &operator in graph.label_operators(label) {
                let slot = &mut saturated[operator.index()];
                *slot = slot.max(need);
            }
        }
    }
    Ok(saturated)
}

/// What a transition from goal distance `source_h` to `target_h` needs of
/// its cost `residual_budget`; `None` if either end has no finite distance.
pub(crate) fn saturation_need(
    source_h: f64,
    target_h: f64,
    residual_budget: f64,
    context: &str,
) -> Result<Option<f64>> {
    if !source_h.is_finite() || !target_h.is_finite() {
        return Ok(None);
    }
    let needed = (source_h - target_h).max(0.0);
    ensure!(
        needed <= residual_budget + float_tolerance::AGREEMENT_EPSILON,
        "{context} {needed} exceeds residual cost {residual_budget}"
    );
    Ok(Some(needed))
}
