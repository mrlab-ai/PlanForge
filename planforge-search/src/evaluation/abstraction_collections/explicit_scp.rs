use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use ordered_float::NotNan;
use planforge_sas::numeric_task::ExplicitFact;
use planforge_sas::utils::float_tolerance;

use super::region::{
    StateRegion, TransitionRegion, merge_transition_region, transition_region_key,
};
use super::{MAX_ABSTRACT_OPERATOR_REDUCTION_PIECES, TransitionResidualCosts};

#[derive(Debug, Clone, PartialEq)]
pub struct AbstractTransition {
    pub transition_id: usize,
    pub abstract_op_id: usize,
    pub concrete_op_ids: Vec<usize>,
    pub source_hash: usize,
    pub target_hash: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AbstractOperatorFootprint {
    pub labels: Vec<ConcreteOperatorFootprint>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConcreteOperatorFootprint {
    pub concrete_op_id: usize,
    pub source_region: Arc<StateRegion>,
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
    pub state_regions: Vec<Arc<StateRegion>>,
}

impl AbstractTransitionSystem {
    pub fn transition_region(&self, transition: &AbstractTransition) -> Result<TransitionRegion> {
        let source = self
            .state_regions
            .get(transition.source_hash)
            .with_context(|| {
                format!(
                    "missing source state region {} for transition {}",
                    transition.source_hash, transition.transition_id
                )
            })?
            .clone();
        let target = self
            .state_regions
            .get(transition.target_hash)
            .with_context(|| {
                format!(
                    "missing target state region {} for transition {}",
                    transition.target_hash, transition.transition_id
                )
            })?
            .clone();
        Ok(TransitionRegion { source, target })
    }

    pub fn abstract_operator_regions(&self) -> Vec<Option<TransitionRegion>> {
        assert!(
            !self.state_regions.is_empty(),
            "abstract transition system has no materialized state regions"
        );
        let num_abstract_ops = self
            .transitions
            .iter()
            .map(|transition| transition.abstract_op_id)
            .max()
            .map_or(0, |max_id| max_id + 1);
        let mut regions: Vec<Option<TransitionRegion>> = vec![None; num_abstract_ops];
        for transition in &self.transitions {
            let source = self.state_regions[transition.source_hash].clone();
            let target = self.state_regions[transition.target_hash].clone();
            let transition_region = TransitionRegion { source, target };
            match &mut regions[transition.abstract_op_id] {
                Some(region) => merge_transition_region(region, &transition_region),
                None => regions[transition.abstract_op_id] = Some(transition_region),
            }
        }
        regions
    }

    pub fn abstract_operator_region_covers(&self) -> Vec<Vec<TransitionRegion>> {
        assert!(
            !self.state_regions.is_empty(),
            "abstract transition system has no materialized state regions"
        );
        let num_abstract_ops = self
            .transitions
            .iter()
            .map(|transition| transition.abstract_op_id)
            .max()
            .map_or(0, |max_id| max_id + 1);
        let mut covers = vec![Vec::new(); num_abstract_ops];
        let mut seen = vec![std::collections::HashSet::new(); num_abstract_ops];
        for transition in &self.transitions {
            let region = TransitionRegion {
                source: self.state_regions[transition.source_hash].clone(),
                target: self.state_regions[transition.target_hash].clone(),
            };
            let key = transition_region_key(&region);
            if seen[transition.abstract_op_id].insert(key) {
                covers[transition.abstract_op_id].push(region);
            }
        }
        for (abstract_op_id, cover) in covers.iter_mut().enumerate() {
            if cover.len() > MAX_ABSTRACT_OPERATOR_REDUCTION_PIECES {
                let mut hull = cover[0].clone();
                for region in cover.iter().skip(1) {
                    merge_transition_region(&mut hull, region);
                }
                tracing::debug!(
                    "abstract operator {abstract_op_id} reduction cover exceeded {} pieces; using hull fallback",
                    MAX_ABSTRACT_OPERATOR_REDUCTION_PIECES
                );
                cover.clear();
                cover.push(hull);
            }
        }
        covers
    }

    pub fn concrete_operator_ids_by_abstract_operator(&self) -> Vec<Vec<usize>> {
        let num_abstract_ops = self
            .transitions
            .iter()
            .map(|transition| transition.abstract_op_id)
            .max()
            .map_or(0, |max_id| max_id + 1);
        let mut concrete_op_ids = vec![Vec::new(); num_abstract_ops];
        for transition in &self.transitions {
            concrete_op_ids[transition.abstract_op_id]
                .extend(transition.concrete_op_ids.iter().copied());
        }
        for ids in &mut concrete_op_ids {
            ids.sort_unstable();
            ids.dedup();
        }
        concrete_op_ids
    }

    pub(super) fn transition_counts_by_abstract_operator(
        &self,
        num_abstract_ops: usize,
    ) -> Vec<usize> {
        let mut counts = vec![0usize; num_abstract_ops];
        for transition in &self.transitions {
            if let Some(count) = counts.get_mut(transition.abstract_op_id) {
                *count = count.saturating_add(1);
            }
        }
        counts
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AbstractTransitionCostFunction {
    pub transition_costs: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AbstractOperatorCostFunction {
    pub operator_costs: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct RegionalCostAllocation {
    entries: Vec<RegionalCostAllocationEntry>,
}

#[derive(Debug, Clone)]
pub struct RegionalCostAllocationEntry {
    pub footprint: ConcreteOperatorFootprint,
    pub amount: f64,
}

impl RegionalCostAllocation {
    pub fn new(entries: Vec<RegionalCostAllocationEntry>) -> Self {
        Self { entries }
    }

    pub fn entries(&self) -> &[RegionalCostAllocationEntry] {
        &self.entries
    }
}

pub fn build_explicit_label_cost_partitioning_table(
    transition_system: &AbstractTransitionSystem,
    operator_costs: &[f64],
    cap_state_id: Option<usize>,
    deadline: Option<Instant>,
) -> Result<(Vec<f64>, Vec<f64>)> {
    ensure_scp_table_deadline(deadline)?;
    let transition_costs = transition_system
        .transitions
        .iter()
        .map(|transition| {
            ensure!(
                !transition.concrete_op_ids.is_empty(),
                "abstract transition {} has no concrete operator labels",
                transition.transition_id
            );
            transition
                .concrete_op_ids
                .iter()
                .map(|&operator_id| {
                    operator_costs.get(operator_id).copied().with_context(|| {
                        format!("missing residual cost for concrete operator {operator_id}")
                    })
                })
                .collect::<Result<Vec<_>>>()
                .map(|costs| costs.into_iter().fold(f64::INFINITY, f64::min))
        })
        .collect::<Result<Vec<_>>>()?;
    let distances =
        build_explicit_goal_distances(transition_system, &transition_costs, deadline, None)?;
    let saturation_table = capped_saturation_table(&distances, cap_state_id)?;
    let saturated = saturated_label_costs(
        transition_system,
        &transition_costs,
        operator_costs.len(),
        &saturation_table,
    )?;

    if cap_state_id.is_none() {
        return Ok((distances, saturated));
    }
    let saturated_transition_costs = transition_system
        .transitions
        .iter()
        .map(|transition| {
            transition
                .concrete_op_ids
                .iter()
                .map(|&operator_id| saturated[operator_id])
                .fold(f64::INFINITY, f64::min)
        })
        .collect::<Vec<_>>();
    let global_distances = build_explicit_goal_distances(
        transition_system,
        &saturated_transition_costs,
        deadline,
        None,
    )?;
    Ok((global_distances, saturated))
}

pub fn build_explicit_regional_cost_partitioning_table(
    transition_system: &AbstractTransitionSystem,
    footprints: &[AbstractOperatorFootprint],
    residual_costs: &TransitionResidualCosts,
    abstraction_id: usize,
    cap_state_id: Option<usize>,
    deadline: Option<Instant>,
) -> Result<(Vec<f64>, AbstractOperatorCostFunction)> {
    let operator_costs = abstract_operator_costs_from_footprints(
        footprints.len(),
        footprints,
        residual_costs,
        abstraction_id,
        deadline,
    )?;
    let transition_costs =
        transition_costs_from_abstract_operator_costs(transition_system, &operator_costs)?;
    let distances =
        build_explicit_goal_distances(transition_system, &transition_costs, deadline, None)?;
    let saturation_table = capped_saturation_table(&distances, cap_state_id)?;
    let saturated =
        saturated_abstract_operator_costs(transition_system, &operator_costs, &saturation_table)?;

    if cap_state_id.is_none() {
        return Ok((
            distances,
            AbstractOperatorCostFunction {
                operator_costs: saturated,
            },
        ));
    }
    let saturated_transition_costs =
        transition_costs_from_abstract_operator_costs(transition_system, &saturated)?;
    let global_distances = build_explicit_goal_distances(
        transition_system,
        &saturated_transition_costs,
        deadline,
        None,
    )?;
    Ok((
        global_distances,
        AbstractOperatorCostFunction {
            operator_costs: saturated,
        },
    ))
}

pub(super) fn ensure_scp_table_deadline(deadline: Option<Instant>) -> Result<()> {
    crate::resource_limits::ensure_before_deadline(deadline, "SCP table construction")
}

pub(crate) fn build_explicit_goal_distances(
    transition_system: &AbstractTransitionSystem,
    transition_costs: &[f64],
    deadline: Option<Instant>,
    generating: Option<&mut [Option<usize>]>,
) -> Result<Vec<f64>> {
    backward_goal_distances(
        transition_system,
        transition_costs,
        || ensure_scp_table_deadline(deadline),
        generating,
    )
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

fn capped_saturation_table(distances: &[f64], cap_state_id: Option<usize>) -> Result<Vec<f64>> {
    let Some(cap_state_id) = cap_state_id else {
        return Ok(distances.to_vec());
    };
    let h_cap = distances.get(cap_state_id).copied().with_context(|| {
        format!(
            "perimeter cap state {cap_state_id} out of bounds for {} states",
            distances.len()
        )
    })?;
    let mut capped = distances.to_vec();
    if h_cap.is_finite() {
        for value in &mut capped {
            if !value.is_finite() || *value > h_cap {
                *value = f64::NEG_INFINITY;
            }
        }
    }
    Ok(capped)
}

fn saturated_label_costs(
    transition_system: &AbstractTransitionSystem,
    transition_costs: &[f64],
    num_operators: usize,
    distances: &[f64],
) -> Result<Vec<f64>> {
    let mut saturated = vec![0.0_f64; num_operators];
    for transition in &transition_system.transitions {
        let source_h = distances[transition.source_hash];
        let target_h = distances[transition.target_hash];
        let Some(needed) = saturation_need(
            source_h,
            target_h,
            transition_costs[transition.transition_id],
            "saturated transition cost",
        )?
        else {
            continue;
        };
        for &operator_id in &transition.concrete_op_ids {
            let slot = saturated.get_mut(operator_id).with_context(|| {
                format!("transition references missing concrete operator {operator_id}")
            })?;
            *slot = slot.max(needed);
        }
    }
    Ok(saturated)
}

fn transition_costs_from_abstract_operator_costs(
    transition_system: &AbstractTransitionSystem,
    operator_costs: &[f64],
) -> Result<Vec<f64>> {
    transition_system
        .transitions
        .iter()
        .map(|transition| {
            operator_costs
                .get(transition.abstract_op_id)
                .copied()
                .with_context(|| {
                    format!(
                        "transition {} references missing abstract operator {}",
                        transition.transition_id, transition.abstract_op_id
                    )
                })
        })
        .collect()
}

pub(crate) fn saturated_abstract_operator_costs(
    transition_system: &AbstractTransitionSystem,
    operator_costs: &[f64],
    distances: &[f64],
) -> Result<Vec<f64>> {
    let mut saturated = vec![0.0_f64; operator_costs.len()];
    for transition in &transition_system.transitions {
        let source_h = distances[transition.source_hash];
        let target_h = distances[transition.target_hash];
        let operator_cost = *operator_costs
            .get(transition.abstract_op_id)
            .with_context(|| {
                format!(
                    "transition {} references missing abstract operator {}",
                    transition.transition_id, transition.abstract_op_id
                )
            })?;
        let Some(needed) = saturation_need(
            source_h,
            target_h,
            operator_cost,
            "saturated abstract-operator cost",
        )?
        else {
            continue;
        };
        saturated[transition.abstract_op_id] = saturated[transition.abstract_op_id].max(needed);
    }
    Ok(saturated)
}

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
        needed <= residual_budget + 1e-7,
        "{context} {needed} exceeds residual cost {residual_budget}"
    );
    Ok(Some(needed))
}

fn abstract_operator_costs_from_footprints(
    num_operators: usize,
    footprints: &[AbstractOperatorFootprint],
    residual_costs: &TransitionResidualCosts,
    abstraction_id: usize,
    deadline: Option<Instant>,
) -> Result<Vec<f64>> {
    ensure!(
        footprints.len() >= num_operators,
        "abstract-operator footprint/operator size mismatch: footprints={} operators={num_operators}",
        footprints.len()
    );
    let has_reductions = residual_costs.has_reductions();
    let mut operator_costs = vec![f64::INFINITY; num_operators];
    for abstract_op_id in 0..num_operators {
        if abstract_op_id.is_multiple_of(64) {
            ensure_scp_table_deadline(deadline)?;
        }
        let footprint = &footprints[abstract_op_id];
        ensure!(
            !footprint.labels.is_empty(),
            "abstract operator {abstract_op_id} has no concrete footprint labels"
        );
        operator_costs[abstract_op_id] = footprint
            .labels
            .iter()
            .map(|label| {
                let residual = if has_reductions {
                    residual_costs.cost_for_operator_footprint(
                        abstraction_id,
                        abstract_op_id,
                        label,
                    )
                } else {
                    residual_costs.base_cost(label.concrete_op_id)
                };
                residual.min(residual_costs.base_cost(label.concrete_op_id))
            })
            .fold(f64::INFINITY, f64::min);
        ensure!(
            operator_costs[abstract_op_id].is_finite(),
            "residual cost for abstract operator {abstract_op_id} is not finite"
        );
    }
    Ok(operator_costs)
}
