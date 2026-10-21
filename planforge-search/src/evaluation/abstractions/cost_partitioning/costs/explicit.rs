use anyhow::{Context, Result, ensure};

use super::{RemainingCosts, SCP_TABLE_CONSTRUCTION};
use crate::evaluation::abstractions::common::transition_system::{
    AbstractOperatorSourceSets, AbstractTransitionSystem, CostedTransitionSystem, LabelSourceSet,
    backward_goal_distances,
};
use crate::resource_limits::Deadline;

#[derive(Debug, Clone, PartialEq)]
pub struct AbstractTransitionCostFunction {
    pub transition_costs: Vec<f64>,
}

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

pub fn build_explicit_label_cost_partitioning_table(
    transition_system: &AbstractTransitionSystem,
    operator_costs: &[f64],
    cap_state_id: Option<usize>,
    deadline: Deadline,
) -> Result<(Vec<f64>, Vec<f64>)> {
    deadline.ensure(SCP_TABLE_CONSTRUCTION)?;
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
                    operator_costs
                        .get(operator_id.index())
                        .copied()
                        .with_context(|| {
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
                .map(|&operator_id| saturated[operator_id.index()])
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
    source_sets: &[AbstractOperatorSourceSets],
    residual_costs: &RemainingCosts,
    abstraction_id: usize,
    cap_state_id: Option<usize>,
    deadline: Deadline,
) -> Result<(Vec<f64>, SourceSetCostAllocation)> {
    let operator_costs = abstract_operator_costs_from_source_sets(
        source_sets.len(),
        source_sets,
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

    let allocation =
        residual_costs.source_set_allocation(abstraction_id, source_sets, &saturated, deadline)?;
    if cap_state_id.is_none() {
        return Ok((distances, allocation));
    }
    let saturated_transition_costs =
        transition_costs_from_abstract_operator_costs(transition_system, &saturated)?;
    let global_distances = build_explicit_goal_distances(
        transition_system,
        &saturated_transition_costs,
        deadline,
        None,
    )?;
    Ok((global_distances, allocation))
}

pub(crate) fn build_explicit_goal_distances(
    transition_system: &AbstractTransitionSystem,
    transition_costs: &[f64],
    deadline: Deadline,
    generating: Option<&mut [Option<usize>]>,
) -> Result<Vec<f64>> {
    ensure!(
        transition_system.transitions.len() == transition_costs.len(),
        "transition system/cost vector size mismatch: {} vs {}",
        transition_system.transitions.len(),
        transition_costs.len()
    );
    backward_goal_distances(
        &mut CostedTransitionSystem {
            system: transition_system,
            transition_costs,
        },
        &transition_system.goal_state_hashes,
        || deadline.ensure(SCP_TABLE_CONSTRUCTION),
        generating,
    )
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
            let slot = saturated.get_mut(operator_id.index()).with_context(|| {
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

fn abstract_operator_costs_from_source_sets(
    num_operators: usize,
    source_sets: &[AbstractOperatorSourceSets],
    residual_costs: &RemainingCosts,
    abstraction_id: usize,
    deadline: Deadline,
) -> Result<Vec<f64>> {
    ensure!(
        source_sets.len() >= num_operators,
        "abstract-operator region/operator size mismatch: source_sets={} operators={num_operators}",
        source_sets.len()
    );
    let has_reductions = residual_costs.has_reductions();
    let mut operator_costs = vec![f64::INFINITY; num_operators];
    for abstract_op_id in 0..num_operators {
        if abstract_op_id.is_multiple_of(64) {
            deadline.ensure(SCP_TABLE_CONSTRUCTION)?;
        }
        let source_set = &source_sets[abstract_op_id];
        ensure!(
            !source_set.labels.is_empty(),
            "abstract operator {abstract_op_id} has no concrete operator-region labels"
        );
        operator_costs[abstract_op_id] = source_set
            .labels
            .iter()
            .map(|label| {
                let residual = if has_reductions {
                    residual_costs.cost_for_source_set(abstraction_id, abstract_op_id, label)
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
