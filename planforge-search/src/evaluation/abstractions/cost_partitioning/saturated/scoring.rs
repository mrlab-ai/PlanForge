use std::collections::BTreeMap;

use planforge_sas::utils::float_tolerance;

use crate::evaluation::abstractions::common::cartesian_set::CartesianSet;
use crate::evaluation::abstractions::component::AbstractionComponent;
use crate::evaluation::evaluator::EvaluationError;

/// The greedy-order score of an abstraction (Seipp 2017): its heuristic
/// value per cost it steals from the others.
pub(super) fn compute_score(h: f64, stolen_costs: f64) -> f64 {
    h / stolen_costs.max(1.0)
}

/// How much of an operator's cost an abstraction that wants `wanted` takes
/// from the others, given the operator's `surplus` (its cost minus everything
/// all abstractions want).
pub(super) fn compute_stolen_costs(wanted: f64, surplus: f64) -> f64 {
    // Saturated costs never exceed finite operator costs, but are -inf for an
    // abstraction that is indifferent to the operator: it wants nothing, so it
    // steals nothing.
    if wanted == f64::NEG_INFINITY {
        return 0.0;
    }
    assert!(
        wanted.is_finite(),
        "saturated costs are finite or -inf, got {wanted}"
    );
    // `compute_surplus_cost` reports +inf for an operator whose cost or total
    // finite want is not finite; nothing is short there, so nothing is stolen.
    if surplus == f64::INFINITY {
        return 0.0;
    }
    assert!(
        surplus.is_finite(),
        "surplus costs are finite or +inf, got {surplus}"
    );
    let rest = surplus + wanted;
    if rest >= 0.0 {
        (0.0_f64).max(wanted - rest)
    } else {
        wanted.max(rest)
    }
}

pub(super) fn compute_costs_stolen_by_heuristic(saturated: &[f64], surplus: &[f64]) -> f64 {
    saturated
        .iter()
        .zip(surplus.iter())
        .map(|(&s, &su)| compute_stolen_costs(s, su))
        .sum()
}

/// The source sets of one component's labels, grouped by concrete operator
/// in increasing operator order.
type SourcesByOperator<'a> = Vec<(usize, Vec<&'a CartesianSet>)>;

/// The regional greedy-order score of every component with source sets (the
/// stolen costs of [`compute_score`]); `None` for a pattern database. Two
/// components compete for an operator when together they saturate more of
/// it than it costs and their source sets for it overlap; each is charged
/// the excess. The sum runs over component pairs and then operators in
/// increasing order, so it is the same in every run.
pub(super) fn region_stolen_costs(
    components: &[AbstractionComponent<'_>],
    saturated_by_component: &[Vec<f64>],
    operator_costs: &[f64],
) -> Result<Vec<Option<f64>>, EvaluationError> {
    let mut sources_by_component = Vec::with_capacity(components.len());
    for (component_id, component) in components.iter().enumerate() {
        let Some(abstraction) = component.cegar() else {
            sources_by_component.push(None);
            continue;
        };
        let source_sets = abstraction.abstract_source_sets().ok_or_else(|| {
            EvaluationError::ComputationFailed(format!(
                "region SCP component {component_id} has no source sets"
            ))
        })?;
        let expected_source_sets = abstraction.num_abstract_operators();
        if source_sets.len() != expected_source_sets {
            return Err(EvaluationError::ComputationFailed(format!(
                "region SCP component {component_id} has {} operator regions for {expected_source_sets} retained abstract transitions/operators",
                source_sets.len()
            )));
        }
        let mut by_operator = BTreeMap::<usize, Vec<&CartesianSet>>::new();
        for label in source_sets.iter().flat_map(|source_set| &source_set.labels) {
            if label.concrete_op_id.index() >= operator_costs.len() {
                return Err(EvaluationError::ComputationFailed(format!(
                    "region SCP component {component_id} operator region references missing operator {}",
                    label.concrete_op_id
                )));
            }
            by_operator
                .entry(label.concrete_op_id.index())
                .or_default()
                .push(&label.source);
        }
        sources_by_component.push(Some(by_operator.into_iter().collect::<SourcesByOperator>()));
    }

    let mut scores = sources_by_component
        .iter()
        .map(|sources| sources.as_ref().map(|_| 0.0))
        .collect::<Vec<_>>();
    for left in 0..components.len() {
        let Some(left_by_operator) = &sources_by_component[left] else {
            continue;
        };
        for right in left + 1..components.len() {
            let Some(right_by_operator) = &sources_by_component[right] else {
                continue;
            };
            let conflict = pairwise_component_stolen_cost(
                left_by_operator,
                &saturated_by_component[left],
                right_by_operator,
                &saturated_by_component[right],
                operator_costs,
            );
            *scores[left]
                .as_mut()
                .expect("a component with source sets has a score") += conflict;
            *scores[right]
                .as_mut()
                .expect("a component with source sets has a score") += conflict;
        }
    }
    Ok(scores)
}

/// What two components steal from each other over the operators both have
/// source sets for, merged in increasing operator order.
fn pairwise_component_stolen_cost(
    left_by_operator: &SourcesByOperator<'_>,
    left_saturated: &[f64],
    right_by_operator: &SourcesByOperator<'_>,
    right_saturated: &[f64],
    operator_costs: &[f64],
) -> f64 {
    let mut conflict = 0.0;
    let mut right_iter = right_by_operator.iter().peekable();
    for (operator_id, left_sources) in left_by_operator {
        while right_iter
            .next_if(|(right_id, _)| right_id < operator_id)
            .is_some()
        {}
        let Some((_, right_sources)) = right_iter.next_if(|(right_id, _)| right_id == operator_id)
        else {
            continue;
        };
        let operator_id = *operator_id;
        let (left_amount, right_amount, base_cost) = (
            left_saturated[operator_id],
            right_saturated[operator_id],
            operator_costs[operator_id],
        );
        assert!(
            left_amount.is_finite() && right_amount.is_finite() && base_cost.is_finite(),
            "label saturation of an operator with source sets is finite, got {left_amount} and {right_amount} of the cost {base_cost} of operator {operator_id}"
        );
        let stolen = pairwise_region_stolen_cost(
            left_sources,
            left_amount,
            right_sources,
            right_amount,
            base_cost,
        );
        if stolen > float_tolerance::SEARCH_EPSILON {
            conflict += stolen;
        }
    }
    conflict
}

/// What two components saturating `left_amount` and `right_amount` of an
/// operator costing `base_cost` steal from each other: the excess over its
/// cost if their source sets for it overlap, and nothing otherwise.
pub(super) fn pairwise_region_stolen_cost(
    left_regions: &[&CartesianSet],
    left_amount: f64,
    right_regions: &[&CartesianSet],
    right_amount: f64,
    base_cost: f64,
) -> f64 {
    let excess = left_amount + right_amount - base_cost;
    if excess <= float_tolerance::SEARCH_EPSILON {
        return 0.0;
    }
    if left_regions
        .iter()
        .any(|left| right_regions.iter().any(|right| left.overlaps(right)))
    {
        excess
    } else {
        0.0
    }
}

pub(super) fn compute_surplus_cost(
    saturated_by_abs: &[Vec<f64>],
    op_id: usize,
    remaining_cost: f64,
) -> f64 {
    let sum: f64 = saturated_by_abs
        .iter()
        .map(|costs| costs[op_id])
        .filter(|&w| w > f64::NEG_INFINITY)
        .sum();
    if !remaining_cost.is_finite() || !sum.is_finite() {
        return f64::INFINITY;
    }
    remaining_cost - sum
}

pub(super) fn compute_all_surplus_costs(costs: &[f64], saturated_by_abs: &[Vec<f64>]) -> Vec<f64> {
    (0..costs.len())
        .map(|op_id| compute_surplus_cost(saturated_by_abs, op_id, costs[op_id]))
        .collect()
}
