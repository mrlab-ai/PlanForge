use super::*;

pub(super) fn compute_score(h: f64, stolen_costs: f64, scoring_function: ScoringFunction) -> f64 {
    match scoring_function {
        ScoringFunction::MaxHeuristic => h,
        ScoringFunction::MinStolenCosts => -stolen_costs,
        ScoringFunction::MaxHeuristicPerStolenCosts => h / stolen_costs.max(1.0),
    }
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

pub(super) fn interval_stolen_costs(
    components: &[AbstractionComponent<'_>],
    saturated_by_component: &[Vec<f64>],
    operator_costs: &[f64],
) -> Result<Vec<Option<f64>>, EvaluationError> {
    let mut regions_by_component = Vec::with_capacity(components.len());
    for (component_id, component) in components.iter().enumerate() {
        let (operator_regions, expected_operator_regions) = match component {
            AbstractionComponent::Domain(heuristic) => {
                let abstraction = heuristic.abstraction();
                (
                    abstraction.abstract_operator_regions.as_slice(),
                    abstraction.abstract_operators.len(),
                )
            }
            AbstractionComponent::Cartesian(heuristic) => {
                let abstraction = heuristic.abstraction();
                (
                    abstraction.abstract_operator_regions.as_slice(),
                    abstraction.transition_system.transitions.len(),
                )
            }
            AbstractionComponent::PatternDatabase(_) => {
                regions_by_component.push(None);
                continue;
            }
        };
        if operator_regions.len() != expected_operator_regions {
            return Err(EvaluationError::ComputationFailed(format!(
                "region SCP component {component_id} has {} operator regions for {expected_operator_regions} retained abstract transitions/operators",
                operator_regions.len()
            )));
        }
        let mut by_operator = HashMap::<usize, Vec<&StateRegion>>::new();
        for operator_region in operator_regions {
            for label in &operator_region.labels {
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
        }
        regions_by_component.push(Some(by_operator));
    }

    let mut scores = regions_by_component
        .iter()
        .map(|regions| regions.as_ref().map(|_| 0.0))
        .collect::<Vec<_>>();
    for left in 0..components.len() {
        let Some(left_by_operator) = &regions_by_component[left] else {
            continue;
        };
        for right in left + 1..components.len() {
            let Some(right_by_operator) = &regions_by_component[right] else {
                continue;
            };
            let left_saturated = &saturated_by_component[left];
            let right_saturated = &saturated_by_component[right];
            for (&operator_id, left_regions) in left_by_operator {
                let Some(right_regions) = right_by_operator.get(&operator_id) else {
                    continue;
                };
                let left_amount = left_saturated
                    .get(operator_id)
                    .copied()
                    .unwrap_or(f64::NEG_INFINITY);
                let right_amount = right_saturated
                    .get(operator_id)
                    .copied()
                    .unwrap_or(f64::NEG_INFINITY);
                let base_cost = operator_costs[operator_id];
                if !left_amount.is_finite() || !right_amount.is_finite() || !base_cost.is_finite() {
                    continue;
                }
                let conflict = pairwise_interval_stolen_cost(
                    left_regions,
                    left_amount,
                    right_regions,
                    right_amount,
                    base_cost,
                );
                if conflict > 1e-9 {
                    *scores[left]
                        .as_mut()
                        .expect("regional component must have a conflict score") += conflict;
                    *scores[right]
                        .as_mut()
                        .expect("regional component must have a conflict score") += conflict;
                }
            }
        }
    }
    Ok(scores)
}

pub(super) fn pairwise_interval_stolen_cost(
    left_regions: &[&StateRegion],
    left_amount: f64,
    right_regions: &[&StateRegion],
    right_amount: f64,
    base_cost: f64,
) -> f64 {
    let excess = (left_amount + right_amount - base_cost).max(0.0);
    if excess <= 1e-9 {
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
        .map(|costs| costs.get(op_id).copied().unwrap_or(f64::NEG_INFINITY))
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
