//! What every reachable state satisfies: bounds on the features and, with
//! the AIBR closure, the facts that can hold.

use super::*;

/// What holds in every reachable state.
pub(super) struct ReachableBounds {
    /// Per feature, the bounds it stays in.
    pub(super) features: Vec<FeatureBounds>,
    /// Per variable and value, whether the fact can hold; known with the AIBR
    /// closure only.
    pub(super) facts: Option<Vec<Vec<bool>>>,
}

/// The bounds every reachable state keeps each feature in -- from the simple
/// action bounds, from monotone features, and from the AIBR closure, as
/// configured -- and, with the AIBR closure, the reachable facts.
pub(super) fn global_feature_bounds(
    task: &Task,
    helper: &NumericTaskHelper,
    features: &[NumericFeature],
    assignment_target_features: &[bool],
    effects: &PrimitiveEffects,
    bounds_provider: BoundsProvider,
    simple_action_bounds: bool,
) -> Result<ReachableBounds, String> {
    let mut global_feature_bounds = vec![FeatureBounds::default(); features.len()];
    if simple_action_bounds {
        let mut primitive_bounds = vec![FeatureBounds::default(); task.numeric_variables().len()];
        for (local_id, task_id) in task.regular_numeric_variable_ids().into_iter().enumerate() {
            if effects.assignment_targets[task_id.index()] {
                continue;
            }
            if let Some((lower, upper)) = helper.numeric_variable_bounds(local_id) {
                primitive_bounds[task_id.index()] = FeatureBounds {
                    lower: if lower > -9_999_999.0 {
                        lower
                    } else {
                        f64::NEG_INFINITY
                    },
                    upper: if upper < 9_999_999.0 {
                        upper
                    } else {
                        f64::INFINITY
                    },
                };
            }
        }
        for (feature_id, feature) in features.iter().enumerate() {
            if feature.affine
                && feature.coefficients.len() == 1
                && feature.constant == 0.0
                && task.numeric_type(feature.source_numeric_var) == NumericType::Regular
            {
                global_feature_bounds[feature_id] =
                    primitive_bounds[feature.source_numeric_var.index()];
            }
        }
    }
    if matches!(
        bounds_provider,
        BoundsProvider::Monotone | BoundsProvider::All
    ) {
        let initial = task.initial_numeric_values();
        for (feature_id, feature) in features.iter().enumerate() {
            if assignment_target_features[feature_id] {
                continue;
            }
            let all_nonnegative = effects
                .deltas
                .iter()
                .all(|deltas| dot_sparse(&feature.coefficients, deltas) >= 0.0);
            let all_nonpositive = effects
                .deltas
                .iter()
                .all(|deltas| dot_sparse(&feature.coefficients, deltas) <= 0.0);
            let initial_value = if feature.affine {
                NumericValue::affine(
                    feature
                        .coefficients
                        .iter()
                        .map(|&(var, coefficient)| (coefficient, initial[var.index()])),
                    feature.constant,
                )
                .value()
            } else {
                initial[feature.source_numeric_var.index()].value()
            };
            if all_nonnegative {
                global_feature_bounds[feature_id].lower = initial_value;
            }
            if all_nonpositive {
                global_feature_bounds[feature_id].upper = initial_value;
            }
        }
    }
    let mut reachable_facts = None;
    if matches!(bounds_provider, BoundsProvider::Aibr | BoundsProvider::All) {
        let closure = compute_aibr_closure(task)?;
        let mut contribution = vec![FeatureBounds::default(); features.len()];
        map_primitive_bounds_to_features(features, &closure.numeric_bounds, &mut contribution);
        intersect_bounds(&mut global_feature_bounds, &contribution);
        reachable_facts = Some(closure.reachable_facts);
    }
    Ok(ReachableBounds {
        features: global_feature_bounds,
        facts: reachable_facts,
    })
}

/// Marks the operators no reachable state can apply -- a propositional
/// precondition outside `reachable_facts`, or a numeric precondition outside
/// the global feature bounds -- and narrows the others' numeric precondition
/// bounds to the global ones.
pub(super) fn restrict_operators_to_reachable(
    operators: &mut [PotentialOperator],
    global_feature_bounds: &[FeatureBounds],
    reachable_facts: Option<&[Vec<bool>]>,
) {
    for operator in operators {
        operator.reachable = operator
            .preconditions
            .iter()
            .all(|fact| reachable_facts.is_none_or(|facts| facts[fact.var()][fact.value()]))
            && operator
                .numeric_precondition_bounds
                .iter()
                .all(|(feature_id, precondition)| {
                    let global = global_feature_bounds[*feature_id];
                    precondition.lower <= global.upper && precondition.upper >= global.lower
                });
        operator.numeric_precondition_bounds =
            intersect_sparse_bounds(&operator.numeric_precondition_bounds, global_feature_bounds);
    }
}

pub(super) struct AibrClosure {
    pub(super) numeric_bounds: Vec<FeatureBounds>,
    pub(super) reachable_facts: Vec<Vec<bool>>,
}

pub(super) fn compute_aibr_closure(task: &Task) -> Result<AibrClosure, String> {
    let (initial_props, initial_numeric) = task
        .evaluate_initial_state_values()
        .map_err(|error| format!("numeric bounds could not evaluate initial state: {error}"))?;
    let mut reachable_facts: Vec<Vec<bool>> = task
        .variables()
        .iter()
        .map(|variable| vec![false; variable.domain_size()])
        .collect();
    for (var_id, value) in initial_props.into_iter().enumerate() {
        reachable_facts[var_id][value.index()] = true;
    }
    let mut intervals: Vec<FeatureBounds> = initial_numeric
        .into_iter()
        .map(|value| FeatureBounds {
            lower: value.value(),
            upper: value.value(),
        })
        .collect();
    // This closure has to *over*approximate reachability -- an operator whose
    // preconditions it declares unreachable loses its LP constraint, and a
    // propositional goal it declares unreachable makes the heuristic infinite --
    // so it needs the rules describing how a derived variable takes its default
    // value, exactly like the other relaxations. Since issue454 the task does not
    // carry them and it asks for them itself.
    let default_value_axioms =
        default_value_axioms(task, DefaultValueAxiomMode::ApproximateNegativeCycles);

    loop {
        let previous_facts = reachable_facts.clone();
        let previous_intervals = intervals.clone();

        for axiom in task.assignment_axioms() {
            let value = match &axiom.expression {
                NumericExpression::Linear(terms) => terms.iter().fold(
                    FeatureBounds {
                        lower: 0.0,
                        upper: 0.0,
                    },
                    |sum, (var, coefficient)| {
                        interval_add(
                            sum,
                            interval_product(
                                intervals[var.index()],
                                FeatureBounds {
                                    lower: coefficient.value(),
                                    upper: coefficient.value(),
                                },
                            ),
                        )
                    },
                ),
                NumericExpression::Binary {
                    operator,
                    left,
                    right,
                } => {
                    let (left, right) = (intervals[left.index()], intervals[right.index()]);
                    match operator {
                        CalOperator::Sum => interval_add(left, right),
                        CalOperator::Difference => interval_subtract(left, right),
                        CalOperator::Product => interval_product(left, right),
                        CalOperator::Division => interval_divide(left, right),
                    }
                }
            };
            hull_into(&mut intervals[axiom.affected_var_id().index()], value);
        }
        for axiom in task.comparison_axioms() {
            if comparison_may_hold(
                intervals[axiom.left_var_id().index()],
                intervals[axiom.right_var_id().index()],
                axiom.operator(),
            ) {
                reachable_facts[axiom.affected_var_id().index()][0] = true;
            }
        }
        for axiom in task.axioms().iter().chain(&default_value_axioms) {
            if axiom
                .conditions()
                .iter()
                .all(|fact| reachable_facts[fact.var()][fact.value()])
            {
                reachable_facts[axiom.var_id().index()][axiom.effect_value().index()] = true;
            }
        }
        for operator in task.operators() {
            if !operator
                .preconditions()
                .iter()
                .all(|fact| reachable_facts[fact.var()][fact.value()])
            {
                continue;
            }
            for effect in operator.effects() {
                reachable_facts[effect.var_id().index()][effect.value().index()] = true;
            }
            for effect in operator.assignment_effects() {
                let source = intervals[effect.var_id().index()];
                let target = &mut intervals[effect.affected_var_id().index()];
                match effect.operation() {
                    AssignmentOperation::Assign => hull_into(target, source),
                    AssignmentOperation::Plus => {
                        if source.lower < 0.0 {
                            target.lower = f64::NEG_INFINITY;
                        }
                        if source.upper > 0.0 {
                            target.upper = f64::INFINITY;
                        }
                    }
                    AssignmentOperation::Minus => {
                        if source.upper > 0.0 {
                            target.lower = f64::NEG_INFINITY;
                        }
                        if source.lower < 0.0 {
                            target.upper = f64::INFINITY;
                        }
                    }
                    AssignmentOperation::Times | AssignmentOperation::Divide => {
                        *target = FeatureBounds::default();
                    }
                }
            }
        }

        if reachable_facts == previous_facts && intervals == previous_intervals {
            break;
        }
    }
    Ok(AibrClosure {
        numeric_bounds: intervals,
        reachable_facts,
    })
}

pub(super) fn hull_into(target: &mut FeatureBounds, value: FeatureBounds) {
    target.lower = target.lower.min(value.lower);
    target.upper = target.upper.max(value.upper);
}

pub(super) fn interval_add(left: FeatureBounds, right: FeatureBounds) -> FeatureBounds {
    FeatureBounds {
        lower: left.lower + right.lower,
        upper: left.upper + right.upper,
    }
}

pub(super) fn interval_subtract(left: FeatureBounds, right: FeatureBounds) -> FeatureBounds {
    FeatureBounds {
        lower: left.lower - right.upper,
        upper: left.upper - right.lower,
    }
}

pub(super) fn interval_product(left: FeatureBounds, right: FeatureBounds) -> FeatureBounds {
    let values = [
        left.lower * right.lower,
        left.lower * right.upper,
        left.upper * right.lower,
        left.upper * right.upper,
    ];
    if values.iter().any(|value| value.is_nan()) {
        return FeatureBounds::default();
    }
    FeatureBounds {
        lower: values.iter().copied().fold(f64::INFINITY, f64::min),
        upper: values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    }
}

pub(super) fn interval_divide(left: FeatureBounds, right: FeatureBounds) -> FeatureBounds {
    if right.lower <= 0.0 && right.upper >= 0.0 {
        FeatureBounds::default()
    } else {
        interval_product(
            left,
            FeatureBounds {
                lower: 1.0 / right.upper,
                upper: 1.0 / right.lower,
            },
        )
    }
}

pub(super) fn comparison_may_hold(
    left: FeatureBounds,
    right: FeatureBounds,
    operator: &ComparisonOperator,
) -> bool {
    match operator {
        ComparisonOperator::LessThan => left.lower < right.upper,
        ComparisonOperator::LessThanOrEqual => left.lower <= right.upper,
        ComparisonOperator::Equal => left.lower <= right.upper && right.lower <= left.upper,
        ComparisonOperator::GreaterThanOrEqual => left.upper >= right.lower,
        ComparisonOperator::GreaterThan => left.upper > right.lower,
        ComparisonOperator::UnEqual => {
            left.lower != left.upper || right.lower != right.upper || left.lower != right.lower
        }
    }
}

pub(super) fn intersect_bounds(target: &mut [FeatureBounds], contribution: &[FeatureBounds]) {
    assert_eq!(target.len(), contribution.len());
    for (target, contribution) in target.iter_mut().zip(contribution) {
        target.lower = target.lower.max(contribution.lower);
        target.upper = target.upper.min(contribution.upper);
    }
}
