//! Backward-direction primitives shared with the unified flaw-emission path.
//!
//! The functions exported here are the building blocks used by
//! `progression::get_progression_flaws` when called with
//! [`super::SplitDirection::Backward`]: they place split values at the
//! boundaries derived from the regressed-target / required interval, rather
//! than at the concrete value that produced the flaw.

use std::collections::HashMap;

use planforge_sas::{
    numeric_task::{AbstractNumericTask, ExplicitFact, NumericType},
    utils::linear_effects::linearize_numeric_var,
};

use super::{NumericFlaw, can_split_numeric_var, numeric_requirement_for_comparison_fact};
pub(super) use crate::evaluation::domain_abstractions::additive_numeric_views::numeric_effect_deltas;
use crate::evaluation::domain_abstractions::{
    comparison_expression::{CompOp, Interval},
    domain_abstraction::{ComparisonAxiomIndex, NumericPartitions},
};

/// Backward-direction split helper for deviation flaws.
///
/// When a concrete progression-state lands outside the abstract-state's
/// expected interval `expected_interval`, regress the boundary closest to
/// `concrete_successor` by the operator's effect `delta` and return the
/// corresponding split point in source space. Returns `None` if no boundary
/// of `expected_interval` is finite on the relevant side.
pub(super) fn preimage_split_for_expected_successor(
    expected_interval: Interval,
    concrete_successor: f64,
    delta: f64,
) -> Option<(f64, bool)> {
    if expected_interval.is_empty() {
        return None;
    }
    if concrete_successor < expected_interval.lower
        || (concrete_successor == expected_interval.lower && !expected_interval.lower_closed)
    {
        if expected_interval.lower.is_finite() {
            return Some((
                expected_interval.lower - delta,
                !expected_interval.lower_closed,
            ));
        }
    }
    if concrete_successor > expected_interval.upper
        || (concrete_successor == expected_interval.upper && !expected_interval.upper_closed)
    {
        if expected_interval.upper.is_finite() {
            return Some((
                expected_interval.upper - delta,
                expected_interval.upper_closed,
            ));
        }
    }
    None
}

/// Backward-direction dependent-flaw helper used by the unified precondition
/// and goal flaw emission paths.
///
/// For a comparison-axiom propositional fact, decode the required numeric
/// interval and emit a stack of boundary-aligned shell flaws covering the
/// concrete value's distance to the closest boundary.
pub(super) fn dependent_numeric_flaws_backward(
    task: &dyn AbstractNumericTask,
    deltas: &HashMap<usize, Vec<f64>>,
    partitions: &NumericPartitions,
    comparison_index: &ComparisonAxiomIndex,
    fact: &ExplicitFact,
    numeric_state: &[f64],
    step: usize,
) -> Vec<NumericFlaw> {
    if let Some((numeric_var_id, required_interval)) =
        numeric_requirement_for_comparison_fact(task, comparison_index, fact)
    {
        let Some(&concrete_value) = numeric_state.get(numeric_var_id) else {
            return Vec::new();
        };
        return target_centered_shell_flaws(
            deltas,
            partitions,
            numeric_var_id,
            required_interval,
            concrete_value,
            step,
        );
    }

    target_centered_linear_comparison_flaws(
        task,
        deltas,
        partitions,
        comparison_index,
        fact,
        numeric_state,
        step,
    )
}

fn target_centered_linear_comparison_flaws(
    task: &dyn AbstractNumericTask,
    deltas: &HashMap<usize, Vec<f64>>,
    partitions: &NumericPartitions,
    comparison_index: &ComparisonAxiomIndex,
    fact: &ExplicitFact,
    numeric_state: &[f64],
    step: usize,
) -> Vec<NumericFlaw> {
    let Some(tree) = comparison_index.comparison_tree(fact.var()) else {
        return Vec::new();
    };
    let Ok(left) = linearize_numeric_var(task, tree.left_numeric_var_id) else {
        return Vec::new();
    };
    let Ok(right) = linearize_numeric_var(task, tree.right_numeric_var_id) else {
        return Vec::new();
    };
    let expression = left.subtract(&right);
    let Some(required_op) = required_comparison_op(tree.op, fact.value()) else {
        return Vec::new();
    };

    let mut flaws = Vec::new();
    for (numeric_var_id, &coefficient) in expression.coefficients.iter().enumerate() {
        if coefficient.abs() < 1e-12 {
            continue;
        }
        if task
            .numeric_variables()
            .get(numeric_var_id)
            .is_none_or(|variable| variable.get_type() != &NumericType::Regular)
        {
            continue;
        }
        let Some(&concrete_value) = numeric_state.get(numeric_var_id) else {
            continue;
        };
        let fixed_constant = expression.constant
            + expression
                .coefficients
                .iter()
                .enumerate()
                .filter(|(var, _)| *var != numeric_var_id)
                .map(|(var, other_coefficient)| {
                    other_coefficient * numeric_state.get(var).copied().unwrap_or(0.0)
                })
                .sum::<f64>();
        let Some(required_interval) = single_var_interval(coefficient, fixed_constant, required_op)
        else {
            continue;
        };
        flaws.extend(target_centered_shell_flaws(
            deltas,
            partitions,
            numeric_var_id,
            required_interval,
            concrete_value,
            step,
        ));
    }
    flaws
}

fn target_centered_shell_flaws(
    deltas: &HashMap<usize, Vec<f64>>,
    partitions: &NumericPartitions,
    numeric_var_id: usize,
    required_interval: Interval,
    concrete_value: f64,
    step: usize,
) -> Vec<NumericFlaw> {
    let Some((boundary, include_in_lower)) =
        split_for_missing_numeric_requirement(required_interval, concrete_value)
    else {
        return Vec::new();
    };
    let mut flaws = Vec::new();
    push_numeric_flaw_if_possible(
        partitions,
        numeric_var_id,
        boundary,
        include_in_lower,
        step,
        &mut flaws,
    );

    let Some(var_deltas) = deltas.get(&numeric_var_id) else {
        return flaws;
    };
    let shell_delta = if concrete_value < boundary {
        var_deltas
            .iter()
            .copied()
            .filter(|delta| *delta > 1e-12)
            .min_by(|left, right| left.total_cmp(right))
    } else {
        var_deltas
            .iter()
            .copied()
            .filter(|delta| *delta < -1e-12)
            .max_by(|left, right| left.total_cmp(right))
    };
    let Some(shell_delta) = shell_delta else {
        return flaws;
    };

    let mut value = boundary - shell_delta;
    for _ in 0..128 {
        if !value.is_finite() {
            break;
        }
        if shell_delta > 0.0 && value <= concrete_value {
            break;
        }
        if shell_delta < 0.0 && value >= concrete_value {
            break;
        }
        push_numeric_flaw_if_possible(
            partitions,
            numeric_var_id,
            value,
            include_in_lower,
            step,
            &mut flaws,
        );
        value -= shell_delta;
    }
    flaws
}

fn push_numeric_flaw_if_possible(
    partitions: &NumericPartitions,
    numeric_var_id: usize,
    value: f64,
    include_in_lower: bool,
    step: usize,
    flaws: &mut Vec<NumericFlaw>,
) {
    if can_split_numeric_var(partitions, numeric_var_id, value, include_in_lower) {
        flaws.push(NumericFlaw {
            numeric_var_id,
            value,
            include_in_lower,
            step,
        });
    }
}

fn required_comparison_op(op: CompOp, prop_value: usize) -> Option<CompOp> {
    match prop_value {
        0 => Some(op),
        1 => Some(match op {
            CompOp::Lt => CompOp::Ge,
            CompOp::Le => CompOp::Gt,
            CompOp::Gt => CompOp::Le,
            CompOp::Ge => CompOp::Lt,
            CompOp::Eq => CompOp::Ne,
            CompOp::Ne => CompOp::Eq,
        }),
        _ => None,
    }
}

fn single_var_interval(coefficient: f64, constant: f64, op: CompOp) -> Option<Interval> {
    if coefficient.abs() < 1e-12 || op == CompOp::Ne {
        return None;
    }
    let threshold = -constant / coefficient;
    if !threshold.is_finite() {
        return None;
    }
    Some(match (op, coefficient.is_sign_positive()) {
        (CompOp::Lt, true) | (CompOp::Gt, false) => {
            Interval::new(f64::NEG_INFINITY, threshold, false, false)
        }
        (CompOp::Le, true) | (CompOp::Ge, false) => {
            Interval::new(f64::NEG_INFINITY, threshold, false, true)
        }
        (CompOp::Gt, true) | (CompOp::Lt, false) => {
            Interval::new(threshold, f64::INFINITY, false, false)
        }
        (CompOp::Ge, true) | (CompOp::Le, false) => {
            Interval::new(threshold, f64::INFINITY, true, false)
        }
        (CompOp::Eq, _) => Interval::singleton(threshold),
        (CompOp::Ne, _) => return None,
    })
}

fn split_for_missing_numeric_requirement(
    requirement: Interval,
    concrete_value: f64,
) -> Option<(f64, bool)> {
    if concrete_value < requirement.lower
        || (concrete_value == requirement.lower && !requirement.lower_closed)
    {
        if requirement.lower.is_finite() {
            return Some((requirement.lower, !requirement.lower_closed));
        }
    }
    if concrete_value > requirement.upper
        || (concrete_value == requirement.upper && !requirement.upper_closed)
    {
        if requirement.upper.is_finite() {
            return Some((requirement.upper, requirement.upper_closed));
        }
    }
    None
}
