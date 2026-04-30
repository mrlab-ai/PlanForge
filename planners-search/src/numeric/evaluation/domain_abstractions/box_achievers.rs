#[cfg(test)]
mod tests;

use std::collections::HashMap;

use planners_sas::numeric::axioms::ComparisonOperator;
use planners_sas::numeric::numeric_task::{AbstractNumericTask, ExplicitFact, NumericType};

use super::comparison_expression::Interval;
use super::domain_abstraction_heuristic::{COMPARISON_FALSE_VAL, COMPARISON_TRUE_VAL};

#[derive(Debug, Clone, PartialEq)]
pub struct NumericBoxAchiever {
    pub operator_id: usize,
    pub achieved_facts: Vec<ExplicitFact>,
    pub bounds: Vec<(usize, Interval)>,
}

pub fn detect_numeric_box_achievers(task: &dyn AbstractNumericTask) -> Vec<NumericBoxAchiever> {
    let goal_facts = goal_relevant_facts(task);
    if goal_facts.is_empty() {
        return Vec::new();
    }

    let initial_numeric_values = task.get_initial_numeric_state_values();
    let mut out = Vec::new();
    for (operator_id, operator) in task.get_operators().iter().enumerate() {
        let achieved_facts: Vec<ExplicitFact> = operator
            .effects()
            .iter()
            .filter_map(|effect| {
                effect.precondition_value().map(|pre| {
                    let _ = pre;
                    ExplicitFact::new(effect.var_id(), effect.value())
                })
            })
            .filter(|fact| goal_facts.contains(fact))
            .collect();
        if achieved_facts.is_empty() {
            continue;
        }

        let mut bounds_by_var: HashMap<usize, Interval> = HashMap::new();
        for precondition in operator.preconditions() {
            let Some(comparison_axiom) = task
                .comparison_axioms()
                .iter()
                .find(|axiom| axiom.get_affected_var_id() == precondition.var)
            else {
                continue;
            };
            let desired_truth = match precondition.value {
                COMPARISON_TRUE_VAL => Some(true),
                COMPARISON_FALSE_VAL => Some(false),
                _ => None,
            };
            let Some(desired_truth) = desired_truth else {
                continue;
            };

            let Some((numeric_var_id, bound)) = comparison_bound_interval(
                task,
                comparison_axiom.get_left_var_id(),
                comparison_axiom.get_right_var_id(),
                comparison_axiom.get_operator(),
                desired_truth,
                &initial_numeric_values,
            ) else {
                continue;
            };

            bounds_by_var
                .entry(numeric_var_id)
                .and_modify(|current| {
                    if let Some(intersection) = intersect_intervals(*current, bound) {
                        *current = intersection;
                    }
                })
                .or_insert(bound);
        }

        if bounds_by_var.is_empty() {
            continue;
        }

        let mut bounds: Vec<(usize, Interval)> = bounds_by_var.into_iter().collect();
        bounds.sort_by_key(|(numeric_var_id, _)| *numeric_var_id);
        out.push(NumericBoxAchiever {
            operator_id,
            achieved_facts,
            bounds,
        });
    }
    out
}

fn goal_relevant_facts(task: &dyn AbstractNumericTask) -> Vec<ExplicitFact> {
    let mut goal_axiom_map: HashMap<usize, usize> = HashMap::new();
    for (idx, axiom) in task.axioms().iter().enumerate() {
        if !axiom.conditions().is_empty() {
            goal_axiom_map.insert(axiom.var_id(), idx);
        }
    }

    let mut out = Vec::new();
    for goal_id in 0..task.get_num_goals() {
        let goal = task.get_goal_fact(goal_id);
        if let Some(&axiom_id) = goal_axiom_map.get(&goal.var) {
            out.extend(task.axioms()[axiom_id].conditions().iter().cloned());
        } else {
            out.push(goal.clone());
        }
    }
    out.sort();
    out.dedup();
    out
}

fn comparison_bound_interval(
    task: &dyn AbstractNumericTask,
    left_var_id: usize,
    right_var_id: usize,
    operator: &ComparisonOperator,
    desired_truth: bool,
    initial_numeric_values: &[f64],
) -> Option<(usize, Interval)> {
    let left = task.numeric_variables().get(left_var_id)?;
    let right = task.numeric_variables().get(right_var_id)?;

    match (left.get_type(), right.get_type()) {
        (NumericType::Regular, NumericType::Constant) => {
            let threshold = *initial_numeric_values.get(right_var_id)?;
            interval_for_comparison(operator, threshold, true, desired_truth)
                .map(|interval| (left_var_id, interval))
        }
        (NumericType::Constant, NumericType::Regular) => {
            let threshold = *initial_numeric_values.get(left_var_id)?;
            interval_for_comparison(operator, threshold, false, desired_truth)
                .map(|interval| (right_var_id, interval))
        }
        _ => None,
    }
}

fn interval_for_comparison(
    operator: &ComparisonOperator,
    threshold: f64,
    variable_on_left: bool,
    desired_truth: bool,
) -> Option<Interval> {
    let normalized = match (operator, variable_on_left, desired_truth) {
        (ComparisonOperator::LessThan, true, true) => {
            Interval::new(f64::NEG_INFINITY, threshold, false, false)
        }
        (ComparisonOperator::LessThan, true, false) => {
            Interval::new(threshold, f64::INFINITY, true, false)
        }
        (ComparisonOperator::LessThanOrEqual, true, true) => {
            Interval::new(f64::NEG_INFINITY, threshold, false, true)
        }
        (ComparisonOperator::LessThanOrEqual, true, false) => {
            Interval::new(threshold, f64::INFINITY, false, false)
        }
        (ComparisonOperator::GreaterThan, true, true) => {
            Interval::new(threshold, f64::INFINITY, false, false)
        }
        (ComparisonOperator::GreaterThan, true, false) => {
            Interval::new(f64::NEG_INFINITY, threshold, false, true)
        }
        (ComparisonOperator::GreaterThanOrEqual, true, true) => {
            Interval::new(threshold, f64::INFINITY, true, false)
        }
        (ComparisonOperator::GreaterThanOrEqual, true, false) => {
            Interval::new(f64::NEG_INFINITY, threshold, false, false)
        }
        (ComparisonOperator::LessThan, false, truth) => {
            interval_for_comparison(&ComparisonOperator::GreaterThan, threshold, true, truth)?
        }
        (ComparisonOperator::LessThanOrEqual, false, truth) => interval_for_comparison(
            &ComparisonOperator::GreaterThanOrEqual,
            threshold,
            true,
            truth,
        )?,
        (ComparisonOperator::GreaterThan, false, truth) => {
            interval_for_comparison(&ComparisonOperator::LessThan, threshold, true, truth)?
        }
        (ComparisonOperator::GreaterThanOrEqual, false, truth) => interval_for_comparison(
            &ComparisonOperator::LessThanOrEqual,
            threshold,
            true,
            truth,
        )?,
        (ComparisonOperator::Equal, _, true) => Interval::singleton(threshold),
        (ComparisonOperator::Equal, _, false) => return None,
        (ComparisonOperator::UnEqual, _, true) => return None,
        (ComparisonOperator::UnEqual, _, false) => Interval::singleton(threshold),
    };

    Some(normalized)
}

fn intersect_intervals(left: Interval, right: Interval) -> Option<Interval> {
    if !left.intersects(&right) {
        return None;
    }

    let (lower, lower_closed) = if left.lower > right.lower {
        (left.lower, left.lower_closed)
    } else if right.lower > left.lower {
        (right.lower, right.lower_closed)
    } else {
        (left.lower, left.lower_closed && right.lower_closed)
    };

    let (upper, upper_closed) = if left.upper < right.upper {
        (left.upper, left.upper_closed)
    } else if right.upper < left.upper {
        (right.upper, right.upper_closed)
    } else {
        (left.upper, left.upper_closed && right.upper_closed)
    };

    let intersection = Interval::new(lower, upper, lower_closed, upper_closed);
    (!intersection.is_empty()).then_some(intersection)
}