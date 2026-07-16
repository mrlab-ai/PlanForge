#[cfg(test)]
mod tests;

use std::collections::{HashMap, HashSet};

use planforge_sas::numeric_task::{AbstractNumericTask, AssignmentOperation, NumericType};

use crate::task_restriction::validate_restricted_task;

#[derive(Debug, Clone, Copy)]
struct NumericCondition {
    var_id: usize,
    constant: f64,
}

pub struct NumericSizeEstimator {
    approximate_domain_sizes: Vec<usize>,
}

impl NumericSizeEstimator {
    pub fn new(task: &dyn AbstractNumericTask) -> Self {
        validate_restricted_task(task)
            .expect("numeric PDB size estimation requires a restricted task");
        let base_initial_numeric_values = task.get_initial_numeric_state_values().to_vec();
        let conditions = collect_numeric_conditions(task, &base_initial_numeric_values);

        Self {
            approximate_domain_sizes: (0..task.numeric_variables().len())
                .map(|numeric_var_id| {
                    estimate_numeric_domain_size(
                        task,
                        numeric_var_id,
                        &base_initial_numeric_values,
                        &conditions,
                    )
                })
                .collect(),
        }
    }

    pub fn estimate_domain_size(&self, numeric_var_id: usize) -> usize {
        self.approximate_domain_sizes
            .get(numeric_var_id)
            .copied()
            .expect("numeric size estimate requires a valid numeric variable ID")
            .max(1)
    }
}

fn collect_numeric_conditions(
    task: &dyn AbstractNumericTask,
    base_initial_numeric_values: &[f64],
) -> Vec<NumericCondition> {
    let mut comparison_axioms_by_var = HashMap::new();
    let mut numeric_condition_vars = HashSet::new();

    for (comparison_axiom_id, comparison_axiom) in task.comparison_axioms().iter().enumerate() {
        let affected_var_id = comparison_axiom.get_affected_var_id();
        comparison_axioms_by_var.insert(affected_var_id, comparison_axiom_id);
        numeric_condition_vars.insert(affected_var_id);
    }

    let mut conditions = Vec::new();
    for operator in task.get_operators() {
        for fact in operator.preconditions() {
            let fact_var_id = fact.var();
            if !numeric_condition_vars.contains(&fact_var_id) {
                continue;
            }
            if let Some(&comparison_axiom_id) = comparison_axioms_by_var.get(&fact_var_id)
                && let Some(condition) =
                    build_numeric_condition(task, comparison_axiom_id, base_initial_numeric_values)
            {
                conditions.push(condition);
            }
        }
    }

    for goal_id in 0..task.get_num_goals() {
        let goal = task.get_goal_fact(goal_id);
        let goal_var_id = goal.var();
        if !numeric_condition_vars.contains(&goal_var_id) {
            continue;
        }
        if let Some(&comparison_axiom_id) = comparison_axioms_by_var.get(&goal_var_id)
            && let Some(condition) =
                build_numeric_condition(task, comparison_axiom_id, base_initial_numeric_values)
        {
            conditions.push(condition);
        }
    }

    conditions
}

fn build_numeric_condition(
    task: &dyn AbstractNumericTask,
    comparison_axiom_id: usize,
    initial_numeric_values: &[f64],
) -> Option<NumericCondition> {
    let comparison_axiom = &task.comparison_axioms()[comparison_axiom_id];
    let left = comparison_axiom.get_left_var_id();
    let right = comparison_axiom.get_right_var_id();
    let left_type = task.numeric_variables()[left].get_type();
    let right_type = task.numeric_variables()[right].get_type();

    match (left_type, right_type) {
        (NumericType::Regular, NumericType::Constant | NumericType::Cost) => {
            Some(NumericCondition {
                var_id: left,
                constant: initial_numeric_values[right],
            })
        }
        (NumericType::Constant | NumericType::Cost, NumericType::Regular) => {
            Some(NumericCondition {
                var_id: right,
                constant: initial_numeric_values[left],
            })
        }
        (NumericType::Regular, NumericType::Regular)
        | (NumericType::Constant | NumericType::Cost, NumericType::Constant | NumericType::Cost) => {
            None
        }
        (NumericType::Derived, _) | (_, NumericType::Derived) => {
            unreachable!("restricted-task validation excludes derived comparison operands")
        }
    }
}

fn estimate_numeric_domain_size(
    task: &dyn AbstractNumericTask,
    numeric_var_id: usize,
    base_initial_numeric_values: &[f64],
    conditions: &[NumericCondition],
) -> usize {
    assert!(
        numeric_var_id < task.numeric_variables().len(),
        "numeric size estimation requires a valid numeric variable ID"
    );

    let mut increments = Vec::new();
    let mut decrements = Vec::new();
    let mut min_const = 0.0_f64;
    let mut max_const = 0.0_f64;
    let mut min_change = f64::INFINITY;
    let mut max_pos_change = 0.0_f64;
    let mut max_neg_change = 0.0_f64;

    for operator in task.get_operators() {
        let mut additive_change = 0.0;
        let mut assigned_value = None;
        for effect in operator
            .assignment_effects()
            .iter()
            .filter(|effect| effect.affected_var_id() == numeric_var_id)
        {
            if effect.is_conditional() {
                return usize::MAX;
            }
            let source_var_id = effect.var_id();
            let source_type = task.numeric_variables()[source_var_id].get_type();
            if !matches!(source_type, NumericType::Constant | NumericType::Cost) {
                return usize::MAX;
            }
            let source_value = base_initial_numeric_values[source_var_id];
            match effect.operation() {
                AssignmentOperation::Plus if assigned_value.is_none() => {
                    additive_change += source_value;
                }
                AssignmentOperation::Minus if assigned_value.is_none() => {
                    additive_change -= source_value;
                }
                AssignmentOperation::Assign
                    if assigned_value.is_none() && additive_change == 0.0 =>
                {
                    assigned_value = Some(source_value);
                }
                AssignmentOperation::Plus
                | AssignmentOperation::Minus
                | AssignmentOperation::Assign
                | AssignmentOperation::Times
                | AssignmentOperation::Divide => return usize::MAX,
            }
        }

        if let Some(assigned_value) = assigned_value {
            min_const = min_const.min(assigned_value);
            max_const = max_const.max(assigned_value);
        } else if additive_change > 0.0 {
            increments.push(additive_change);
            min_change = min_change.min(additive_change);
            max_pos_change = max_pos_change.max(additive_change);
        } else if additive_change < 0.0 {
            decrements.push(additive_change);
            min_change = min_change.min(additive_change.abs());
            max_neg_change = max_neg_change.min(additive_change);
        }
    }

    let initial_value = base_initial_numeric_values[numeric_var_id];
    min_const = min_const.min(initial_value);
    max_const = max_const.max(initial_value);

    for condition in conditions {
        if condition.var_id == numeric_var_id {
            min_const = min_const.min(condition.constant);
            max_const = max_const.max(condition.constant);
        }
    }

    min_const += max_neg_change;
    max_const += max_pos_change;

    let min_increment = if !increments.is_empty() && !decrements.is_empty() {
        let mut best_increment = f64::INFINITY;
        for increment in &increments {
            for decrement in &decrements {
                best_increment = best_increment.min((increment + decrement).abs());
            }
        }
        if best_increment <= 1e-12 {
            min_change
        } else {
            best_increment
        }
    } else {
        min_change
    };

    if !min_increment.is_finite() || min_increment <= 1e-12 {
        return 1;
    }

    ((((max_const - min_const).abs() / min_increment) + 1.0) as usize).max(1)
}
