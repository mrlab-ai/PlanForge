//! The helper's bounds on the numeric variables and the epsilons that make
//! strict comparisons closed.

use super::*;

impl NumericTaskHelper {
    pub(super) fn calculates_bounds_numeric_variables(
        &mut self,
        task: &Task,
        precision: f64,
        infinity: f64,
    ) {
        let initial_numeric_values = task.initial_numeric_values();
        self.numeric_variable_lower_bounds = self
            .numeric_variable_ids
            .iter()
            .map(|&task_var_id| initial_numeric_values[task_var_id.index()].value())
            .collect();
        self.numeric_variable_upper_bounds = self.numeric_variable_lower_bounds.clone();

        for local_var_id in 0..self.numeric_variable_ids.len() {
            for action_model in self.action_models.iter().take(task.operators().len()) {
                let effect = action_model.simple_effects[local_var_id];
                if effect.abs() < precision {
                    continue;
                }
                let mut bound_for_action = false;
                let mut bound = 0.0;
                for &group_id in &action_model.numeric_precondition_group_ids {
                    let Some(condition_ids) = self.condition_group_condition_ids(group_id) else {
                        continue;
                    };
                    for &condition_id in condition_ids {
                        let Some(condition) = self.normalized_condition(condition_id) else {
                            continue;
                        };
                        let task_var_id = self.numeric_variable_ids[local_var_id];
                        let weight = condition.coefficients[task_var_id.index()];
                        if !simple_on_regular_var(
                            condition,
                            &self.numeric_variable_ids,
                            task_var_id,
                            precision,
                        ) {
                            continue;
                        }
                        if weight * effect > 0.0 {
                            continue;
                        }
                        bound_for_action = true;
                        bound = -condition.constant / weight + effect;
                    }
                }
                if bound_for_action {
                    if effect > 0.0 {
                        self.numeric_variable_upper_bounds[local_var_id] =
                            self.numeric_variable_upper_bounds[local_var_id].max(bound);
                    } else {
                        self.numeric_variable_lower_bounds[local_var_id] =
                            self.numeric_variable_lower_bounds[local_var_id].min(bound);
                    }
                } else if effect > 0.0 {
                    self.numeric_variable_upper_bounds[local_var_id] = infinity;
                } else {
                    self.numeric_variable_lower_bounds[local_var_id] = -infinity;
                }
            }
        }
    }

    pub(super) fn calculate_epsilons(
        &mut self,
        num_operators: usize,
        precision: f64,
        default_epsilon: f64,
    ) {
        self.condition_epsilons = vec![0.0; self.normalized_conditions.len()];
        for (condition_id, condition) in self.normalized_conditions.iter().enumerate() {
            if !condition.is_strictly_greater {
                continue;
            }
            let mut min_epsilon = calculate_epsilon_value(condition.constant, precision);
            let mut use_default_epsilon = false;
            for action_model in self.action_models.iter().take(num_operators) {
                for linear_effect in &action_model.linear_effects {
                    let task_var_id =
                        self.numeric_variable_ids[linear_effect.target_local_var_id.index()];
                    if condition.coefficients[task_var_id.index()].abs() >= precision {
                        use_default_epsilon = true;
                        break;
                    }
                }
                if use_default_epsilon {
                    break;
                }
                for (local_var_id, &is_assignment) in action_model.is_assignment.iter().enumerate()
                {
                    if is_assignment {
                        min_epsilon = min_epsilon.min(calculate_epsilon_value(
                            action_model.assignment_values[local_var_id],
                            precision,
                        ));
                    }
                }
                let effect = self
                    .numeric_variable_ids
                    .iter()
                    .enumerate()
                    .map(|(local_var_id, &task_var_id)| {
                        condition.coefficients[task_var_id.index()]
                            * action_model.simple_effects[local_var_id]
                    })
                    .sum::<f64>();
                min_epsilon = min_epsilon.min(calculate_epsilon_value(effect, precision));
            }
            self.condition_epsilons[condition_id] = if use_default_epsilon {
                default_epsilon
            } else {
                min_epsilon.max(default_epsilon)
            };
        }
    }
}
