//! Constant assignments: whether an operator sets a condition's variables to
//! constants, and the change in the condition's value that causes.

use super::*;

impl LandmarkCutLandmarks {
    pub(super) fn has_constant_assignment_effect(
        &self,
        original_operator_id: OperatorIndex,
        coefficients: &[f64],
        use_bounded_linear: bool,
    ) -> Result<bool, String> {
        let helper_is_assignment = self
            .numeric_helper
            .get_action_is_assignment(original_operator_id.index())
            .ok_or_else(|| {
                format!("LM-cut helper action is_assignment {original_operator_id} is missing")
            })?;
        let helper_conditional_assignments = self
            .numeric_helper
            .get_action_conditional_assign_list(original_operator_id.index())
            .ok_or_else(|| {
                format!("LM-cut helper conditional assignments {original_operator_id} are missing")
            })?;
        for (local_var_id, &is_assignment) in helper_is_assignment.iter().enumerate() {
            if !is_assignment {
                continue;
            }
            let actual_var_id = *self.regular_numeric_variable_ids.get(local_var_id).ok_or_else(|| {
                format!(
                    "LM-cut helper assignment target {local_var_id} is invalid for operator {original_operator_id}"
                )
            })?;
            let weight: f64 = coefficients[actual_var_id.index()];
            if weight.abs() < self.config.precision {
                continue;
            }
            let local_var_id = VariableIndex::from_usize(local_var_id);
            if self.use_bounds
                && ((weight >= self.config.precision
                    && self
                        .numeric_bound()
                        .has_no_increasing_assignment_effect(original_operator_id, local_var_id))
                    || (weight <= -self.config.precision
                        && self.numeric_bound().has_no_decreasing_assignment_effect(
                            original_operator_id,
                            local_var_id,
                        )))
            {
                continue;
            }

            return Ok(true);
        }

        for conditional_assignment in helper_conditional_assignments {
            let local_var_id = conditional_assignment.target_local_var_id;
            let actual_var_id = *self.regular_numeric_variable_ids.get(local_var_id.index()).ok_or_else(|| {
                format!(
                    "LM-cut helper conditional assignment target {local_var_id} is invalid for operator {original_operator_id}"
                )
            })?;
            let weight = coefficients[actual_var_id.index()];
            if weight.abs() < self.config.precision {
                continue;
            }
            if self.use_bounds
                && ((weight >= self.config.precision
                    && self
                        .numeric_bound()
                        .has_no_increasing_assignment_effect(original_operator_id, local_var_id))
                    || (weight <= -self.config.precision
                        && self.numeric_bound().has_no_decreasing_assignment_effect(
                            original_operator_id,
                            local_var_id,
                        )))
            {
                continue;
            }
            return Ok(true);
        }

        if use_bounded_linear {
            for (local_var_id, &actual_var_id) in
                self.regular_numeric_variable_ids.iter().enumerate()
            {
                let local_var_id = VariableIndex::from_usize(local_var_id);
                let weight = coefficients[actual_var_id.index()];
                if (weight >= self.config.precision
                    && self
                        .numeric_bound()
                        .get_assignment_has_ub(original_operator_id, local_var_id)
                    && (!self.use_bounds
                        || !self.numeric_bound().has_no_increasing_assignment_effect(
                            original_operator_id,
                            local_var_id,
                        )))
                    || (weight <= -self.config.precision
                        && self
                            .numeric_bound()
                            .get_assignment_has_lb(original_operator_id, local_var_id)
                        && (!self.use_bounds
                            || !self.numeric_bound().has_no_decreasing_assignment_effect(
                                original_operator_id,
                                local_var_id,
                            )))
                {
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    pub(super) fn calculate_constant_assignment_effect(
        &self,
        original_operator_id: OperatorIndex,
        coefficients: &[f64],
        numeric_values: &[NumericValue],
        use_bounded_linear: bool,
    ) -> Result<f64, String> {
        let helper_is_assignment = self
            .numeric_helper
            .get_action_is_assignment(original_operator_id.index())
            .ok_or_else(|| {
                format!("LM-cut helper action is_assignment {original_operator_id} is missing")
            })?;
        let helper_assign_list = self
            .numeric_helper
            .get_action_assign_list(original_operator_id.index())
            .ok_or_else(|| {
                format!("LM-cut helper action assign_list {original_operator_id} is missing")
            })?;
        let helper_conditional_assignments = self
            .numeric_helper
            .get_action_conditional_assign_list(original_operator_id.index())
            .ok_or_else(|| {
                format!("LM-cut helper conditional assignments {original_operator_id} are missing")
            })?;
        let mut net = 0.0;

        for (local_var_id, &is_assignment) in helper_is_assignment.iter().enumerate() {
            if !is_assignment {
                continue;
            }
            let actual_var_id = *self.regular_numeric_variable_ids.get(local_var_id).ok_or_else(|| {
                format!(
                    "LM-cut helper assignment target {local_var_id} is invalid for operator {original_operator_id}"
                )
            })?;
            let weight: f64 = coefficients[actual_var_id.index()];
            if weight.abs() < self.config.precision {
                continue;
            }
            if self.use_bounds
                && ((weight >= self.config.precision
                    && self.numeric_bound().has_no_increasing_assignment_effect(
                        original_operator_id,
                        VariableIndex::from_usize(local_var_id),
                    ))
                    || (weight <= -self.config.precision
                        && self.numeric_bound().has_no_decreasing_assignment_effect(
                            original_operator_id,
                            VariableIndex::from_usize(local_var_id),
                        )))
            {
                continue;
            }

            let constant_target = helper_assign_list[local_var_id];
            let state_value = numeric_values[actual_var_id.index()];
            if (weight >= self.config.precision && constant_target > state_value.value())
                || (weight <= -self.config.precision && constant_target < state_value.value())
            {
                net += weight * (constant_target - state_value.value());
            }
        }

        for conditional_assignment in helper_conditional_assignments {
            let local_var_id = conditional_assignment.target_local_var_id;
            let actual_var_id = *self.regular_numeric_variable_ids.get(local_var_id.index()).ok_or_else(|| {
                format!(
                    "LM-cut helper conditional assignment target {local_var_id} is invalid for operator {original_operator_id}"
                )
            })?;
            let weight: f64 = coefficients[actual_var_id.index()];
            if weight.abs() < self.config.precision {
                continue;
            }
            if self.use_bounds
                && ((weight >= self.config.precision
                    && self
                        .numeric_bound()
                        .has_no_increasing_assignment_effect(original_operator_id, local_var_id))
                    || (weight <= -self.config.precision
                        && self.numeric_bound().has_no_decreasing_assignment_effect(
                            original_operator_id,
                            local_var_id,
                        )))
            {
                continue;
            }

            let constant_target = conditional_assignment.assigned_value;
            let state_value = numeric_values[actual_var_id.index()];
            if (weight >= self.config.precision && constant_target > state_value.value())
                || (weight <= -self.config.precision && constant_target < state_value.value())
            {
                net += weight * (constant_target - state_value.value());
            }
        }

        if use_bounded_linear {
            for (local_var_id, &actual_var_id) in
                self.regular_numeric_variable_ids.iter().enumerate()
            {
                let local_var_id = VariableIndex::from_usize(local_var_id);
                let weight = coefficients[actual_var_id.index()];
                if self.use_bounds
                    && ((weight >= self.config.precision
                        && self.numeric_bound().has_no_increasing_assignment_effect(
                            original_operator_id,
                            local_var_id,
                        ))
                        || (weight <= -self.config.precision
                            && self.numeric_bound().has_no_decreasing_assignment_effect(
                                original_operator_id,
                                local_var_id,
                            )))
                {
                    continue;
                }

                if weight >= self.config.precision
                    && self
                        .numeric_bound()
                        .get_assignment_has_ub(original_operator_id, local_var_id)
                {
                    let mut contribution = (self
                        .numeric_bound()
                        .get_assignment_ub(original_operator_id, local_var_id)
                        - numeric_values[actual_var_id.index()].value())
                    .max(0.0);
                    if self
                        .numeric_bound()
                        .get_effect_has_ub(original_operator_id, local_var_id)
                    {
                        contribution = contribution.min(
                            self.numeric_bound()
                                .get_effect_ub(original_operator_id, local_var_id),
                        );
                    }
                    net += weight * contribution;
                } else if weight <= -self.config.precision
                    && self
                        .numeric_bound()
                        .get_assignment_has_lb(original_operator_id, local_var_id)
                {
                    let mut contribution = (self
                        .numeric_bound()
                        .get_assignment_lb(original_operator_id, local_var_id)
                        - numeric_values[actual_var_id.index()].value())
                    .min(0.0);
                    if self
                        .numeric_bound()
                        .get_effect_has_lb(original_operator_id, local_var_id)
                    {
                        contribution = contribution.max(
                            self.numeric_bound()
                                .get_effect_lb(original_operator_id, local_var_id),
                        );
                    }
                    net += weight * contribution;
                }
            }
        }

        Ok(net)
    }

    #[inline(always)]
    pub(super) fn calculate_constant_assignment_effect_infallible(
        &self,
        original_operator_id: OperatorIndex,
        coefficients: &[f64],
        numeric_values: &[NumericValue],
        use_bounded_linear: bool,
    ) -> f64 {
        self.calculate_constant_assignment_effect(
            original_operator_id,
            coefficients,
            numeric_values,
            use_bounded_linear,
        )
        .unwrap_or_else(|error| panic!("{error}"))
    }
}
