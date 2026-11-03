//! What a relaxed operator does to a numeric condition: simple and linear
//! effects, and the change in a condition an original operator causes;
//! [`super::sose_effects`] and [`super::constant_assignments`] cover the rest.

use super::*;

impl LandmarkCutLandmarks {
    pub(super) fn build_simple_effects(&mut self) -> Result<(), String> {
        let operator_count = self.task.operators().len();

        for relaxed_operator_id in 0..self.relaxed_operators.len() {
            let original_op_id: OperatorIndex = {
                let relaxed_operator = &self.relaxed_operators[relaxed_operator_id];
                // PARITY(numeric-fd): keep this aligned with C++ `build_simple_effects()`,
                // which checks only `!conditional && op_id_1 == -1 && op_id_2 < n_actions`.
                // Infinite operators must still be considered here because an action with both
                // a linear effect and a simple/assignment-like effect can legitimately
                // contribute simple effects to its infinite relaxed operator.
                if relaxed_operator.conditional || relaxed_operator.original_op_id_1.is_some() {
                    continue;
                }
                match relaxed_operator.original_op_id_2 {
                    Some(original_id) if original_id.index() < operator_count => original_id,
                    _ => continue,
                }
            };

            let mut additional_effect_ids = Vec::new();
            let mut seen: BTreeSet<usize> = self.relaxed_operators[relaxed_operator_id]
                .effect_ids
                .iter()
                .copied()
                .collect();

            for condition_id in 0..self.conditions.len() {
                let has_supported_sose = self
                    .operator_condition_eval
                    .get(original_op_id, condition_id)
                    .has_sose;
                let (has_simple_effect, simple_effect) = self.calculate_simple_effect_constant(
                    original_op_id,
                    &self.conditions[condition_id].coefficients,
                    self.use_bounds && !has_supported_sose,
                )?;
                let has_constant_assignment_effect = self.config.use_constant_assignment
                    && self.has_constant_assignment_effect(
                        original_op_id,
                        &self.conditions[condition_id].coefficients,
                        self.use_bounds && !has_supported_sose,
                    )?;

                if !has_simple_effect && !has_supported_sose && !has_constant_assignment_effect {
                    continue;
                }

                self.operator_condition_eval
                    .get_mut(original_op_id, condition_id)
                    .simple_effect = Some(simple_effect);
                let proposition_id = self.get_numeric_proposition_id(condition_id)?;
                if seen.insert(proposition_id) {
                    additional_effect_ids.push(proposition_id);
                }
            }

            if !additional_effect_ids.is_empty() {
                self.relaxed_operators[relaxed_operator_id]
                    .effect_ids
                    .extend(additional_effect_ids);
            }
        }

        Ok(())
    }

    pub(super) fn calculate_simple_effect_constant(
        &self,
        original_operator_id: OperatorIndex,
        coefficients: &[f64],
        use_bounded_linear: bool,
    ) -> Result<(bool, f64), String> {
        let helper_simple_effects = self
            .numeric_helper
            .get_action_eff_list(original_operator_id.index())
            .ok_or_else(|| {
                format!("LM-cut helper action eff_list {original_operator_id} is missing")
            })?;
        let helper_conditional_numeric_effects = self
            .numeric_helper
            .get_action_conditional_eff_list(original_operator_id.index())
            .ok_or_else(|| {
                format!(
                    "LM-cut helper conditional numeric effects {original_operator_id} are missing"
                )
            })?;
        let helper_linear_effects = self
            .numeric_helper
            .get_action_linear_effects(original_operator_id.index())
            .ok_or_else(|| {
                format!("LM-cut helper linear effects {original_operator_id} are missing")
            })?;
        let mut has_simple_effect = false;
        let mut net = 0.0;

        for (local_var_id, &simple_effect) in helper_simple_effects.iter().enumerate() {
            let actual_var_id = *self.regular_numeric_variable_ids.get(local_var_id).ok_or_else(|| {
                format!(
                    "LM-cut helper simple effect target {local_var_id} is invalid for operator {original_operator_id}"
                )
            })?;
            let weight = coefficients
                .get(actual_var_id.index())
                .copied()
                .unwrap_or(0.0);
            if weight.abs() < self.config.precision {
                continue;
            }
            net += weight * simple_effect;
        }

        for conditional_effect in helper_conditional_numeric_effects {
            let actual_var_id = *self.regular_numeric_variable_ids
                .get(conditional_effect.target_local_var_id.index())
                .ok_or_else(|| {
                    format!(
                        "LM-cut helper conditional numeric effect target {} is invalid for operator {original_operator_id}",
                        conditional_effect.target_local_var_id
                    )
                })?;
            let weight = coefficients
                .get(actual_var_id.index())
                .copied()
                .unwrap_or(0.0);
            if weight.abs() < self.config.precision {
                continue;
            }
            let contribution = weight * conditional_effect.delta;
            if contribution >= self.config.precision {
                net += contribution;
            }
        }

        for linear_effect in helper_linear_effects {
            let local_var_id = linear_effect.target_local_var_id;
            let actual_var_id = *self.regular_numeric_variable_ids.get(local_var_id.index()).ok_or_else(|| {
                format!(
                    "LM-cut helper linear effect target {local_var_id} is invalid for operator {original_operator_id}"
                )
            })?;
            let weight = coefficients
                .get(actual_var_id.index())
                .copied()
                .unwrap_or(0.0);
            if weight.abs() < self.config.precision {
                continue;
            }

            let conditional = !linear_effect.preconditions.propositional_facts.is_empty()
                || !linear_effect.preconditions.numeric_group_ids.is_empty();
            let mut contribution = weight * linear_effect.constant;

            if use_bounded_linear
                && weight >= self.config.precision
                && self
                    .numeric_bound()
                    .get_effect_has_ub(original_operator_id, local_var_id)
                && (!self.config.use_constant_assignment
                    || !self
                        .numeric_bound()
                        .get_assignment_has_ub(original_operator_id, local_var_id))
            {
                contribution = weight
                    * self
                        .numeric_bound()
                        .get_effect_ub(original_operator_id, local_var_id);
            } else if use_bounded_linear
                && weight <= -self.config.precision
                && self
                    .numeric_bound()
                    .get_effect_has_lb(original_operator_id, local_var_id)
                && (!self.config.use_constant_assignment
                    || !self
                        .numeric_bound()
                        .get_assignment_has_lb(original_operator_id, local_var_id))
            {
                contribution = weight
                    * self
                        .numeric_bound()
                        .get_effect_lb(original_operator_id, local_var_id);
            } else if use_bounded_linear
                && self.config.use_constant_assignment
                && ((weight >= self.config.precision
                    && self
                        .numeric_bound()
                        .get_assignment_has_ub(original_operator_id, local_var_id))
                    || (weight <= -self.config.precision
                        && self
                            .numeric_bound()
                            .get_assignment_has_lb(original_operator_id, local_var_id)))
            {
                contribution = 0.0;
                has_simple_effect = true;
            }

            if !conditional || contribution >= self.config.precision {
                net += contribution;
            }
        }

        if !has_simple_effect {
            has_simple_effect = net >= self.config.precision;
        }

        Ok((has_simple_effect, net))
    }

    pub(super) fn build_linear_operators(
        &mut self,
        operator_id: OperatorIndex,
        operator: &Operator,
        base_cost: f64,
        base_precondition_ids: &[usize],
        linearized_assignment_effects: &[Option<LinearNumericEffect>],
    ) -> Result<(), String> {
        let helper_linear_effects = self
            .numeric_helper
            .get_action_linear_effects(operator_id.index())
            .map(|effects| effects.to_vec())
            .ok_or_else(|| format!("LM-cut helper linear effects {operator_id} are missing"))?;
        if self
            .numeric_helper
            .get_action_n_linear_eff(operator_id.index())
            == 0
        {
            return Ok(());
        }
        for helper_linear_effect in helper_linear_effects {
            let assignment_effect_id = helper_linear_effect.source_assignment_effect_id;
            let linear_effect = linearized_assignment_effects
                .get(assignment_effect_id)
                .and_then(|effect| effect.clone())
                .ok_or_else(|| {
                    format!(
                        "LM-cut linearized assignment effect {assignment_effect_id} for operator {operator_id} is missing"
                    )
                })?;
            let helper_effect_preconditions = helper_linear_effect.preconditions.clone();

            let mut precondition_ids = base_precondition_ids.to_vec();
            let mut seen: BTreeSet<usize> = precondition_ids.iter().copied().collect();
            let assignment_condition_groups = self.precondition_proposition_id_groups(
                &helper_effect_preconditions.propositional_facts,
            );
            for group in &assignment_condition_groups {
                for &proposition_id in group {
                    if seen.insert(proposition_id) {
                        precondition_ids.push(proposition_id);
                    }
                }
            }
            self.append_numeric_condition_propositions(
                &helper_effect_preconditions.numeric_group_ids,
                &mut precondition_ids,
            );
            if precondition_ids.is_empty() {
                precondition_ids.push(self.artificial_precondition_id);
            }

            let mut plus_effect_ids = Vec::new();
            let mut minus_effect_ids = Vec::new();
            for condition_id in 0..self.conditions.len() {
                let condition = self
                    .conditions
                    .get(condition_id)
                    .ok_or_else(|| format!("LM-cut numeric condition {condition_id} is invalid"))?;
                if !self.has_linear_effect(
                    operator_id,
                    &condition.coefficients,
                    self.use_bounds,
                    false,
                )? {
                    continue;
                }
                let weight = condition
                    .coefficients
                    .get(linear_effect.affected_var_id.index())
                    .copied()
                    .unwrap_or(0.0);
                if weight > self.config.precision {
                    plus_effect_ids.push(self.get_numeric_proposition_id(condition_id)?);
                } else if weight < -self.config.precision {
                    minus_effect_ids.push(self.get_numeric_proposition_id(condition_id)?);
                }
            }

            if !plus_effect_ids.is_empty() {
                let mut relaxed_operator = RelaxedOperator::new(
                    {
                        let mut guarded_preconditions = precondition_ids.clone();
                        for &condition_id in &self.linear_effect_to_conditions_plus
                            [operator_id.index()][assignment_effect_id]
                        {
                            guarded_preconditions
                                .push(self.get_numeric_proposition_id(condition_id)?);
                        }
                        guarded_preconditions
                    },
                    plus_effect_ids,
                    operator_id,
                    base_cost,
                    format!("{} {} +inf", operator.name(), linear_effect.affected_var_id),
                    true,
                );
                relaxed_operator.infinite = true;
                relaxed_operator.assignment_effect_ids = vec![assignment_effect_id];
                relaxed_operator.linear_assignment_effects = vec![linear_effect.clone()];
                relaxed_operator.assert_well_formed();
                self.relaxed_operators.push(relaxed_operator);
            }

            if !minus_effect_ids.is_empty() {
                let mut relaxed_operator = RelaxedOperator::new(
                    {
                        let mut guarded_preconditions = precondition_ids;
                        for &condition_id in &self.linear_effect_to_conditions_minus
                            [operator_id.index()][assignment_effect_id]
                        {
                            guarded_preconditions
                                .push(self.get_numeric_proposition_id(condition_id)?);
                        }
                        guarded_preconditions
                    },
                    minus_effect_ids,
                    operator_id,
                    base_cost,
                    format!("{} {} -inf", operator.name(), linear_effect.affected_var_id),
                    true,
                );
                relaxed_operator.infinite = true;
                relaxed_operator.assignment_effect_ids = vec![assignment_effect_id];
                relaxed_operator.linear_assignment_effects = vec![linear_effect];
                relaxed_operator.assert_well_formed();
                self.relaxed_operators.push(relaxed_operator);
            }
        }

        Ok(())
    }

    pub(super) fn has_linear_effect(
        &self,
        original_operator_id: OperatorIndex,
        coefficients: &[f64],
        use_bounded_linear: bool,
        only_conditional: bool,
    ) -> Result<bool, String> {
        let helper_linear_effects = self
            .numeric_helper
            .get_action_linear_effects(original_operator_id.index())
            .ok_or_else(|| {
                format!("LM-cut helper linear effects {original_operator_id} are missing")
            })?;
        for linear_effect in helper_linear_effects {
            let actual_var_id = *self.regular_numeric_variable_ids
                .get(linear_effect.target_local_var_id.index())
                .ok_or_else(|| {
                    format!(
                        "LM-cut helper linear effect target {} is invalid for operator {original_operator_id}",
                        linear_effect.target_local_var_id
                    )
                })?;
            let weight: f64 = coefficients
                .get(actual_var_id.index())
                .copied()
                .unwrap_or(0.0);
            if weight.abs() < self.config.precision {
                continue;
            }
            if only_conditional
                && linear_effect.preconditions.propositional_facts.is_empty()
                && linear_effect.preconditions.numeric_group_ids.is_empty()
            {
                continue;
            }

            if !use_bounded_linear {
                return Ok(true);
            }

            let local_var_id = Some(linear_effect.target_local_var_id);

            if weight >= self.config.precision {
                let blocked_by_upper_bound = local_var_id
                    .map(|var_id| {
                        self.numeric_bound()
                            .get_effect_has_ub(original_operator_id, var_id)
                            || (self.config.use_constant_assignment
                                && self
                                    .numeric_bound()
                                    .get_assignment_has_ub(original_operator_id, var_id))
                    })
                    .unwrap_or(false);
                if !blocked_by_upper_bound {
                    return Ok(true);
                }
                continue;
            }

            if weight <= -self.config.precision {
                let blocked_by_lower_bound = local_var_id
                    .map(|var_id| {
                        self.numeric_bound()
                            .get_effect_has_lb(original_operator_id, var_id)
                            || (self.config.use_constant_assignment
                                && self
                                    .numeric_bound()
                                    .get_assignment_has_lb(original_operator_id, var_id))
                    })
                    .unwrap_or(false);
                if !blocked_by_lower_bound {
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    pub(super) fn has_effect(
        &self,
        original_operator_id: OperatorIndex,
        coefficients: &[f64],
    ) -> Result<bool, String> {
        if self.has_linear_effect(original_operator_id, coefficients, false, false)? {
            return Ok(true);
        }

        if self.config.use_constant_assignment
            && self.has_constant_assignment_effect(original_operator_id, coefficients, false)?
        {
            return Ok(true);
        }

        let (has_simple_effect, _) =
            self.calculate_simple_effect_constant(original_operator_id, coefficients, false)?;
        Ok(has_simple_effect)
    }

    pub(super) fn original_operator_condition_delta_expression(
        &self,
        original_operator_id: OperatorIndex,
        condition_id: usize,
    ) -> Result<LinearExpression, String> {
        let helper_linear_effects = self
            .numeric_helper
            .get_action_linear_effects(original_operator_id.index())
            .ok_or_else(|| {
                format!("LM-cut helper linear effects {original_operator_id} are missing")
            })?;
        let reconstructed_linear_effects = helper_linear_effects
            .iter()
            .map(|linear_effect| {
                self.numeric_helper
                    .linearized_effect_for_action_assignment(
                        original_operator_id.index(),
                        linear_effect.source_assignment_effect_id,
                    )
                    .ok_or_else(|| {
                        format!(
                            "LM-cut helper linearized effect {} is missing for operator {original_operator_id}",
                            linear_effect.source_assignment_effect_id
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.condition_delta_expression(condition_id, reconstructed_linear_effects.iter())
    }

    pub(super) fn condition_delta_expression<'a>(
        &self,
        condition_id: usize,
        linear_effects: impl Iterator<Item = &'a LinearNumericEffect>,
    ) -> Result<LinearExpression, String> {
        let condition = self
            .conditions
            .get(condition_id)
            .ok_or_else(|| format!("LM-cut numeric condition {condition_id} is invalid"))?;

        let mut expression = LinearExpression::zero(self.task.numeric_variables().len());
        for linear_effect in linear_effects {
            let target_coefficient: f64 = condition
                .coefficients
                .get(linear_effect.affected_var_id.index())
                .copied()
                .unwrap_or(0.0);
            if target_coefficient.abs() < self.config.precision {
                continue;
            }
            expression = expression.add(&linear_effect.delta.scale(target_coefficient));
        }
        Ok(expression)
    }
}
