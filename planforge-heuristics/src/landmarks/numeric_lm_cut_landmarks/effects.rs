//! What a relaxed operator does to a numeric condition: simple effects,
//! linear effects and second-order simple effects (SOSE), and constant
//! assignments.

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

    pub(super) fn build_supported_sose_operators(&mut self) -> Result<(), String> {
        let operator_count = self.task.operators().len();
        self.operator_condition_eval = OperatorConditionTable::new(
            self.config.condition_table,
            operator_count,
            self.conditions.len(),
        );
        if !self.config.use_second_order_simple {
            return Ok(());
        }

        let mut base_relaxed_by_original = vec![None; operator_count];
        for (relaxed_operator_id, operator) in self.relaxed_operators.iter().enumerate() {
            if operator.original_op_id_1.is_some() || operator.conditional || operator.infinite {
                continue;
            }
            if let Some(original_id) = operator
                .original_op_id_2
                .filter(|&id| id.index() < operator_count)
            {
                base_relaxed_by_original[original_id.index()] = Some(relaxed_operator_id);
            }
        }

        let mut new_operators = Vec::new();
        for op2_id in 0..operator_count {
            let Some(op2_relaxed_id) = base_relaxed_by_original[op2_id] else {
                continue;
            };
            let op2_id = OperatorIndex::from_usize(op2_id);

            let mut supporter_to_effects: BTreeMap<usize, Vec<(usize, f64)>> = BTreeMap::new();
            for condition_id in 0..self.conditions.len() {
                let condition = self
                    .conditions
                    .get(condition_id)
                    .ok_or_else(|| format!("LM-cut numeric condition {condition_id} is invalid"))?;
                if !self.has_linear_effect(op2_id, &condition.coefficients, false, false)?
                    || self.has_linear_effect(op2_id, &condition.coefficients, false, true)?
                {
                    continue;
                }

                let base_expression =
                    self.original_operator_condition_delta_expression(op2_id, condition_id)?;
                let composite_expression = LinearExpression {
                    coefficients: base_expression.coefficients.clone(),
                    constant: 0.0,
                };
                if self.has_effect(op2_id, &composite_expression.coefficients)? {
                    continue;
                }

                // PARITY(numeric-fd): get_sose_supporters() iterates ALL operators for the
                // linear-supporter check. Any op with a linear effect on composite_coefficients
                // (even one that has no base relaxed op) invalidates SOSE. The supporter
                // collection then only gathers ops that also have a simple/constant-assignment
                // effect. We must not gate the linear check behind base_relaxed_by_original.
                let mut condition_supporters = Vec::new();
                let mut invalid_support = false;
                for (op1_id, &op1_base_relaxed_id) in base_relaxed_by_original.iter().enumerate() {
                    let op1_id = OperatorIndex::from_usize(op1_id);
                    // Linear-effect check: scan ALL operators (C++ iterates all operators here).
                    if self.has_linear_effect(
                        op1_id,
                        &composite_expression.coefficients,
                        self.use_bounds,
                        false,
                    )? {
                        invalid_support = true;
                        break;
                    }

                    // Supporter collection: only ops that have a base relaxed op can become
                    // SOSE supporters (because we need a relaxed precondition set for them).
                    let Some(op1_relaxed_id) = op1_base_relaxed_id else {
                        continue;
                    };

                    let (has_simple_effect, simple_effect) = self
                        .calculate_simple_effect_constant(
                            op1_id,
                            &composite_expression.coefficients,
                            self.use_bounds,
                        )?;
                    let has_constant_assignment_effect = self.config.use_constant_assignment
                        && self.has_constant_assignment_effect(
                            op1_id,
                            &composite_expression.coefficients,
                            self.use_bounds,
                        )?;
                    if !has_simple_effect && !has_constant_assignment_effect {
                        continue;
                    }

                    if self.has_effect(op1_id, &condition.coefficients)? {
                        invalid_support = true;
                        break;
                    }

                    condition_supporters.push((op1_relaxed_id, simple_effect));
                }

                if invalid_support {
                    continue;
                }

                self.operator_condition_eval
                    .get_mut(op2_id, condition_id)
                    .has_sose = true;
                self.operator_condition_eval
                    .get_mut(op2_id, condition_id)
                    .composite_expression = Some(composite_expression);
                if self.use_bounds {
                    let projected_coefficients = self
                        .regular_numeric_variable_ids
                        .iter()
                        .map(|&numeric_var_id| {
                            base_expression
                                .coefficients
                                .get(numeric_var_id.index())
                                .copied()
                                .unwrap_or(0.0)
                        })
                        .collect::<Vec<_>>();
                    let mut has_bound = true;
                    let mut upper_bound = 0.0;
                    for (regular_var_id, &weight) in projected_coefficients.iter().enumerate() {
                        let regular_var_id = VariableIndex::from_usize(regular_var_id);
                        if weight >= self.config.precision
                            && self
                                .numeric_bound()
                                .get_variable_before_action_has_ub(regular_var_id, op2_id)
                        {
                            upper_bound += weight
                                * self
                                    .numeric_bound()
                                    .get_variable_before_action_ub(regular_var_id, op2_id);
                        } else if weight <= -self.config.precision
                            && self
                                .numeric_bound()
                                .get_variable_before_action_has_lb(regular_var_id, op2_id)
                        {
                            upper_bound += weight
                                * self
                                    .numeric_bound()
                                    .get_variable_before_action_lb(regular_var_id, op2_id);
                        } else if weight.abs() >= self.config.precision {
                            has_bound = false;
                            break;
                        }
                    }

                    if has_bound {
                        self.operator_condition_eval
                            .get_mut(op2_id, condition_id)
                            .has_upper_bound = true;
                        self.operator_condition_eval
                            .get_mut(op2_id, condition_id)
                            .upper_bound = upper_bound;
                    }
                }

                for (op1_relaxed_id, sose_constant) in condition_supporters {
                    supporter_to_effects
                        .entry(op1_relaxed_id)
                        .or_default()
                        .push((condition_id, sose_constant));
                }
            }

            for (op1_relaxed_id, effects) in supporter_to_effects {
                let op1 = self.relaxed_operators[op1_relaxed_id].clone();
                let op2 = self.relaxed_operators[op2_relaxed_id].clone();
                let original_op_id_1 = op1
                        .original_op_id_2
                        .filter(|&id| id.index() < operator_count)
                        .ok_or_else(|| {
                            format!(
                                "LM-cut SOSE supporter relaxed operator {op1_relaxed_id} is missing its concrete operator id"
                            )
                        })?;
                let original_op_id_2 = op2
                        .original_op_id_2
                        .filter(|&id| id.index() < operator_count)
                        .ok_or_else(|| {
                            format!(
                                "LM-cut SOSE target relaxed operator {op2_relaxed_id} is missing its concrete operator id"
                            )
                        })?;

                let mut precondition_ids = op1.precondition_ids.clone();
                precondition_ids.extend(op2.precondition_ids);
                if precondition_ids.is_empty() {
                    precondition_ids.push(self.artificial_precondition_id);
                }

                let mut effect_ids = Vec::new();
                let mut sose_constants = vec![0.0; self.conditions.len()];
                for (condition_id, sose_constant) in effects {
                    effect_ids.push(self.get_numeric_proposition_id(condition_id)?);
                    sose_constants[condition_id] = sose_constant;
                }

                let mut sose_operator = RelaxedOperator::new(
                    precondition_ids,
                    effect_ids,
                    original_op_id_2,
                    op2.base_cost_2,
                    format!("{} {}", op1.name, op2.name),
                    false,
                );
                sose_operator.original_op_id_1 = Some(original_op_id_1);
                sose_operator.original_op_id_2 = Some(original_op_id_2);
                sose_operator.base_cost_1 = op1.base_cost_2;
                sose_operator.base_cost_2 = op2.base_cost_2;
                sose_operator.cost_1 = sose_operator.base_cost_1;
                sose_operator.cost_2 = sose_operator.base_cost_2;
                sose_operator.sose_constants = sose_constants;
                sose_operator.linear_assignment_effects = op2.linear_assignment_effects.clone();
                sose_operator.assert_well_formed();
                new_operators.push(sose_operator);
            }
        }

        self.relaxed_operators.extend(new_operators);
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
            let weight: f64 = coefficients
                .get(actual_var_id.index())
                .copied()
                .unwrap_or(0.0);
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
            let weight = coefficients
                .get(actual_var_id.index())
                .copied()
                .unwrap_or(0.0);
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
                let weight = coefficients
                    .get(actual_var_id.index())
                    .copied()
                    .unwrap_or(0.0);
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
            let weight: f64 = coefficients
                .get(actual_var_id.index())
                .copied()
                .unwrap_or(0.0);
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
            let state_value = numeric_values
                .get(actual_var_id.index())
                .copied()
                .unwrap_or(ZERO_VALUE);
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
            let weight: f64 = coefficients
                .get(actual_var_id.index())
                .copied()
                .unwrap_or(0.0);
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
            let state_value = numeric_values
                .get(actual_var_id.index())
                .copied()
                .unwrap_or(ZERO_VALUE);
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
                let weight = coefficients
                    .get(actual_var_id.index())
                    .copied()
                    .unwrap_or(0.0);
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
                        - numeric_values
                            .get(actual_var_id.index())
                            .map(NumericValue::value)
                            .unwrap_or(0.0))
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
                        - numeric_values
                            .get(actual_var_id.index())
                            .map(NumericValue::value)
                            .unwrap_or(0.0))
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
