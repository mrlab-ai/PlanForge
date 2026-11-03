//! Second-order simple effects (SOSE): the relaxed operators that model an
//! effect whose change depends on a numeric variable the condition does not
//! read.

use super::*;

impl LandmarkCutLandmarks {
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
                        .map(|&numeric_var_id| base_expression.coefficients[numeric_var_id.index()])
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
}
