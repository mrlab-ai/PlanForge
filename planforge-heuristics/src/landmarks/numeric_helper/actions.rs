//! The helper's actions: each operator and axiom with its propositional and
//! numeric precondition lists -- redundant condition groups materialized --
//! and its effects classified into simple, linear and assignment effects.

use super::*;

impl NumericTaskHelper {
    pub(crate) fn build_precondition_lists(
        &self,
        preconditions: &[ExplicitFact],
    ) -> (Vec<ExplicitFact>, Vec<usize>) {
        let mut propositional_facts = Vec::new();
        let mut numeric_group_ids = Vec::new();
        let mut seen_propositional = BTreeSet::new();
        let mut seen_numeric = BTreeSet::new();

        for condition in preconditions {
            let var_id = condition.var_index();
            if !self.is_numeric_axiom_var(var_id) {
                if seen_propositional.insert((condition.var_index(), condition.value_index())) {
                    propositional_facts.push(*condition);
                }
                continue;
            }

            if condition.value() > 0 {
                continue;
            }

            for group_id in self.condition_group_ids_for_numeric_fact(condition) {
                if seen_numeric.insert(group_id) {
                    numeric_group_ids.push(group_id);
                }
            }
        }

        (propositional_facts, numeric_group_ids)
    }

    pub(crate) fn build_base_precondition_lists_with_redundancy(
        &mut self,
        preconditions: &[ExplicitFact],
        precision: f64,
    ) -> HelperPreconditionLists {
        let (propositional_facts, _numeric_group_ids) =
            self.build_precondition_lists(preconditions);
        let numeric_group_ids =
            self.build_numeric_precondition_group_ids_with_redundancy(preconditions, precision);
        HelperPreconditionLists {
            propositional_facts,
            numeric_group_ids,
        }
    }

    pub(crate) fn build_conditional_precondition_lists_with_redundancy(
        &mut self,
        base_preconditions: &[ExplicitFact],
        conditions: &[ExplicitFact],
        precision: f64,
    ) -> HelperPreconditionLists {
        // PARITY(numeric-fd): mirrors the split storage used by `build_action()` for conditional
        // effects/numeric effects/assignment effects/linear effects: propositional facts from the
        // local condition list, plus numeric local list with pairwise redundancy and raw cross-list
        // redundancy against the already-built base action numeric list.
        let base_numeric_group_ids = self
            .build_numeric_precondition_group_ids_with_redundancy(base_preconditions, precision);
        let (propositional_facts, _local_numeric_group_ids) =
            self.build_precondition_lists(conditions);
        let numeric_group_ids = self.build_conditional_numeric_group_ids_with_redundancy(
            conditions,
            &base_numeric_group_ids,
            precision,
        );
        HelperPreconditionLists {
            propositional_facts,
            numeric_group_ids,
        }
    }

    pub(crate) fn condition_group_ids_for_numeric_fact(&self, fact: &ExplicitFact) -> Vec<usize> {
        let var_id = fact.var_index();
        if let Some(group_ids) =
            self.comparison_fact_condition_group_ids(var_id, fact.value_index())
        {
            return group_ids.to_vec();
        }
        if let Some(group_ids) = self.goal_helper_numeric_condition_group_ids(var_id) {
            return group_ids.to_vec();
        }
        Vec::new()
    }

    pub(crate) fn build_numeric_precondition_group_ids_with_redundancy(
        &mut self,
        preconditions: &[ExplicitFact],
        precision: f64,
    ) -> Vec<usize> {
        // PARITY(numeric-fd): mirrors the `actions[op_id].num_list` population followed by
        // `build_redundant_constraints(original_list, actions[op_id].num_list)`.
        let (_propositional_facts, numeric_group_ids) =
            self.build_precondition_lists(preconditions);
        let mut result = numeric_group_ids.clone();
        let redundant_group_ids =
            self.materialize_pairwise_redundant_condition_groups(&numeric_group_ids, precision);
        append_unique_ids(&mut result, &redundant_group_ids);
        result
    }

    pub(crate) fn build_conditional_numeric_group_ids_with_redundancy(
        &mut self,
        conditions: &[ExplicitFact],
        base_numeric_group_ids: &[usize],
        precision: f64,
    ) -> Vec<usize> {
        // PARITY(numeric-fd): mirrors the helper-side population of conditional numeric lists like
        // `eff_num_conditions[index]`, `num_eff_num_conditions[index]`,
        // `assign_eff_num_conditions[index]`, and `linear_eff_num_conditions[index]`, followed by
        // the pairwise overload on the local list and the raw cross-list overload against the base
        // action numeric preconditions.
        let (_propositional_facts, local_numeric_group_ids) =
            self.build_precondition_lists(conditions);
        let mut result = local_numeric_group_ids.clone();
        let pairwise_group_ids = self
            .materialize_pairwise_redundant_condition_groups(&local_numeric_group_ids, precision);
        append_unique_ids(&mut result, &pairwise_group_ids);
        let cross_group_ids = self.materialize_cross_redundant_condition_groups_raw(
            &local_numeric_group_ids,
            base_numeric_group_ids,
            precision,
        );
        append_unique_ids(&mut result, &cross_group_ids);
        result
    }

    pub(crate) fn materialize_pairwise_redundant_condition_groups(
        &mut self,
        group_ids: &[usize],
        precision: f64,
    ) -> Vec<usize> {
        // PARITY(numeric-fd): this mirrors the helper-side overload that expands each input
        // `numeric_conditions_id` entry into all of its normalized members before generating
        // pairwise redundant constraints. The separate cross-list overload has intentionally different
        // raw-id semantics in the reference implementation and is modeled separately below.
        let mut redundant_group_ids = Vec::new();
        for left_index in 0..group_ids.len() {
            for right_index in (left_index + 1)..group_ids.len() {
                let Some(left_condition_ids) = self
                    .condition_group_condition_ids(group_ids[left_index])
                    .map(|ids| ids.to_vec())
                else {
                    continue;
                };
                let Some(right_condition_ids) = self
                    .condition_group_condition_ids(group_ids[right_index])
                    .map(|ids| ids.to_vec())
                else {
                    continue;
                };

                for left_condition_id in left_condition_ids {
                    for &right_condition_id in &right_condition_ids {
                        let Some(left_condition) =
                            self.normalized_condition(left_condition_id).cloned()
                        else {
                            continue;
                        };
                        let Some(right_condition) = self.normalized_condition(right_condition_id)
                        else {
                            continue;
                        };
                        let redundant_condition = left_condition.add(right_condition);
                        if redundant_condition.is_empty(precision) {
                            continue;
                        }
                        redundant_group_ids
                            .push(self.register_condition_group(vec![redundant_condition]));
                    }
                }
            }
        }
        redundant_group_ids
    }

    pub(crate) fn materialize_cross_redundant_condition_groups_raw(
        &mut self,
        list1_group_ids: &[usize],
        list2_group_ids: &[usize],
        precision: f64,
    ) -> Vec<usize> {
        // PARITY(numeric-fd): this mirrors `build_redundant_constraints(list1, list2, target)`.
        // The reference does *not* expand `numeric_conditions_id[x]` / `numeric_conditions_id[y]`
        // here. Instead, it feeds the raw ids into `add_redundant_constraint(x, y, ...)`, which then
        // indexes `numeric_conditions[x]` and `numeric_conditions[y]` directly. In practice that means
        // "use the representative/first normalized condition" for multi-condition groups such as `eq`.
        let mut redundant_group_ids = Vec::new();
        for &left_group_id in list1_group_ids {
            for &right_group_id in list2_group_ids {
                if left_group_id == right_group_id {
                    continue;
                }
                let Some(left_condition_id) =
                    self.condition_group_representative_condition_id(left_group_id)
                else {
                    continue;
                };
                let Some(right_condition_id) =
                    self.condition_group_representative_condition_id(right_group_id)
                else {
                    continue;
                };
                let Some(left_condition) = self.normalized_condition(left_condition_id).cloned()
                else {
                    continue;
                };
                let Some(right_condition) = self.normalized_condition(right_condition_id) else {
                    continue;
                };
                let redundant_condition = left_condition.add(right_condition);
                if redundant_condition.is_empty(precision) {
                    continue;
                }
                redundant_group_ids.push(self.register_condition_group(vec![redundant_condition]));
            }
        }
        redundant_group_ids
    }

    pub(super) fn build_actions(
        &mut self,
        task: &Task,
        default_value_axioms: &[PropositionalAxiom],
        precision: f64,
        separate_constant_assignment: bool,
    ) {
        self.action_models.clear();
        for (operator_id, operator) in task.operators().iter().enumerate() {
            let action_model = self.build_action(
                task,
                operator,
                OperatorIndex::from_usize(operator_id),
                precision,
                separate_constant_assignment,
            );
            self.action_models.push(action_model);
        }
        for axiom in task.axioms().iter().chain(default_value_axioms) {
            let action_model = self.build_axiom_action(axiom, precision);
            self.action_models.push(action_model);
        }
    }

    pub(super) fn build_propositions(&mut self, task: &Task) {
        self.proposition_ids_by_var_value.clear();
        self.proposition_facts.clear();
        self.proposition_var_ids.clear();
        self.proposition_names.clear();
        for var_id in 0..task.variables().len() {
            let domain_size = task
                .variable(VariableIndex::from_usize(var_id))
                .domain_size();
            let mut ids = Vec::with_capacity(domain_size);
            for value in 0..domain_size {
                let proposition_id = self.proposition_facts.len();
                ids.push(proposition_id);
                let fact = task.numeric_conditions().fact(var_id, value);
                self.proposition_facts.push(fact);
                self.proposition_var_ids.push(var_id);
                self.proposition_names
                    .push(task.fact_name(&fact).to_string());
            }
            self.proposition_ids_by_var_value.push(ids);
        }
    }

    pub(super) fn build_action(
        &mut self,
        task: &Task,
        operator: &Operator,
        operator_id: OperatorIndex,
        precision: f64,
        separate_constant_assignment: bool,
    ) -> HelperActionModel {
        let base_preconditions =
            self.build_base_precondition_lists_with_redundancy(operator.preconditions(), precision);
        let mut action_model = HelperActionModel {
            propositional_preconditions: base_preconditions.propositional_facts.clone(),
            numeric_precondition_group_ids: base_preconditions.numeric_group_ids.clone(),
            simple_effects: vec![0.0; self.numeric_variable_ids.len()],
            is_assignment: vec![false; self.numeric_variable_ids.len()],
            assignment_values: vec![0.0; self.numeric_variable_ids.len()],
            cost: task.operator_cost(operator_id),
            ..HelperActionModel::default()
        };

        let mut base_precondition_values = BTreeMap::new();
        for precondition in operator.preconditions() {
            base_precondition_values.insert(precondition.var_index(), precondition.value_index());
        }

        for effect in operator.effects() {
            let add_fact =
                ExplicitFact::propositional(effect.var_id().index(), effect.value().index());
            if effect.conditions().is_empty() {
                action_model.add_facts.push(add_fact);
                if let Some(&pre_value) = base_precondition_values.get(&(effect.var_id())) {
                    action_model.del_facts.push(ExplicitFact::propositional(
                        effect.var_id().index(),
                        pre_value.index(),
                    ));
                }
            } else {
                let conditional_preconditions = self
                    .build_conditional_precondition_lists_with_redundancy(
                        operator.preconditions(),
                        effect.conditions(),
                        precision,
                    );
                action_model
                    .conditional_fact_effects
                    .push(HelperConditionalExplicitFactEffect {
                        preconditions: conditional_preconditions,
                        add_fact,
                    });
            }
        }

        action_model.pre_del_facts = intersect_fact_lists(
            &action_model.propositional_preconditions,
            &action_model.del_facts,
        );

        if task.is_linear_cost_operator(operator_id) {
            action_model.linear_cost = true;
            action_model.cost = 0.0;
            action_model.cost_coefficients = task.operator_cost_coefficients(operator_id);
            action_model.cost_constant = task.operator_cost_constant(operator_id);
        }

        let linearized_assignment_effects = task
            .linearized_assignment_effects(operator_id)
            .unwrap_or_else(|error| {
                panic!("failed to build helper action model for operator {operator_id}: {error}")
            });
        for (assignment_effect_id, assignment_effect) in
            operator.assignment_effects().iter().enumerate()
        {
            let linear_effect = linearized_assignment_effects
                .get(assignment_effect_id)
                .expect("helper linearized assignment effect id must be valid");
            self.classify_assignment_effect_into_action(
                task,
                operator.preconditions(),
                ClassifiedAssignmentEffect {
                    id: assignment_effect_id,
                    effect: assignment_effect,
                    linear: linear_effect,
                },
                precision,
                separate_constant_assignment,
                &mut action_model,
            );
        }

        action_model
    }

    pub(super) fn build_axiom_action(
        &mut self,
        axiom: &PropositionalAxiom,
        precision: f64,
    ) -> HelperActionModel {
        let base_preconditions =
            self.build_base_precondition_lists_with_redundancy(axiom.conditions(), precision);
        let mut action_model = HelperActionModel {
            propositional_preconditions: base_preconditions.propositional_facts,
            numeric_precondition_group_ids: base_preconditions.numeric_group_ids,
            simple_effects: vec![0.0; self.numeric_variable_ids.len()],
            is_assignment: vec![false; self.numeric_variable_ids.len()],
            assignment_values: vec![0.0; self.numeric_variable_ids.len()],
            cost: 0.0,
            ..HelperActionModel::default()
        };
        action_model.add_facts.push(ExplicitFact::propositional(
            axiom.var_id().index(),
            axiom.effect_value().index(),
        ));
        action_model.del_facts.push(ExplicitFact::propositional(
            axiom.var_id().index(),
            axiom.precondition_value().index(),
        ));
        action_model.pre_del_facts = intersect_fact_lists(
            &action_model.propositional_preconditions,
            &action_model.del_facts,
        );
        action_model
    }

    pub(super) fn classify_assignment_effect_into_action(
        &mut self,
        task: &Task,
        base_preconditions: &[ExplicitFact],
        classified: ClassifiedAssignmentEffect<'_>,
        precision: f64,
        separate_constant_assignment: bool,
        action_model: &mut HelperActionModel,
    ) {
        let ClassifiedAssignmentEffect {
            id: assignment_effect_id,
            effect: assignment_effect,
            linear: linear_effect,
        } = classified;
        let affected_var_id = linear_effect.affected_var_id;
        let Some(local_var_id) = self.local_numeric_var_id(affected_var_id) else {
            return;
        };

        let final_expression = final_expression_from_effect(linear_effect);
        let is_conditional =
            assignment_effect.is_conditional() || !assignment_effect.conditions().is_empty();

        if is_simple_numeric_effect(linear_effect) {
            let delta = linear_effect.delta.constant;
            if is_conditional {
                action_model
                    .conditional_numeric_effects
                    .push(HelperConditionalNumericEffect {
                        target_local_var_id: local_var_id,
                        delta,
                    });
            } else {
                action_model.simple_effects[local_var_id.index()] = delta;
            }
            return;
        }

        if is_constant_assignment_like_effect(
            task,
            linear_effect,
            &final_expression,
            precision,
            separate_constant_assignment,
        ) {
            let assigned_value = final_expression.constant;
            if is_conditional {
                action_model.conditional_assignment_effects.push(
                    HelperConditionalAssignmentEffect {
                        target_local_var_id: local_var_id,
                        assigned_value,
                    },
                );
            } else {
                action_model.is_assignment[local_var_id.index()] = true;
                action_model.assignment_values[local_var_id.index()] = assigned_value;
            }
            return;
        }

        let coefficients = self
            .numeric_variable_ids
            .iter()
            .map(|&task_var_id| final_expression.coefficients[task_var_id.index()])
            .collect::<Vec<_>>();
        action_model.linear_effects.push(HelperLinearEffect {
            source_assignment_effect_id: assignment_effect_id,
            source_var_id: assignment_effect.var_id(),
            operation: assignment_effect.operation().clone(),
            preconditions: if is_conditional {
                self.build_conditional_precondition_lists_with_redundancy(
                    base_preconditions,
                    assignment_effect.conditions(),
                    precision,
                )
            } else {
                HelperPreconditionLists::default()
            },
            target_local_var_id: local_var_id,
            coefficients,
            constant: final_expression.constant,
        });
    }
}
