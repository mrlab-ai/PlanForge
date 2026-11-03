//! The propositions of the relaxed task: one per fact, and one per numeric
//! condition an operator precondition, an axiom or the goal names, together
//! with the linear conditions that combine them.

use super::*;

impl LandmarkCutLandmarks {
    pub(super) fn build_propositional_propositions(&mut self) {
        for variable_id in 0..self.num_variables {
            let domain_size = self
                .task
                .variable(VariableIndex::from_usize(variable_id))
                .domain_size();
            self.proposition_index[variable_id].reserve(domain_size);
            for value in 0..domain_size {
                let helper_proposition_id = self
                    .numeric_helper
                    .get_proposition(
                        VariableIndex::from_usize(variable_id),
                        ExplicitValueIndex::from_usize(value),
                    )
                    .expect("helper proposition id must exist");
                let proposition_id = self.propositions.len();
                let proposition = RelaxedProposition::new(
                    proposition_id,
                    self.numeric_helper
                        .get_proposition_name(helper_proposition_id)
                        .unwrap_or("")
                        .to_string(),
                );
                self.propositions.push(proposition);
                self.proposition_index[variable_id].push(proposition_id);
                self.num_propositions += 1;
            }
        }
    }

    pub(super) fn build_numeric_conditions(&mut self) {
        if self.config.ignore_numeric {
            return;
        }

        for condition_id in 0..self.numeric_helper.get_n_numeric_conditions() {
            let condition = self
                .numeric_helper
                .get_condition(condition_id)
                .cloned()
                .expect("helper numeric condition must exist");
            let epsilon = self.numeric_helper.get_epsilon(condition_id).unwrap_or(
                if condition.is_strictly_greater {
                    self.config.epsilon
                } else {
                    0.0
                },
            );
            self.add_numeric_condition_proposition_with_epsilon(condition, epsilon);
        }
    }

    pub(super) fn build_precondition_ids(&self, preconditions: &[ExplicitFact]) -> Vec<usize> {
        self.flatten_precondition_groups(&self.precondition_proposition_id_groups(preconditions))
    }

    pub(super) fn precondition_proposition_id_groups(
        &self,
        preconditions: &[ExplicitFact],
    ) -> Vec<Vec<usize>> {
        preconditions
            .iter()
            .map(|precondition| self.precondition_proposition_ids(precondition))
            .filter(|group| !group.is_empty())
            .collect()
    }

    pub(super) fn flatten_precondition_groups(&self, groups: &[Vec<usize>]) -> Vec<usize> {
        let mut result = Vec::new();
        let mut seen = BTreeSet::new();
        for group in groups {
            for &proposition_id in group {
                if seen.insert(proposition_id) {
                    result.push(proposition_id);
                }
            }
        }
        result
    }

    pub(super) fn append_numeric_condition_propositions(
        &mut self,
        condition_group_ids: &[usize],
        target_ids: &mut Vec<usize>,
    ) {
        for condition_id in self
            .numeric_helper
            .get_condition_ids_from_group_ids(condition_group_ids)
        {
            let Ok(proposition_id) = self.get_numeric_proposition_id(condition_id) else {
                continue;
            };
            if !target_ids.contains(&proposition_id) {
                target_ids.push(proposition_id);
            }
        }
    }

    pub(super) fn build_comparison_fact_condition_ids(&mut self) {
        if self.config.ignore_numeric {
            return;
        }

        for comparison_axiom in self.task.comparison_axioms().iter() {
            let affected_var_id = comparison_axiom.affected_var_id();
            let condition_ids = self
                .numeric_helper
                .get_comparison_fact_condition_ids(affected_var_id, COMPARISON_FALSE_FACT_VALUE);
            if !condition_ids.is_empty() {
                self.comparison_false_fact_condition_ids
                    .insert(affected_var_id, condition_ids);
            }
        }
    }

    pub(super) fn add_linear_conditions(&mut self) {
        if self.config.ignore_numeric {
            return;
        }

        let task = Arc::clone(&self.task);
        self.linear_effect_to_conditions_plus = vec![Vec::new(); task.operators().len()];
        self.linear_effect_to_conditions_minus = vec![Vec::new(); task.operators().len()];

        for (operator_id, operator) in task.operators().iter().enumerate() {
            let helper_conditional_fact_effects = self
                .numeric_helper
                .get_action_conditional_fact_effects(operator_id)
                .map(|effects| effects.to_vec())
                .expect("helper conditional fact effects must exist for operator");
            let helper_linear_effects = self
                .numeric_helper
                .get_action_linear_effects(operator_id)
                .map(|effects| effects.to_vec())
                .expect("helper linear effects must exist for operator");
            self.linear_effect_to_conditions_plus[operator_id] =
                vec![Vec::new(); operator.assignment_effects().len()];
            self.linear_effect_to_conditions_minus[operator_id] =
                vec![Vec::new(); operator.assignment_effects().len()];

            let base_precondition_groups = self.precondition_proposition_id_groups(
                self.numeric_helper
                    .get_action_pre_list(operator_id)
                    .expect("helper action pre-list must exist for operator"),
            );
            let helper_num_list = self
                .numeric_helper
                .get_action_num_list(operator_id)
                .map(|ids| ids.to_vec())
                .expect("helper action numeric pre-list must exist for operator");
            let mut expanded_base_precondition_ids =
                self.flatten_precondition_groups(&base_precondition_groups);
            self.append_numeric_condition_propositions(
                &helper_num_list,
                &mut expanded_base_precondition_ids,
            );

            if !base_precondition_groups.is_empty() {
                let mut global_base_precondition_ids =
                    self.flatten_precondition_groups(&base_precondition_groups);
                self.append_numeric_condition_propositions(
                    &helper_num_list,
                    &mut global_base_precondition_ids,
                );
                // Keep this explicit even though the local vector is not reused below:
                // helper-owned action precondition shaping still materializes the redundant
                // conditions globally through `add_numeric_condition_proposition()`, mirroring the
                // reference-side global condition growth in `numeric_helper::build_action()`.
            }

            for conditional_effect in &helper_conditional_fact_effects {
                let effect_condition_groups = self.precondition_proposition_id_groups(
                    &conditional_effect.preconditions.propositional_facts,
                );
                if effect_condition_groups.is_empty() {
                    continue;
                }

                let mut expanded_effect_condition_ids =
                    self.flatten_precondition_groups(&effect_condition_groups);
                self.append_numeric_condition_propositions(
                    &conditional_effect.preconditions.numeric_group_ids,
                    &mut expanded_effect_condition_ids,
                );
                // The expanded ids are transient here, but the redundant numeric conditions are
                // still materialized globally through `append_*_redundant_numeric_conditions()`.
            }

            for linear_effect in &helper_linear_effects {
                let assignment_effect_id = linear_effect.source_assignment_effect_id;
                let helper_effect_preconditions = linear_effect.preconditions.clone();
                let mut extended_numeric_group_ids =
                    helper_effect_preconditions.numeric_group_ids.clone();
                for &group_id in &helper_num_list {
                    if !extended_numeric_group_ids.contains(&group_id) {
                        extended_numeric_group_ids.push(group_id);
                    }
                }
                let assignment_condition_groups = self.precondition_proposition_id_groups(
                    &helper_effect_preconditions.propositional_facts,
                );
                if !assignment_condition_groups.is_empty()
                    || !helper_effect_preconditions.numeric_group_ids.is_empty()
                {
                    let mut expanded_effect_condition_ids =
                        self.flatten_precondition_groups(&assignment_condition_groups);
                    self.append_numeric_condition_propositions(
                        &helper_effect_preconditions.numeric_group_ids,
                        &mut expanded_effect_condition_ids,
                    );
                    // Same as above: the local vector is disposable, but the condition material-
                    // ization side effects still persist globally in `self.conditions`.
                }

                let mut extended_precondition_ids = expanded_base_precondition_ids.clone();
                let mut seen_preconditions: BTreeSet<usize> =
                    extended_precondition_ids.iter().copied().collect();
                for group in &assignment_condition_groups {
                    for &proposition_id in group {
                        if seen_preconditions.insert(proposition_id) {
                            extended_precondition_ids.push(proposition_id);
                        }
                    }
                }
                self.append_numeric_condition_propositions(
                    &helper_effect_preconditions.numeric_group_ids,
                    &mut extended_precondition_ids,
                );

                let reconstructed_linear_effect = self
                    .numeric_helper
                    .linearized_effect_for_action_assignment(operator_id, assignment_effect_id)
                    .unwrap_or_else(|| {
                        panic!(
                            "LM-cut helper linearized effect {assignment_effect_id} is missing for operator {operator_id}"
                        )
                    });
                let affected_var_id = reconstructed_linear_effect.affected_var_id;
                let delta_expression = reconstructed_linear_effect.delta;

                let plus_proposition_id =
                    self.add_numeric_condition_proposition(NumericCondition::from_expression(
                        delta_expression.clone(),
                        true,
                        format!(
                            "numeric ({} {} +inf guard)",
                            operator.name(),
                            affected_var_id
                        ),
                    ));
                let plus_condition_id = self.propositions[plus_proposition_id]
                    .id_numeric_condition
                    .expect("new +inf condition proposition must reference its condition");
                self.linear_effect_to_conditions_plus[operator_id][assignment_effect_id]
                    .push(plus_condition_id);

                let plus_redundant_conditions = self
                    .conditions
                    .get(plus_condition_id)
                    .map(|base_condition| {
                        let other_conditions = self
                            .representative_numeric_conditions_for_group_ids(
                                &extended_numeric_group_ids,
                            );
                        self.numeric_helper
                            .combine_condition_with_conditions(base_condition, &other_conditions)
                    })
                    .unwrap_or_default();
                for redundant_condition in plus_redundant_conditions {
                    let redundant_proposition_id =
                        self.add_numeric_condition_proposition(redundant_condition);
                    let redundant_condition_id = self.propositions[redundant_proposition_id]
                        .id_numeric_condition
                        .expect(
                            "new redundant +inf condition proposition must reference its condition",
                        );
                    self.linear_effect_to_conditions_plus[operator_id][assignment_effect_id]
                        .push(redundant_condition_id);
                }

                let minus_proposition_id =
                    self.add_numeric_condition_proposition(NumericCondition::from_expression(
                        delta_expression.scale(-1.0),
                        true,
                        format!(
                            "numeric ({} {} -inf guard)",
                            operator.name(),
                            affected_var_id
                        ),
                    ));
                let minus_condition_id = self.propositions[minus_proposition_id]
                    .id_numeric_condition
                    .expect("new -inf condition proposition must reference its condition");
                self.linear_effect_to_conditions_minus[operator_id][assignment_effect_id]
                    .push(minus_condition_id);

                let minus_redundant_conditions = self
                    .conditions
                    .get(minus_condition_id)
                    .map(|base_condition| {
                        let other_conditions = self
                            .representative_numeric_conditions_for_group_ids(
                                &extended_numeric_group_ids,
                            );
                        self.numeric_helper
                            .combine_condition_with_conditions(base_condition, &other_conditions)
                    })
                    .unwrap_or_default();
                for redundant_condition in minus_redundant_conditions {
                    let redundant_proposition_id =
                        self.add_numeric_condition_proposition(redundant_condition);
                    let redundant_condition_id = self.propositions[redundant_proposition_id]
                        .id_numeric_condition
                        .expect(
                            "new redundant -inf condition proposition must reference its condition",
                        );
                    self.linear_effect_to_conditions_minus[operator_id][assignment_effect_id]
                        .push(redundant_condition_id);
                }
            }
        }
    }

    pub(super) fn representative_numeric_conditions_for_group_ids(
        &self,
        group_ids: &[usize],
    ) -> Vec<NumericCondition> {
        group_ids
            .iter()
            .filter_map(|&group_id| {
                self.numeric_helper
                    .condition_group_representative_condition_id(group_id)
                    .and_then(|condition_id| self.numeric_helper.get_condition(condition_id))
                    .cloned()
            })
            .collect()
    }

    pub(super) fn add_numeric_condition_proposition(
        &mut self,
        condition: NumericCondition,
    ) -> usize {
        let epsilon = if condition.is_strictly_greater {
            self.config.epsilon
        } else {
            0.0
        };
        self.add_numeric_condition_proposition_with_epsilon(condition, epsilon)
    }

    pub(super) fn add_numeric_condition_proposition_with_epsilon(
        &mut self,
        condition: NumericCondition,
        epsilon: f64,
    ) -> usize {
        let condition_id = self.conditions.len();
        let proposition_id = self.propositions.len();
        let mut proposition = RelaxedProposition::new(proposition_id, condition.name.clone());
        proposition.is_numeric_condition = true;
        proposition.id_numeric_condition = Some(condition_id);
        self.propositions.push(proposition);
        self.numeric_condition_proposition_ids.push(proposition_id);
        self.conditions.push(condition);
        self.epsilons.push(epsilon);
        self.num_propositions += 1;
        proposition_id
    }

    pub(super) fn get_numeric_proposition_id(&self, condition_id: usize) -> Result<usize, String> {
        self.numeric_condition_proposition_ids
            .get(condition_id)
            .copied()
            .ok_or_else(|| format!("LM-cut numeric condition {condition_id} has no proposition"))
    }

    #[inline(always)]
    pub(super) fn get_numeric_proposition_id_infallible(&self, condition_id: usize) -> usize {
        self.numeric_condition_proposition_ids[condition_id]
    }

    pub(super) fn precondition_proposition_ids(&self, fact: &ExplicitFact) -> Vec<usize> {
        if !self.config.ignore_numeric
            && self
                .numeric_helper
                .is_comparison_axiom_var(fact.var_index())
        {
            if fact.value_index() != COMPARISON_FALSE_FACT_VALUE {
                return Vec::new();
            }
            // Every key of the map is the affected variable of a comparison
            // axiom, which is exactly what `is_comparison_axiom_var` reports.
            if let Some(condition_ids) = self
                .comparison_false_fact_condition_ids
                .get(&fact.var_index())
            {
                return condition_ids
                    .iter()
                    .map(|&condition_id| {
                        self.get_numeric_proposition_id(condition_id)
                            .expect("comparison fact condition must have proposition")
                    })
                    .collect();
            }
        }

        if self.is_numeric_axiom_var(fact.var_index()) {
            // Reached when either `ignore_numeric=true` (numeric tracking is disabled
            // wholesale) or the fact references a numeric-axiom var that isn't a
            // comparison axiom registered in `comparison_false_fact_condition_ids`
            // (e.g. an assignment-axiom var, or a comparison var whose value=0
            // entry produced an empty condition list at build time).
            //
            // Dropping the precondition is an admissible relaxation: an operator
            // whose numeric/axiom precondition we cannot model becomes easier to
            // apply, never harder. Reachability over-approximates and the LM-cut
            // sum of cut costs is still a valid lower bound on optimal plan cost.
            // The prior `panic!` aborted whole searches on plant-watering/6_2_2
            // and similar tasks where assignment-axiom preconditions surface.
            return Vec::new();
        }

        vec![self.get_proposition_id(fact)]
    }
}
