//! Assembling the relaxed task LM-cut explores: a relaxed operator per
//! operator and per axiom, the goal operator, and the cross references and
//! packed adjacency the explorations walk.

use super::*;

impl LandmarkCutLandmarks {
    pub(super) fn build_relaxed_operators(&mut self) {
        let task = Arc::clone(&self.task);
        let operators = task.operators();
        for (operator_id, operator) in operators.iter().enumerate() {
            let operator_id = OperatorIndex::from_usize(operator_id);
            let base_cost = self.calculate_base_operator_cost(operator_id);
            self.build_relaxed_operator_for_operator(operator_id, operator, base_cost)
                .expect("LM-cut numeric operator construction must succeed");
        }

        let relaxed_axioms = Arc::clone(&self.relaxed_axioms);
        for (axiom_offset, axiom) in relaxed_axioms.iter().enumerate() {
            let operator_id = OperatorIndex::from_usize(operators.len() + axiom_offset);
            self.build_relaxed_operator_for_axiom(operator_id, axiom);
        }

        self.build_supported_sose_operators()
            .expect("LM-cut supported SOSE operator construction must succeed");
        self.prune_infinite_effects_for_supported_sose();

        self.build_simple_effects()
            .expect("LM-cut simple-effect construction must succeed");

        self.delete_noops();
    }

    pub(super) fn delete_noops(&mut self) {
        self.relaxed_operators
            .retain(|operator| !operator.effect_ids.is_empty());
    }

    pub(super) fn prune_infinite_effects_for_supported_sose(&mut self) {
        let operator_count = self.task.operators().len();
        for relaxed_operator in &mut self.relaxed_operators {
            if !relaxed_operator.infinite {
                continue;
            }

            let Some(original_op_id) = relaxed_operator
                .original_op_id_2
                .filter(|&operator_id| operator_id.index() < operator_count)
            else {
                continue;
            };

            relaxed_operator.effect_ids.retain(|&effect_id| {
                let Some(condition_id) = self
                    .propositions
                    .get(effect_id)
                    .and_then(|proposition| proposition.id_numeric_condition)
                else {
                    return true;
                };

                !self
                    .operator_condition_eval
                    .get(original_op_id, condition_id)
                    .has_sose
            });
        }
    }

    pub(super) fn build_goal_operator(&mut self) {
        let mut goal_preconditions = self.goal_precondition_ids.clone();
        if goal_preconditions.is_empty() {
            goal_preconditions.push(self.artificial_precondition_id);
        }

        let mut goal_operator = RelaxedOperator::new(
            goal_preconditions,
            vec![self.artificial_goal_id],
            OperatorIndex::new(u32::MAX),
            0.0,
            "goal".to_string(),
            false,
        );
        goal_operator.original_op_id_2 = None;
        goal_operator.assert_well_formed();
        self.relaxed_operators.push(goal_operator);
    }

    pub(super) fn prepare_goal_preconditions(&mut self) {
        let mut goal_preconditions = Vec::new();
        let mut seen = BTreeSet::new();
        for goal_index in 0..self.task.goals().len() {
            let goal = self.task.goals()[goal_index];
            let helper_propositional_ids = self
                .numeric_helper
                .get_propositional_goals(goal_index)
                .map(|goals| self.build_precondition_ids(goals))
                .unwrap_or_default();
            // When `ignore_numeric=true`, `build_comparison_fact_condition_ids` and
            // `add_linear_conditions` both return early without registering any numeric
            // condition propositions, so helper goals over numeric vars would fail the
            // lookup. Drop them silently in that mode — the heuristic intentionally
            // ignores numerics, so omitting them is admissible (under-approximates the
            // goal, never over-approximates).
            let helper_numeric_condition_ids: Vec<usize> = if self.config.ignore_numeric {
                Vec::new()
            } else {
                self.numeric_helper.get_numeric_goals(goal_index).to_vec()
            };
            let helper_numeric_ids = helper_numeric_condition_ids
                .iter()
                .map(|&condition_id| {
                    self.get_numeric_proposition_id(condition_id).expect(
                        "helper goal numeric condition must already have a canonical LM-cut proposition",
                    )
                })
                .collect::<Vec<_>>();

            if helper_propositional_ids.is_empty() && helper_numeric_ids.is_empty() {
                // A goal on a comparison variable (the translator names a
                // numeric goal condition directly) maps to its numeric
                // conditions exactly as a precondition does. Dropping it would
                // leave the goal operator without the goal: h = 0.
                let goal_proposition_ids = self.precondition_proposition_ids(&goal);
                assert!(
                    !goal_proposition_ids.is_empty()
                        || self.config.ignore_numeric
                        || !self.is_numeric_axiom_var(goal.var_index()),
                    "LM-cut cannot model goal fact {goal:?}; dropping it would drop the goal"
                );
                for proposition_id in goal_proposition_ids {
                    if seen.insert(proposition_id) {
                        goal_preconditions.push(proposition_id);
                    }
                }
                continue;
            }

            // PARITY(numeric-fd): when a goal is compiled through numeric helper axioms,
            // the C++ goal operator uses the helper propositional/numeric conditions produced by
            // `build_numeric_goals()` rather than also keeping the derived goal fact itself as a
            // precondition. Keeping both over-constrains the goal operator and can create false
            // dead ends.
            // The shared helper now owns the reference-side `fact_to_axiom_map = -2`
            // goal classification, so later state/precondition handling skips the direct fact.

            for proposition_id in helper_propositional_ids
                .iter()
                .chain(helper_numeric_ids.iter())
                .copied()
            {
                if seen.insert(proposition_id) {
                    goal_preconditions.push(proposition_id);
                }
            }
        }
        self.goal_precondition_ids = goal_preconditions;
    }

    pub(super) fn build_cross_references(&mut self) {
        for proposition in &mut self.propositions {
            proposition.precondition_of.clear();
            proposition.effect_of.clear();
        }

        for (operator_index, operator) in self.relaxed_operators.iter().enumerate() {
            let operator_index = OperatorIndex::from_usize(operator_index);
            for &proposition_id in &operator.precondition_ids {
                let proposition = self
                    .propositions
                    .get_mut(proposition_id)
                    .expect("precondition proposition id must be valid");
                proposition.precondition_of.push(operator_index);
            }
            for &proposition_id in &operator.effect_ids {
                let proposition = self
                    .propositions
                    .get_mut(proposition_id)
                    .expect("effect proposition id must be valid");
                proposition.effect_of.push(operator_index);
            }
        }
    }

    pub(super) fn build_packed_runtime_adjacency(&mut self) {
        self.proposition_precondition_of_data.clear();
        self.proposition_precondition_of_ranges.clear();
        self.proposition_precondition_of_ranges
            .reserve(self.propositions.len());
        for proposition in &self.propositions {
            let start = self.proposition_precondition_of_data.len();
            self.proposition_precondition_of_data
                .extend_from_slice(&proposition.precondition_of);
            let end = self.proposition_precondition_of_data.len();
            self.proposition_precondition_of_ranges.push((start, end));
        }

        self.proposition_effect_of_data.clear();
        self.proposition_effect_of_ranges.clear();
        self.proposition_effect_of_ranges
            .reserve(self.propositions.len());
        for proposition in &self.propositions {
            let start = self.proposition_effect_of_data.len();
            self.proposition_effect_of_data
                .extend_from_slice(&proposition.effect_of);
            let end = self.proposition_effect_of_data.len();
            self.proposition_effect_of_ranges.push((start, end));
        }

        self.operator_precondition_id_data.clear();
        self.operator_precondition_id_ranges.clear();
        self.operator_precondition_id_ranges
            .reserve(self.relaxed_operators.len());
        self.operator_effect_id_data.clear();
        self.operator_effect_id_ranges.clear();
        self.operator_effect_id_ranges
            .reserve(self.relaxed_operators.len());
        for operator in &self.relaxed_operators {
            let precondition_start = self.operator_precondition_id_data.len();
            self.operator_precondition_id_data
                .extend_from_slice(&operator.precondition_ids);
            let precondition_end = self.operator_precondition_id_data.len();
            self.operator_precondition_id_ranges
                .push((precondition_start, precondition_end));

            let effect_start = self.operator_effect_id_data.len();
            self.operator_effect_id_data
                .extend_from_slice(&operator.effect_ids);
            let effect_end = self.operator_effect_id_data.len();
            self.operator_effect_id_ranges
                .push((effect_start, effect_end));
        }
    }

    pub(super) fn build_original_to_relaxed_index(&mut self) {
        let operator_count = self.task.operators().len() + self.relaxed_axioms.len();
        self.original_to_relaxed_operators = vec![Vec::new(); operator_count];
        for (relaxed_operator_id, operator) in self.relaxed_operators.iter().enumerate() {
            let relaxed_operator_id = OperatorIndex::from_usize(relaxed_operator_id);
            if let Some(original_id) = operator.original_op_id_1
                && let Some(mapped) = self
                    .original_to_relaxed_operators
                    .get_mut(original_id.index())
            {
                mapped.push(relaxed_operator_id);
            }
            if let Some(original_id) = operator.original_op_id_2
                && let Some(mapped) = self
                    .original_to_relaxed_operators
                    .get_mut(original_id.index())
            {
                mapped.push(relaxed_operator_id);
            }
        }
    }

    pub(super) fn calculate_base_operator_cost(&self, operator_id: OperatorIndex) -> f64 {
        assert!(
            operator_id.index() < self.task.operators().len(),
            "base operator cost is only defined for concrete operators"
        );
        let mut operator_cost = self.task.operator_cost(operator_id);

        if self.task.is_linear_cost_operator(operator_id) && self.use_bounds {
            let coefficients = self.task.operator_cost_coefficients(operator_id);
            operator_cost = self.task.operator_cost_constant(operator_id);

            for (numeric_var_id, &weight) in coefficients.iter().enumerate() {
                if weight >= self.config.precision
                    && self.numeric_bound().get_variable_before_action_has_lb(
                        VariableIndex::from_usize(numeric_var_id),
                        operator_id,
                    )
                {
                    operator_cost += weight
                        * self.numeric_bound().get_variable_before_action_lb(
                            VariableIndex::from_usize(numeric_var_id),
                            operator_id,
                        );
                } else if weight <= -self.config.precision
                    && self.numeric_bound().get_variable_before_action_has_ub(
                        VariableIndex::from_usize(numeric_var_id),
                        operator_id,
                    )
                {
                    operator_cost += weight
                        * self.numeric_bound().get_variable_before_action_ub(
                            VariableIndex::from_usize(numeric_var_id),
                            operator_id,
                        );
                } else if weight.abs() >= self.config.precision {
                    operator_cost = 0.0;
                    break;
                }
            }
        }

        operator_cost.max(0.0)
    }

    pub(super) fn build_relaxed_operator_for_operator(
        &mut self,
        operator_id: OperatorIndex,
        operator: &Operator,
        base_cost: f64,
    ) -> Result<(), String> {
        let helper_linearized_assignment_effects =
            self.numeric_helper.linearized_effects_for_action(
                operator_id.index(),
                operator.assignment_effects().len(),
            )?;
        let helper_conditional_fact_effects = self
            .numeric_helper
            .get_action_conditional_fact_effects(operator_id.index())
            .map(|effects| effects.to_vec())
            .ok_or_else(|| {
                format!("LM-cut helper conditional fact effects {operator_id} are missing")
            })?;
        let helper_linear_effects = self
            .numeric_helper
            .get_action_linear_effects(operator_id.index())
            .map(|effects| effects.to_vec())
            .ok_or_else(|| format!("LM-cut helper linear effects {operator_id} are missing"))?;
        let helper_pre_list = self
            .numeric_helper
            .get_action_pre_list(operator_id.index())
            .expect("helper action pre-list must exist for operator");
        let helper_num_list = self
            .numeric_helper
            .get_action_num_list(operator_id.index())
            .map(|ids| ids.to_vec())
            .expect("helper action numeric pre-list must exist for operator");
        let precondition_groups = self.precondition_proposition_id_groups(helper_pre_list);
        let mut precondition_ids = self.flatten_precondition_groups(&precondition_groups);
        self.append_numeric_condition_propositions(&helper_num_list, &mut precondition_ids);
        let unconditional_linear_assignment_effect_ids = helper_linear_effects
            .iter()
            .filter(|effect| {
                effect.preconditions.propositional_facts.is_empty()
                    && effect.preconditions.numeric_group_ids.is_empty()
            })
            .map(|effect| effect.source_assignment_effect_id)
            .collect::<Vec<_>>();

        for conditional_effect in &helper_conditional_fact_effects {
            if self.is_numeric_axiom_var(conditional_effect.add_fact.var_index()) {
                continue;
            }
            let mut extended_preconditions = precondition_ids.clone();
            let mut seen: BTreeSet<usize> = extended_preconditions.iter().copied().collect();
            let effect_condition_groups = self.precondition_proposition_id_groups(
                &conditional_effect.preconditions.propositional_facts,
            );
            for group in &effect_condition_groups {
                for &proposition_id in group {
                    if seen.insert(proposition_id) {
                        extended_preconditions.push(proposition_id);
                    }
                }
            }
            self.append_numeric_condition_propositions(
                &conditional_effect.preconditions.numeric_group_ids,
                &mut extended_preconditions,
            );
            let conditional_name = format!(
                "{} {}",
                operator.name(),
                self.get_proposition_name(
                    conditional_effect.add_fact.var_index(),
                    conditional_effect.add_fact.value_index()
                )
            );
            let conditional_operator = RelaxedOperator::new(
                extended_preconditions,
                vec![self.get_proposition_id(&conditional_effect.add_fact)],
                operator_id,
                base_cost,
                conditional_name,
                true,
            );
            conditional_operator.assert_well_formed();
            self.relaxed_operators.push(conditional_operator);
        }

        let base_effect_ids = self
            .numeric_helper
            .get_action_add_list(operator_id.index())
            .into_iter()
            .flatten()
            .filter(|fact| !self.is_numeric_axiom_var(fact.var_index()))
            .map(|fact| self.get_proposition_id(fact))
            .collect::<Vec<_>>();

        let mut relaxed_operator = RelaxedOperator::new(
            if precondition_ids.is_empty() {
                vec![self.artificial_precondition_id]
            } else {
                precondition_ids.clone()
            },
            base_effect_ids.clone(),
            operator_id,
            base_cost,
            operator.name().to_string(),
            false,
        );
        relaxed_operator.assignment_effect_ids = unconditional_linear_assignment_effect_ids.clone();
        relaxed_operator.linear_assignment_effects = unconditional_linear_assignment_effect_ids
            .iter()
            .map(|&assignment_effect_id| {
                helper_linearized_assignment_effects
                    .get(assignment_effect_id)
                    .and_then(|effect| effect.clone())
                    .expect("helper linearized assignment effect id must be valid")
            })
            .collect();
        relaxed_operator.assert_well_formed();
        self.relaxed_operators.push(relaxed_operator);

        self.build_linear_operators(
            operator_id,
            operator,
            base_cost,
            &precondition_ids,
            &helper_linearized_assignment_effects,
        )?;
        Ok(())
    }

    pub(super) fn build_relaxed_operator_for_axiom(
        &mut self,
        operator_id: OperatorIndex,
        axiom: &PropositionalAxiom,
    ) {
        let helper_preconditions = self
            .numeric_helper
            .get_action_pre_list(operator_id.index())
            .expect("helper action pre-list for axiom must exist");
        let helper_num_list = self
            .numeric_helper
            .get_action_num_list(operator_id.index())
            .map(|ids| ids.to_vec())
            .expect("helper action numeric pre-list for axiom must exist");
        let precondition_groups = self.precondition_proposition_id_groups(helper_preconditions);
        let mut precondition_ids = self.flatten_precondition_groups(&precondition_groups);
        self.append_numeric_condition_propositions(&helper_num_list, &mut precondition_ids);
        let effect_fact = self
            .numeric_helper
            .get_action_add_list(operator_id.index())
            .and_then(|add_facts| add_facts.first())
            .cloned()
            .unwrap_or_else(|| {
                ExplicitFact::propositional(axiom.var_id().index(), axiom.effect_value().index())
            });
        let effect_ids = if self.is_numeric_axiom_var(axiom.var_id()) {
            Vec::new()
        } else {
            vec![self.get_proposition_id(&effect_fact)]
        };
        let relaxed_operator = RelaxedOperator::new(
            if precondition_ids.is_empty() {
                vec![self.artificial_precondition_id]
            } else {
                precondition_ids
            },
            effect_ids,
            operator_id,
            0.0,
            format!(
                "axiom {}",
                self.get_proposition_name(effect_fact.var_index(), effect_fact.value_index())
            ),
            false,
        );
        relaxed_operator.assert_well_formed();
        self.relaxed_operators.push(relaxed_operator);
    }
}
