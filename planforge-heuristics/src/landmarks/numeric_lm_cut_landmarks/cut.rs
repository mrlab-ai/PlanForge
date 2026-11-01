//! Cut computation: the goal plateau, the second exploration that finds the
//! cut, and the landmark loop that sums the cut costs into the estimate.

use super::*;

impl LandmarkCutLandmarks {
    pub(super) fn mark_goal_plateau(
        &mut self,
        propositional_values: &[ExplicitValueIndex],
        numeric_values: &[NumericValue],
        proposition_id: usize,
    ) {
        if self.proposition_status(proposition_id) == PropositionStatus::GoalZone {
            return;
        }
        // Depth-first over zero-cost applicable achievers, with an explicit
        // stack of (proposition, next achiever index, end) in the order the
        // recursive definition visits: a deep plateau would overflow the call
        // stack.
        self.set_proposition_status(proposition_id, PropositionStatus::GoalZone);
        let (start, end) = self.proposition_effect_of_range(proposition_id);
        let mut stack = std::mem::take(&mut self.plateau_stack);
        stack.clear();
        stack.push((proposition_id, start, end));
        while let Some(top) = stack.last_mut() {
            let (proposition_id, achiever_index, end) = *top;
            if achiever_index == end {
                stack.pop();
                continue;
            }
            top.1 += 1;
            let achiever_id = self.proposition_effect_of_data[achiever_index];
            let is_zero_cost_applicable = self.operator_cost_1(achiever_id) < self.config.precision
                && self.operator_cost_2(achiever_id) < self.config.precision
                && self.operator_unsatisfied_preconditions(achiever_id) == 0;
            if !is_zero_cost_applicable {
                continue;
            }
            let Some(supporter_id) = self.operator_h_max_supporter(achiever_id) else {
                continue;
            };
            let ms = self.calculate_numeric_times(
                propositional_values,
                numeric_values,
                proposition_id,
                achiever_id,
                !self.config.disable_ma,
            );
            if !self.multiplier_allows_traversal(achiever_id, ms)
                || self.proposition_status(supporter_id) == PropositionStatus::GoalZone
            {
                continue;
            }
            self.set_proposition_status(supporter_id, PropositionStatus::GoalZone);
            let (start, end) = self.proposition_effect_of_range(supporter_id);
            stack.push((supporter_id, start, end));
        }
        self.plateau_stack = stack;
    }

    pub(super) fn second_exploration(
        &mut self,
        propositional_values: &[ExplicitValueIndex],
        numeric_values: &[NumericValue],
        cut: &mut Vec<OperatorIndex>,
        m_list: &mut Vec<(f64, f64)>,
    ) {
        assert!(
            cut.is_empty(),
            "LM-cut second exploration requires empty cut"
        );
        assert!(
            m_list.is_empty(),
            "LM-cut second exploration requires empty multiplier list"
        );
        self.start_cut_marking();
        let mut queue = std::mem::take(&mut self.second_exploration_queue_scratch);
        queue.clear();
        self.set_proposition_status(
            self.artificial_precondition_id,
            PropositionStatus::BeforeGoalZone,
        );
        queue.push(self.artificial_precondition_id);

        for (variable_id, &value) in propositional_values.iter().enumerate() {
            if self.is_numeric_axiom_var(VariableIndex::from_usize(variable_id))
                && !self.config.ignore_numeric
            {
                continue;
            }
            let fact = self
                .numeric_helper
                .fact(VariableIndex::from_usize(variable_id), value);
            let proposition_id = self.get_proposition_id(&fact);
            if self.proposition_status(proposition_id) != PropositionStatus::BeforeGoalZone {
                self.set_proposition_status(proposition_id, PropositionStatus::BeforeGoalZone);
                queue.push(proposition_id);
            }
        }

        if !self.config.ignore_numeric {
            for condition_id in 0..self.conditions.len() {
                if self.numeric_initial_state[condition_id] < self.config.precision {
                    let proposition_id = self.get_numeric_proposition_id_infallible(condition_id);
                    if self.proposition_status(proposition_id) != PropositionStatus::BeforeGoalZone
                    {
                        self.set_proposition_status(
                            proposition_id,
                            PropositionStatus::BeforeGoalZone,
                        );
                        queue.push(proposition_id);
                    }
                }
            }
        }

        while let Some(proposition_id) = queue.pop() {
            let (triggered_start, triggered_end) =
                self.proposition_precondition_of_range(proposition_id);
            for triggered_index in triggered_start..triggered_end {
                let operator_id = self.proposition_precondition_of_data[triggered_index];
                let should_process = {
                    self.operator_h_max_supporter(operator_id) == Some(proposition_id)
                        && !self.is_cut_marked(operator_id)
                };
                if !should_process {
                    continue;
                }

                let (effect_start, effect_end) = self.operator_effect_id_range(operator_id);
                let mut min_cut_cost = f64::INFINITY;

                for effect_index in effect_start..effect_end {
                    let effect_id = self.operator_effect_id_data[effect_index];
                    let effect_status = self.proposition_status(effect_id);
                    if effect_status == PropositionStatus::GoalZone {
                        let ms = self.calculate_numeric_times(
                            propositional_values,
                            numeric_values,
                            effect_id,
                            operator_id,
                            !self.config.disable_ma,
                        );
                        let operator = &self.relaxed_operators[operator_id.index()];
                        if (operator.original_op_id_1.is_some() && ms.0 >= self.config.precision)
                            || (operator.original_op_id_1.is_none()
                                && ms.1 >= self.config.precision)
                        {
                            let edge_cost = self.edge_cost(operator_id, ms);
                            cut.push(operator_id);
                            m_list.push(ms);
                            self.mark_cut(operator_id);
                            min_cut_cost = min_cut_cost.min(edge_cost);
                        }
                    }
                }

                for effect_index in effect_start..effect_end {
                    let effect_id = self.operator_effect_id_data[effect_index];
                    let effect_status = self.proposition_status(effect_id);
                    if effect_status == PropositionStatus::BeforeGoalZone
                        || effect_status == PropositionStatus::GoalZone
                    {
                        continue;
                    }
                    let ms = self.calculate_numeric_times(
                        propositional_values,
                        numeric_values,
                        effect_id,
                        operator_id,
                        !self.config.disable_ma,
                    );
                    let operator = &self.relaxed_operators[operator_id.index()];
                    if (operator.original_op_id_1.is_some() && ms.0 >= self.config.precision)
                        || (operator.original_op_id_1.is_none() && ms.1 >= self.config.precision)
                    {
                        let edge_cost = self.edge_cost(operator_id, ms);
                        if edge_cost < min_cut_cost {
                            assert_eq!(effect_status, PropositionStatus::Reached);
                            self.set_proposition_status(
                                effect_id,
                                PropositionStatus::BeforeGoalZone,
                            );
                            queue.push(effect_id);
                        }
                    }
                }
            }
        }
        queue.clear();
        self.second_exploration_queue_scratch = queue;
    }

    #[inline(always)]
    pub(super) fn multiplier_allows_traversal(
        &self,
        operator_id: OperatorIndex,
        ms: (f64, f64),
    ) -> bool {
        let operator = &self.relaxed_operators[operator_id.index()];
        (operator.original_op_id_1.is_some() && ms.0 >= self.config.precision)
            || ms.1 >= self.config.precision
    }

    pub(super) fn edge_cost(&self, operator_id: OperatorIndex, ms: (f64, f64)) -> f64 {
        let operator = &self.relaxed_operators[operator_id.index()];
        let mut edge_cost = ms.1 * self.operator_cost_2(operator_id);
        if operator.original_op_id_1.is_some() {
            edge_cost += ms.0 * self.operator_cost_1(operator_id);
        }
        edge_cost
    }

    pub(super) fn reset_goal_zone_statuses(&mut self) {
        for &proposition_id in &self.goal_zone_proposition_ids {
            if self.proposition_status(proposition_id) == PropositionStatus::GoalZone
                || self.proposition_status(proposition_id) == PropositionStatus::BeforeGoalZone
            {
                self.proposition_runtime[proposition_id].status = PropositionStatus::Reached;
            }
            self.proposition_runtime[proposition_id].goal_zone_touched = false;
        }
        self.goal_zone_proposition_ids.clear();
    }

    pub(super) fn compute_landmarks_impl(
        &mut self,
        propositional_values: &[ExplicitValueIndex],
        numeric_values: &[NumericValue],
    ) -> Result<(bool, f64), String> {
        assert!(
            self.initialized,
            "LM-cut landmarks used before initialization"
        );
        assert_eq!(
            numeric_values.len(),
            self.task.numeric_variables().len(),
            "LM-cut scaffold received the wrong number of numeric values"
        );
        let mut cut = std::mem::take(&mut self.cut_scratch);
        cut.clear();
        let mut m_list = std::mem::take(&mut self.multiplier_scratch);
        m_list.clear();
        let result = (|| {
            self.start_state_numeric_tracking();
            self.first_exploration(propositional_values, numeric_values)?;
            if self.proposition_status(self.artificial_goal_id) == PropositionStatus::Unreached {
                return Ok((true, f64::INFINITY));
            }

            let mut total_cost = 0.0;

            while self.proposition_h_max_cost(self.artificial_goal_id) >= self.config.precision {
                self.start_cut_iteration_tracking();
                self.mark_goal_plateau(
                    propositional_values,
                    numeric_values,
                    self.artificial_goal_id,
                );
                self.second_exploration(
                    propositional_values,
                    numeric_values,
                    &mut cut,
                    &mut m_list,
                );
                assert!(!cut.is_empty(), "LM-cut must find a non-empty cut");

                let mut cut_cost = f64::INFINITY;

                for (cut_index, &operator_id) in cut.iter().enumerate() {
                    let multiplier = m_list[cut_index];
                    let current_cut_cost = self.edge_cost(operator_id, multiplier);
                    let (original_op_id_1, original_op_id_2) = {
                        let operator = &self.relaxed_operators[operator_id.index()];
                        (operator.original_op_id_1, operator.original_op_id_2)
                    };
                    if multiplier.0 >= self.config.precision
                        && let Some(original_id) = original_op_id_1
                    {
                        self.update_original_operator_min_cut_cost(original_id, current_cut_cost);
                    }
                    if let Some(original_id) = original_op_id_2 {
                        self.update_original_operator_min_cut_cost(original_id, current_cut_cost);
                    }
                    cut_cost = cut_cost.min(current_cut_cost);
                }

                if !cut_cost.is_finite() {
                    let cut_details = cut
                    .iter()
                    .zip(m_list.iter())
                    .map(|(&operator_id, &multiplier)| {
                        let operator = &self.relaxed_operators[operator_id.index()];
                        let edge_cost = self.edge_cost(operator_id, multiplier);
                        let supporter = operator
                            .original_op_id_2
                            .and(self.operator_h_max_supporter(operator_id))
                            .or(self.operator_h_max_supporter(operator_id))
                            .or(None)
                            .and_then(|supporter_id| self.propositions.get(supporter_id).map(|p| {
                                format!(
                                    "{}:{}:{:?}",
                                    supporter_id,
                                    p.name,
                                    self.proposition_status(supporter_id)
                                )
                            }))
                            .or_else(|| self.operator_h_max_supporter(operator_id).and_then(|supporter_id| self.propositions.get(supporter_id).map(|p| {
                                format!(
                                    "{}:{}:{:?}",
                                    supporter_id,
                                    p.name,
                                    self.proposition_status(supporter_id)
                                )
                            })))
                            .unwrap_or_else(|| "none".to_string());
                        let effects = operator
                            .effect_ids
                            .iter()
                            .filter_map(|&effect_id| {
                                self.propositions.get(effect_id).map(|p| {
                                    format!("{}:{}:{:?}", effect_id, p.name, self.proposition_status(effect_id))
                                })
                            })
                            .collect::<Vec<_>>();
                        let preconditions = operator
                            .precondition_ids
                            .iter()
                            .filter_map(|&precondition_id| {
                                self.propositions.get(precondition_id).map(|p| {
                                    format!(
                                        "{}:{}:{:?}:h={}",
                                        precondition_id,
                                        p.name,
                                        self.proposition_status(precondition_id),
                                        self.proposition_h_max_cost(precondition_id),
                                    )
                                })
                            })
                            .collect::<Vec<_>>();
                        format!(
                            "id={operator_id} name={} unsat={} edge_cost={} m=({},{}) cost=({},{}) orig=({:?},{:?}) supporter={} preconditions=[{}] effects=[{}]",
                            operator.name,
                            self.operator_unsatisfied_preconditions(operator_id),
                            edge_cost,
                            multiplier.0,
                            multiplier.1,
                            self.operator_cost_1(operator_id),
                            self.operator_cost_2(operator_id),
                            operator.original_op_id_1,
                            operator.original_op_id_2,
                            supporter,
                            preconditions.join(", "),
                            effects.join(", "),
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(" | ");
                    return Err(format!(
                        "LM-cut produced an invalid cut: cut_cost={cut_cost}, cut_size={}, goal_h_max={}, cut=[{}].",
                        cut.len(),
                        self.proposition_h_max_cost(self.artificial_goal_id),
                        cut_details,
                    ));
                }

                total_cost += cut_cost;

                // PARITY(numeric-fd): the reference implementation has no bailout for repeated
                // zero-cost cuts here. Returning an error from this loop turns a parity bug into a
                // synthetic dead end, so the faithful port must continue the LM-cut iterations.
                for touched_index in 0..self.touched_original_operator_ids.len() {
                    let original_id = self.touched_original_operator_ids[touched_index];
                    let min_cost = self.original_operator_min_cut_costs[original_id.index()];
                    if min_cost < self.config.precision {
                        continue;
                    }
                    let mapped_operator_count =
                        self.original_to_relaxed_operators[original_id.index()].len();
                    for mapped_index in 0..mapped_operator_count {
                        let relaxed_operator_id =
                            self.original_to_relaxed_operators[original_id.index()][mapped_index];
                        let mut multiplier = min_cost;
                        {
                            let relaxed_operator =
                                &self.relaxed_operators[relaxed_operator_id.index()];
                            let runtime = &mut self.operator_runtime[relaxed_operator_id.index()];
                            if relaxed_operator.original_op_id_1 == Some(original_id)
                                && runtime.cost_1 >= self.config.precision
                            {
                                multiplier /= runtime.cost_1;
                                runtime.cost_1 = (runtime.cost_1 - cut_cost / multiplier).max(0.0);
                            }
                            if relaxed_operator.original_op_id_2 == Some(original_id)
                                && runtime.cost_2 >= self.config.precision
                            {
                                multiplier /= runtime.cost_2;
                                runtime.cost_2 = (runtime.cost_2 - cut_cost / multiplier).max(0.0);
                            }
                        }
                    }
                }

                self.first_exploration_incremental(propositional_values, numeric_values, &cut)?;
                cut.clear();
                m_list.clear();
                self.reset_goal_zone_statuses();
                self.set_proposition_status(self.artificial_goal_id, PropositionStatus::Reached);
                self.set_proposition_status(
                    self.artificial_precondition_id,
                    PropositionStatus::Reached,
                );
            }

            Ok((false, total_cost))
        })();
        cut.clear();
        m_list.clear();
        self.cut_scratch = cut;
        self.multiplier_scratch = m_list;
        result
    }

    pub fn compute_landmark_cost(
        &mut self,
        propositional_values: &[ExplicitValueIndex],
        numeric_values: &[NumericValue],
    ) -> Result<(bool, f64), String> {
        self.compute_landmarks_impl(propositional_values, numeric_values)
    }
}
