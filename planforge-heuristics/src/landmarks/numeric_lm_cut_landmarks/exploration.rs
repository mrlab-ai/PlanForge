//! The h^max exploration of the relaxed task: the first exploration from a
//! state and its incremental reruns after each cut, and the numeric times
//! that price reaching a numeric condition.

use super::*;

impl LandmarkCutLandmarks {
    pub(super) fn setup_exploration_queue(&mut self) {
        self.priority_queue.clear();
        for &proposition_id in &self.touched_proposition_ids {
            let runtime = &mut self.proposition_runtime[proposition_id];
            runtime.status = PropositionStatus::Unreached;
            runtime.h_max_cost = f64::INFINITY;
        }

        for &operator_id in &self.touched_operator_ids {
            self.operator_runtime[operator_id.index()] =
                OperatorRuntime::new(&self.relaxed_operators[operator_id.index()]);
        }

        self.start_exploration_runtime_tracking();
    }

    pub(super) fn setup_exploration_queue_state(
        &mut self,
        propositional_values: &[ExplicitValueIndex],
        numeric_values: &[NumericValue],
    ) -> Result<(), String> {
        assert_eq!(
            propositional_values.len(),
            self.num_variables,
            "LM-cut exploration received the wrong number of propositional values"
        );
        self.numeric_initial_state.clear();
        self.numeric_initial_state
            .resize(self.conditions.len(), 0.0);

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
            self.enqueue_if_necessary(proposition_id, 0.0);
        }

        if !self.config.ignore_numeric {
            for condition_id in 0..self.conditions.len() {
                let slack = self.evaluate_numeric_condition(condition_id, numeric_values)?;
                self.numeric_initial_state[condition_id] = -slack;
                if slack > -self.config.precision {
                    let proposition_id = self.get_numeric_proposition_id(condition_id)?;
                    self.enqueue_if_necessary(proposition_id, 0.0);
                }
            }
        }

        self.enqueue_if_necessary(self.artificial_precondition_id, 0.0);
        Ok(())
    }

    pub(super) fn first_exploration(
        &mut self,
        propositional_values: &[ExplicitValueIndex],
        numeric_values: &[NumericValue],
    ) -> Result<(), String> {
        assert!(
            self.priority_queue.is_empty(),
            "LM-cut first exploration requires an empty queue"
        );
        self.setup_exploration_queue();
        self.setup_exploration_queue_state(propositional_values, numeric_values)?;

        while let Some(entry) = self.priority_queue.pop() {
            let popped_cost = entry.cost;
            let proposition_id = entry.proposition_id;
            let proposition_cost = self.proposition_h_max_cost(proposition_id);
            assert!(
                proposition_cost <= popped_cost,
                "LM-cut queue popped a cost smaller than the proposition h_max"
            );
            if proposition_cost < popped_cost {
                continue;
            }

            let (triggered_start, triggered_end) =
                self.proposition_precondition_of_range(proposition_id);
            for triggered_index in triggered_start..triggered_end {
                let operator_id = self.proposition_precondition_of_data[triggered_index];
                let effect_count = {
                    assert!(
                        self.operator_unsatisfied_preconditions(operator_id) > 0,
                        "LM-cut operator precondition counter underflow"
                    );
                    let remaining = self.decrement_operator_unsatisfied_preconditions(operator_id);

                    if remaining == 0 {
                        self.set_operator_h_max_supporter(operator_id, Some(proposition_id));
                        self.set_operator_h_max_supporter_cost(operator_id, proposition_cost);
                        self.relaxed_operators[operator_id.index()].effect_ids.len()
                    } else {
                        0
                    }
                };

                if effect_count > 0 {
                    let (effect_start, _) = self.operator_effect_id_range(operator_id);
                    for effect_index in effect_start..(effect_start + effect_count) {
                        let effect_id = self.operator_effect_id_data[effect_index];
                        self.update_queue(
                            propositional_values,
                            numeric_values,
                            operator_id,
                            proposition_id,
                            effect_id,
                        );
                    }
                }
            }
        }

        Ok(())
    }

    pub(super) fn first_exploration_incremental(
        &mut self,
        propositional_values: &[ExplicitValueIndex],
        numeric_values: &[NumericValue],
        cut: &[OperatorIndex],
    ) -> Result<(), String> {
        assert!(
            self.priority_queue.is_empty(),
            "LM-cut incremental exploration requires an empty queue"
        );
        let mut original_ids_to_update =
            std::mem::take(&mut self.incremental_original_operator_ids_scratch);
        original_ids_to_update.clear();
        Self::advance_epoch(
            &mut self.incremental_original_operator_epoch,
            &mut self.incremental_original_operator_marks,
        );
        let result = (|| {
            for &relaxed_operator_id in cut {
                let original_ids = {
                    let operator = self
                        .relaxed_operators
                        .get(relaxed_operator_id.index())
                        .ok_or_else(|| {
                            format!("LM-cut cut operator id {relaxed_operator_id} is invalid")
                        })?;
                    [operator.original_op_id_1, operator.original_op_id_2]
                };

                for original_id in original_ids.into_iter().flatten() {
                    let mark = self
                        .incremental_original_operator_marks
                        .get_mut(original_id.index())
                        .ok_or_else(|| {
                            format!("LM-cut original operator id {original_id} is invalid")
                        })?;
                    if *mark != self.incremental_original_operator_epoch {
                        *mark = self.incremental_original_operator_epoch;
                        original_ids_to_update.push(original_id);
                    }
                }
            }

            for &original_id in &original_ids_to_update {
                let mapped_operator_count = self
                    .original_to_relaxed_operators
                    .get(original_id.index())
                    .ok_or_else(|| format!("LM-cut original operator id {original_id} is invalid"))?
                    .len();
                for mapped_index in 0..mapped_operator_count {
                    let mapped_operator_id =
                        self.original_to_relaxed_operators[original_id.index()][mapped_index];
                    let operator = self
                        .relaxed_operators
                        .get(mapped_operator_id.index())
                        .ok_or_else(|| {
                            format!(
                                "LM-cut mapped relaxed operator id {mapped_operator_id} is invalid"
                            )
                        })?;
                    if self.operator_unsatisfied_preconditions(mapped_operator_id) == 0 {
                        let supporter_id = self
                            .operator_h_max_supporter(mapped_operator_id)
                            .ok_or_else(|| {
                                format!(
                                    "LM-cut reachable operator {} must have an h_max supporter",
                                    operator.name
                                )
                            })?;
                        let (effect_start, effect_end) =
                            self.operator_effect_id_range(mapped_operator_id);
                        for effect_index in effect_start..effect_end {
                            let effect_id = self.operator_effect_id_data[effect_index];
                            self.update_queue(
                                propositional_values,
                                numeric_values,
                                mapped_operator_id,
                                supporter_id,
                                effect_id,
                            );
                        }
                    }
                }
            }

            while let Some(entry) = self.priority_queue.pop() {
                let popped_cost = entry.cost;
                let proposition_id = entry.proposition_id;
                let proposition_cost = self.proposition_h_max_cost(proposition_id);
                assert!(
                    proposition_cost <= popped_cost,
                    "LM-cut incremental queue popped a cost smaller than the proposition h_max"
                );
                if proposition_cost < popped_cost {
                    continue;
                }

                let (triggered_start, triggered_end) =
                    self.proposition_precondition_of_range(proposition_id);
                for triggered_index in triggered_start..triggered_end {
                    let operator_id = self.proposition_precondition_of_data[triggered_index];
                    let update = {
                        if self.operator_h_max_supporter(operator_id) == Some(proposition_id) {
                            let old_supporter_cost =
                                self.operator_h_max_supporter_cost(operator_id);
                            if old_supporter_cost > proposition_cost {
                                let new_supporter = self.update_h_max_supporter(operator_id);
                                if let Some((new_supporter_id, new_cost)) = new_supporter {
                                    Some((
                                        new_supporter_id,
                                        new_cost,
                                        new_cost != old_supporter_cost,
                                    ))
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    };

                    if let Some((new_supporter_id, new_cost, needs_effect_update)) = update {
                        self.set_operator_h_max_supporter(operator_id, Some(new_supporter_id));
                        self.set_operator_h_max_supporter_cost(operator_id, new_cost);
                        if needs_effect_update {
                            let (effect_start, effect_end) =
                                self.operator_effect_id_range(operator_id);
                            for effect_index in effect_start..effect_end {
                                let effect_id = self.operator_effect_id_data[effect_index];
                                self.update_queue(
                                    propositional_values,
                                    numeric_values,
                                    operator_id,
                                    new_supporter_id,
                                    effect_id,
                                );
                            }
                        }
                    }
                }
            }

            Ok(())
        })();
        original_ids_to_update.clear();
        self.incremental_original_operator_ids_scratch = original_ids_to_update;
        result
    }

    #[inline(always)]
    pub(super) fn update_h_max_supporter(
        &mut self,
        operator_id: OperatorIndex,
    ) -> Option<(usize, f64)> {
        debug_assert!(operator_id.index() < self.relaxed_operators.len());
        if self.operator_unsatisfied_preconditions(operator_id) != 0 {
            return None;
        }

        let mut best_supporter = self.operator_h_max_supporter(operator_id);
        let mut best_cost = best_supporter
            .map(|supporter_id| self.proposition_h_max_cost(supporter_id))
            .unwrap_or(f64::NEG_INFINITY);

        if let Some(supporter_id) = best_supporter
            && self.proposition_status(supporter_id) == PropositionStatus::Unreached
        {
            return None;
        }

        let (precondition_start, precondition_end) =
            self.operator_precondition_id_range(operator_id);
        for precondition_index in precondition_start..precondition_end {
            let precondition_id = self.operator_precondition_id_data[precondition_index];
            if self.proposition_status(precondition_id) == PropositionStatus::Unreached {
                return None;
            }
            let precondition_cost = self.proposition_h_max_cost(precondition_id);
            if best_supporter.is_none() || precondition_cost > best_cost {
                best_supporter = Some(precondition_id);
                best_cost = precondition_cost;
            }
        }

        if let Some(supporter_id) = best_supporter {
            self.set_operator_h_max_supporter(operator_id, Some(supporter_id));
            self.set_operator_h_max_supporter_cost(operator_id, best_cost);
            Some((supporter_id, best_cost))
        } else {
            None
        }
    }

    #[inline(always)]
    pub(super) fn calculate_numeric_times_sose(
        &mut self,
        numeric_values: &[NumericValue],
        condition_id: usize,
        operator_id: OperatorIndex,
        operator_runtime: OperatorRuntime,
        original_op_id_1: OperatorIndex,
        original_op_id_2: OperatorIndex,
    ) -> (f64, f64) {
        let eval_row = self
            .operator_condition_eval
            .get(original_op_id_2, condition_id);
        let c = eval_row.simple_effect.unwrap_or(0.0);
        let has_upper_bound = eval_row.has_upper_bound;
        let upper_bound = eval_row.upper_bound;
        let composite_coefficients = eval_row
            .composite_expression
            .as_ref()
            .map(|expression| expression.coefficients.as_slice());

        let mut c_u = *self.relaxed_operators[operator_id.index()]
            .sose_constants
            .get(condition_id)
            .expect("LM-cut SOSE operator must store condition constants");
        if self.config.use_constant_assignment {
            c_u += self.calculate_constant_assignment_effect_infallible(
                original_op_id_1,
                composite_coefficients.unwrap_or_else(|| {
                    panic!(
                        "LM-cut SOSE target operator {original_op_id_2} is missing composite coefficients for condition {condition_id}"
                    )
                }),
                numeric_values,
                self.use_bounds,
            );
        }
        if c_u < self.config.precision {
            return (-1.0, -1.0);
        }
        if operator_runtime.cost_1 < self.config.precision {
            return (1.0, 1.0);
        }

        let s_u =
            self.cached_composite_expression_value(original_op_id_2, condition_id, numeric_values);

        if operator_runtime.cost_2 < self.config.precision {
            if (c + s_u).abs() < self.config.precision {
                return (1.0, 1.0);
            }
            if c + s_u > 0.0 {
                return (-1.0, -1.0);
            }
            let mut m_1 = -(c + s_u) / c_u;
            if self.config.ceiling_less_than_one {
                m_1 = m_1.max(1.0);
            }
            return (m_1, 1.0);
        }

        let mut u_target =
            (self.numeric_initial_state[condition_id] * c_u * operator_runtime.cost_2
                / operator_runtime.cost_1)
                .sqrt()
                - c;
        if self.use_bounds && has_upper_bound {
            u_target = u_target.min(upper_bound);
        }
        if u_target - s_u < self.config.precision || c + u_target < self.config.precision {
            return (-1.0, -1.0);
        }

        let mut m_1 = (u_target - s_u) / c_u;
        let mut m_2 = self.numeric_initial_state[condition_id] / (c + u_target);
        if self.config.ceiling_less_than_one {
            m_1 = m_1.max(1.0);
            m_2 = m_2.max(1.0);
        }
        (m_1, m_2)
    }

    #[inline(always)]
    pub(super) fn calculate_numeric_times_simple_numeric(
        &mut self,
        numeric_values: &[NumericValue],
        condition_id: usize,
        original_op_id_2: Option<OperatorIndex>,
    ) -> (f64, f64) {
        let mut net = 0.0;
        if let Some(original_id) = original_op_id_2 {
            let eval_row = self.operator_condition_eval.get(original_id, condition_id);
            net += eval_row.simple_effect.unwrap_or(0.0);
            if eval_row.composite_expression.is_some() {
                net += self.cached_composite_expression_value(
                    original_id,
                    condition_id,
                    numeric_values,
                );
            }
        }
        if self.config.use_constant_assignment {
            let original_operator_id = original_op_id_2
                .expect("LM-cut relaxed operator must store its concrete operator id");
            let has_supported_sose = self
                .operator_condition_eval
                .get(original_operator_id, condition_id)
                .has_sose;
            net += self.calculate_constant_assignment_effect_infallible(
                original_operator_id,
                &self.conditions[condition_id].coefficients,
                numeric_values,
                self.use_bounds && !has_supported_sose,
            );
        }
        if net < self.config.precision {
            return (-1.0, -1.0);
        }

        let mut m = self.numeric_initial_state[condition_id] / net;
        if m < self.config.precision {
            return (0.0, 0.0);
        }
        if self.config.ceiling_less_than_one {
            m = m.max(1.0);
        }
        (0.0, m)
    }

    #[inline(never)]
    pub(super) fn calculate_numeric_times(
        &mut self,
        _propositional_values: &[ExplicitValueIndex],
        numeric_values: &[NumericValue],
        effect_id: usize,
        operator_id: OperatorIndex,
        use_ma: bool,
    ) -> (f64, f64) {
        debug_assert!(effect_id < self.propositions.len());
        debug_assert!(operator_id.index() < self.relaxed_operators.len());
        let effect = &self.propositions[effect_id];
        let operator_runtime = self.operator_runtime[operator_id.index()];
        let operator = &self.relaxed_operators[operator_id.index()];
        if !use_ma || !effect.is_numeric_condition || operator.infinite {
            return (0.0, 1.0);
        }

        let condition_id = effect
            .id_numeric_condition
            .expect("LM-cut numeric proposition must store its condition id");
        match (operator.original_op_id_1, operator.original_op_id_2) {
            (Some(original_op_id_1), Some(original_op_id_2)) => self.calculate_numeric_times_sose(
                numeric_values,
                condition_id,
                operator_id,
                operator_runtime,
                original_op_id_1,
                original_op_id_2,
            ),
            (_, original_op_id_2) => self.calculate_numeric_times_simple_numeric(
                numeric_values,
                condition_id,
                original_op_id_2,
            ),
        }
    }

    #[inline(always)]
    pub(super) fn update_queue(
        &mut self,
        propositional_values: &[ExplicitValueIndex],
        numeric_values: &[NumericValue],
        operator_id: OperatorIndex,
        supporter_id: usize,
        effect_id: usize,
    ) {
        debug_assert!(effect_id < self.propositions.len());
        debug_assert!(operator_id.index() < self.relaxed_operators.len());
        debug_assert!(supporter_id < self.propositions.len());
        let effect = &self.propositions[effect_id];
        if effect.is_numeric_condition {
            let condition_id = effect
                .id_numeric_condition
                .expect("LM-cut numeric proposition must store its condition id");
            if self.numeric_initial_state[condition_id] < self.config.precision {
                return;
            }
            let ms = self.calculate_numeric_times(
                propositional_values,
                numeric_values,
                effect_id,
                operator_id,
                !self.config.irmax,
            );
            let operator = &self.relaxed_operators[operator_id.index()];
            if operator.original_op_id_1.is_some() {
                if ms.0 >= self.config.precision {
                    let target_cost = self.proposition_h_max_cost(supporter_id)
                        + (ms.0 * self.operator_cost_1(operator_id))
                        + (ms.1 * self.operator_cost_2(operator_id));
                    self.enqueue_if_necessary(effect_id, target_cost);
                }
            } else if ms.1 >= self.config.precision {
                let target_cost = self.proposition_h_max_cost(supporter_id)
                    + (ms.1 * self.operator_cost_2(operator_id));
                self.enqueue_if_necessary(effect_id, target_cost);
            }
            return;
        }
        let target_cost =
            self.proposition_h_max_cost(supporter_id) + self.operator_cost_2(operator_id);
        self.enqueue_if_necessary(effect_id, target_cost);
    }

    #[inline(always)]
    pub(super) fn enqueue_if_necessary(&mut self, proposition_id: usize, cost: f64) -> bool {
        assert!(cost >= 0.0, "LM-cut enqueue cost must be non-negative");
        assert!(!cost.is_nan(), "LM-cut enqueue cost must not be NaN");
        debug_assert!(proposition_id < self.propositions.len());
        // PARITY(numeric-fd): C++ uses the strict comparison `h_max_cost > cost` here.
        // A `+ precision` tolerance suppresses small but real h_max decreases during
        // `first_exploration_incremental()`, which can keep `goal_h_max` artificially high
        // after zero-cost cuts and surface as false dead-end reports.
        if self.proposition_status(proposition_id) == PropositionStatus::Unreached
            || self.proposition_h_max_cost(proposition_id) > cost
        {
            self.set_proposition_status(proposition_id, PropositionStatus::Reached);
            self.set_proposition_h_max_cost(proposition_id, cost);
            self.priority_queue.push(QueueEntry {
                cost,
                proposition_id,
            });
            return true;
        }
        false
    }
}
