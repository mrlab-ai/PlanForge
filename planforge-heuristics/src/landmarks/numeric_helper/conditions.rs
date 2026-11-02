//! The helper's numeric conditions: one per comparison a precondition, an
//! axiom or the goal names, grouped, and the numeric goals.

use super::*;

impl NumericTaskHelper {
    pub(super) fn build_numeric_conditions(&mut self, task: &Task) {
        for (comparison_axiom_id, comparison_axiom) in task.comparison_axioms().iter().enumerate() {
            let affected_var_id = comparison_axiom.affected_var_id();
            self.comparison_axiom_by_var
                .insert(affected_var_id, comparison_axiom_id);
            if let Some(marker) = self.fact_to_axiom_marker.get_mut(affected_var_id.index()) {
                *marker = Some(comparison_axiom_id);
            }

            // PARITY(numeric-fd): `NumericTaskProxy::build_numeric_conditions()` only materializes
            // the comparison-axiom-side numeric conditions that correspond to the kept numeric
            // precondition polarity. The Rust LM-cut port currently follows the same one-sided
            // lookup and filters unsupported inequality facts earlier.
            if let Some(conditions) = self.build_conditions_for_fact_value(
                task,
                comparison_axiom,
                affected_var_id,
                ExplicitValueIndex::new(0),
            ) {
                let group_id = self.register_condition_group(conditions.clone());
                self.comparison_fact_condition_group_ids.insert(
                    (affected_var_id, ExplicitValueIndex::new(0)),
                    vec![group_id],
                );
                self.comparison_fact_conditions
                    .insert((affected_var_id, ExplicitValueIndex::new(0)), conditions);
            }
        }
    }

    pub(super) fn build_numeric_goals(&mut self, task: &Task, precision: f64) {
        self.goal_models = vec![HelperPreconditionLists::default(); task.goals().len()];
        let mut axiom_table: BTreeMap<VariableIndex, Vec<VariableIndex>> = BTreeMap::new();
        let mut fact_table: BTreeMap<VariableIndex, Vec<ExplicitFact>> = BTreeMap::new();

        for axiom in task.axioms() {
            let effect_var_id = axiom.var_id();
            for precondition in axiom.conditions() {
                let precondition_var_id = precondition.var_index();
                if self.is_comparison_axiom_var(precondition_var_id) {
                    axiom_table
                        .entry(effect_var_id)
                        .or_default()
                        .push(precondition_var_id);
                } else {
                    fact_table
                        .entry(effect_var_id)
                        .or_default()
                        .push(*precondition);
                }
            }
        }

        for (goal_index, goal) in task.goals().iter().enumerate() {
            let goal_var_id = goal.var_index();
            let Some(helper_numeric_vars) = axiom_table.get(&goal_var_id) else {
                continue;
            };

            let propositional_facts = fact_table.get(&goal_var_id).cloned().unwrap_or_default();
            self.goal_models[goal_index].propositional_facts = propositional_facts.clone();
            if !propositional_facts.is_empty() {
                self.goal_helper_propositional_facts
                    .insert(goal_var_id, propositional_facts);
            }

            let mut numeric_conditions = Vec::new();
            let mut numeric_condition_group_ids = Vec::new();
            for &helper_numeric_var_id in helper_numeric_vars {
                if let Some(group_ids) = self
                    .comparison_fact_condition_group_ids(
                        helper_numeric_var_id,
                        ExplicitValueIndex::new(0),
                    )
                    .map(|group_ids| group_ids.to_vec())
                {
                    numeric_condition_group_ids.extend(group_ids);
                }
                if let Some(conditions) = self
                    .comparison_fact_conditions(helper_numeric_var_id, ExplicitValueIndex::new(0))
                    .map(|conditions| conditions.to_vec())
                {
                    for condition in conditions {
                        if !condition.is_empty(precision) {
                            numeric_conditions.push(condition);
                            self.numeric_goal_helper_vars.insert(goal_var_id);
                            if let Some(marker) =
                                self.fact_to_axiom_marker.get_mut(goal_var_id.index())
                            {
                                *marker = None;
                            }
                        }
                    }
                }
            }

            if numeric_conditions.is_empty() {
                continue;
            }

            let redundant_group_ids = self.materialize_pairwise_redundant_condition_groups(
                &numeric_condition_group_ids,
                precision,
            );
            numeric_condition_group_ids.extend(redundant_group_ids);

            let original_numeric_conditions = numeric_conditions.clone();
            for (index, left_condition) in original_numeric_conditions.iter().enumerate() {
                for right_condition in original_numeric_conditions.iter().skip(index + 1) {
                    let redundant_condition = left_condition.add(right_condition);
                    if !redundant_condition.is_empty(precision) {
                        numeric_conditions.push(redundant_condition);
                    }
                }
            }

            self.goal_helper_numeric_conditions
                .insert(goal_var_id, numeric_conditions);
            self.goal_helper_numeric_condition_group_ids
                .insert(goal_var_id, numeric_condition_group_ids);
            self.goal_models[goal_index].numeric_group_ids = self
                .goal_helper_numeric_condition_group_ids
                .get(&goal_var_id)
                .cloned()
                .unwrap_or_default();
        }

        // A goal on a comparison variable names its numeric condition directly
        // instead of through a goal axiom. Model all such goals together, as
        // the axiom encoding did, so they also get their pairwise redundant
        // conditions. Only the goal models change: the variables stay ordinary
        // comparison variables for the preconditions that use them.
        let direct_goal_indices = task
            .goals()
            .iter()
            .enumerate()
            .filter(|(_, goal)| self.is_comparison_axiom_var(goal.var_index()))
            .map(|(goal_index, _)| goal_index)
            .collect::<Vec<_>>();
        if direct_goal_indices.is_empty() {
            return;
        }
        let mut group_ids = Vec::new();
        for &goal_index in &direct_goal_indices {
            let goal = task.goals()[goal_index];
            let goal_group_ids = self
                .comparison_fact_condition_group_ids(goal.var_index(), goal.value_index())
                .unwrap_or_else(|| {
                    panic!("comparison goal fact {goal:?} has no numeric conditions")
                })
                .to_vec();
            group_ids.extend(goal_group_ids);
        }
        let redundant_group_ids =
            self.materialize_pairwise_redundant_condition_groups(&group_ids, precision);
        group_ids.extend(redundant_group_ids);
        for goal_index in direct_goal_indices {
            self.goal_models[goal_index].numeric_group_ids = group_ids.clone();
        }
    }

    pub(super) fn register_condition_group(
        &mut self,
        conditions: Vec<LinearNumericCondition>,
    ) -> usize {
        assert!(
            !conditions.is_empty(),
            "numeric helper condition groups must contain at least one normalized condition"
        );
        let mut condition_ids = Vec::with_capacity(conditions.len());
        for condition in conditions {
            let condition_id = self.normalized_conditions.len();
            self.normalized_conditions.push(condition);
            condition_ids.push(condition_id);
        }
        let group_id = self.condition_group_condition_ids.len();
        self.condition_group_representative_condition_ids
            .push(condition_ids[0]);
        self.condition_group_condition_ids.push(condition_ids);
        group_id
    }

    pub(super) fn build_conditions_for_fact_value(
        &self,
        task: &Task,
        comparison_axiom: &ComparisonAxiom,
        affected_var_id: VariableIndex,
        fact_value: ExplicitValueIndex,
    ) -> Option<Vec<LinearNumericCondition>> {
        let operator = comparison_operator_for_fact_value(comparison_axiom, fact_value)?;
        let lhs = comparison_axiom.left_var_id();
        let rhs = comparison_axiom.right_var_id();
        // A comparison axiom's affected variable is by definition the carrier of
        // a numeric condition.
        let fact = ExplicitFact::condition(affected_var_id.index(), fact_value.index());
        let fact_name = match task.fact_name(&fact) {
            "" => format!("comparison fact {affected_var_id}={fact_value}"),
            name => name.to_string(),
        };

        match operator {
            ComparisonOperator::GreaterThan | ComparisonOperator::GreaterThanOrEqual => {
                Some(vec![self.build_condition(
                    task,
                    lhs,
                    rhs,
                    matches!(operator, ComparisonOperator::GreaterThan),
                    fact_name,
                )?])
            }
            ComparisonOperator::LessThan | ComparisonOperator::LessThanOrEqual => {
                Some(vec![self.build_condition(
                    task,
                    rhs,
                    lhs,
                    matches!(operator, ComparisonOperator::LessThan),
                    fact_name,
                )?])
            }
            ComparisonOperator::Equal => Some(vec![
                self.build_condition(task, lhs, rhs, false, fact_name.clone())?,
                self.build_condition(task, rhs, lhs, false, fact_name)?,
            ]),
            ComparisonOperator::UnEqual => None,
        }
    }

    pub(super) fn build_condition(
        &self,
        task: &Task,
        positive_var_id: VariableIndex,
        negative_var_id: VariableIndex,
        is_strictly_greater: bool,
        name: String,
    ) -> Option<LinearNumericCondition> {
        let positive_expression = match task.linearize_numeric_var(positive_var_id) {
            Ok(expression) => expression,
            Err(error) => {
                tracing::warn!(
                    "Ignoring nonlinear numeric comparison feature `{name}`: variable {positive_var_id} cannot be linearized ({error})"
                );
                return None;
            }
        };
        let negative_expression = match task.linearize_numeric_var(negative_var_id) {
            Ok(expression) => expression,
            Err(error) => {
                tracing::warn!(
                    "Ignoring nonlinear numeric comparison feature `{name}`: variable {negative_var_id} cannot be linearized ({error})"
                );
                return None;
            }
        };
        let expression = positive_expression.subtract(&negative_expression);
        Some(LinearNumericCondition::from_expression(
            expression,
            is_strictly_greater,
            format!("numeric ({name})"),
        ))
    }
}
