use planners_sas::numeric::axioms::{ComparisonAxiom, ComparisonOperator, PropositionalAxiom};
use planners_sas::numeric::numeric_task::{
    AbstractNumericTask, AssignmentEffect, AssignmentOperation, ExplicitFact, NumericType, Operator,
};
use planners_sas::numeric::utils::linear_effects::{LinearExpression, LinearNumericEffect};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LinearNumericCondition {
    pub(crate) coefficients: Vec<f64>,
    pub(crate) constant: f64,
    pub(crate) is_strictly_greater: bool,
    pub(crate) name: String,
}

impl LinearNumericCondition {
    pub(crate) fn from_expression(
        expression: LinearExpression,
        is_strictly_greater: bool,
        name: String,
    ) -> Self {
        Self {
            coefficients: expression.coefficients,
            constant: expression.constant,
            is_strictly_greater,
            name,
        }
    }

    pub(crate) fn add(&self, other: &Self) -> Self {
        Self {
            coefficients: self
                .coefficients
                .iter()
                .zip(other.coefficients.iter())
                .map(|(left, right)| left + right)
                .collect(),
            constant: self.constant + other.constant,
            is_strictly_greater: self.is_strictly_greater || other.is_strictly_greater,
            name: format!("{} + {}", self.name, other.name),
        }
    }

    pub(crate) fn is_empty(&self, _precision: f64) -> bool {
        self.coefficients
            .iter()
            .all(|&coefficient| coefficient == 0.0)
    }

    pub(crate) fn evaluate_slack(&self, numeric_values: &[f64], epsilon: f64) -> f64 {
        let mut net = self.constant;
        if self.is_strictly_greater {
            net -= epsilon;
        }
        for (coefficient, value) in self.coefficients.iter().zip(numeric_values.iter()) {
            net += coefficient * value;
        }
        net
    }

    pub(crate) fn dominates(&self, other: &Self, precision: f64) -> bool {
        assert_eq!(self.coefficients.len(), other.coefficients.len());
        let mut ratio: f64 = 0.0;
        for (&rhs, &lhs) in self.coefficients.iter().zip(other.coefficients.iter()) {
            if lhs.abs() < precision {
                if rhs.abs() >= precision {
                    return false;
                }
            } else {
                if ratio.abs() < precision {
                    ratio = rhs / lhs;
                }
                if ratio < precision {
                    return false;
                }
                if ((rhs / lhs) - ratio).abs() >= precision {
                    return false;
                }
            }
        }
        (self.constant.abs() - (ratio * other.constant).abs()) >= -precision
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct NumericTaskHelper {
    normalized_conditions: Vec<LinearNumericCondition>,
    condition_group_condition_ids: Vec<Vec<usize>>,
    condition_group_representative_condition_ids: Vec<usize>,
    fact_to_axiom_marker: Vec<Option<usize>>,
    numeric_variable_ids: Vec<usize>,
    numeric_variable_index_by_task_id: Vec<Option<usize>>,
    comparison_axiom_by_var: BTreeMap<usize, usize>,
    comparison_fact_condition_group_ids: BTreeMap<(usize, usize), Vec<usize>>,
    comparison_fact_conditions: BTreeMap<(usize, usize), Vec<LinearNumericCondition>>,
    goal_helper_propositional_facts: BTreeMap<usize, Vec<ExplicitFact>>,
    goal_helper_numeric_condition_group_ids: BTreeMap<usize, Vec<usize>>,
    goal_helper_numeric_conditions: BTreeMap<usize, Vec<LinearNumericCondition>>,
    goal_models: Vec<HelperPreconditionLists>,
    numeric_goal_helper_vars: BTreeSet<usize>,
    action_models: Vec<HelperActionModel>,
    condition_achievers: Vec<Vec<usize>>,
    numeric_variable_lower_bounds: Vec<f64>,
    numeric_variable_upper_bounds: Vec<f64>,
    condition_small_m: Vec<f64>,
    condition_epsilons: Vec<f64>,
    dominance_conditions: Vec<Vec<bool>>,
    proposition_ids_by_var_value: Vec<Vec<usize>>,
    proposition_facts: Vec<ExplicitFact>,
    proposition_var_ids: Vec<usize>,
    proposition_add_action_ids: Vec<Vec<usize>>,
    proposition_names: Vec<String>,
    mutex_action_ids: Vec<Vec<usize>>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct HelperPreconditionLists {
    pub(crate) propositional_facts: Vec<ExplicitFact>,
    pub(crate) numeric_group_ids: Vec<usize>,
}

#[derive(Debug, Clone)]
#[allow(unused)]
pub(crate) struct HelperConditionalExplicitFactEffect {
    pub(crate) preconditions: HelperPreconditionLists,
    pub(crate) add_fact: ExplicitFact,
    pub(crate) del_fact: Option<ExplicitFact>,
}

#[derive(Debug, Clone)]
#[allow(unused)]
pub(crate) struct HelperConditionalNumericEffect {
    pub(crate) source_assignment_effect_id: usize,
    pub(crate) preconditions: HelperPreconditionLists,
    pub(crate) target_local_var_id: usize,
    pub(crate) delta: f64,
}

#[derive(Debug, Clone)]
#[allow(unused)]
pub(crate) struct HelperConditionalAssignmentEffect {
    pub(crate) source_assignment_effect_id: usize,
    pub(crate) preconditions: HelperPreconditionLists,
    pub(crate) target_local_var_id: usize,
    pub(crate) assigned_value: f64,
}

#[derive(Debug, Clone)]
pub(crate) struct HelperLinearEffect {
    pub(crate) source_assignment_effect_id: usize,
    pub(crate) source_var_id: usize,
    pub(crate) operation: AssignmentOperation,
    pub(crate) preconditions: HelperPreconditionLists,
    pub(crate) target_local_var_id: usize,
    pub(crate) coefficients: Vec<f64>,
    pub(crate) constant: f64,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct HelperActionModel {
    pub(crate) propositional_preconditions: Vec<ExplicitFact>,
    pub(crate) numeric_precondition_group_ids: Vec<usize>,
    pub(crate) add_facts: Vec<ExplicitFact>,
    pub(crate) del_facts: Vec<ExplicitFact>,
    pub(crate) pre_del_facts: Vec<ExplicitFact>,
    pub(crate) conditional_fact_effects: Vec<HelperConditionalExplicitFactEffect>,
    pub(crate) simple_effects: Vec<f64>,
    pub(crate) is_assignment: Vec<bool>,
    pub(crate) assignment_values: Vec<f64>,
    pub(crate) conditional_numeric_effects: Vec<HelperConditionalNumericEffect>,
    pub(crate) conditional_assignment_effects: Vec<HelperConditionalAssignmentEffect>,
    pub(crate) linear_effects: Vec<HelperLinearEffect>,
    pub(crate) possible_add_condition_ids: Vec<usize>,
    pub(crate) cost: f64,
    pub(crate) linear_cost: bool,
    pub(crate) cost_coefficients: Vec<f64>,
    pub(crate) cost_constant: f64,
}

#[allow(unused)]
impl NumericTaskHelper {
    pub(crate) fn new(
        task: &dyn AbstractNumericTask,
        precision: f64,
        default_epsilon: f64,
        separate_constant_assignment: bool,
    ) -> Self {
        Self::new_with_options(
            task,
            precision,
            default_epsilon,
            separate_constant_assignment,
            true,
            true,
        )
    }

    pub(crate) fn new_lmcut(
        task: &dyn AbstractNumericTask,
        precision: f64,
        default_epsilon: f64,
        separate_constant_assignment: bool,
    ) -> Self {
        Self::new_with_options(
            task,
            precision,
            default_epsilon,
            separate_constant_assignment,
            false,
            true,
        )
    }

    fn new_with_options(
        task: &dyn AbstractNumericTask,
        precision: f64,
        default_epsilon: f64,
        separate_constant_assignment: bool,
        build_mutex_and_dominance: bool,
        build_bound_metadata: bool,
    ) -> Self {
        let mut helper = Self {
            fact_to_axiom_marker: vec![None; task.get_num_variables()],
            numeric_variable_index_by_task_id: vec![None; task.numeric_variables().len()],
            ..Self::default()
        };
        helper.numeric_variable_ids = task.regular_numeric_variable_ids();
        for (local_var_id, &task_var_id) in helper.numeric_variable_ids.iter().enumerate() {
            if let Some(index) = helper
                .numeric_variable_index_by_task_id
                .get_mut(task_var_id)
            {
                *index = Some(local_var_id);
            }
        }
        helper.build_numeric_conditions(task);
        helper.build_numeric_goals(task, precision);
        helper.build_actions(task, precision, separate_constant_assignment);
        helper.build_propositions(task);
        if build_mutex_and_dominance {
            helper.build_mutex_actions(task);
        }
        if build_bound_metadata {
            helper.calculates_bounds_numeric_variables(task, precision, 9_999_999.0);
            helper.calculate_small_m(precision, 9_999_999.0);
        }
        helper.calculate_epsilons(task.get_operators().len(), precision, default_epsilon);
        if build_mutex_and_dominance {
            helper.calculate_dominance(precision);
        }
        helper
    }

    pub(crate) fn action_models(&self) -> &[HelperActionModel] {
        self.action_models.as_slice()
    }

    pub(crate) fn action_model(&self, action_id: usize) -> Option<&HelperActionModel> {
        self.action_models.get(action_id)
    }

    pub(crate) fn get_action_pre_list(&self, action_id: usize) -> Option<&[ExplicitFact]> {
        self.action_model(action_id)
            .map(|action_model| action_model.propositional_preconditions.as_slice())
    }

    pub(crate) fn get_action_num_list(&self, action_id: usize) -> Option<&[usize]> {
        self.action_model(action_id)
            .map(|action_model| action_model.numeric_precondition_group_ids.as_slice())
    }

    pub(crate) fn get_action_add_list(&self, action_id: usize) -> Option<&[ExplicitFact]> {
        self.action_model(action_id)
            .map(|action_model| action_model.add_facts.as_slice())
    }

    pub(crate) fn get_action_eff_list(&self, action_id: usize) -> Option<&[f64]> {
        self.action_model(action_id)
            .map(|action_model| action_model.simple_effects.as_slice())
    }

    pub(crate) fn get_action_is_assignment(&self, action_id: usize) -> Option<&[bool]> {
        self.action_model(action_id)
            .map(|action_model| action_model.is_assignment.as_slice())
    }

    pub(crate) fn get_action_assign_list(&self, action_id: usize) -> Option<&[f64]> {
        self.action_model(action_id)
            .map(|action_model| action_model.assignment_values.as_slice())
    }

    pub(crate) fn get_action_conditional_fact_effects(
        &self,
        action_id: usize,
    ) -> Option<&[HelperConditionalExplicitFactEffect]> {
        self.action_model(action_id)
            .map(|action_model| action_model.conditional_fact_effects.as_slice())
    }

    pub(crate) fn get_action_conditional_eff_list(
        &self,
        action_id: usize,
    ) -> Option<&[HelperConditionalNumericEffect]> {
        self.action_model(action_id)
            .map(|action_model| action_model.conditional_numeric_effects.as_slice())
    }

    pub(crate) fn get_action_conditional_assign_list(
        &self,
        action_id: usize,
    ) -> Option<&[HelperConditionalAssignmentEffect]> {
        self.action_model(action_id)
            .map(|action_model| action_model.conditional_assignment_effects.as_slice())
    }

    pub(crate) fn get_action_linear_effects(
        &self,
        action_id: usize,
    ) -> Option<&[HelperLinearEffect]> {
        self.action_model(action_id)
            .map(|action_model| action_model.linear_effects.as_slice())
    }

    pub(crate) fn get_action_n_linear_eff(&self, action_id: usize) -> usize {
        self.get_action_linear_effects(action_id)
            .map(|effects| effects.len())
            .unwrap_or(0)
    }

    pub(crate) fn get_propositional_goals(&self, goal_id: usize) -> Option<&[ExplicitFact]> {
        self.goal_model(goal_id)
            .map(|goal_model| goal_model.propositional_facts.as_slice())
    }

    pub(crate) fn get_n_numeric_conditions(&self) -> usize {
        self.normalized_conditions.len()
    }

    pub(crate) fn linearized_effect_for_action_assignment(
        &self,
        action_id: usize,
        assignment_effect_id: usize,
    ) -> Option<LinearNumericEffect> {
        let linear_effect =
            self.linear_effect_for_assignment_effect(action_id, assignment_effect_id)?;
        let affected_var_id = *self
            .numeric_variable_ids
            .get(linear_effect.target_local_var_id)?;
        let mut delta = LinearExpression::zero(self.numeric_variable_index_by_task_id.len());
        for (local_var_id, &coefficient) in linear_effect.coefficients.iter().enumerate() {
            let task_var_id = *self.numeric_variable_ids.get(local_var_id)?;
            delta.coefficients[task_var_id] = coefficient;
        }
        delta.coefficients[affected_var_id] -= 1.0;
        delta.constant = linear_effect.constant;

        Some(LinearNumericEffect {
            affected_var_id,
            source_var_id: linear_effect.source_var_id,
            operation: linear_effect.operation.clone(),
            conditions: linear_effect.preconditions.propositional_facts.clone(),
            is_conditional: !linear_effect.preconditions.propositional_facts.is_empty()
                || !linear_effect.preconditions.numeric_group_ids.is_empty(),
            delta,
        })
    }

    pub(crate) fn linearized_effects_for_action(
        &self,
        action_id: usize,
        assignment_effect_count: usize,
    ) -> Result<Vec<Option<LinearNumericEffect>>, String> {
        let mut effects = vec![None; assignment_effect_count];
        let Some(action_model) = self.action_model(action_id) else {
            return Err(format!(
                "numeric helper action model {action_id} is missing"
            ));
        };

        for linear_effect in &action_model.linear_effects {
            let assignment_effect_id = linear_effect.source_assignment_effect_id;
            if assignment_effect_id >= effects.len() {
                return Err(format!(
                    "numeric helper linear effect source assignment id {assignment_effect_id} is invalid for action {action_id}"
                ));
            }
            effects[assignment_effect_id] =
                self.linearized_effect_for_action_assignment(action_id, assignment_effect_id);
        }

        Ok(effects)
    }

    pub(crate) fn get_achievers(&self, condition_id: usize) -> Option<&[usize]> {
        self.condition_achievers
            .get(condition_id)
            .map(Vec::as_slice)
    }

    pub(crate) fn get_small_m(&self, condition_id: usize) -> Option<f64> {
        self.condition_small_m.get(condition_id).copied()
    }

    pub(crate) fn get_epsilon(&self, condition_id: usize) -> Option<f64> {
        self.condition_epsilons.get(condition_id).copied()
    }

    pub(crate) fn get_dominance(&self, left_id: usize, right_id: usize) -> Option<bool> {
        self.dominance_conditions
            .get(left_id)
            .and_then(|row| row.get(right_id))
            .copied()
    }

    pub(crate) fn local_numeric_var_id(&self, task_numeric_var_id: usize) -> Option<usize> {
        self.numeric_variable_index_by_task_id
            .get(task_numeric_var_id)
            .copied()
            .unwrap_or(None)
    }

    pub(crate) fn get_proposition(&self, var_id: usize, value: usize) -> Option<usize> {
        self.proposition_ids_by_var_value
            .get(var_id)
            .and_then(|values| values.get(value))
            .copied()
    }

    pub(crate) fn proposition_fact(&self, proposition_id: usize) -> Option<&ExplicitFact> {
        self.proposition_facts.get(proposition_id)
    }

    pub(crate) fn proposition_var_id(&self, proposition_id: usize) -> Option<usize> {
        self.proposition_var_ids.get(proposition_id).copied()
    }

    pub(crate) fn get_add_actions(&self, proposition_id: usize) -> Option<&[usize]> {
        self.proposition_add_action_ids
            .get(proposition_id)
            .map(Vec::as_slice)
    }

    pub(crate) fn get_proposition_name(&self, proposition_id: usize) -> Option<&str> {
        self.proposition_names
            .get(proposition_id)
            .map(String::as_str)
    }

    pub(crate) fn get_mutex_actions(&self, action_id: usize) -> Option<&[usize]> {
        self.mutex_action_ids.get(action_id).map(Vec::as_slice)
    }

    pub(crate) fn comparison_axiom_id_for_var(&self, variable_id: usize) -> Option<usize> {
        self.comparison_axiom_by_var.get(&variable_id).copied()
    }

    pub(crate) fn fact_to_axiom_marker(&self, variable_id: usize) -> Option<Option<usize>> {
        self.fact_to_axiom_marker.get(variable_id).copied()
    }

    pub(crate) fn is_comparison_axiom_var(&self, variable_id: usize) -> bool {
        self.comparison_axiom_by_var.contains_key(&variable_id)
    }

    pub(crate) fn is_numeric_axiom_var(&self, variable_id: usize) -> bool {
        self.fact_to_axiom_marker(variable_id)
            .map(|marker| marker.is_some())
            .unwrap_or(false)
    }

    pub(crate) fn comparison_fact_conditions(
        &self,
        variable_id: usize,
        fact_value: usize,
    ) -> Option<&[LinearNumericCondition]> {
        self.comparison_fact_conditions
            .get(&(variable_id, fact_value))
            .map(Vec::as_slice)
    }

    pub(crate) fn comparison_fact_condition_group_ids(
        &self,
        variable_id: usize,
        fact_value: usize,
    ) -> Option<&[usize]> {
        self.comparison_fact_condition_group_ids
            .get(&(variable_id, fact_value))
            .map(Vec::as_slice)
    }

    pub(crate) fn goal_helper_propositional_facts(
        &self,
        variable_id: usize,
    ) -> Option<&[ExplicitFact]> {
        self.goal_helper_propositional_facts
            .get(&variable_id)
            .map(Vec::as_slice)
    }

    pub(crate) fn goal_helper_numeric_conditions(
        &self,
        variable_id: usize,
    ) -> Option<&[LinearNumericCondition]> {
        self.goal_helper_numeric_conditions
            .get(&variable_id)
            .map(Vec::as_slice)
    }

    pub(crate) fn goal_helper_numeric_condition_group_ids(
        &self,
        variable_id: usize,
    ) -> Option<&[usize]> {
        self.goal_helper_numeric_condition_group_ids
            .get(&variable_id)
            .map(Vec::as_slice)
    }

    pub(crate) fn goal_model(&self, goal_id: usize) -> Option<&HelperPreconditionLists> {
        self.goal_models.get(goal_id)
    }

    pub(crate) fn get_numeric_goals(&self, goal_id: usize) -> Vec<usize> {
        self.goal_model(goal_id)
            .into_iter()
            .flat_map(|goal_model| goal_model.numeric_group_ids.iter().copied())
            .flat_map(|pre_id| {
                self.get_numeric_conditions_id(pre_id)
                    .into_iter()
                    .flatten()
                    .copied()
            })
            .collect()
    }

    pub(crate) fn goal_materialized_conditions(
        &self,
        goal_id: usize,
    ) -> Vec<LinearNumericCondition> {
        self.goal_model(goal_id)
            .map(|goal_model| {
                self.materialized_conditions_for_group_ids(&goal_model.numeric_group_ids)
            })
            .unwrap_or_default()
    }

    pub(crate) fn condition_group_condition_ids(&self, group_id: usize) -> Option<&[usize]> {
        self.condition_group_condition_ids
            .get(group_id)
            .map(Vec::as_slice)
    }

    pub(crate) fn get_numeric_conditions_id(&self, pre_id: usize) -> Option<&[usize]> {
        self.condition_group_condition_ids(pre_id)
    }

    pub(crate) fn get_condition_ids_from_group_ids(&self, group_ids: &[usize]) -> Vec<usize> {
        let mut condition_ids = Vec::new();
        let mut seen = BTreeSet::new();
        for &group_id in group_ids {
            let Some(ids) = self.get_numeric_conditions_id(group_id) else {
                continue;
            };
            for &condition_id in ids {
                if seen.insert(condition_id) {
                    condition_ids.push(condition_id);
                }
            }
        }
        condition_ids
    }

    pub(crate) fn get_comparison_fact_condition_ids(
        &self,
        variable_id: usize,
        fact_value: usize,
    ) -> Vec<usize> {
        self.comparison_fact_condition_group_ids(variable_id, fact_value)
            .map(|group_ids| self.get_condition_ids_from_group_ids(group_ids))
            .unwrap_or_default()
    }

    pub(crate) fn materialized_conditions_for_group_id(
        &self,
        group_id: usize,
    ) -> Vec<LinearNumericCondition> {
        self.condition_group_condition_ids(group_id)
            .into_iter()
            .flatten()
            .filter_map(|&condition_id| self.normalized_condition(condition_id).cloned())
            .collect()
    }

    pub(crate) fn materialized_conditions_for_group_ids(
        &self,
        group_ids: &[usize],
    ) -> Vec<LinearNumericCondition> {
        group_ids
            .iter()
            .flat_map(|&group_id| self.materialized_conditions_for_group_id(group_id))
            .collect()
    }

    pub(crate) fn condition_group_representative_condition_id(
        &self,
        group_id: usize,
    ) -> Option<usize> {
        self.condition_group_representative_condition_ids
            .get(group_id)
            .copied()
    }

    pub(crate) fn normalized_condition(
        &self,
        condition_id: usize,
    ) -> Option<&LinearNumericCondition> {
        self.normalized_conditions.get(condition_id)
    }

    pub(crate) fn get_condition(&self, condition_id: usize) -> Option<&LinearNumericCondition> {
        self.normalized_condition(condition_id)
    }

    pub(crate) fn comparison_fact_materialized_conditions(
        &self,
        variable_id: usize,
        fact_value: usize,
    ) -> Vec<LinearNumericCondition> {
        self.comparison_fact_condition_group_ids(variable_id, fact_value)
            .map(|group_ids| self.materialized_conditions_for_group_ids(group_ids))
            .unwrap_or_default()
    }

    pub(crate) fn combine_condition_with_conditions(
        &self,
        base_condition: &LinearNumericCondition,
        other_conditions: &[LinearNumericCondition],
    ) -> Vec<LinearNumericCondition> {
        other_conditions
            .iter()
            .map(|condition| base_condition.add(condition))
            .collect()
    }

    pub(crate) fn pairwise_combined_conditions(
        &self,
        conditions: &[LinearNumericCondition],
        precision: f64,
    ) -> Vec<LinearNumericCondition> {
        let mut combined_conditions = Vec::new();
        for left_index in 0..conditions.len() {
            for right_index in (left_index + 1)..conditions.len() {
                let combined = conditions[left_index].add(&conditions[right_index]);
                if combined.is_empty(precision) {
                    continue;
                }
                combined_conditions.push(combined);
            }
        }
        combined_conditions
    }

    pub(crate) fn preconditions_for_assignment_effect(
        &self,
        action_id: usize,
        assignment_effect_id: usize,
    ) -> Option<&HelperPreconditionLists> {
        let action_model = self.action_model(action_id)?;
        action_model
            .linear_effects
            .iter()
            .find(|effect| effect.source_assignment_effect_id == assignment_effect_id)
            .map(|effect| &effect.preconditions)
            .or_else(|| {
                action_model
                    .conditional_numeric_effects
                    .iter()
                    .find(|effect| effect.source_assignment_effect_id == assignment_effect_id)
                    .map(|effect| &effect.preconditions)
            })
            .or_else(|| {
                action_model
                    .conditional_assignment_effects
                    .iter()
                    .find(|effect| effect.source_assignment_effect_id == assignment_effect_id)
                    .map(|effect| &effect.preconditions)
            })
    }

    pub(crate) fn linear_effect_for_assignment_effect(
        &self,
        action_id: usize,
        assignment_effect_id: usize,
    ) -> Option<&HelperLinearEffect> {
        self.action_model(action_id)?
            .linear_effects
            .iter()
            .find(|effect| effect.source_assignment_effect_id == assignment_effect_id)
    }

    pub(crate) fn build_precondition_lists(
        &self,
        preconditions: &[ExplicitFact],
    ) -> (Vec<ExplicitFact>, Vec<usize>) {
        let mut propositional_facts = Vec::new();
        let mut numeric_group_ids = Vec::new();
        let mut seen_propositional = BTreeSet::new();
        let mut seen_numeric = BTreeSet::new();

        for condition in preconditions {
            let var_id = condition.var();
            if !self.is_numeric_axiom_var(var_id) {
                if seen_propositional.insert((condition.var(), condition.value())) {
                    propositional_facts.push(condition.clone());
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
        let var_id = fact.var();
        if let Some(group_ids) = self.comparison_fact_condition_group_ids(var_id, fact.value()) {
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

    fn build_actions(
        &mut self,
        task: &dyn AbstractNumericTask,
        precision: f64,
        separate_constant_assignment: bool,
    ) {
        self.action_models.clear();
        for (operator_id, operator) in task.get_operators().iter().enumerate() {
            let action_model = self.build_action(
                task,
                operator,
                operator_id,
                precision,
                separate_constant_assignment,
            );
            self.action_models.push(action_model);
        }
        for axiom in task.axioms() {
            let action_model = self.build_axiom_action(axiom, precision);
            self.action_models.push(action_model);
        }
        self.build_possible_add_lists();
    }

    fn build_propositions(&mut self, task: &dyn AbstractNumericTask) {
        self.proposition_ids_by_var_value.clear();
        self.proposition_facts.clear();
        self.proposition_var_ids.clear();
        self.proposition_names.clear();
        for var_id in 0..task.get_num_variables() {
            let domain_size = task
                .get_variable_domain_size(var_id)
                .expect("helper proposition domain size must exist");
            let mut ids = Vec::with_capacity(domain_size);
            for value in 0..domain_size {
                let proposition_id = self.proposition_facts.len();
                ids.push(proposition_id);
                let fact = ExplicitFact::new(var_id, value);
                self.proposition_facts.push(fact.clone());
                self.proposition_var_ids.push(var_id);
                self.proposition_names
                    .push(task.get_fact_name(&fact).to_string());
            }
            self.proposition_ids_by_var_value.push(ids);
        }

        self.proposition_add_action_ids = vec![Vec::new(); self.proposition_facts.len()];
        for (action_id, action_model) in self.action_models.iter().enumerate() {
            for fact in action_model.add_facts.iter().chain(
                action_model
                    .conditional_fact_effects
                    .iter()
                    .map(|effect| &effect.add_fact),
            ) {
                if let Some(proposition_id) = self.get_proposition(fact.var(), fact.value()) {
                    self.proposition_add_action_ids[proposition_id].push(action_id);
                }
            }
        }
    }

    fn build_mutex_actions(&mut self, task: &dyn AbstractNumericTask) {
        let operators = task.get_operators();
        let num_variables = task.get_num_variables();
        self.mutex_action_ids = vec![Vec::new(); operators.len()];
        for (op_id, operator) in operators.iter().enumerate() {
            let mut precondition = vec![None; num_variables];
            let mut postcondition = vec![None; num_variables];
            for condition in operator.preconditions() {
                precondition[condition.var()] = Some(condition.value());
            }
            for effect in operator.effects() {
                postcondition[effect.var_id()] = Some(effect.value());
            }

            for (other_op_id, other_operator) in operators.iter().enumerate() {
                if op_id == other_op_id {
                    continue;
                }
                let mut is_mutex = false;
                for effect in other_operator.effects() {
                    let var_id = effect.var_id();
                    let post = effect.value();
                    if (precondition[var_id].is_some() && precondition[var_id].unwrap() != post)
                        || postcondition[var_id].is_none()
                        || postcondition[var_id].unwrap() != post
                    {
                        is_mutex = true;
                        break;
                    }
                }
                if is_mutex {
                    self.mutex_action_ids[op_id].push(other_op_id);
                }
            }
        }
    }

    fn build_action(
        &mut self,
        task: &dyn AbstractNumericTask,
        operator: &Operator,
        operator_id: usize,
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
            cost: operator.cost() as f64,
            ..HelperActionModel::default()
        };

        let mut base_precondition_values = BTreeMap::new();
        for precondition in operator.preconditions() {
            base_precondition_values.insert(precondition.var(), precondition.value());
        }

        for effect in operator.effects() {
            let add_fact = ExplicitFact::new(effect.var_id(), effect.value());
            if effect.conditions().is_empty() {
                action_model.add_facts.push(add_fact.clone());
                if let Some(&pre_value) = base_precondition_values.get(&(effect.var_id())) {
                    action_model
                        .del_facts
                        .push(ExplicitFact::new(effect.var_id(), pre_value));
                }
            } else {
                let conditional_preconditions = self
                    .build_conditional_precondition_lists_with_redundancy(
                        operator.preconditions(),
                        effect.conditions(),
                        precision,
                    );
                let mut extended_precondition_values = base_precondition_values.clone();
                for condition in effect.conditions() {
                    extended_precondition_values.insert(condition.var(), condition.value());
                }
                let del_fact = extended_precondition_values
                    .get(&(effect.var_id()))
                    .copied()
                    .map(|pre_value| ExplicitFact::new(effect.var_id(), pre_value));
                action_model
                    .conditional_fact_effects
                    .push(HelperConditionalExplicitFactEffect {
                        preconditions: conditional_preconditions,
                        add_fact,
                        del_fact,
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
                assignment_effect_id,
                assignment_effect,
                linear_effect,
                precision,
                separate_constant_assignment,
                &mut action_model,
            );
        }

        action_model
    }

    fn build_axiom_action(
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
        action_model
            .add_facts
            .push(ExplicitFact::new(axiom.var_id(), axiom.effect_value()));
        action_model.del_facts.push(ExplicitFact::new(
            axiom.var_id(),
            axiom.precondition_value(),
        ));
        action_model.pre_del_facts = intersect_fact_lists(
            &action_model.propositional_preconditions,
            &action_model.del_facts,
        );
        action_model
    }

    #[allow(clippy::too_many_arguments)]
    fn classify_assignment_effect_into_action(
        &mut self,
        task: &dyn AbstractNumericTask,
        base_preconditions: &[ExplicitFact],
        assignment_effect_id: usize,
        assignment_effect: &AssignmentEffect,
        linear_effect: &LinearNumericEffect,
        precision: f64,
        separate_constant_assignment: bool,
        action_model: &mut HelperActionModel,
    ) {
        let affected_var_id = linear_effect.affected_var_id;
        let Some(local_var_id) = self.local_numeric_var_id(affected_var_id) else {
            return;
        };

        let final_expression = final_expression_from_effect(linear_effect);
        let conditional_preconditions =
            if assignment_effect.is_conditional() || !assignment_effect.conditions().is_empty() {
                Some(self.build_conditional_precondition_lists_with_redundancy(
                    base_preconditions,
                    assignment_effect.conditions(),
                    precision,
                ))
            } else {
                None
            };

        if is_simple_numeric_effect(linear_effect) {
            let delta = linear_effect.delta.constant;
            if let Some(preconditions) = conditional_preconditions {
                action_model
                    .conditional_numeric_effects
                    .push(HelperConditionalNumericEffect {
                        source_assignment_effect_id: assignment_effect_id,
                        preconditions,
                        target_local_var_id: local_var_id,
                        delta,
                    });
            } else {
                action_model.simple_effects[local_var_id] = delta;
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
            if let Some(preconditions) = conditional_preconditions {
                action_model.conditional_assignment_effects.push(
                    HelperConditionalAssignmentEffect {
                        source_assignment_effect_id: assignment_effect_id,
                        preconditions,
                        target_local_var_id: local_var_id,
                        assigned_value,
                    },
                );
            } else {
                action_model.is_assignment[local_var_id] = true;
                action_model.assignment_values[local_var_id] = assigned_value;
            }
            return;
        }

        let coefficients = self
            .numeric_variable_ids
            .iter()
            .map(|&task_var_id| final_expression.coefficients[task_var_id])
            .collect::<Vec<_>>();
        action_model.linear_effects.push(HelperLinearEffect {
            source_assignment_effect_id: assignment_effect_id,
            source_var_id: assignment_effect.var_id(),
            operation: assignment_effect.operation().clone(),
            preconditions: conditional_preconditions.unwrap_or_default(),
            target_local_var_id: local_var_id,
            coefficients,
            constant: final_expression.constant,
        });
    }

    fn build_possible_add_lists(&mut self) {
        self.condition_achievers = vec![Vec::new(); self.normalized_conditions.len()];
        for (action_id, action_model) in self.action_models.iter_mut().enumerate() {
            action_model.possible_add_condition_ids.clear();
            for (condition_id, condition) in self.normalized_conditions.iter().enumerate() {
                let cumulative_effect = self
                    .numeric_variable_ids
                    .iter()
                    .enumerate()
                    .map(|(local_var_id, &task_var_id)| {
                        condition.coefficients[task_var_id]
                            * action_model.simple_effects[local_var_id]
                    })
                    .sum::<f64>();
                if cumulative_effect > 0.0 {
                    action_model.possible_add_condition_ids.push(condition_id);
                    self.condition_achievers[condition_id].push(action_id);
                }
            }
        }
    }

    fn calculates_bounds_numeric_variables(
        &mut self,
        task: &dyn AbstractNumericTask,
        precision: f64,
        infinity: f64,
    ) {
        let initial_numeric_values = task.get_initial_numeric_state_values();
        self.numeric_variable_lower_bounds = self
            .numeric_variable_ids
            .iter()
            .map(|&task_var_id| {
                initial_numeric_values
                    .get(task_var_id)
                    .copied()
                    .unwrap_or(0.0)
            })
            .collect();
        self.numeric_variable_upper_bounds = self.numeric_variable_lower_bounds.clone();

        for local_var_id in 0..self.numeric_variable_ids.len() {
            for action_model in self.action_models.iter().take(task.get_operators().len()) {
                let effect = action_model.simple_effects[local_var_id];
                if effect.abs() < precision {
                    continue;
                }
                let mut bound_for_action = false;
                let mut bound = 0.0;
                for &group_id in &action_model.numeric_precondition_group_ids {
                    let Some(condition_ids) = self.condition_group_condition_ids(group_id) else {
                        continue;
                    };
                    for &condition_id in condition_ids {
                        let Some(condition) = self.normalized_condition(condition_id) else {
                            continue;
                        };
                        let task_var_id = self.numeric_variable_ids[local_var_id];
                        let weight = condition.coefficients[task_var_id];
                        if !simple_on_regular_var(
                            condition,
                            &self.numeric_variable_ids,
                            task_var_id,
                            precision,
                        ) {
                            continue;
                        }
                        if weight * effect > 0.0 {
                            continue;
                        }
                        bound_for_action = true;
                        bound = -condition.constant / weight + effect;
                    }
                }
                if bound_for_action {
                    if effect > 0.0 {
                        self.numeric_variable_upper_bounds[local_var_id] =
                            self.numeric_variable_upper_bounds[local_var_id].max(bound);
                    } else {
                        self.numeric_variable_lower_bounds[local_var_id] =
                            self.numeric_variable_lower_bounds[local_var_id].min(bound);
                    }
                } else if effect > 0.0 {
                    self.numeric_variable_upper_bounds[local_var_id] = infinity;
                } else {
                    self.numeric_variable_lower_bounds[local_var_id] = -infinity;
                }
            }
        }
    }

    fn calculate_small_m(&mut self, precision: f64, infinity: f64) {
        self.condition_small_m = vec![-infinity; self.normalized_conditions.len()];
        for (condition_id, condition) in self.normalized_conditions.iter().enumerate() {
            let mut lower_bound = condition.constant;
            for (local_var_id, &task_var_id) in self.numeric_variable_ids.iter().enumerate() {
                let coefficient = condition.coefficients[task_var_id];
                if coefficient >= precision {
                    let variable_lb = self.numeric_variable_lower_bounds[local_var_id];
                    if variable_lb > -infinity {
                        lower_bound += coefficient * variable_lb;
                    } else {
                        lower_bound = -infinity;
                        break;
                    }
                } else if coefficient <= -precision {
                    let variable_ub = self.numeric_variable_upper_bounds[local_var_id];
                    if variable_ub < infinity {
                        lower_bound += coefficient * variable_ub;
                    } else {
                        lower_bound = -infinity;
                        break;
                    }
                }
            }
            self.condition_small_m[condition_id] = lower_bound;
        }
    }

    fn calculate_epsilons(&mut self, num_operators: usize, precision: f64, default_epsilon: f64) {
        self.condition_epsilons = vec![0.0; self.normalized_conditions.len()];
        for (condition_id, condition) in self.normalized_conditions.iter().enumerate() {
            if !condition.is_strictly_greater {
                continue;
            }
            let mut min_epsilon = calculate_epsilon_value(condition.constant, precision);
            let mut use_default_epsilon = false;
            for action_model in self.action_models.iter().take(num_operators) {
                for linear_effect in &action_model.linear_effects {
                    let task_var_id = self.numeric_variable_ids[linear_effect.target_local_var_id];
                    if condition.coefficients[task_var_id].abs() >= precision {
                        use_default_epsilon = true;
                        break;
                    }
                }
                if use_default_epsilon {
                    break;
                }
                for (local_var_id, &is_assignment) in action_model.is_assignment.iter().enumerate()
                {
                    if is_assignment {
                        min_epsilon = min_epsilon.min(calculate_epsilon_value(
                            action_model.assignment_values[local_var_id],
                            precision,
                        ));
                    }
                }
                let effect = self
                    .numeric_variable_ids
                    .iter()
                    .enumerate()
                    .map(|(local_var_id, &task_var_id)| {
                        condition.coefficients[task_var_id]
                            * action_model.simple_effects[local_var_id]
                    })
                    .sum::<f64>();
                min_epsilon = min_epsilon.min(calculate_epsilon_value(effect, precision));
            }
            self.condition_epsilons[condition_id] = if use_default_epsilon {
                default_epsilon
            } else {
                min_epsilon.max(default_epsilon)
            };
        }
    }

    fn calculate_dominance(&mut self, precision: f64) {
        let size = self.normalized_conditions.len();
        self.dominance_conditions = vec![vec![false; size]; size];
        for left_id in 0..size {
            for right_id in left_id..size {
                if self.normalized_conditions[left_id]
                    .dominates(&self.normalized_conditions[right_id], precision)
                {
                    self.dominance_conditions[left_id][right_id] = true;
                }
                if self.normalized_conditions[right_id]
                    .dominates(&self.normalized_conditions[left_id], precision)
                {
                    self.dominance_conditions[right_id][left_id] = true;
                }
            }
        }
    }

    fn build_numeric_conditions(&mut self, task: &dyn AbstractNumericTask) {
        for (comparison_axiom_id, comparison_axiom) in task.comparison_axioms().iter().enumerate() {
            let affected_var_id = comparison_axiom.get_affected_var_id();
            self.comparison_axiom_by_var
                .insert(affected_var_id, comparison_axiom_id);
            if let Some(marker) = self.fact_to_axiom_marker.get_mut(affected_var_id) {
                *marker = Some(comparison_axiom_id);
            }

            // PARITY(numeric-fd): `NumericTaskProxy::build_numeric_conditions()` only materializes
            // the comparison-axiom-side numeric conditions that correspond to the kept numeric
            // precondition polarity. The Rust LM-cut port currently follows the same one-sided
            // lookup and filters unsupported inequality facts earlier.
            if let Some(conditions) =
                self.build_conditions_for_fact_value(task, comparison_axiom, affected_var_id, 0)
            {
                let group_id = self.register_condition_group(conditions.clone());
                self.comparison_fact_condition_group_ids
                    .insert((affected_var_id, 0), vec![group_id]);
                self.comparison_fact_conditions
                    .insert((affected_var_id, 0), conditions);
            }
        }
    }

    fn build_numeric_goals(&mut self, task: &dyn AbstractNumericTask, precision: f64) {
        self.goal_models = vec![HelperPreconditionLists::default(); task.get_num_goals()];
        let mut axiom_table: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        let mut fact_table: BTreeMap<usize, Vec<ExplicitFact>> = BTreeMap::new();

        for axiom in task.axioms() {
            let effect_var_id = axiom.var_id();
            for precondition in axiom.conditions() {
                let precondition_var_id = precondition.var();
                if self.is_comparison_axiom_var(precondition_var_id) {
                    axiom_table
                        .entry(effect_var_id)
                        .or_default()
                        .push(precondition_var_id);
                } else {
                    fact_table
                        .entry(effect_var_id)
                        .or_default()
                        .push(precondition.clone());
                }
            }
        }

        for goal_index in 0..task.get_num_goals() {
            let goal = task.get_goal_fact(goal_index);
            let goal_var_id = goal.var();
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
                    .comparison_fact_condition_group_ids(helper_numeric_var_id, 0)
                    .map(|group_ids| group_ids.to_vec())
                {
                    numeric_condition_group_ids.extend(group_ids);
                }
                if let Some(conditions) = self
                    .comparison_fact_conditions(helper_numeric_var_id, 0)
                    .map(|conditions| conditions.to_vec())
                {
                    for condition in conditions {
                        if !condition.is_empty(precision) {
                            numeric_conditions.push(condition);
                            self.numeric_goal_helper_vars.insert(goal_var_id);
                            if let Some(marker) = self.fact_to_axiom_marker.get_mut(goal_var_id) {
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
    }

    fn register_condition_group(&mut self, conditions: Vec<LinearNumericCondition>) -> usize {
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

    fn build_conditions_for_fact_value(
        &self,
        task: &dyn AbstractNumericTask,
        comparison_axiom: &ComparisonAxiom,
        affected_var_id: usize,
        fact_value: usize,
    ) -> Option<Vec<LinearNumericCondition>> {
        let operator = comparison_operator_for_fact_value(comparison_axiom, fact_value)?;
        let lhs = comparison_axiom.get_left_var_id();
        let rhs = comparison_axiom.get_right_var_id();
        let fact = ExplicitFact::new(affected_var_id, fact_value);
        let fact_name = task.get_fact_name(&fact).to_string();

        match operator {
            ComparisonOperator::GreaterThan | ComparisonOperator::GreaterThanOrEqual => {
                Some(vec![self.build_condition(
                    task,
                    lhs,
                    rhs,
                    matches!(operator, ComparisonOperator::GreaterThan),
                    fact_name,
                )])
            }
            ComparisonOperator::LessThan | ComparisonOperator::LessThanOrEqual => {
                Some(vec![self.build_condition(
                    task,
                    rhs,
                    lhs,
                    matches!(operator, ComparisonOperator::LessThan),
                    fact_name,
                )])
            }
            ComparisonOperator::Equal => Some(vec![
                self.build_condition(task, lhs, rhs, false, fact_name.clone()),
                self.build_condition(task, rhs, lhs, false, fact_name),
            ]),
            ComparisonOperator::UnEqual => None,
        }
    }

    fn build_condition(
        &self,
        task: &dyn AbstractNumericTask,
        positive_var_id: usize,
        negative_var_id: usize,
        is_strictly_greater: bool,
        name: String,
    ) -> LinearNumericCondition {
        let positive_expression =
            task.linearize_numeric_var(positive_var_id)
                .unwrap_or_else(|error| {
                    panic!(
                        "failed to linearize numeric helper lhs variable {positive_var_id}: {error}"
                    )
                });
        let negative_expression =
            task.linearize_numeric_var(negative_var_id)
                .unwrap_or_else(|error| {
                    panic!(
                        "failed to linearize numeric helper rhs variable {negative_var_id}: {error}"
                    )
                });
        let expression = positive_expression.subtract(&negative_expression);
        LinearNumericCondition::from_expression(
            expression,
            is_strictly_greater,
            format!("numeric ({name})"),
        )
    }
}

fn invert_comparison_operator(operator: &ComparisonOperator) -> ComparisonOperator {
    match operator {
        ComparisonOperator::LessThan => ComparisonOperator::GreaterThanOrEqual,
        ComparisonOperator::LessThanOrEqual => ComparisonOperator::GreaterThan,
        ComparisonOperator::Equal => ComparisonOperator::UnEqual,
        ComparisonOperator::GreaterThanOrEqual => ComparisonOperator::LessThan,
        ComparisonOperator::GreaterThan => ComparisonOperator::LessThanOrEqual,
        ComparisonOperator::UnEqual => ComparisonOperator::Equal,
    }
}

fn comparison_operator_for_fact_value(
    comparison_axiom: &ComparisonAxiom,
    fact_value: usize,
) -> Option<ComparisonOperator> {
    assert!(
        fact_value == 0 || fact_value == 1,
        "comparison fact value must be boolean-like, got {fact_value}"
    );

    let operator = if fact_value == 0 {
        comparison_axiom.get_operator().clone()
    } else {
        invert_comparison_operator(comparison_axiom.get_operator())
    };

    match operator {
        ComparisonOperator::UnEqual => None,
        supported => Some(supported),
    }
}

fn intersect_fact_lists(left: &[ExplicitFact], right: &[ExplicitFact]) -> Vec<ExplicitFact> {
    let right_set = right
        .iter()
        .map(|fact| (fact.var(), fact.value()))
        .collect::<BTreeSet<_>>();
    left.iter()
        .filter(|fact| right_set.contains(&(fact.var(), fact.value())))
        .cloned()
        .collect()
}

fn final_expression_from_effect(linear_effect: &LinearNumericEffect) -> LinearExpression {
    let mut expression = linear_effect.delta.clone();
    if let Some(coefficient) = expression
        .coefficients
        .get_mut(linear_effect.affected_var_id)
    {
        *coefficient += 1.0;
    }
    expression
}

fn is_simple_numeric_effect(linear_effect: &LinearNumericEffect) -> bool {
    matches!(
        linear_effect.operation,
        AssignmentOperation::Plus | AssignmentOperation::Minus
    ) && linear_effect.delta.is_constant()
}

fn is_constant_assignment_like_effect(
    task: &dyn AbstractNumericTask,
    linear_effect: &LinearNumericEffect,
    final_expression: &LinearExpression,
    precision: f64,
    separate_constant_assignment: bool,
) -> bool {
    if !separate_constant_assignment {
        return false;
    }
    if task
        .numeric_variables()
        .get(linear_effect.affected_var_id)
        .map(|numeric_var| numeric_var.get_type() != &NumericType::Regular)
        .unwrap_or(true)
    {
        return false;
    }
    final_expression
        .coefficients
        .iter()
        .all(|coefficient| coefficient.abs() < precision)
}

fn simple_on_regular_var(
    condition: &LinearNumericCondition,
    regular_numeric_variable_ids: &[usize],
    target_task_var_id: usize,
    precision: f64,
) -> bool {
    for &task_var_id in regular_numeric_variable_ids {
        if task_var_id != target_task_var_id
            && condition.coefficients[task_var_id].abs() > precision
        {
            return false;
        }
    }
    true
}

fn calculate_epsilon_value(mut value: f64, precision: f64) -> f64 {
    let mut epsilon = 1.0;
    let mut fractional_part = (value.round() - value).abs();
    while fractional_part >= precision {
        value *= 10.0;
        epsilon /= 10.0;
        fractional_part = (value.round() - value).abs();
    }
    epsilon
}

fn append_unique_ids(target: &mut Vec<usize>, source: &[usize]) {
    let mut seen: BTreeSet<usize> = target.iter().copied().collect();
    for &value in source {
        if seen.insert(value) {
            target.push(value);
        }
    }
}
