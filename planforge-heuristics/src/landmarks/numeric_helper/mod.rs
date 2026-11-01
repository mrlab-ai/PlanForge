mod actions;
mod bounds;
mod conditions;

use planforge_sas::axioms::{ComparisonAxiom, ComparisonOperator, PropositionalAxiom};
use planforge_sas::numeric_task::Task;
use planforge_sas::numeric_task::{
    AssignmentEffect, AssignmentOperation, ExplicitFact, ExplicitValueIndex, NumericType,
    NumericValue, Operator, OperatorIndex, VariableIndex, ZERO_VALUE,
};
use planforge_sas::utils::linear_effects::{LinearExpression, LinearNumericEffect};
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

    pub(crate) fn evaluate_slack(&self, numeric_values: &[NumericValue], epsilon: f64) -> f64 {
        assert!(
            numeric_values.len() >= self.coefficients.len(),
            "numeric state too short for condition slack"
        );
        let net = NumericValue::affine(
            self.coefficients
                .iter()
                .copied()
                .zip(numeric_values.iter().copied()),
            self.constant,
        )
        .value();
        if self.is_strictly_greater {
            net - epsilon
        } else {
            net
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct NumericTaskHelper {
    normalized_conditions: Vec<LinearNumericCondition>,
    condition_group_condition_ids: Vec<Vec<usize>>,
    condition_group_representative_condition_ids: Vec<usize>,
    fact_to_axiom_marker: Vec<Option<usize>>,
    numeric_variable_ids: Vec<VariableIndex>,
    numeric_variable_index_by_task_id: Vec<Option<VariableIndex>>,
    comparison_axiom_by_var: BTreeMap<VariableIndex, usize>,
    comparison_fact_condition_group_ids: BTreeMap<(VariableIndex, ExplicitValueIndex), Vec<usize>>,
    comparison_fact_conditions:
        BTreeMap<(VariableIndex, ExplicitValueIndex), Vec<LinearNumericCondition>>,
    goal_helper_propositional_facts: BTreeMap<VariableIndex, Vec<ExplicitFact>>,
    goal_helper_numeric_condition_group_ids: BTreeMap<VariableIndex, Vec<usize>>,
    goal_helper_numeric_conditions: BTreeMap<VariableIndex, Vec<LinearNumericCondition>>,
    goal_models: Vec<HelperPreconditionLists>,
    numeric_goal_helper_vars: BTreeSet<VariableIndex>,
    action_models: Vec<HelperActionModel>,
    numeric_variable_lower_bounds: Vec<f64>,
    numeric_variable_upper_bounds: Vec<f64>,
    condition_epsilons: Vec<f64>,
    proposition_ids_by_var_value: Vec<Vec<usize>>,
    proposition_facts: Vec<ExplicitFact>,
    proposition_var_ids: Vec<usize>,
    proposition_names: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct HelperPreconditionLists {
    pub(crate) propositional_facts: Vec<ExplicitFact>,
    pub(crate) numeric_group_ids: Vec<usize>,
}

#[derive(Debug, Clone)]
pub(crate) struct HelperConditionalExplicitFactEffect {
    pub(crate) preconditions: HelperPreconditionLists,
    pub(crate) add_fact: ExplicitFact,
}

/// A conditional additive effect. Its condition is relaxed away: the
/// heuristics count its change wherever that helps them.
#[derive(Debug, Clone)]
pub(crate) struct HelperConditionalNumericEffect {
    pub(crate) target_local_var_id: VariableIndex,
    pub(crate) delta: f64,
}

/// A conditional constant assignment, its condition relaxed away as for
/// [`HelperConditionalNumericEffect`].
#[derive(Debug, Clone)]
pub(crate) struct HelperConditionalAssignmentEffect {
    pub(crate) target_local_var_id: VariableIndex,
    pub(crate) assigned_value: f64,
}

#[derive(Debug, Clone)]
pub(crate) struct HelperLinearEffect {
    pub(crate) source_assignment_effect_id: usize,
    pub(crate) source_var_id: VariableIndex,
    pub(crate) operation: AssignmentOperation,
    pub(crate) preconditions: HelperPreconditionLists,
    pub(crate) target_local_var_id: VariableIndex,
    pub(crate) coefficients: Vec<f64>,
    pub(crate) constant: f64,
}

/// One assignment effect of an operator in the three forms classification needs
/// it: its index in the operator, the effect itself, and its linearized form.
#[derive(Clone, Copy)]
struct ClassifiedAssignmentEffect<'a> {
    id: usize,
    effect: &'a AssignmentEffect,
    linear: &'a LinearNumericEffect,
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
    pub(crate) cost: f64,
    pub(crate) linear_cost: bool,
    pub(crate) cost_coefficients: Vec<f64>,
    pub(crate) cost_constant: f64,
}

impl NumericTaskHelper {
    /// `default_value_axioms` are the rules describing how a derived variable
    /// takes its default value. They are not part of the task -- the consumer
    /// that reads the axioms as relaxed operators derives them for itself -- and
    /// they get an action model each, after the task's own axioms, so that a
    /// caller can index both by one axiom offset.
    pub(crate) fn new_lmcut(
        task: &Task,
        default_value_axioms: &[PropositionalAxiom],
        precision: f64,
        default_epsilon: f64,
        separate_constant_assignment: bool,
    ) -> Self {
        Self::new_with_options(
            task,
            default_value_axioms,
            precision,
            default_epsilon,
            separate_constant_assignment,
            true,
        )
    }

    #[cfg(feature = "cplex")]
    /// Build the task metadata required by numeric potentials.
    ///
    /// Potentials consume materialized comparison conditions and numeric
    /// variable bounds, but never the landmark-only mutex-action matrix or
    /// quadratic condition-dominance table. Avoiding those unused structures
    /// is essential on tasks with thousands of numeric comparisons.
    pub(crate) fn new_potentials(
        task: &Task,
        precision: f64,
        default_epsilon: f64,
        build_bound_metadata: bool,
    ) -> Self {
        let mut helper = Self {
            fact_to_axiom_marker: vec![None; task.variables().len()],
            numeric_variable_index_by_task_id: vec![None; task.numeric_variables().len()],
            ..Self::default()
        };
        helper.numeric_variable_ids = task.regular_numeric_variable_ids();
        for (local_var_id, &task_var_id) in helper.numeric_variable_ids.iter().enumerate() {
            if let Some(index) = helper
                .numeric_variable_index_by_task_id
                .get_mut(task_var_id.index())
            {
                *index = Some(VariableIndex::from_usize(local_var_id));
            }
        }
        helper.build_numeric_conditions(task);
        if build_bound_metadata {
            helper.build_numeric_goals(task, precision);
            helper.build_actions(task, &[], precision, false);
            helper.build_propositions(task);
            helper.calculates_bounds_numeric_variables(task, precision, 9_999_999.0);
            helper.calculate_epsilons(task.operators().len(), precision, default_epsilon);
        }
        helper
    }

    fn new_with_options(
        task: &Task,
        default_value_axioms: &[PropositionalAxiom],
        precision: f64,
        default_epsilon: f64,
        separate_constant_assignment: bool,
        build_bound_metadata: bool,
    ) -> Self {
        let mut helper = Self {
            fact_to_axiom_marker: vec![None; task.variables().len()],
            numeric_variable_index_by_task_id: vec![None; task.numeric_variables().len()],
            ..Self::default()
        };
        helper.numeric_variable_ids = task.regular_numeric_variable_ids();
        for (local_var_id, &task_var_id) in helper.numeric_variable_ids.iter().enumerate() {
            if let Some(index) = helper
                .numeric_variable_index_by_task_id
                .get_mut(task_var_id.index())
            {
                *index = Some(VariableIndex::from_usize(local_var_id));
            }
        }
        helper.build_numeric_conditions(task);
        // Deliberately the task's own axioms only: this reconstructs the
        // conditions hidden behind a *derived goal*, and a rule refuting that
        // goal's variable would contribute the negation of those conditions.
        helper.build_numeric_goals(task, precision);
        helper.build_actions(
            task,
            default_value_axioms,
            precision,
            separate_constant_assignment,
        );
        helper.build_propositions(task);
        if build_bound_metadata {
            helper.calculates_bounds_numeric_variables(task, precision, 9_999_999.0);
        }
        helper.calculate_epsilons(task.operators().len(), precision, default_epsilon);
        helper
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

    #[cfg(feature = "cplex")]
    pub(crate) fn numeric_variable_bounds(&self, local_var_id: usize) -> Option<(f64, f64)> {
        Some((
            *self.numeric_variable_lower_bounds.get(local_var_id)?,
            *self.numeric_variable_upper_bounds.get(local_var_id)?,
        ))
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
            .get(linear_effect.target_local_var_id.index())?;
        let mut delta = LinearExpression::zero(self.numeric_variable_index_by_task_id.len());
        for (local_var_id, &coefficient) in linear_effect.coefficients.iter().enumerate() {
            let task_var_id = *self.numeric_variable_ids.get(local_var_id)?;
            delta.coefficients[task_var_id.index()] = coefficient;
        }
        delta.coefficients[affected_var_id.index()] -= 1.0;
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

    pub(crate) fn get_epsilon(&self, condition_id: usize) -> Option<f64> {
        self.condition_epsilons.get(condition_id).copied()
    }

    pub(crate) fn local_numeric_var_id(
        &self,
        task_numeric_var_id: VariableIndex,
    ) -> Option<VariableIndex> {
        self.numeric_variable_index_by_task_id
            .get(task_numeric_var_id.index())
            .copied()
            .unwrap_or(None)
    }

    pub(crate) fn get_proposition(
        &self,
        var_id: VariableIndex,
        value: ExplicitValueIndex,
    ) -> Option<usize> {
        self.proposition_ids_by_var_value
            .get(var_id.index())
            .and_then(|values| values.get(value.index()))
            .copied()
    }

    pub(crate) fn get_proposition_name(&self, proposition_id: usize) -> Option<&str> {
        self.proposition_names
            .get(proposition_id)
            .map(String::as_str)
    }

    pub(crate) fn fact_to_axiom_marker(&self, variable_id: VariableIndex) -> Option<Option<usize>> {
        self.fact_to_axiom_marker.get(variable_id.index()).copied()
    }

    pub(crate) fn is_comparison_axiom_var(&self, variable_id: VariableIndex) -> bool {
        self.comparison_axiom_by_var.contains_key(&variable_id)
    }

    /// A fact on `variable_id`, tagged with the namespace the helper's own
    /// comparison-axiom map puts that variable in.
    pub(crate) fn fact(
        &self,
        variable_id: VariableIndex,
        value: ExplicitValueIndex,
    ) -> ExplicitFact {
        if self.is_comparison_axiom_var(variable_id) {
            ExplicitFact::condition(variable_id.index(), value.index())
        } else {
            ExplicitFact::propositional(variable_id.index(), value.index())
        }
    }
    /// A fact on `variable_id`, tagged with the namespace the helper's own
    /// comparison-axiom map puts that variable in.
    pub(crate) fn is_numeric_axiom_var(&self, variable_id: VariableIndex) -> bool {
        self.fact_to_axiom_marker(variable_id)
            .map(|marker| marker.is_some())
            .unwrap_or(false)
    }

    pub(crate) fn comparison_fact_conditions(
        &self,
        variable_id: VariableIndex,
        fact_value: ExplicitValueIndex,
    ) -> Option<&[LinearNumericCondition]> {
        self.comparison_fact_conditions
            .get(&(variable_id, fact_value))
            .map(Vec::as_slice)
    }

    pub(crate) fn comparison_fact_condition_group_ids(
        &self,
        variable_id: VariableIndex,
        fact_value: ExplicitValueIndex,
    ) -> Option<&[usize]> {
        self.comparison_fact_condition_group_ids
            .get(&(variable_id, fact_value))
            .map(Vec::as_slice)
    }

    pub(crate) fn goal_helper_numeric_condition_group_ids(
        &self,
        variable_id: VariableIndex,
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
        variable_id: VariableIndex,
        fact_value: ExplicitValueIndex,
    ) -> Vec<usize> {
        self.comparison_fact_condition_group_ids(variable_id, fact_value)
            .map(|group_ids| self.get_condition_ids_from_group_ids(group_ids))
            .unwrap_or_default()
    }

    #[cfg(feature = "cplex")]
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

    #[cfg(feature = "cplex")]
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

    #[cfg(feature = "cplex")]
    pub(crate) fn comparison_fact_materialized_conditions(
        &self,
        variable_id: VariableIndex,
        fact_value: ExplicitValueIndex,
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
    fact_value: ExplicitValueIndex,
) -> Option<ComparisonOperator> {
    assert!(
        fact_value.index() == 0 || fact_value.index() == 1,
        "comparison fact value must be boolean-like, got {fact_value}"
    );

    let operator = if fact_value.index() == 0 {
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
        .get_mut(linear_effect.affected_var_id.index())
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
    task: &Task,
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
        .get(linear_effect.affected_var_id.index())
        .map(|numeric_var| numeric_var.numeric_type() != NumericType::Regular)
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
    regular_numeric_variable_ids: &[VariableIndex],
    target_task_var_id: VariableIndex,
    precision: f64,
) -> bool {
    for &task_var_id in regular_numeric_variable_ids {
        if task_var_id != target_task_var_id
            && condition.coefficients[task_var_id.index()].abs() > precision
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

#[cfg(test)]
mod affine_slack_tests {
    use super::*;

    #[test]
    fn state_expression_is_rounded_before_the_relaxation_epsilon() {
        let condition = LinearNumericCondition::from_expression(
            LinearExpression {
                coefficients: vec![0.7, 0.7],
                constant: 0.0,
            },
            true,
            "small affine condition".into(),
        );
        let slack = condition.evaluate_slack(&[NumericValue::new(2e-12); 2], 1e-13);
        assert_eq!(slack, NumericValue::new(3e-12).value() - 1e-13);
    }
}
