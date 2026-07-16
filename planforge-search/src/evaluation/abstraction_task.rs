use std::cell::{Ref, RefMut};

use anyhow::{Context, Result, ensure};
use planforge_sas::axioms::{AssignmentAxiom, ComparisonAxiom, PropositionalAxiom};
use planforge_sas::numeric_task::{
    AbstractNumericTask, AssignmentOperation, ExplicitFact, ExplicitVariable, Metric, NumericType,
    NumericVariable, Operator,
};

/// A task view that changes only the goal condition.
///
/// Operator IDs, state representations, costs, and ancestor conversions remain
/// identical to the base task, so abstractions built from this view can be
/// combined directly for the base task.
pub(crate) struct SingleGoalTask<'task> {
    base: &'task dyn AbstractNumericTask,
    goal: ExplicitFact,
}

impl<'task> SingleGoalTask<'task> {
    pub(crate) fn new(base: &'task dyn AbstractNumericTask, goal: ExplicitFact) -> Self {
        Self { base, goal }
    }
}

impl AbstractNumericTask for SingleGoalTask<'_> {
    fn variables(&self) -> &Vec<ExplicitVariable> {
        self.base.variables()
    }

    fn numeric_variables(&self) -> &Vec<NumericVariable> {
        self.base.numeric_variables()
    }

    fn assignment_axioms(&self) -> &Vec<AssignmentAxiom> {
        self.base.assignment_axioms()
    }

    fn comparison_axioms(&self) -> &Vec<ComparisonAxiom> {
        self.base.comparison_axioms()
    }

    fn axioms(&self) -> &Vec<PropositionalAxiom> {
        self.base.axioms()
    }

    fn metric(&self) -> &Metric {
        self.base.metric()
    }

    fn get_num_variables(&self) -> usize {
        self.base.get_num_variables()
    }

    fn get_variable_name(&self, index: usize) -> Result<&str, &str> {
        self.base.get_variable_name(index)
    }

    fn get_variable_domain_size(&self, index: usize) -> Result<usize, &str> {
        self.base.get_variable_domain_size(index)
    }

    fn get_variable_axiom_layer(&self, index: usize) -> Result<Option<usize>, &str> {
        self.base.get_variable_axiom_layer(index)
    }

    fn get_variable_default_axiom_value(&self, index: usize) -> Result<usize, &str> {
        self.base.get_variable_default_axiom_value(index)
    }

    fn get_fact_name(&self, fact: &ExplicitFact) -> &str {
        self.base.get_fact_name(fact)
    }

    fn are_facts_mutex(&self, fact1: &ExplicitFact, fact2: &ExplicitFact) -> bool {
        self.base.are_facts_mutex(fact1, fact2)
    }

    fn get_operators(&self) -> &Vec<Operator> {
        self.base.get_operators()
    }

    fn get_operator_cost(&self, index: usize, is_axiom: bool) -> u64 {
        self.base.get_operator_cost(index, is_axiom)
    }

    fn get_operator_name(&self, index: usize, is_axiom: bool) -> &str {
        self.base.get_operator_name(index, is_axiom)
    }

    fn get_num_operators(&self) -> usize {
        self.base.get_num_operators()
    }

    fn get_num_operator_preconditions(&self, index: usize, is_axiom: bool) -> usize {
        self.base.get_num_operator_preconditions(index, is_axiom)
    }

    fn get_operator_precondition(
        &self,
        index: usize,
        precond_index: usize,
        is_axiom: bool,
    ) -> &ExplicitFact {
        self.base
            .get_operator_precondition(index, precond_index, is_axiom)
    }

    fn get_num_operator_effects(&self, index: usize, is_axiom: bool) -> usize {
        self.base.get_num_operator_effects(index, is_axiom)
    }

    fn get_num_operator_effect_conditions(
        &self,
        index: usize,
        eff_index: usize,
        is_axiom: bool,
    ) -> usize {
        self.base
            .get_num_operator_effect_conditions(index, eff_index, is_axiom)
    }

    fn get_operator_effect_condition(
        &self,
        index: usize,
        eff_index: usize,
        cond_index: usize,
        is_axiom: bool,
    ) -> &ExplicitFact {
        self.base
            .get_operator_effect_condition(index, eff_index, cond_index, is_axiom)
    }

    fn get_operator_effect(&self, index: usize, eff_index: usize, is_axiom: bool) -> &ExplicitFact {
        self.base.get_operator_effect(index, eff_index, is_axiom)
    }

    fn convert_operator_index(&self, index: usize, ancestor_task: &dyn AbstractNumericTask) {
        self.base.convert_operator_index(index, ancestor_task)
    }

    fn get_num_axioms(&self) -> usize {
        self.base.get_num_axioms()
    }

    fn get_num_goals(&self) -> usize {
        1
    }

    fn get_goal_fact(&self, index: usize) -> &ExplicitFact {
        assert_eq!(index, 0, "SingleGoalTask only exposes one goal");
        &self.goal
    }

    fn get_initial_propositional_state_values(&self) -> Ref<'_, Vec<usize>> {
        self.base.get_initial_propositional_state_values()
    }

    fn get_initial_numeric_state_values(&self) -> Ref<'_, Vec<f64>> {
        self.base.get_initial_numeric_state_values()
    }

    fn get_initial_propositional_state_values_mut(&self) -> RefMut<'_, Vec<usize>> {
        self.base.get_initial_propositional_state_values_mut()
    }

    fn get_initial_numeric_state_values_mut(&self) -> RefMut<'_, Vec<f64>> {
        self.base.get_initial_numeric_state_values_mut()
    }

    fn set_initial_numeric_state_values(&self, values: Vec<f64>) {
        self.base.set_initial_numeric_state_values(values)
    }

    fn set_initial_propositional_state_values(&self, values: Vec<usize>) {
        self.base.set_initial_propositional_state_values(values)
    }

    fn convert_ancestor_state_values(
        &self,
        ancestor_state_values: &[usize],
        ancestor_task: &dyn AbstractNumericTask,
    ) -> Vec<usize> {
        self.base
            .convert_ancestor_state_values(ancestor_state_values, ancestor_task)
    }

    fn get_num_cmp_axioms(&self) -> usize {
        self.base.get_num_cmp_axioms()
    }

    fn abstract_state_values(
        &self,
        propositional_values: &[usize],
        numeric_values: &[f64],
    ) -> Result<(Vec<usize>, Vec<f64>), String> {
        self.base
            .abstract_state_values(propositional_values, numeric_values)
    }

    fn evaluated_initial_abstract_state_values(&self) -> Result<(Vec<usize>, Vec<f64>), String> {
        self.base.evaluated_initial_abstract_state_values()
    }

    fn abstract_operator_cost(&self, operator_id: usize) -> f64 {
        self.base.abstract_operator_cost(operator_id)
    }
}

/// Validates the concrete-operator fragment shared by domain and Cartesian
/// abstractions. Unsupported task input is rejected before either backend
/// constructs transitions.
pub(crate) fn validate_abstraction_operator(
    task: &dyn AbstractNumericTask,
    operator: &Operator,
    operator_id: usize,
) -> Result<()> {
    let mut propositional_effect_by_var = vec![None; task.get_num_variables()];
    for (effect_id, effect) in operator.effects().iter().enumerate() {
        ensure!(
            effect.var_id() < task.get_num_variables(),
            "operator {operator_id} ({}) propositional effect {effect_id} targets missing variable {}",
            operator.name(),
            effect.var_id()
        );
        ensure!(
            effect.conditions().is_empty(),
            "numeric-fd parity: conditional propositional or numeric effects are unsupported in abstraction generation"
        );
        ensure!(
            propositional_effect_by_var[effect.var_id()]
                .replace(effect_id)
                .is_none(),
            "operator {operator_id} ({}) has multiple propositional effects on variable {}",
            operator.name(),
            effect.var_id()
        );
    }

    let numeric_variables = task.numeric_variables();
    let initial_numeric = task.get_initial_numeric_state_values();
    let mut numeric_effect_by_var = vec![None; numeric_variables.len()];
    for (effect_id, effect) in operator.assignment_effects().iter().enumerate() {
        ensure!(
            !effect.is_conditional() && effect.conditions().is_empty(),
            "numeric-fd parity: conditional propositional or numeric effects are unsupported in abstraction generation"
        );
        ensure!(
            effect.affected_var_id() < numeric_variables.len(),
            "operator {operator_id} ({}) numeric effect {effect_id} targets missing variable {}",
            operator.name(),
            effect.affected_var_id()
        );
        ensure!(
            numeric_effect_by_var[effect.affected_var_id()]
                .replace(effect_id)
                .is_none(),
            "operator {operator_id} ({}) has multiple numeric effects on variable {}",
            operator.name(),
            effect.affected_var_id()
        );
        let affected_type = numeric_variables[effect.affected_var_id()].get_type();
        ensure!(
            matches!(affected_type, NumericType::Regular | NumericType::Cost),
            "operator {operator_id} ({}) numeric effect {effect_id} targets {:?} variable {}",
            operator.name(),
            affected_type,
            effect.affected_var_id()
        );
        let rhs_var_id = effect.var_id();
        let rhs_variable = numeric_variables.get(rhs_var_id).with_context(|| {
            format!(
                "operator {operator_id} ({}) numeric effect {effect_id} reads missing RHS variable {rhs_var_id}",
                operator.name()
            )
        })?;
        ensure!(
            rhs_variable.get_type() == &NumericType::Constant,
            "numeric-fd parity: assignment effects require constant RHS, got {:?} for numeric var {}",
            rhs_variable.get_type(),
            rhs_var_id
        );
        let rhs = *initial_numeric.get(rhs_var_id).with_context(|| {
            format!("missing initial value for constant numeric variable {rhs_var_id}")
        })?;
        ensure!(
            rhs.is_finite(),
            "operator {operator_id} ({}) numeric effect {effect_id} has non-finite constant RHS {rhs}",
            operator.name()
        );
        ensure!(
            !matches!(effect.operation(), AssignmentOperation::Divide) || rhs != 0.0,
            "operator {operator_id} ({}) numeric effect {effect_id} divides by zero",
            operator.name()
        );
    }
    Ok(())
}
