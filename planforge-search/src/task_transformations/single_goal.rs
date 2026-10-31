//! A view of a task with a single goal fact: the members of a goal-focused
//! abstraction collection each refine for one goal of the task.

use std::sync::Arc;

use planforge_sas::axioms::{AssignmentAxiom, ComparisonAxiom, PropositionalAxiom};
use planforge_sas::numeric_conditions::NumericConditions;
use planforge_sas::numeric_task::{
    AbstractNumericTask, ExplicitFact, ExplicitValueIndex, ExplicitVariable, Metric, NumericValue,
    NumericVariable, Operator, OperatorIndex,
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
    fn variables(&self) -> &[ExplicitVariable] {
        self.base.variables()
    }

    fn numeric_variables(&self) -> &[NumericVariable] {
        self.base.numeric_variables()
    }

    fn assignment_axioms(&self) -> &[AssignmentAxiom] {
        self.base.assignment_axioms()
    }

    fn comparison_axioms(&self) -> &[ComparisonAxiom] {
        self.base.comparison_axioms()
    }

    fn numeric_conditions(&self) -> &Arc<NumericConditions> {
        self.base.numeric_conditions()
    }

    fn axioms(&self) -> &[PropositionalAxiom] {
        self.base.axioms()
    }

    fn metric(&self) -> &Metric {
        self.base.metric()
    }

    fn fact_name(&self, fact: &ExplicitFact) -> &str {
        self.base.fact_name(fact)
    }

    fn are_facts_mutex(&self, fact1: &ExplicitFact, fact2: &ExplicitFact) -> bool {
        self.base.are_facts_mutex(fact1, fact2)
    }

    fn operators(&self) -> &[Operator] {
        self.base.operators()
    }

    fn goals(&self) -> &[ExplicitFact] {
        std::slice::from_ref(&self.goal)
    }

    fn initial_propositional_values(&self) -> &[ExplicitValueIndex] {
        self.base.initial_propositional_values()
    }

    fn initial_numeric_values(&self) -> &[NumericValue] {
        self.base.initial_numeric_values()
    }

    fn evaluate_initial_state_values(
        &self,
    ) -> Result<(Vec<ExplicitValueIndex>, Vec<NumericValue>), String> {
        self.base.evaluate_initial_state_values()
    }

    fn effective_operator_costs(&self) -> std::borrow::Cow<'_, [f64]> {
        self.base.effective_operator_costs()
    }

    fn operator_cost(&self, operator_id: OperatorIndex) -> f64 {
        self.base.operator_cost(operator_id)
    }
}
