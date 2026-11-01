//! Small tasks shared by unit tests.

use std::sync::Arc;

use planforge_sas::numeric_task::{
    Effect, ExplicitFact, ExplicitValueIndex, ExplicitVariable, Metric, Operator, OperatorCost,
    Task, TaskParts, VariableIndex,
};

/// One variable `location` in {start, goal}, starting at `start` with the
/// goal `goal`, and one unit-cost operator `finish` from `start` to `goal`.
pub fn one_step_task() -> Arc<Task> {
    Arc::new(Task::new(TaskParts {
        metric: Metric::new(false, None),
        variables: vec![ExplicitVariable::new(
            2,
            "location".to_string(),
            vec!["start".to_string(), "goal".to_string()],
            None,
            ExplicitValueIndex::new(0),
        )],
        numeric_variables: vec![],
        goals: vec![ExplicitFact::propositional(0, 1)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![],
        operators: vec![Operator::new(
            "finish".to_string(),
            vec![ExplicitFact::propositional(0, 0)],
            vec![Effect::new(
                vec![],
                VariableIndex::from_usize(0),
                Some(ExplicitValueIndex::new(0)),
                ExplicitValueIndex::new(1),
            )],
            vec![],
            OperatorCost::new(1),
        )],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: Some(ExplicitFact::propositional(0, 0)),
    }))
}
