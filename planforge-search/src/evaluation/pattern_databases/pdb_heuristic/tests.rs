use planforge_sas::axioms::{AssignmentAxiom, CalOperator};
use planforge_sas::numeric_task::{
    Effect, ExplicitFact, ExplicitValueIndex, ExplicitVariable, Metric, NumericRootTask,
    NumericRootTaskParts, NumericType, NumericValue, NumericVariable, Operator, OperatorCost,
    VariableIndex,
};
use planforge_sas::state_registry::StateRegistry;

use crate::evaluation::evaluator::EvaluationState;
use crate::evaluation::heuristic::Heuristic;
use crate::evaluation::pattern_databases::pattern_generator_greedy::GreedyPatternGeneratorConfig;

use super::GreedyNumericPdbHeuristic;

fn initial_goal_task() -> NumericRootTask {
    NumericRootTask::new(NumericRootTaskParts {
        version: 1,
        metric: Metric::new(true, None),
        variables: vec![ExplicitVariable::new(
            2,
            "p".to_string(),
            vec!["p=0".to_string(), "p=1".to_string()],
            None,
            ExplicitValueIndex::new(0),
        )],
        numeric_variables: vec![NumericVariable::new(
            "x".to_string(),
            NumericType::Regular,
            None,
        )],
        goals: vec![ExplicitFact::propositional(0, 1)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(1)],
        numeric_state: vec![NumericValue::new(0.0)],
        operators: vec![Operator::new(
            "leave-goal".to_string(),
            vec![ExplicitFact::propositional(0, 1)],
            vec![Effect::new(
                vec![],
                VariableIndex::new(0),
                Some(ExplicitValueIndex::new(1)),
                ExplicitValueIndex::new(0),
            )],
            vec![],
            OperatorCost::new(1),
        )],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![AssignmentAxiom::new(
            VariableIndex::from_usize(0),
            CalOperator::Sum,
            VariableIndex::from_usize(0),
            VariableIndex::from_usize(0),
        )],
        global_constraint: ExplicitFact::propositional(0, 0),
    })
}

#[test]
fn greedy_numeric_pdb_returns_zero_for_concrete_goal_state() {
    let task = initial_goal_task();
    let mut state_registry = StateRegistry::for_task(std::sync::Arc::new(&task));
    let initial_state = state_registry.get_initial_state();
    let heuristic = GreedyNumericPdbHeuristic::new(
        &task,
        GreedyPatternGeneratorConfig {
            max_pdb_states: 16,
            ..GreedyPatternGeneratorConfig::default()
        },
    )
    .expect("greedy numeric PDB should build for simple goal task");

    let mut eval_state = EvaluationState::new(&initial_state, &task, &state_registry);
    eval_state.set_is_goal(true);
    let value = heuristic
        .compute_heuristic(&eval_state)
        .expect("goal evaluation should succeed");

    assert_eq!(value, 0.0);
}
