use planforge_sas::numeric_task::{
    Effect, ExplicitFact, ExplicitValueIndex, ExplicitVariable, Metric, NumericType, NumericValue,
    NumericVariable, Operator, OperatorCost, Task, TaskParts, VariableIndex,
};

use super::*;
use planforge_search::task_transformations::projection::Pattern;

fn simple_var(name: &str) -> ExplicitVariable {
    ExplicitVariable::new(
        2,
        name.to_string(),
        vec![format!("{name}=0"), format!("{name}=1")],
        None,
        ExplicitValueIndex::new(1),
    )
}

fn sample_task() -> Task {
    Task::new(TaskParts {
        metric: Metric::new(true, None),
        variables: vec![simple_var("p"), simple_var("q")],
        numeric_variables: vec![NumericVariable::new(
            "x".to_string(),
            NumericType::Regular,
            None,
        )],
        goals: vec![ExplicitFact::propositional(1, 1)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0), ExplicitValueIndex::new(0)],
        numeric_state: vec![NumericValue::new(0.0)],
        operators: vec![Operator::new(
            "advance".to_string(),
            vec![ExplicitFact::propositional(0, 1)],
            vec![Effect::new(
                vec![],
                VariableIndex::new(1),
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
    })
}

#[test]
fn builds_one_pattern_database_per_pattern() {
    let task = sample_task();
    let patterns = PatternCollection::new(vec![
        Pattern::new(vec![VariableIndex::from_usize(1)], vec![]),
        Pattern::new(
            vec![VariableIndex::from_usize(0), VariableIndex::from_usize(1)],
            vec![],
        ),
    ]);

    let pdbs = build_pattern_databases(
        &task,
        &patterns,
        PdbBuildConfig {
            max_states: 32,
            heuristics: Default::default(),
            retain_transitions: false,
        },
    )
    .unwrap();

    assert_eq!(pdbs.len(), 2);
}
