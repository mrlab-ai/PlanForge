use planforge_sas::numeric_task::{
    Effect, ExplicitFact, ExplicitVariable, Metric, NumericRootTask, NumericRootTaskParts,
    NumericType, NumericVariable, Operator,
};

use super::*;

fn simple_var(name: &str) -> ExplicitVariable {
    ExplicitVariable::new(
        2,
        name.to_string(),
        vec![format!("{name}=0"), format!("{name}=1")],
        None,
        1,
    )
}

fn sample_task() -> NumericRootTask {
    NumericRootTask::new(NumericRootTaskParts {
        version: 1,
        metric: Metric::new(true, None),
        variables: vec![simple_var("p"), simple_var("q")],
        numeric_variables: vec![NumericVariable::new(
            "x".to_string(),
            NumericType::Regular,
            None,
        )],
        goals: vec![ExplicitFact::propositional(1, 1)],
        mutexes: vec![],
        state: vec![0, 0],
        numeric_state: vec![0.0],
        operators: vec![Operator::new(
            "advance".to_string(),
            vec![ExplicitFact::propositional(0, 1)],
            vec![Effect::new(vec![], 1, Some(0), 1)],
            vec![],
            1,
        )],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    })
}

#[test]
fn pdb_collection_builds_all_patterns() {
    let task = sample_task();
    let patterns = PatternCollection::new(vec![
        Pattern::new(vec![1], vec![]),
        Pattern::new(vec![0, 1], vec![]),
    ]);

    let collection = PdbCollection::new(&task, patterns, 32).unwrap();

    assert_eq!(collection.len(), 2);
    assert_eq!(
        collection.singleton_additive_subsets(),
        vec![vec![0], vec![1]]
    );
}
