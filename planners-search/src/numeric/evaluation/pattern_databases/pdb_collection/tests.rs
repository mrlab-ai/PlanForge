use planners_sas::numeric::numeric_task::{
    Effect, ExplicitFact, ExplicitVariable, Metric, NumericRootTask, NumericType, NumericVariable,
    Operator,
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
    NumericRootTask::new(
        1,
        Metric::new(true, None),
        vec![simple_var("p"), simple_var("q")],
        vec![NumericVariable::new(
            "x".to_string(),
            NumericType::Regular,
            None,
        )],
        vec![ExplicitFact::new(1, 1)],
        vec![],
        vec![0, 0],
        vec![0.0],
        vec![Operator::new(
            "advance".to_string(),
            vec![ExplicitFact::new(0, 1)],
            vec![Effect::new(vec![], 1, Some(0), 1)],
            vec![],
            1,
        )],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    )
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
