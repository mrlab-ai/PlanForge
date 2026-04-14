use planners_sas::numeric::numeric_task::{
    Effect, ExplicitFact, ExplicitVariable, Metric, NumericRootTask, NumericType, NumericVariable,
    Operator,
};

use super::*;
use crate::numeric::evaluation::pattern_databases::projected_task::Pattern;

fn simple_var(name: &str) -> ExplicitVariable {
    ExplicitVariable::new(
        2,
        name.to_string(),
        vec![format!("{name}=0"), format!("{name}=1")],
        None,
        1,
    )
}

fn canonical_sample_task() -> NumericRootTask {
    NumericRootTask::new(
        1,
        Metric::new(true, None),
        vec![simple_var("p"), simple_var("q")],
        vec![NumericVariable::new(
            "x".to_string(),
            NumericType::Regular,
            None,
        )],
        vec![ExplicitFact::new(0, 1), ExplicitFact::new(1, 1)],
        vec![],
        vec![0, 0],
        vec![0.0],
        vec![
            Operator::new(
                "set-p".to_string(),
                vec![],
                vec![Effect::new(vec![], 0, Some(0), 1)],
                vec![],
                2,
            ),
            Operator::new(
                "set-q".to_string(),
                vec![],
                vec![Effect::new(vec![], 1, Some(0), 1)],
                vec![],
                3,
            ),
        ],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    )
}

#[test]
fn canonical_collection_information_uses_explicit_subsets() {
    let task = canonical_sample_task();
    let patterns = PatternCollection::new(vec![
        Pattern::new(vec![0], vec![]),
        Pattern::new(vec![1], vec![]),
    ]);
    let pdb_collection = PdbCollection::new(&task, patterns, 32).unwrap();
    let collection_information = CanonicalPdbCollectionInformation::with_explicit_subsets(
        pdb_collection,
        vec![vec![0, 1], vec![0], vec![1]],
    );
    let mut pdb_value_cache = Vec::new();

    let value = collection_information
        .evaluate_projected_state_values(&[0, 0], &[0.0], &mut pdb_value_cache)
        .unwrap();

    assert_eq!(value, 5.0);
}

#[test]
fn canonical_collection_computes_max_additive_subset() {
    let task = canonical_sample_task();
    let patterns = PatternCollection::new(vec![
        Pattern::new(vec![0], vec![]),
        Pattern::new(vec![1], vec![]),
    ]);

    let collection_information =
        CanonicalPdbCollectionInformation::new(&task, patterns, 32).unwrap();

    assert_eq!(collection_information.max_additive_subsets(), &[vec![0, 1]]);
}
