use planforge_sas::numeric_task::{
    Effect, ExplicitFact, ExplicitValueIndex, ExplicitVariable, Metric, NumericRootTask,
    NumericRootTaskParts, NumericType, NumericVariable, Operator, OperatorCost, VariableIndex,
};

use super::*;
use crate::evaluation::pattern_databases::projected_task::Pattern;

fn simple_var(name: &str) -> ExplicitVariable {
    ExplicitVariable::new(
        2,
        name.to_string(),
        vec![format!("{name}=0"), format!("{name}=1")],
        None,
        ExplicitValueIndex::new(1),
    )
}

fn canonical_sample_task() -> NumericRootTask {
    NumericRootTask::new(NumericRootTaskParts {
        version: 1,
        metric: Metric::new(true, None),
        variables: vec![simple_var("p"), simple_var("q")],
        numeric_variables: vec![NumericVariable::new(
            "x".to_string(),
            NumericType::Regular,
            None,
        )],
        goals: vec![
            ExplicitFact::propositional(0, 1),
            ExplicitFact::propositional(1, 1),
        ],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0), ExplicitValueIndex::new(0)],
        numeric_state: vec![NumericValue::new(0.0)],
        operators: vec![
            Operator::new(
                "set-p".to_string(),
                vec![],
                vec![Effect::new(
                    vec![],
                    VariableIndex::new(0),
                    Some(ExplicitValueIndex::new(0)),
                    ExplicitValueIndex::new(1),
                )],
                vec![],
                OperatorCost::new(2),
            ),
            Operator::new(
                "set-q".to_string(),
                vec![],
                vec![Effect::new(
                    vec![],
                    VariableIndex::new(1),
                    Some(ExplicitValueIndex::new(0)),
                    ExplicitValueIndex::new(1),
                )],
                vec![],
                OperatorCost::new(3),
            ),
        ],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    })
}

#[test]
fn canonical_collection_information_uses_explicit_subsets() {
    let task = canonical_sample_task();
    let patterns = PatternCollection::new(vec![
        Pattern::new(vec![VariableIndex::from_usize(0)], vec![]),
        Pattern::new(vec![VariableIndex::from_usize(1)], vec![]),
    ]);
    let pdb_collection = PdbCollection::new(&task, patterns, 32).unwrap();
    let collection_information = CanonicalPdbCollectionInformation::with_explicit_subsets(
        pdb_collection,
        vec![vec![0, 1], vec![0], vec![1]],
    );
    let mut pdb_value_cache = PdbValueCache::default();

    let value = collection_information.evaluate_projected_state_values(
        &[ExplicitValueIndex::new(0), ExplicitValueIndex::new(0)],
        &[NumericValue::new(0.0)],
        &mut pdb_value_cache,
    );

    assert_eq!(value, 5.0);
}

#[test]
fn canonical_collection_computes_max_additive_subset() {
    let task = canonical_sample_task();
    let patterns = PatternCollection::new(vec![
        Pattern::new(vec![VariableIndex::from_usize(0)], vec![]),
        Pattern::new(vec![VariableIndex::from_usize(1)], vec![]),
    ]);

    let collection_information =
        CanonicalPdbCollectionInformation::new(&task, patterns, 32, PdbHeuristicConfig::default())
            .unwrap();

    assert_eq!(collection_information.max_additive_subsets(), &[vec![0, 1]]);
}
