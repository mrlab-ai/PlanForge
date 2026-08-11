use planforge_sas::numeric_task::{
    Effect, ExplicitFact, ExplicitVariable, Metric, NumericRootTask, NumericRootTaskParts,
    NumericType, NumericVariable, Operator,
};

use super::*;
use crate::evaluation::pattern_databases::projected_task::Pattern;

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
        goals: vec![
            ExplicitFact::propositional(0, 1),
            ExplicitFact::propositional(1, 1),
        ],
        mutexes: vec![],
        state: vec![0, 0],
        numeric_state: vec![0.0],
        operators: vec![
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
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    })
}

#[test]
fn collection_information_builds_pdbs_and_additive_subsets() {
    let task = sample_task();
    let info = PatternCollectionInformation::new(
        &task,
        PatternCollection::new(vec![
            Pattern::new(vec![0], vec![]),
            Pattern::new(vec![1], vec![]),
        ]),
        32,
    );

    assert_eq!(info.get_pdbs().unwrap().len(), 2);
    assert_eq!(info.get_max_additive_subsets().unwrap(), &[vec![0, 1]]);
}
