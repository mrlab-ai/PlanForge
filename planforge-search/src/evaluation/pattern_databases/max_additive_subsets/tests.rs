use planforge_sas::numeric_task::{
    AssignmentEffect, Effect, ExplicitFact, ExplicitVariable, Metric, NumericRootTask,
    NumericRootTaskParts, NumericType, NumericVariable, Operator,
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

fn disjoint_effect_task() -> NumericRootTask {
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
                1,
            ),
            Operator::new(
                "set-q".to_string(),
                vec![],
                vec![Effect::new(vec![], 1, Some(0), 1)],
                vec![],
                1,
            ),
        ],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    })
}

fn shared_effect_task() -> NumericRootTask {
    NumericRootTask::new(NumericRootTaskParts {
        version: 1,
        metric: Metric::new(true, None),
        variables: vec![simple_var("p"), simple_var("q")],
        numeric_variables: vec![
            NumericVariable::new("c".to_string(), NumericType::Constant, None),
            NumericVariable::new("x".to_string(), NumericType::Regular, None),
            NumericVariable::new("y".to_string(), NumericType::Regular, None),
        ],
        goals: vec![],
        mutexes: vec![],
        state: vec![0, 0],
        numeric_state: vec![1.0, 0.0, 0.0],
        operators: vec![Operator::new(
            "touch-both".to_string(),
            vec![],
            vec![Effect::new(vec![], 0, Some(0), 1)],
            vec![AssignmentEffect::new(
                1,
                AssignmentOperation::Plus,
                0,
                false,
                vec![],
            )],
            1,
        )],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    })
}

fn zero_additive_effect_task() -> NumericRootTask {
    NumericRootTask::new(NumericRootTaskParts {
        version: 1,
        metric: Metric::new(true, None),
        variables: vec![simple_var("p")],
        numeric_variables: vec![
            NumericVariable::new("zero".to_string(), NumericType::Constant, None),
            NumericVariable::new("x".to_string(), NumericType::Regular, None),
        ],
        goals: vec![],
        mutexes: vec![],
        state: vec![0],
        numeric_state: vec![0.0, 0.0],
        operators: vec![Operator::new(
            "set-p-and-add-zero".to_string(),
            vec![],
            vec![Effect::new(vec![], 0, Some(0), 1)],
            vec![AssignmentEffect::new(
                1,
                AssignmentOperation::Plus,
                0,
                false,
                vec![],
            )],
            1,
        )],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    })
}

#[test]
fn computes_additive_patterns_for_disjoint_effects() {
    let task = disjoint_effect_task();
    let patterns = PatternCollection::new(vec![
        Pattern::new(vec![0], vec![]),
        Pattern::new(vec![1], vec![]),
    ]);

    let additivity = compute_additive_vars(&task);
    let subsets = compute_max_additive_subsets(&patterns, &additivity);

    assert_eq!(subsets, vec![vec![0, 1]]);
}

#[test]
fn marks_prop_and_numeric_as_non_additive_when_same_operator_touches_both() {
    let task = shared_effect_task();
    let additivity = compute_additive_vars(&task);

    assert!(!additivity.prop_to_num[0][1]);
    assert!(!additivity.num_to_prop[1][0]);
}

#[test]
fn zero_constant_additive_effect_does_not_break_additivity_like_fd() {
    let task = zero_additive_effect_task();
    let additivity = compute_additive_vars(&task);

    assert!(additivity.prop_to_num[0][1]);
    assert!(additivity.num_to_prop[1][0]);
}
