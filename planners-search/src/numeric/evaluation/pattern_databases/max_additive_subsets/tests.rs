use planners_sas::numeric::numeric_task::{
    AssignmentEffect, Effect, ExplicitFact, ExplicitVariable, Metric, NumericRootTask, NumericType,
    NumericVariable, Operator,
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
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    )
}

fn shared_effect_task() -> NumericRootTask {
    NumericRootTask::new(
        1,
        Metric::new(true, None),
        vec![simple_var("p"), simple_var("q")],
        vec![
            NumericVariable::new("c".to_string(), NumericType::Constant, None),
            NumericVariable::new("x".to_string(), NumericType::Regular, None),
            NumericVariable::new("y".to_string(), NumericType::Regular, None),
        ],
        vec![],
        vec![],
        vec![0, 0],
        vec![1.0, 0.0, 0.0],
        vec![Operator::new(
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
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    )
}

fn zero_additive_effect_task() -> NumericRootTask {
    NumericRootTask::new(
        1,
        Metric::new(true, None),
        vec![simple_var("p")],
        vec![
            NumericVariable::new("zero".to_string(), NumericType::Constant, None),
            NumericVariable::new("x".to_string(), NumericType::Regular, None),
        ],
        vec![],
        vec![],
        vec![0],
        vec![0.0, 0.0],
        vec![Operator::new(
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
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    )
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
