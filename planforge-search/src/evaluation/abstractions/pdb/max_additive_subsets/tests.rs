use planforge_sas::numeric_task::{
    AssignmentEffect, Effect, ExplicitFact, ExplicitValueIndex, ExplicitVariable, Metric,
    NumericRootTask, NumericRootTaskParts, NumericType, NumericValue, NumericVariable, Operator,
    OperatorCost, VariableIndex,
};

use super::*;

fn simple_var(name: &str) -> ExplicitVariable {
    ExplicitVariable::new(
        2,
        name.to_string(),
        vec![format!("{name}=0"), format!("{name}=1")],
        None,
        ExplicitValueIndex::new(1),
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
                OperatorCost::new(1),
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
                OperatorCost::new(1),
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
        state: vec![ExplicitValueIndex::new(0), ExplicitValueIndex::new(0)],
        numeric_state: vec![
            NumericValue::new(1.0),
            NumericValue::new(0.0),
            NumericValue::new(0.0),
        ],
        operators: vec![Operator::new(
            "touch-both".to_string(),
            vec![],
            vec![Effect::new(
                vec![],
                VariableIndex::new(0),
                Some(ExplicitValueIndex::new(0)),
                ExplicitValueIndex::new(1),
            )],
            vec![AssignmentEffect::new(
                VariableIndex::from_usize(1),
                AssignmentOperation::Plus,
                VariableIndex::from_usize(0),
                false,
                vec![],
            )],
            OperatorCost::new(1),
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
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![NumericValue::new(0.0), NumericValue::new(0.0)],
        operators: vec![Operator::new(
            "set-p-and-add-zero".to_string(),
            vec![],
            vec![Effect::new(
                vec![],
                VariableIndex::new(0),
                Some(ExplicitValueIndex::new(0)),
                ExplicitValueIndex::new(1),
            )],
            vec![AssignmentEffect::new(
                VariableIndex::from_usize(1),
                AssignmentOperation::Plus,
                VariableIndex::from_usize(0),
                false,
                vec![],
            )],
            OperatorCost::new(1),
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
        Pattern::new(vec![VariableIndex::from_usize(0)], vec![]),
        Pattern::new(vec![VariableIndex::from_usize(1)], vec![]),
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
