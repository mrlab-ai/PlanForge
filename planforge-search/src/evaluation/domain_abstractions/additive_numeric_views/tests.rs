use planforge_sas::axioms::{AssignmentAxiom, CalOperator};
use planforge_sas::numeric_task::{
    AssignmentEffect, AssignmentOperation, ExplicitFact, ExplicitValueIndex, ExplicitVariable,
    Metric, NumericRootTask, NumericRootTaskParts, NumericType, NumericVariable, ONE_VALUE,
    Operator, OperatorCost,
};

use super::*;

fn global_constraint_variable() -> ExplicitVariable {
    ExplicitVariable::new(
        1,
        "global-constraint".into(),
        vec!["true".into()],
        None,
        ExplicitValueIndex::new(0),
    )
}

fn affine_sum_task(operation: AssignmentOperation) -> NumericRootTask {
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("y".into(), NumericType::Regular, None),
        NumericVariable::new("x_plus_y".into(), NumericType::Derived, None),
        NumericVariable::new("one".into(), NumericType::Constant, None),
    ];
    let operator = Operator::new(
        "change-x".into(),
        vec![],
        vec![],
        vec![AssignmentEffect::new(
            VariableIndex::from_usize(0),
            operation,
            VariableIndex::from_usize(3),
            false,
            vec![],
        )],
        OperatorCost::new(1),
    );
    NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, None),
        variables: vec![global_constraint_variable()],
        numeric_variables,
        goals: vec![],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![
            NumericValue::new(0.0),
            NumericValue::new(0.0),
            NumericValue::new(0.0),
            NumericValue::new(1.0),
        ],
        operators: vec![operator],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![AssignmentAxiom::new(
            VariableIndex::from_usize(2),
            CalOperator::Sum,
            VariableIndex::from_usize(0),
            VariableIndex::from_usize(1),
        )],
        global_constraint: ExplicitFact::propositional(0, 0),
    })
}

#[test]
fn affine_sum_is_an_exact_additive_coordinate() {
    let task = affine_sum_task(AssignmentOperation::Plus);
    let view = analyze_additive_numeric_view(&task, VariableIndex::from_usize(2))
        .expect("x+y should be additive");

    assert_eq!(view.expression.coefficients, vec![1.0, 1.0, 0.0, 0.0]);
    assert_eq!(
        view.operator_delta(OperatorIndex::new(0)).unwrap(),
        ONE_VALUE
    );
}

#[test]
fn non_additive_effect_rejects_affine_coordinate() {
    let task = affine_sum_task(AssignmentOperation::Assign);

    assert!(analyze_additive_numeric_view(&task, VariableIndex::from_usize(2)).is_none());
    assert!(AdditiveNumericViews::for_active_dimensions(&task, &[1, 1, 2, 1]).is_err());
}

#[test]
fn cost_dependent_expression_is_not_an_additive_view() {
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("accumulated-cost".into(), NumericType::Cost, None),
        NumericVariable::new("x-plus-cost".into(), NumericType::Derived, None),
    ];
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, Some(VariableIndex::from_usize(1))),
        variables: vec![global_constraint_variable()],
        numeric_variables,
        goals: vec![],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![
            NumericValue::new(0.0),
            NumericValue::new(0.0),
            NumericValue::new(0.0),
        ],
        operators: vec![],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![AssignmentAxiom::new(
            VariableIndex::from_usize(2),
            CalOperator::Sum,
            VariableIndex::from_usize(0),
            VariableIndex::from_usize(1),
        )],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    assert!(analyze_additive_numeric_view(&task, VariableIndex::from_usize(2)).is_none());
}
