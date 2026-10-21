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
    affine_sum_task_with_step(operation, 1.0)
}

/// `x`, `y` and the derived `x + y`; one operator changes `x` by `step`.
fn affine_sum_task_with_step(operation: AssignmentOperation, step: f64) -> NumericRootTask {
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
            NumericValue::new(step),
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
    assert_eq!(view.operator_delta(OperatorIndex::new(0)), ONE_VALUE);
}

#[test]
fn non_additive_effect_rejects_affine_coordinate() {
    let task = affine_sum_task(AssignmentOperation::Assign);

    assert!(analyze_additive_numeric_view(&task, VariableIndex::from_usize(2)).is_none());
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

#[test]
fn a_coordinate_preimage_lies_on_the_canonical_lattice() {
    // -900 - 91.2 is -991.2 in f64, which is not canonical; the canonical
    // source -991.1999999999999 reaches exactly -900, outside the open target.
    let task = affine_sum_task_with_step(AssignmentOperation::Plus, 91.2);
    let coordinates = Coordinates::new(&task);
    let target = Interval::new(NumericValue::new(-900.0), INF_VALUE, false, false);

    let preimage = coordinates
        .preimage(target, OperatorIndex::new(0), VariableIndex::from_usize(2))
        .unwrap()
        .expect("an additive coordinate has a preimage");

    assert_eq!(preimage, preimage.canonicalized());
    assert!(!preimage.contains(float_tolerance::canonicalize(NumericValue::new(-991.2))));
}
