use planforge_sas::axioms::{AssignmentAxiom, CalOperator, ComparisonAxiom, ComparisonOperator};
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
    derived_task(
        operation,
        step,
        CalOperator::Sum,
        VariableIndex::from_usize(1),
    )
}

/// `x`, `y`, the constant `one` and the derived `x <definition> right`; one
/// operator changes `x` by `step`.
fn derived_task(
    operation: AssignmentOperation,
    step: f64,
    definition: CalOperator,
    right: VariableIndex,
) -> NumericRootTask {
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("y".into(), NumericType::Regular, None),
        NumericVariable::new("x_plus_y".into(), NumericType::Derived, Some(0)),
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
        variables: vec![
            global_constraint_variable(),
            ExplicitVariable::new(
                2,
                "comparison".into(),
                vec!["true".into(), "false".into()],
                Some(1),
                ExplicitValueIndex::new(1),
            ),
        ],
        numeric_variables,
        goals: vec![],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0), ExplicitValueIndex::new(1)],
        numeric_state: vec![
            NumericValue::new(0.0),
            NumericValue::new(0.0),
            NumericValue::new(0.0),
            NumericValue::new(step),
        ],
        operators: vec![operator],
        axioms: vec![],
        comparison_axioms: vec![ComparisonAxiom::new(
            VariableIndex::new(1),
            VariableIndex::new(2),
            VariableIndex::new(3),
            ComparisonOperator::GreaterThan,
        )],
        assignment_axioms: vec![AssignmentAxiom::new(
            VariableIndex::from_usize(2),
            definition,
            VariableIndex::from_usize(0),
            right,
        )],
        global_constraint: ExplicitFact::propositional(0, 0),
    })
}

#[test]
fn affine_sum_is_an_exact_additive_coordinate() {
    let task = affine_sum_task(AssignmentOperation::Plus);
    let coordinates = Coordinates::new(&task);
    let view = coordinates.views()[task.comparison_axioms()[0].get_left_var_id().index()]
        .as_ref()
        .expect("x+y should be additive");

    assert_eq!(
        view.expression,
        NumericExpression::Linear(vec![
            (VariableIndex::new(0), ONE_VALUE),
            (VariableIndex::new(1), ONE_VALUE)
        ])
    );
    assert_eq!(view.operator_delta(OperatorIndex::new(0)), ONE_VALUE);
}

#[test]
fn non_additive_effect_rejects_affine_coordinate() {
    let task = affine_sum_task(AssignmentOperation::Assign);

    assert!(
        Coordinates::new(&task).views()[task.comparison_axioms()[0].get_left_var_id().index()]
            .is_none()
    );
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

    assert!(Coordinates::new(&task).views()[2].is_none());
}

#[test]
fn a_coordinate_preimage_lies_on_the_canonical_lattice() {
    // -900 - 91.2 is -991.2 in f64, which is not canonical; the canonical
    // source -991.1999999999999 reaches exactly -900, outside the open target.
    let task = affine_sum_task_with_step(AssignmentOperation::Plus, 91.2);
    let coordinates = Coordinates::new(&task);
    let target = Interval::new(NumericValue::new(-900.0), INF_VALUE, false, false);

    let preimage = coordinates
        .preimage(
            target,
            OperatorIndex::new(0),
            task.comparison_axioms()[0].get_left_var_id(),
        )
        .expect("an additive coordinate has a preimage");

    assert!(!preimage.contains(NumericValue::new(-991.2)));
}

#[test]
fn a_one_variable_expression_is_refined_along_its_variable() {
    // `x - one` is an image of `x` alone.
    let task = derived_task(
        AssignmentOperation::Plus,
        1.0,
        CalOperator::Difference,
        VariableIndex::from_usize(3),
    );
    let coordinates = Coordinates::new(&task);

    assert_eq!(
        task.comparison_axioms()[0].get_left_var_id(),
        VariableIndex::new(0)
    );
    assert!(coordinates.is_coordinate(VariableIndex::new(0)));
    assert!(coordinates.views().iter().all(Option::is_none));
}

#[test]
fn decimal_steps_are_inferred_for_regular_and_shared_linear_coordinates() {
    let task = affine_sum_task_with_step(AssignmentOperation::Plus, 0.1);
    let coordinates = Coordinates::new(&task);
    let lattices = coordinates.value_lattices();
    for variable in [0, task.comparison_axioms()[0].get_left_var_id().index()] {
        assert_eq!(
            lattices[variable].split_bounds(NumericValue::new(140.0), true),
            Some((NumericValue::new(140.0), NumericValue::new(140.1)))
        );
    }
}

#[test]
fn assigned_constants_contribute_differences_from_the_initial_value() {
    let task = affine_sum_task_with_step(AssignmentOperation::Assign, 0.3);
    let coordinates = Coordinates::new(&task);
    assert_eq!(
        coordinates.value_lattices()[0].split_bounds(NumericValue::new(0.1), true),
        Some((ZERO_VALUE, NumericValue::new(0.3)))
    );
}

#[test]
fn multiplicative_and_large_effects_use_the_full_grid() {
    for (operation, step) in [
        (AssignmentOperation::Times, 2.0),
        (AssignmentOperation::Plus, 1e10),
    ] {
        let task = affine_sum_task_with_step(operation, step);
        assert_eq!(
            Coordinates::new(&task).value_lattices()[0],
            ValueLattice::GRID
        );
    }
}
