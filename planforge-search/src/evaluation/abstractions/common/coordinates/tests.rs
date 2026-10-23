use planforge_sas::axioms::{AssignmentAxiom, CalOperator, ComparisonAxiom, ComparisonOperator};
use planforge_sas::numeric_task::{
    AssignmentEffect, AssignmentOperation, ExplicitFact, ExplicitValueIndex, ExplicitVariable,
    Metric, NumericRootTask, NumericRootTaskParts, NumericType, NumericVariable, ONE_VALUE,
    Operator, OperatorCost,
};
use planforge_sas::utils::value_lattice::ValueLattice;

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
    NumericRootTask::new(derived_parts(
        operation,
        step,
        CalOperator::Sum,
        VariableIndex::from_usize(1),
    ))
}

/// `x`, `y`, the constant `one` and the derived `x <definition> right`; one
/// operator changes `x` by `step`.
fn derived_parts(
    operation: AssignmentOperation,
    step: f64,
    definition: CalOperator,
    right: VariableIndex,
) -> NumericRootTaskParts {
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
    NumericRootTaskParts {
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
    }
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
    let task = NumericRootTask::new(derived_parts(
        AssignmentOperation::Plus,
        1.0,
        CalOperator::Difference,
        VariableIndex::from_usize(3),
    ));
    let coordinates = Coordinates::new(&task);

    assert_eq!(
        task.comparison_axioms()[0].get_left_var_id(),
        VariableIndex::new(0)
    );
    assert!(coordinates.is_coordinate(VariableIndex::new(0)));
    assert!(coordinates.views().iter().all(Option::is_none));
}

#[test]
fn an_unbounded_decimal_increment_does_not_claim_a_nominal_lattice() {
    let task = affine_sum_task_with_step(AssignmentOperation::Plus, 0.1);
    let coordinates = Coordinates::new(&task);
    let lattices = coordinates.value_lattices();
    assert_eq!(lattices[0], ValueLattice::GRID);
    let mut value = ZERO_VALUE;
    for _ in 0..40_962 {
        value += NumericValue::new(0.1);
    }
    let nominal = NumericValue::new(4096.2);
    assert_eq!(
        value.grid_units(),
        nominal.grid_units().map(|units| units + 1)
    );
    let (_, upper) = lattices[0].split_bounds(nominal, true).unwrap();
    assert!(
        upper <= value,
        "a drifted reachable value must not be dropped"
    );
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

fn guarded_increment_task(limit: f64) -> NumericRootTask {
    let mut parts = derived_parts(
        AssignmentOperation::Plus,
        0.1,
        CalOperator::Sum,
        VariableIndex::new(1),
    );
    parts.numeric_variables.push(NumericVariable::new(
        "limit".into(),
        NumericType::Constant,
        None,
    ));
    parts.numeric_state.push(NumericValue::new(limit));
    parts.variables.push(ExplicitVariable::new(
        2,
        "below-limit".into(),
        vec!["true".into(), "false".into()],
        Some(1),
        ExplicitValueIndex::new(1),
    ));
    parts.state.push(ExplicitValueIndex::new(1));
    parts.comparison_axioms.push(ComparisonAxiom::new(
        VariableIndex::new(2),
        VariableIndex::new(0),
        VariableIndex::new(4),
        ComparisonOperator::LessThan,
    ));
    let operator = parts.operators.remove(0);
    parts.operators.push(Operator::new(
        operator.name().into(),
        vec![ExplicitFact::propositional(2, 0)],
        vec![],
        operator.assignment_effects().clone(),
        operator.cost(),
    ));
    NumericRootTask::new(parts)
}

#[test]
fn bounded_decimal_increments_keep_the_exact_gcd_lattice() {
    let task = guarded_increment_task(100.0);
    assert_eq!(
        Coordinates::new(&task).value_lattices()[0].split_bounds(NumericValue::new(3.0), true),
        Some((NumericValue::new(3.0), NumericValue::new(3.1)))
    );
    let task = guarded_increment_task(5000.0);
    assert_eq!(
        Coordinates::new(&task).value_lattices()[0],
        ValueLattice::GRID
    );
}

#[test]
fn a_resource_invariant_certifies_a_decimal_affine_coordinate() {
    let mut parts = derived_parts(
        AssignmentOperation::Plus,
        1.0,
        CalOperator::Sum,
        VariableIndex::new(1),
    );
    parts.numeric_state[0] = NumericValue::new(100.0);
    parts.assignment_axioms[0] = AssignmentAxiom::linear(
        VariableIndex::new(2),
        vec![
            (VariableIndex::new(0), ONE_VALUE),
            (VariableIndex::new(1), NumericValue::new(1.1)),
        ],
    );
    parts.variables.push(ExplicitVariable::new(
        2,
        "resource-available".into(),
        vec!["true".into(), "false".into()],
        Some(1),
        ExplicitValueIndex::new(1),
    ));
    parts.state.push(ExplicitValueIndex::new(1));
    parts.comparison_axioms.push(ComparisonAxiom::new(
        VariableIndex::new(2),
        VariableIndex::new(0),
        VariableIndex::new(3),
        ComparisonOperator::GreaterThanOrEqual,
    ));
    parts.operators = vec![Operator::new(
        "transfer".into(),
        vec![ExplicitFact::propositional(2, 0)],
        vec![],
        vec![
            AssignmentEffect::new(
                VariableIndex::new(0),
                AssignmentOperation::Minus,
                VariableIndex::new(3),
                false,
                vec![],
            ),
            AssignmentEffect::new(
                VariableIndex::new(1),
                AssignmentOperation::Plus,
                VariableIndex::new(3),
                false,
                vec![],
            ),
        ],
        OperatorCost::new(1),
    )];
    let task = NumericRootTask::new(parts);
    let coordinate = task.comparison_axioms()[0].get_left_var_id().index();
    assert_eq!(
        Coordinates::new(&task).value_lattices()[coordinate]
            .split_bounds(NumericValue::new(100.0), true),
        Some((NumericValue::new(100.0), NumericValue::new(100.1)))
    );
}

#[test]
fn an_unbounded_decimal_linear_part_does_not_claim_constant_floating_deltas() {
    let mut parts = derived_parts(
        AssignmentOperation::Plus,
        1.0,
        CalOperator::Sum,
        VariableIndex::new(1),
    );
    parts.assignment_axioms[0] = AssignmentAxiom::linear(
        VariableIndex::new(2),
        vec![
            (VariableIndex::new(0), ONE_VALUE),
            (VariableIndex::new(1), NumericValue::new(0.1)),
        ],
    );
    parts.operators = vec![Operator::new(
        "increment-y".into(),
        vec![],
        vec![],
        vec![AssignmentEffect::new(
            VariableIndex::new(1),
            AssignmentOperation::Plus,
            VariableIndex::new(3),
            false,
            vec![],
        )],
        OperatorCost::new(1),
    )];
    let task = NumericRootTask::new(parts);
    let coordinate = task.comparison_axioms()[0].get_left_var_id();
    let coordinates = Coordinates::new(&task);
    assert_eq!(
        coordinates.value_lattices()[coordinate.index()],
        ValueLattice::GRID
    );
    let mut values = task.get_initial_numeric_state_values().to_vec();
    values[1] = NumericValue::new(40_964.0);
    let value = coordinates.views()[coordinate.index()]
        .as_ref()
        .unwrap()
        .evaluate(&values);
    assert_eq!(
        value.grid_units(),
        NumericValue::new(4096.4)
            .grid_units()
            .map(|units| units + 1)
    );
}

#[test]
fn unbounded_sailing_views_cannot_assume_integer_correlations() {
    let mut parts = derived_parts(
        AssignmentOperation::Plus,
        3.0,
        CalOperator::Sum,
        VariableIndex::new(1),
    );
    parts.numeric_variables.push(NumericVariable::new(
        "half-step".into(),
        NumericType::Constant,
        None,
    ));
    parts.numeric_state.push(NumericValue::new(1.5));
    for (name, operation, targets, operand) in [
        ("west", AssignmentOperation::Minus, vec![0], 3),
        ("north-east", AssignmentOperation::Plus, vec![0, 1], 4),
    ] {
        parts.operators.push(Operator::new(
            name.into(),
            vec![],
            vec![],
            targets
                .into_iter()
                .map(|target| {
                    AssignmentEffect::new(
                        VariableIndex::new(target),
                        operation.clone(),
                        VariableIndex::new(operand),
                        false,
                        vec![],
                    )
                })
                .collect(),
            OperatorCost::new(1),
        ));
    }
    let task = NumericRootTask::new(parts);
    let coordinate = task.comparison_axioms()[0].get_left_var_id().index();
    assert_eq!(
        Coordinates::new(&task).value_lattices()[coordinate].step(),
        NumericValue::new(0.5)
    );

    // This multiple of three is reached by exact east moves from zero. The
    // next move rounds at 2^53; north-east then loses the x/y integer phase.
    let mut x = NumericValue::new(((1u64 << 53) - 2) as f64);
    let mut y = ZERO_VALUE;
    x += NumericValue::new(3.0);
    x += NumericValue::new(1.5);
    y += NumericValue::new(1.5);
    x -= NumericValue::new(3.0);
    // Every remaining west move subtracts three from an exactly represented
    // integer below 2^53, so this batch is exact and avoids a huge test loop.
    let west_moves = ((1u64 << 53) - 1) / 3;
    x -= NumericValue::new((3 * west_moves) as f64);
    assert_eq!(x, ONE_VALUE);
    assert_eq!(
        NumericValue::affine([(1.0, x), (1.0, y)], 0.0),
        NumericValue::new(2.5)
    );
}
