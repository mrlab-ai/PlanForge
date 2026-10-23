use super::*;
use crate::axioms::CalOperator;
use crate::numeric_conditions::ConditionValue;

fn task(
    regular: &[f64],
    constants: &[f64],
    definitions: &[(CalOperator, usize, usize)],
    comparisons: &[(usize, usize, ComparisonOperator)],
) -> NumericRootTask {
    let base = regular.len() + constants.len();
    let numeric_variables =
        regular
            .iter()
            .enumerate()
            .map(|(id, _)| NumericVariable::new(format!("x{id}"), NumericType::Regular, None))
            .chain(
                constants.iter().enumerate().map(|(id, _)| {
                    NumericVariable::new(format!("c{id}"), NumericType::Constant, None)
                }),
            )
            .chain(definitions.iter().enumerate().map(|(id, _)| {
                NumericVariable::new(format!("d{id}"), NumericType::Derived, Some(id))
            }))
            .collect();
    let variables = std::iter::once(ExplicitVariable::new(
        1,
        "global".into(),
        vec!["true".into()],
        None,
        ExplicitValueIndex::new(0),
    ))
    .chain(comparisons.iter().enumerate().map(|(id, _)| {
        ExplicitVariable::new(
            2,
            format!("comparison{id}"),
            vec!["true".into(), "false".into()],
            Some(definitions.len()),
            ExplicitValueIndex::new(1),
        )
    }))
    .collect();
    NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, None),
        variables,
        numeric_variables,
        goals: vec![],
        mutexes: vec![],
        state: std::iter::once(ExplicitValueIndex::new(0))
            .chain(comparisons.iter().map(|_| ExplicitValueIndex::new(1)))
            .collect(),
        numeric_state: regular
            .iter()
            .chain(constants)
            .copied()
            .map(NumericValue::new)
            .chain(definitions.iter().map(|_| ZERO_VALUE))
            .collect(),
        operators: vec![],
        axioms: vec![],
        comparison_axioms: comparisons
            .iter()
            .enumerate()
            .map(|(id, (left, right, operator))| {
                ComparisonAxiom::new(
                    VariableIndex::from_usize(id + 1),
                    VariableIndex::from_usize(*left),
                    VariableIndex::from_usize(*right),
                    operator.clone(),
                )
            })
            .collect(),
        assignment_axioms: definitions
            .iter()
            .enumerate()
            .map(|(id, (operator, left, right))| {
                AssignmentAxiom::new(
                    VariableIndex::from_usize(base + id),
                    operator.clone(),
                    VariableIndex::from_usize(*left),
                    VariableIndex::from_usize(*right),
                )
            })
            .collect(),
        global_constraint: ExplicitFact::propositional(0, 0),
    })
}

#[test]
fn shares_linear_parts_across_constants_order_and_positive_scale() {
    use CalOperator::*;
    use ComparisonOperator::*;
    // x+y-4 >= 0, 2*(y+x) > 4 and -(x+y) <= 0.
    let task = task(
        &[2.0, 3.0],
        &[4.0, 0.0, 2.0, -1.0],
        &[
            (Sum, 0, 1),
            (Difference, 6, 2),
            (Sum, 1, 0),
            (Product, 8, 4),
            (Product, 6, 5),
            (Product, 10, 4),
        ],
        &[
            (7, 3, GreaterThanOrEqual),
            (9, 2, GreaterThan),
            (10, 3, LessThanOrEqual),
            (11, 2, LessThanOrEqual),
        ],
    );
    assert_eq!(task.assignment_axioms.len(), 2);
    let comparisons = &task.comparison_axioms;
    assert_eq!(comparisons[0].left_hand_side, comparisons[1].left_hand_side);
    assert_eq!(comparisons[2].left_hand_side, comparisons[3].left_hand_side);
    assert_ne!(comparisons[0].left_hand_side, comparisons[2].left_hand_side);
    assert_eq!(comparisons[2].operator, LessThanOrEqual);
    assert_eq!(
        task.numeric_state[comparisons[0].right_hand_side.index()],
        NumericValue::new(4.0)
    );
    assert_eq!(
        task.numeric_state[comparisons[1].right_hand_side.index()],
        NumericValue::new(2.0)
    );
    assert!(
        task.numeric_conditions
            .iter()
            .all(|condition| condition.evaluate_point(&task.numeric_state))
    );
}

#[test]
fn unary_negative_coefficients_reverse_all_ordered_comparisons() {
    use ComparisonOperator::*;
    let operators = [
        LessThan,
        LessThanOrEqual,
        Equal,
        UnEqual,
        GreaterThanOrEqual,
        GreaterThan,
    ];
    let expected = [
        GreaterThan,
        GreaterThanOrEqual,
        Equal,
        UnEqual,
        LessThanOrEqual,
        LessThan,
    ];
    let task = task(
        &[3.0],
        &[-2.0, 8.0],
        &[(CalOperator::Product, 0, 1)],
        &operators
            .iter()
            .cloned()
            .map(|operator| (3, 2, operator))
            .collect::<Vec<_>>(),
    );
    assert!(task.assignment_axioms.is_empty());
    for (comparison, operator) in task.comparison_axioms.iter().zip(expected) {
        assert_eq!(comparison.left_hand_side, VariableIndex::new(0));
        assert_eq!(comparison.operator, operator);
        assert_eq!(
            task.numeric_state[comparison.right_hand_side.index()],
            NumericValue::new(-4.0)
        );
    }
}

#[test]
fn folds_constant_expressions() {
    let task = task(
        &[],
        &[2.0, 3.0],
        &[(CalOperator::Product, 0, 1)],
        &[(2, 0, ComparisonOperator::GreaterThan)],
    );
    assert!(task.assignment_axioms.is_empty());
    assert_eq!(
        task.state[1],
        ExplicitValueIndex::new(ConditionValue::True.as_u32())
    );
}

#[test]
fn nonlinear_products_remain_binary() {
    let task = task(
        &[2.0, 3.0],
        &[4.0],
        &[(CalOperator::Product, 0, 1)],
        &[(3, 2, ComparisonOperator::GreaterThan)],
    );
    assert!(matches!(
        task.assignment_axioms[0].expression,
        NumericExpression::Binary {
            operator: CalOperator::Product,
            ..
        }
    ));
    assert!(task.numeric_conditions.all()[0].evaluate_point(&task.numeric_state));
}

#[test]
fn unary_comparisons_use_the_decided_arithmetic_contract() {
    let task = task(
        &[1e-12],
        &[0.25, 4.0, 0.0],
        &[(CalOperator::Product, 0, 1), (CalOperator::Product, 4, 2)],
        &[(5, 3, ComparisonOperator::Equal)],
    );
    assert!(task.assignment_axioms.is_empty());
    assert_eq!(
        task.comparison_axioms[0].left_hand_side,
        VariableIndex::new(0)
    );
    assert!(!task.numeric_conditions.all()[0].evaluate_point(&task.numeric_state));
}

#[test]
fn linear_sum_canonicalizes_once_in_both_concrete_evaluators() {
    let terms = vec![
        (VariableIndex::new(0), NumericValue::new(0.25)),
        (VariableIndex::new(1), NumericValue::new(0.25)),
    ];
    let mut values = vec![
        NumericValue::new(2e-12),
        NumericValue::new(2e-12),
        ZERO_VALUE,
        NumericValue::new(1e-12),
    ];
    let axiom = AssignmentAxiom::linear(VariableIndex::new(2), terms);
    assert_eq!(
        axiom.update_values(&mut values).unwrap(),
        NumericValue::new(1e-12)
    );
    let conditions = NumericConditions::build(
        1,
        &[
            NumericVariable::new("x".into(), NumericType::Regular, None),
            NumericVariable::new("y".into(), NumericType::Regular, None),
            NumericVariable::new("sum".into(), NumericType::Derived, Some(0)),
            NumericVariable::new("threshold".into(), NumericType::Constant, None),
        ],
        &[ComparisonAxiom::new(
            VariableIndex::new(0),
            VariableIndex::new(2),
            VariableIndex::new(3),
            ComparisonOperator::Equal,
        )],
        &[axiom],
    )
    .unwrap();
    values[2] = ZERO_VALUE;
    assert!(conditions.all()[0].evaluate_point(&values));
}

#[test]
fn grid_unit_coefficients_are_scaled_before_comparison() {
    let task = task(
        &[4.0],
        &[1e-12, 4e-12],
        &[(CalOperator::Product, 0, 1)],
        &[(3, 2, ComparisonOperator::GreaterThanOrEqual)],
    );
    let comparison = &task.comparison_axioms[0];
    assert_eq!(comparison.left_hand_side, VariableIndex::new(0));
    assert_eq!(
        task.numeric_state[comparison.right_hand_side.index()],
        NumericValue::new(4.0)
    );
}

#[test]
fn folds_constant_subexpressions_inside_nonlinear_comparisons() {
    use CalOperator::*;
    let task = task(
        &[2.0, 3.0],
        &[2.0, 3.0],
        &[(Product, 0, 1), (Sum, 2, 3), (Sum, 4, 5)],
        &[(6, 2, ComparisonOperator::GreaterThan)],
    );
    assert_eq!(task.assignment_axioms.len(), 2);
    assert!(
        task.numeric_variables
            .iter()
            .zip(&task.numeric_state)
            .any(|(var, value)| var.get_type() == &NumericType::Constant
                && *value == NumericValue::new(5.0))
    );
}

#[test]
fn rebuilding_a_normalized_task_preserves_numeric_ids_and_definitions() {
    let original = task(
        &[2.0, 3.0],
        &[4.0],
        &[(CalOperator::Sum, 0, 1)],
        &[(3, 2, ComparisonOperator::GreaterThan)],
    );
    let expected_variables = original.numeric_variables.clone();
    let expected_assignments = original.assignment_axioms.clone();
    let expected_comparisons = original.comparison_axioms.clone();
    let rebuilt = NumericRootTask::new(NumericRootTaskParts {
        version: original.version,
        metric: original.metric,
        variables: original.variables,
        numeric_variables: original.numeric_variables,
        goals: original.goals,
        mutexes: original.mutexes,
        state: original.state,
        numeric_state: original.numeric_state,
        operators: original.operators,
        axioms: original.axioms,
        comparison_axioms: original.comparison_axioms,
        assignment_axioms: original.assignment_axioms,
        global_constraint: original.global_constraint,
    });
    assert_eq!(rebuilt.numeric_variables, expected_variables);
    assert_eq!(rebuilt.assignment_axioms, expected_assignments);
    assert_eq!(rebuilt.comparison_axioms, expected_comparisons);
}

#[test]
fn shared_expression_dags_are_not_expanded_as_trees() {
    // Each definition reads its predecessor twice: expanding as a tree would
    // visit 2^100 occurrences, although the task has only 100 definitions.
    let definitions: Vec<_> = (0..100)
        .map(|id| {
            let operand = if id == 0 { 0 } else { id + 1 };
            (CalOperator::Sum, operand, operand)
        })
        .collect();
    let task = task(
        &[1.0],
        &[0.0],
        &definitions,
        &[(101, 1, ComparisonOperator::GreaterThan)],
    );
    assert!(task.assignment_axioms.is_empty());
    assert_eq!(
        task.comparison_axioms[0].left_hand_side,
        VariableIndex::new(0)
    );
    assert!(task.numeric_conditions.all()[0].evaluate_point(&task.numeric_state));
}

#[test]
fn scaling_preserves_small_terms_in_a_multivariate_comparison() {
    use CalOperator::*;
    // Leading-coefficient scaling would round y's nonzero coefficient away.
    let task = task(
        &[0.0, 1.0],
        &[1e6, 1e-7, 0.0],
        &[(Product, 0, 2), (Product, 1, 3), (Sum, 5, 6)],
        &[(7, 4, ComparisonOperator::GreaterThan)],
    );
    let NumericExpression::Linear(terms) = &task.assignment_axioms[0].expression else {
        panic!("expected a linear sum");
    };
    assert_eq!(terms.len(), 2);
    assert!(
        terms
            .iter()
            .all(|(_, coefficient)| *coefficient != ZERO_VALUE)
    );
    assert!(task.numeric_conditions.all()[0].evaluate_point(&task.numeric_state));
}
