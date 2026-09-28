use super::*;

use crate::axioms::{AssignmentAxiom, CalOperator, ComparisonAxiom, ComparisonOperator};
use crate::numeric_task::{NAN_VALUE, NumericType, NumericVariable, ONE_VALUE, ZERO_VALUE};
use crate::utils::interval::Interval;

fn numeric_var(name: &str, numeric_type: NumericType) -> NumericVariable {
    NumericVariable::new(name.into(), numeric_type, None)
}

/// `d3 = (x0 + x1) * x1`, compared as `d3 > x0`, written into prop var 0.
fn shared_subexpression_conditions() -> NumericConditions {
    let numeric_variables = vec![
        numeric_var("x0", NumericType::Regular),
        numeric_var("x1", NumericType::Regular),
        numeric_var("d2", NumericType::Derived),
        numeric_var("d3", NumericType::Derived),
    ];
    let assignment_axioms = vec![
        AssignmentAxiom::new(
            VariableIndex::new(2),
            CalOperator::Sum,
            VariableIndex::new(0),
            VariableIndex::new(1),
        ),
        AssignmentAxiom::new(
            VariableIndex::new(3),
            CalOperator::Product,
            VariableIndex::new(2),
            VariableIndex::new(1),
        ),
    ];
    let comparison_axioms = vec![ComparisonAxiom::new(
        VariableIndex::new(0),
        VariableIndex::new(3),
        VariableIndex::new(0),
        ComparisonOperator::GreaterThan,
    )];

    NumericConditions::build(
        1,
        &numeric_variables,
        &comparison_axioms,
        &assignment_axioms,
    )
    .unwrap()
}

#[test]
fn build_expands_assignment_axioms_and_collects_regular_dependencies() {
    let conditions = shared_subexpression_conditions();
    assert_eq!(conditions.len(), 1);

    let condition = conditions
        .for_var(VariableIndex::from_usize(0))
        .expect("prop var 0 carries condition");
    assert_eq!(condition.id(), 0);
    assert_eq!(condition.op(), CompOp::Gt);
    assert_eq!(condition.left_numeric_var_id(), VariableIndex::new(3));
    assert_eq!(condition.right_numeric_var_id(), VariableIndex::new(0));
    assert_eq!(
        condition.regular_numeric_var_dependencies(),
        [VariableIndex::new(0), VariableIndex::new(1)]
    );
    assert_eq!(condition.required_numeric_len(), 4);

    match condition.node(condition.left_root()) {
        ConditionNode::Arith {
            result_numeric_var_id,
            op,
            left_numeric_var_id,
            right_numeric_var_id,
            ..
        } => {
            assert_eq!(*result_numeric_var_id, VariableIndex::new(3));
            assert_eq!(*op, ArithOp::Mul);
            assert_eq!(*left_numeric_var_id, VariableIndex::new(2));
            assert_eq!(*right_numeric_var_id, VariableIndex::new(1));
        }
        other => panic!("expected arith node, got {other:?}"),
    }
}

#[test]
fn build_shares_subexpressions_between_operands() {
    // `d = x0 + x1` on both sides expands to a single arena node.
    let numeric_variables = vec![
        numeric_var("x0", NumericType::Regular),
        numeric_var("x1", NumericType::Regular),
        numeric_var("d", NumericType::Derived),
    ];
    let assignment_axioms = vec![AssignmentAxiom::new(
        VariableIndex::new(2),
        CalOperator::Sum,
        VariableIndex::new(0),
        VariableIndex::new(1),
    )];
    let comparison_axioms = vec![ComparisonAxiom::new(
        VariableIndex::new(0),
        VariableIndex::new(2),
        VariableIndex::new(2),
        ComparisonOperator::Equal,
    )];

    let conditions = NumericConditions::build(
        1,
        &numeric_variables,
        &comparison_axioms,
        &assignment_axioms,
    )
    .unwrap();
    let condition = conditions.get(0).unwrap();

    assert_eq!(condition.left_root(), condition.right_root());
    assert_eq!(condition.nodes().len(), 3);
}

#[test]
fn children_precede_parents_in_the_arena() {
    let conditions = shared_subexpression_conditions();
    let condition = conditions.get(0).unwrap();
    for (node_id, node) in condition.nodes().iter().enumerate() {
        if let ConditionNode::Arith { left, right, .. } = node {
            assert!(*left < node_id, "left child {left} must precede {node_id}");
            assert!(
                *right < node_id,
                "right child {right} must precede {node_id}"
            );
        }
    }
}

#[test]
fn point_evaluation_recomputes_derived_variables() {
    let conditions = shared_subexpression_conditions();
    let condition = conditions.get(0).unwrap();

    // Stale derived slots are ignored: (1 + 2) * 2 = 6 > 1.
    assert!(condition.evaluate_point(&[ONE_VALUE, NumericValue::new(2.0), NAN_VALUE, NAN_VALUE]));
    // (1 + 0) * 0 = 0, not > 1.
    assert!(!condition.evaluate_point(&[ONE_VALUE, ZERO_VALUE, ZERO_VALUE, ZERO_VALUE]));
}

#[test]
fn interval_evaluation_is_three_valued() {
    let conditions = shared_subexpression_conditions();
    let condition = conditions.get(0).unwrap();
    let nothing_known = Interval::new(ZERO_VALUE, ZERO_VALUE, false, false);

    let definitely_true = [
        Interval::singleton(ONE_VALUE),
        Interval::singleton(NumericValue::new(2.0)),
        nothing_known,
        nothing_known,
    ];
    assert_eq!(condition.evaluate_interval(&definitely_true), Some(true));
    assert!(condition.admits_true(&definitely_true));
    assert!(!condition.admits_false(&definitely_true));

    let unknown = [
        Interval::closed(ZERO_VALUE, NumericValue::new(4.0)),
        Interval::closed(ZERO_VALUE, ONE_VALUE),
        nothing_known,
        nothing_known,
    ];
    assert_eq!(condition.evaluate_interval(&unknown), None);
    assert!(condition.admits_true(&unknown));
    assert!(condition.admits_false(&unknown));
}

#[test]
fn interval_evaluation_fills_derived_intervals() {
    let numeric_variables = vec![
        numeric_var("x0", NumericType::Regular),
        numeric_var("c1", NumericType::Constant),
        numeric_var("d2", NumericType::Derived),
        numeric_var("d3", NumericType::Derived),
    ];
    let assignment_axioms = vec![
        AssignmentAxiom::new(
            VariableIndex::new(2),
            CalOperator::Sum,
            VariableIndex::new(0),
            VariableIndex::new(1),
        ),
        AssignmentAxiom::new(
            VariableIndex::new(3),
            CalOperator::Product,
            VariableIndex::new(2),
            VariableIndex::new(1),
        ),
    ];
    let comparison_axioms = vec![ComparisonAxiom::new(
        VariableIndex::new(0),
        VariableIndex::new(3),
        VariableIndex::new(1),
        ComparisonOperator::GreaterThan,
    )];

    let conditions = NumericConditions::build(
        1,
        &numeric_variables,
        &comparison_axioms,
        &assignment_axioms,
    )
    .unwrap();
    let condition = conditions.get(0).unwrap();

    let mut intervals = vec![
        Interval::singleton(ONE_VALUE),
        Interval::singleton(NumericValue::new(2.0)),
        Interval::new(ZERO_VALUE, ZERO_VALUE, false, false),
        Interval::new(ZERO_VALUE, ZERO_VALUE, false, false),
    ];
    assert_eq!(
        condition.evaluate_interval_and_fill(&mut intervals),
        Some(true)
    );
    assert_eq!(intervals[2], Interval::singleton(NumericValue::new(3.0)));
    assert_eq!(intervals[3], Interval::singleton(NumericValue::new(6.0)));
}

#[test]
fn lhs_minus_rhs_interval_shifts_the_comparison_to_zero() {
    let conditions = shared_subexpression_conditions();
    let condition = conditions.get(0).unwrap();
    let nothing_known = Interval::new(ZERO_VALUE, ZERO_VALUE, false, false);

    let difference = condition.lhs_minus_rhs_interval(&[
        Interval::singleton(ONE_VALUE),
        Interval::singleton(NumericValue::new(2.0)),
        nothing_known,
        nothing_known,
    ]);
    assert_eq!(difference, Interval::singleton(NumericValue::new(5.0)));
}

#[test]
fn build_rejects_cyclic_assignment_axioms() {
    // d1 = d1 + x0
    let numeric_variables = vec![
        numeric_var("x0", NumericType::Regular),
        numeric_var("d1", NumericType::Derived),
    ];
    let assignment_axioms = vec![AssignmentAxiom::new(
        VariableIndex::new(1),
        CalOperator::Sum,
        VariableIndex::new(1),
        VariableIndex::new(0),
    )];
    let comparison_axioms = vec![ComparisonAxiom::new(
        VariableIndex::new(0),
        VariableIndex::new(1),
        VariableIndex::new(0),
        ComparisonOperator::Equal,
    )];

    assert_eq!(
        NumericConditions::build(
            1,
            &numeric_variables,
            &comparison_axioms,
            &assignment_axioms
        ),
        Err(NumericConditionError::CycleDetected {
            numeric_var_id: VariableIndex::new(1)
        })
    );
}

#[test]
fn build_rejects_duplicate_assignment_targets() {
    let numeric_variables = vec![
        numeric_var("x0", NumericType::Regular),
        numeric_var("d1", NumericType::Derived),
    ];
    let assignment_axioms = vec![
        AssignmentAxiom::new(
            VariableIndex::new(1),
            CalOperator::Sum,
            VariableIndex::new(0),
            VariableIndex::new(0),
        ),
        AssignmentAxiom::new(
            VariableIndex::new(1),
            CalOperator::Product,
            VariableIndex::new(0),
            VariableIndex::new(0),
        ),
    ];
    let comparison_axioms = vec![ComparisonAxiom::new(
        VariableIndex::new(0),
        VariableIndex::new(1),
        VariableIndex::new(0),
        ComparisonOperator::Equal,
    )];

    assert_eq!(
        NumericConditions::build(
            1,
            &numeric_variables,
            &comparison_axioms,
            &assignment_axioms
        ),
        Err(NumericConditionError::DuplicateAssignmentTarget {
            numeric_var_id: VariableIndex::new(1),
            first_assignment_axiom_id: AxiomIndex::new(0),
            second_assignment_axiom_id: AxiomIndex::new(1),
        })
    );
}

#[test]
fn build_rejects_two_axioms_writing_the_same_propositional_var() {
    let numeric_variables = vec![numeric_var("x0", NumericType::Regular)];
    let comparison_axioms = vec![
        ComparisonAxiom::new(
            VariableIndex::new(0),
            VariableIndex::new(0),
            VariableIndex::new(0),
            ComparisonOperator::Equal,
        ),
        ComparisonAxiom::new(
            VariableIndex::new(0),
            VariableIndex::new(0),
            VariableIndex::new(0),
            ComparisonOperator::LessThan,
        ),
    ];

    assert_eq!(
        NumericConditions::build(1, &numeric_variables, &comparison_axioms, &[]),
        Err(NumericConditionError::DuplicatePropositionalVar {
            prop_var_id: VariableIndex::new(0),
            first_comparison_axiom_id: AxiomIndex::new(0),
            second_comparison_axiom_id: AxiomIndex::new(1),
        })
    );
}

#[test]
fn build_rejects_unknown_propositional_var() {
    let numeric_variables = vec![numeric_var("x0", NumericType::Regular)];
    let comparison_axioms = vec![ComparisonAxiom::new(
        VariableIndex::new(7),
        VariableIndex::new(0),
        VariableIndex::new(0),
        ComparisonOperator::Equal,
    )];

    assert_eq!(
        NumericConditions::build(1, &numeric_variables, &comparison_axioms, &[]),
        Err(NumericConditionError::UnknownPropositionalVar {
            comparison_axiom_id: AxiomIndex::new(0),
            provided: VariableIndex::new(7),
            num_propositional_vars: 1,
        })
    );
}

#[test]
fn build_rejects_unknown_numeric_var() {
    let numeric_variables = vec![numeric_var("x0", NumericType::Regular)];
    let comparison_axioms = vec![ComparisonAxiom::new(
        VariableIndex::new(0),
        VariableIndex::new(0),
        VariableIndex::new(3),
        ComparisonOperator::Equal,
    )];

    assert_eq!(
        NumericConditions::build(1, &numeric_variables, &comparison_axioms, &[]),
        Err(NumericConditionError::UnknownNumericVar {
            provided: VariableIndex::new(3),
            num_numeric_vars: 1,
        })
    );
}

#[test]
fn condition_vars_are_distinguished_from_ordinary_prop_vars() {
    let numeric_variables = vec![numeric_var("x0", NumericType::Regular)];
    let comparison_axioms = vec![ComparisonAxiom::new(
        VariableIndex::new(2),
        VariableIndex::new(0),
        VariableIndex::new(0),
        ComparisonOperator::Equal,
    )];

    let conditions =
        NumericConditions::build(4, &numeric_variables, &comparison_axioms, &[]).unwrap();

    assert!(conditions.is_condition_var(VariableIndex::new(2)));
    assert!(!conditions.is_condition_var(VariableIndex::new(0)));
    assert!(!conditions.is_condition_var(VariableIndex::new(9)));
    assert_eq!(conditions.id_for_var(VariableIndex::from_usize(2)), Some(0));
    assert_eq!(conditions.id_for_var(VariableIndex::from_usize(0)), None);
    assert_eq!(conditions.condition_var_ids().collect::<Vec<_>>(), [2]);
}

#[test]
fn precondition_is_contradicted_only_for_condition_vars() {
    let conditions = shared_subexpression_conditions();
    let nothing_known = Interval::new(ZERO_VALUE, ZERO_VALUE, false, false);
    // (1 + 2) * 2 = 6 > 1 always holds here.
    let intervals = [
        Interval::singleton(ONE_VALUE),
        Interval::singleton(NumericValue::new(2.0)),
        nothing_known,
        nothing_known,
    ];

    let holds = ExplicitFact::propositional(0, ConditionValue::True.as_usize());
    let fails = ExplicitFact::propositional(0, ConditionValue::False.as_usize());
    assert!(!conditions.precondition_is_contradicted(&holds, &intervals));
    assert!(conditions.precondition_is_contradicted(&fails, &intervals));

    let ordinary = ExplicitFact::propositional(3, ConditionValue::True.as_usize());
    assert!(!conditions.precondition_is_contradicted(&ordinary, &intervals));
}
