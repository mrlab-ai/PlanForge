use super::*;

use crate::axioms::{AssignmentAxiom, CalOperator, ComparisonAxiom, ComparisonOperator};
use crate::numeric_task::{NumericType, NumericVariable};
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
        AssignmentAxiom::new(2, CalOperator::Sum, 0, 1),
        AssignmentAxiom::new(3, CalOperator::Product, 2, 1),
    ];
    let comparison_axioms = vec![ComparisonAxiom::new(
        0,
        3,
        0,
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

    let condition = conditions.for_var(0).expect("prop var 0 carries condition");
    assert_eq!(condition.id(), 0);
    assert_eq!(condition.op(), CompOp::Gt);
    assert_eq!(condition.left_numeric_var_id(), 3);
    assert_eq!(condition.right_numeric_var_id(), 0);
    assert_eq!(condition.regular_numeric_var_dependencies(), [0, 1]);
    assert_eq!(condition.required_numeric_len(), 4);

    match condition.node(condition.left_root()) {
        ConditionNode::Arith {
            result_numeric_var_id,
            op,
            left_numeric_var_id,
            right_numeric_var_id,
            ..
        } => {
            assert_eq!(*result_numeric_var_id, 3);
            assert_eq!(*op, ArithOp::Mul);
            assert_eq!(*left_numeric_var_id, 2);
            assert_eq!(*right_numeric_var_id, 1);
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
    let assignment_axioms = vec![AssignmentAxiom::new(2, CalOperator::Sum, 0, 1)];
    let comparison_axioms = vec![ComparisonAxiom::new(0, 2, 2, ComparisonOperator::Equal)];

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
    assert!(condition.evaluate_point(&[1.0, 2.0, f64::NAN, f64::NAN]));
    // (1 + 0) * 0 = 0, not > 1.
    assert!(!condition.evaluate_point(&[1.0, 0.0, 0.0, 0.0]));
}

#[test]
fn interval_evaluation_is_three_valued() {
    let conditions = shared_subexpression_conditions();
    let condition = conditions.get(0).unwrap();
    let nothing_known = Interval::new(0.0, 0.0, false, false);

    let definitely_true = [
        Interval::singleton(1.0),
        Interval::singleton(2.0),
        nothing_known,
        nothing_known,
    ];
    assert_eq!(condition.evaluate_interval(&definitely_true), Some(true));
    assert!(condition.admits_true(&definitely_true));
    assert!(!condition.admits_false(&definitely_true));

    let unknown = [
        Interval::closed(0.0, 4.0),
        Interval::closed(0.0, 1.0),
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
        AssignmentAxiom::new(2, CalOperator::Sum, 0, 1),
        AssignmentAxiom::new(3, CalOperator::Product, 2, 1),
    ];
    let comparison_axioms = vec![ComparisonAxiom::new(
        0,
        3,
        1,
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
        Interval::singleton(1.0),
        Interval::singleton(2.0),
        Interval::new(0.0, 0.0, false, false),
        Interval::new(0.0, 0.0, false, false),
    ];
    assert_eq!(
        condition.evaluate_interval_and_fill(&mut intervals),
        Some(true)
    );
    assert_eq!(intervals[2], Interval::singleton(3.0));
    assert_eq!(intervals[3], Interval::singleton(6.0));
}

#[test]
fn lhs_minus_rhs_interval_shifts_the_comparison_to_zero() {
    let conditions = shared_subexpression_conditions();
    let condition = conditions.get(0).unwrap();
    let nothing_known = Interval::new(0.0, 0.0, false, false);

    let difference = condition.lhs_minus_rhs_interval(&[
        Interval::singleton(1.0),
        Interval::singleton(2.0),
        nothing_known,
        nothing_known,
    ]);
    assert_eq!(difference, Interval::singleton(5.0));
}

#[test]
fn lazy_evaluation_agrees_with_bottom_up() {
    let conditions = shared_subexpression_conditions();
    let condition = conditions.get(0).unwrap();
    let inputs = [1.0, 2.0, 0.0, 0.0];

    let mut lazy = condition.lazy_evaluator::<f64>();
    assert_eq!(lazy.evaluate(&inputs), condition.evaluate_point(&inputs));
    // The memo answers repeated probes without re-walking the sub-DAG.
    assert_eq!(lazy.node_value(condition.left_root(), &inputs), 6.0);
    assert_eq!(lazy.node_value(condition.left_root(), &inputs), 6.0);

    let other_inputs = [1.0, 0.0, 0.0, 0.0];
    lazy.reset();
    assert_eq!(
        lazy.evaluate(&other_inputs),
        condition.evaluate_point(&other_inputs)
    );
}

#[test]
fn build_rejects_cyclic_assignment_axioms() {
    // d1 = d1 + x0
    let numeric_variables = vec![
        numeric_var("x0", NumericType::Regular),
        numeric_var("d1", NumericType::Derived),
    ];
    let assignment_axioms = vec![AssignmentAxiom::new(1, CalOperator::Sum, 1, 0)];
    let comparison_axioms = vec![ComparisonAxiom::new(0, 1, 0, ComparisonOperator::Equal)];

    assert_eq!(
        NumericConditions::build(
            1,
            &numeric_variables,
            &comparison_axioms,
            &assignment_axioms
        ),
        Err(NumericConditionError::CycleDetected { numeric_var_id: 1 })
    );
}

#[test]
fn build_rejects_duplicate_assignment_targets() {
    let numeric_variables = vec![
        numeric_var("x0", NumericType::Regular),
        numeric_var("d1", NumericType::Derived),
    ];
    let assignment_axioms = vec![
        AssignmentAxiom::new(1, CalOperator::Sum, 0, 0),
        AssignmentAxiom::new(1, CalOperator::Product, 0, 0),
    ];
    let comparison_axioms = vec![ComparisonAxiom::new(0, 1, 0, ComparisonOperator::Equal)];

    assert_eq!(
        NumericConditions::build(
            1,
            &numeric_variables,
            &comparison_axioms,
            &assignment_axioms
        ),
        Err(NumericConditionError::DuplicateAssignmentTarget {
            numeric_var_id: 1,
            first_assignment_axiom_id: 0,
            second_assignment_axiom_id: 1,
        })
    );
}

#[test]
fn build_rejects_two_axioms_writing_the_same_propositional_var() {
    let numeric_variables = vec![numeric_var("x0", NumericType::Regular)];
    let comparison_axioms = vec![
        ComparisonAxiom::new(0, 0, 0, ComparisonOperator::Equal),
        ComparisonAxiom::new(0, 0, 0, ComparisonOperator::LessThan),
    ];

    assert_eq!(
        NumericConditions::build(1, &numeric_variables, &comparison_axioms, &[]),
        Err(NumericConditionError::DuplicatePropositionalVar {
            prop_var_id: 0,
            first_comparison_axiom_id: 0,
            second_comparison_axiom_id: 1,
        })
    );
}

#[test]
fn build_rejects_unknown_propositional_var() {
    let numeric_variables = vec![numeric_var("x0", NumericType::Regular)];
    let comparison_axioms = vec![ComparisonAxiom::new(7, 0, 0, ComparisonOperator::Equal)];

    assert_eq!(
        NumericConditions::build(1, &numeric_variables, &comparison_axioms, &[]),
        Err(NumericConditionError::UnknownPropositionalVar {
            comparison_axiom_id: 0,
            provided: 7,
            num_propositional_vars: 1,
        })
    );
}

#[test]
fn build_rejects_unknown_numeric_var() {
    let numeric_variables = vec![numeric_var("x0", NumericType::Regular)];
    let comparison_axioms = vec![ComparisonAxiom::new(0, 0, 3, ComparisonOperator::Equal)];

    assert_eq!(
        NumericConditions::build(1, &numeric_variables, &comparison_axioms, &[]),
        Err(NumericConditionError::UnknownNumericVar {
            provided: 3,
            num_numeric_vars: 1,
        })
    );
}

#[test]
fn condition_vars_are_distinguished_from_ordinary_prop_vars() {
    let numeric_variables = vec![numeric_var("x0", NumericType::Regular)];
    let comparison_axioms = vec![ComparisonAxiom::new(2, 0, 0, ComparisonOperator::Equal)];

    let conditions =
        NumericConditions::build(4, &numeric_variables, &comparison_axioms, &[]).unwrap();

    assert!(conditions.is_condition_var(2));
    assert!(!conditions.is_condition_var(0));
    assert!(!conditions.is_condition_var(9));
    assert_eq!(conditions.id_for_var(2), Some(0));
    assert_eq!(conditions.id_for_var(0), None);
    assert_eq!(conditions.condition_var_ids().collect::<Vec<_>>(), [2]);
}

#[test]
fn precondition_is_contradicted_only_for_condition_vars() {
    let conditions = shared_subexpression_conditions();
    let nothing_known = Interval::new(0.0, 0.0, false, false);
    // (1 + 2) * 2 = 6 > 1 always holds here.
    let intervals = [
        Interval::singleton(1.0),
        Interval::singleton(2.0),
        nothing_known,
        nothing_known,
    ];

    let holds = ExplicitFact::new(0, ConditionValue::True.as_usize());
    let fails = ExplicitFact::new(0, ConditionValue::False.as_usize());
    assert!(!conditions.precondition_is_contradicted(&holds, &intervals));
    assert!(conditions.precondition_is_contradicted(&fails, &intervals));

    let ordinary = ExplicitFact::new(3, ConditionValue::True.as_usize());
    assert!(!conditions.precondition_is_contradicted(&ordinary, &intervals));
}
