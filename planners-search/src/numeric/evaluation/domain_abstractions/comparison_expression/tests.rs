use super::*;

use planners_sas::numeric::axioms::{AssignmentAxiom, CalOperator, ComparisonAxiom, ComparisonOperator};
use planners_sas::numeric::numeric_task::{Metric, NumericRootTask, NumericType, NumericVariable};

#[test]
fn example_tree() {
    // Build expression: ((x0 + x1) - x2) < (x3 + x4)
    let mut expr = Expr::new();

    let n0 = expr.add_leaf(0);
    let n1 = expr.add_leaf(1);
    let left_add = expr.add_arith(ArithOp::Add, n0, n1);

    let n2 = expr.add_leaf(2);
    let left_sub = expr.add_arith(ArithOp::Sub, left_add, n2);

    let n3 = expr.add_leaf(3);
    let n4 = expr.add_leaf(4);
    let right_add = expr.add_arith(ArithOp::Add, n3, n4);

    expr.set_root_compare(CompOp::Lt, left_sub, right_add);

    let inputs = [3.0, 4.0, 2.0, 1.5, 2.0];
    // Left: (3+4)-2 = 5, Right: 1.5+2 = 3.5, 5 < 3.5 => false
    assert_eq!(expr.evaluate(&inputs), false);
}

#[test]
fn interval_add_absorbs_unbounded_when_safe() {
    let bounded = Interval::closed(3.0, 4.0);
    let unbounded = Interval::open(-3.0, f64::INFINITY);

    // Fast-path / over-approx: adding a non-negative interval to an interval
    // unbounded above stays within the unbounded interval.
    assert_eq!(bounded + unbounded, unbounded);
    assert_eq!(unbounded + bounded, unbounded);
}

#[test]
fn interval_add_regular() {
    let a = Interval::closed(3.0, 4.0);
    let b = Interval::closed(-3.0, 10.0);
    assert_eq!(a + b, Interval::closed(0.0, 14.0));
}

#[test]
fn interval_comparison_definite_and_unknown() {
    let mut expr = Expr::new();
    let a = expr.add_leaf(0);
    let b = expr.add_leaf(1);
    expr.set_root_compare(CompOp::Lt, a, b);

    // Always true: max(lhs) < min(rhs)
    let inputs = [Interval::closed(0.0, 1.0), Interval::closed(2.0, 3.0)];
    assert_eq!(expr.evaluate_interval(&inputs), Some(true));

    // Always false: min(lhs) >= max(rhs)
    let inputs = [Interval::closed(2.0, 3.0), Interval::closed(0.0, 1.0)];
    assert_eq!(expr.evaluate_interval(&inputs), Some(false));

    // Unknown: intervals overlap
    let inputs = [Interval::closed(0.0, 3.0), Interval::closed(2.0, 4.0)];
    assert_eq!(expr.evaluate_interval(&inputs), None);
}

#[test]
fn interval_eq_and_ne() {
    let mut expr = Expr::new();
    let a = expr.add_leaf(0);
    let b = expr.add_leaf(1);
    expr.set_root_compare(CompOp::Eq, a, b);

    // Singletons equal => definitely true
    let inputs = [Interval::singleton(2.0), Interval::singleton(2.0)];
    assert_eq!(expr.evaluate_interval(&inputs), Some(true));

    // Disjoint => definitely false
    let inputs = [Interval::closed(0.0, 1.0), Interval::closed(2.0, 3.0)];
    assert_eq!(expr.evaluate_interval(&inputs), Some(false));

    // Overlap => unknown
    let inputs = [Interval::closed(0.0, 2.0), Interval::closed(2.0, 3.0)];
    assert_eq!(expr.evaluate_interval(&inputs), None);

    // Ne: disjoint => definitely true
    let mut expr_ne = Expr::new();
    let a = expr_ne.add_leaf(0);
    let b = expr_ne.add_leaf(1);
    expr_ne.set_root_compare(CompOp::Ne, a, b);
    let inputs = [Interval::closed(0.0, 1.0), Interval::closed(2.0, 3.0)];
    assert_eq!(expr_ne.evaluate_interval(&inputs), Some(true));
}

#[test]
fn comparison_tree_build_and_dependencies() {
    let numeric_variables = vec![
        NumericVariable::new("x0".into(), NumericType::Regular, -1),
        NumericVariable::new("x1".into(), NumericType::Regular, -1),
        NumericVariable::new("d2".into(), NumericType::Derived, -1),
        NumericVariable::new("d3".into(), NumericType::Derived, -1),
    ];

    // d2 = x0 + x1
    // d3 = d2 * x1
    let assignment_axioms = vec![
        AssignmentAxiom::new(2, CalOperator::Sum, 0, 1),
        AssignmentAxiom::new(3, CalOperator::Product, 2, 1),
    ];

    // d3 > x0
    let comparison_axioms = vec![ComparisonAxiom::new(
        0,
        3,
        0,
        ComparisonOperator::GreaterThan,
    )];

    let task = NumericRootTask::new(
        4,
        Metric::new(true, -1),
        vec![],
        numeric_variables,
        vec![],
        vec![],
        vec![],
        vec![0.0; 4],
        vec![],
        vec![],
        comparison_axioms,
        assignment_axioms,
        (0, 0),
    );

    let tree = ComparisonTree::from_task(&task, 0).unwrap();
    assert_eq!(tree.op, CompOp::Gt);
    assert_eq!(tree.left_numeric_var_id, 3);
    assert_eq!(tree.right_numeric_var_id, 0);

    let deps = tree.regular_numeric_var_dependencies(&task);
    assert_eq!(deps, vec![0, 1]);

    match &tree.nodes[tree.left_root] {
        ComparisonTreeNode::Arith {
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
fn comparison_tree_cycle_detection() {
    let numeric_variables = vec![
        NumericVariable::new("x0".into(), NumericType::Regular, -1),
        NumericVariable::new("d1".into(), NumericType::Derived, -1),
    ];

    // d1 = d1 + x0 (cycle)
    let assignment_axioms = vec![AssignmentAxiom::new(1, CalOperator::Sum, 1, 0)];

    let comparison_axioms = vec![ComparisonAxiom::new(0, 1, 0, ComparisonOperator::Equal)];

    let task = NumericRootTask::new(
        4,
        Metric::new(true, -1),
        vec![],
        numeric_variables,
        vec![],
        vec![],
        vec![],
        vec![0.0; 2],
        vec![],
        vec![],
        comparison_axioms,
        assignment_axioms,
        (0, 0),
    );

    let err = ComparisonTree::from_task(&task, 0).unwrap_err();
    assert_eq!(
        err,
        ComparisonTreeBuildError::CycleDetected { numeric_var_id: 1 }
    );
}
