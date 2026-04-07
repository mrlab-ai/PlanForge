use super::*;

use planners_sas::numeric::axioms::{
    AssignmentAxiom, CalOperator, ComparisonAxiom, ComparisonOperator,
};
use planners_sas::numeric::numeric_task::{
    ExplicitFact, Metric, NumericRootTask, NumericType, NumericVariable,
};

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
    assert!(!expr.evaluate(&inputs));
}

#[test]
fn interval_add_preserves_exact_bounds() {
    let bounded = Interval::closed(3.0, 4.0);
    let unbounded = Interval::open(-3.0, f64::INFINITY);

    assert_eq!(
        bounded + unbounded,
        Interval::new(0.0, f64::INFINITY, false, false)
    );
    assert_eq!(
        unbounded + bounded,
        Interval::new(0.0, f64::INFINITY, false, false)
    );
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
fn interval_mul_preserves_closed_extrema() {
    let result = ArithOp::Mul.apply_interval(Interval::singleton(2.0), Interval::closed(3.0, 4.0));
    assert_eq!(result, Interval::closed(6.0, 8.0));
}

#[test]
fn interval_le_handles_open_touching_bounds() {
    let mut expr = Expr::new();
    let a = expr.add_leaf(0);
    let b = expr.add_leaf(1);
    expr.set_root_compare(CompOp::Le, a, b);

    let always_true = [Interval::open(1.0, 2.0), Interval::closed(2.0, 3.0)];
    assert_eq!(expr.evaluate_interval(&always_true), Some(true));

    let always_false = [Interval::closed(2.0, 3.0), Interval::open(1.0, 2.0)];
    assert_eq!(expr.evaluate_interval(&always_false), Some(false));
}

#[test]
fn comparison_tree_build_and_dependencies() {
    let numeric_variables = vec![
        NumericVariable::new("x0".into(), NumericType::Regular, None),
        NumericVariable::new("x1".into(), NumericType::Regular, None),
        NumericVariable::new("d2".into(), NumericType::Derived, None),
        NumericVariable::new("d3".into(), NumericType::Derived, None),
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
        Metric::new(true, None),
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
        ExplicitFact::new(0, 0),
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
        NumericVariable::new("x0".into(), NumericType::Regular, None),
        NumericVariable::new("d1".into(), NumericType::Derived, None),
    ];

    // d1 = d1 + x0 (cycle)
    let assignment_axioms = vec![AssignmentAxiom::new(1, CalOperator::Sum, 1, 0)];

    let comparison_axioms = vec![ComparisonAxiom::new(0, 1, 0, ComparisonOperator::Equal)];

    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
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
        ExplicitFact::new(0, 0),
    );

    let err = ComparisonTree::from_task(&task, 0).unwrap_err();
    assert_eq!(
        err,
        ComparisonTreeBuildError::CycleDetected { numeric_var_id: 1 }
    );
}

#[test]
fn comparison_tree_interval_evaluation_fills_derived_intervals() {
    let numeric_variables = vec![
        NumericVariable::new("x0".into(), NumericType::Regular, None),
        NumericVariable::new("c2".into(), NumericType::Constant, None),
        NumericVariable::new("d2".into(), NumericType::Derived, None),
        NumericVariable::new("d3".into(), NumericType::Derived, None),
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

    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        vec![],
        numeric_variables,
        vec![],
        vec![],
        vec![],
        vec![1.0, 2.0, 0.0, 0.0],
        vec![],
        vec![],
        comparison_axioms,
        assignment_axioms,
        ExplicitFact::new(0, 0),
    );

    let tree = ComparisonTree::from_task(&task, 0).unwrap();
    let mut intervals = vec![
        Interval::singleton(1.0),
        Interval::singleton(2.0),
        Interval::new(0.0, 0.0, false, false),
        Interval::new(0.0, 0.0, false, false),
    ];

    assert_eq!(tree.evaluate_interval_and_fill(&mut intervals), Some(true));
    assert_eq!(intervals[2], Interval::singleton(3.0));
    assert_eq!(intervals[3], Interval::singleton(6.0));
}
