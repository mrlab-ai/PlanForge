use super::*;

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
