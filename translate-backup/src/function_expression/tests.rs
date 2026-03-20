use super::*;

#[test]
fn test_parse_constant() {
    let sexpr = SExpr::Atom("42".to_string());
    let expr = parse_functional_expression(&sexpr).unwrap();
    assert!(matches!(expr, FunctionalExpression::Constant(_)));
}

#[test]
fn test_parse_primitive() {
    let sexpr = SExpr::List(vec![
        SExpr::Atom("fuel".to_string()),
        SExpr::Atom("?v".to_string()),
    ]);
    let expr = parse_functional_expression(&sexpr).unwrap();
    assert!(matches!(expr, FunctionalExpression::Primitive(_)));
}

#[test]
fn test_parse_arithmetic() {
    let sexpr = SExpr::List(vec![
        SExpr::Atom("+".to_string()),
        SExpr::Atom("10".to_string()),
        SExpr::Atom("20".to_string()),
    ]);
    let expr = parse_functional_expression(&sexpr).unwrap();
    assert!(matches!(expr, FunctionalExpression::Arithmetic(_)));
}
