use super::*;

#[test]
fn print_nested_list_emits_define() {
    let expr = SExpr::List(vec![
        SExpr::Atom("define".to_string()),
        SExpr::List(vec![
            SExpr::Atom("problem".to_string()),
            SExpr::Atom("p1".to_string()),
        ]),
    ]);
    let printed = print_nested_list(&expr);
    assert!(printed.contains("define"));
    assert!(printed.contains("problem"));
}
