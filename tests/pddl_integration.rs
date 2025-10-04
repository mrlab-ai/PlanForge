// Integration test to force compilation of the `pddl` module and exercise basic APIs

#[test]
fn pddl_module_compiles_and_basic_behaviour() {
    use planners::translate::pddl::{Condition, Literal, Task};
    use planners::translate::pddl::tasks::Requirements;
    // Access f_expression types via module path
    use planners::translate::pddl::f_expression as f_expr_mod;
    use ordered_float::OrderedFloat;

    // PrimitiveNumericExpression: free variables and rename
    let pne = f_expr_mod::PrimitiveNumericExpression::new("f".to_string(), vec!["?x".to_string(), "a".to_string()], 'R');
    let free = pne.free_variables();
    assert!(free.contains("?x"));

    let mut ren = std::collections::HashMap::new();
    ren.insert("?x".to_string(), "?y".to_string());
    let pne2 = pne.rename_variables(&ren);
    assert_eq!(pne2.args[0], "?y");

    // NumericExpression simplification
    let sum = f_expr_mod::NumericExpression::Sum(Box::new(f_expr_mod::NumericExpression::NumericConstant(OrderedFloat(0.0))), Box::new(f_expr_mod::NumericExpression::NumericConstant(OrderedFloat(2.0))));
    assert_eq!(sum.simplified(), f_expr_mod::NumericExpression::NumericConstant(OrderedFloat(2.0)));

    // Literal / Condition basics
    let lit = Literal::new("at".to_string(), vec!["?x".to_string(), "room1".to_string()]);
    assert!(lit.free_variables().contains("?x"));
    let cond = Condition::Literal(lit);
    assert!(cond.free_variables().contains("?x"));

    // Task creation via Requirements
    let req = Requirements::new(vec![]).unwrap();
    let task = Task::new(
        "d".to_string(),
        "p".to_string(),
        req,
        vec![], vec![], vec![], vec![], vec![], vec![],
        Condition::Truth,
        vec![], vec![], None
    );
    assert_eq!(task.domain_name, "d");
}
