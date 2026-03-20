use super::*;

fn const_atom(pred: &str, args: &[&str]) -> Atom {
    Atom {
        predicate: pred.to_string(),
        args: args.iter().map(|s| Arg::Const((*s).to_string())).collect(),
    }
}

#[test]
fn join_rule_produces_effect() {
    // r(X) :- p(X), q(X) with facts p(a), q(a) -> r(a)
    let facts = vec![const_atom("p", &["a"]), const_atom("q", &["a"])];
    let spec = RuleSpec {
        rtype: "join".to_string(),
        effect: SymAtom::new("r", vec!["?x"]),
        conditions: vec![SymAtom::new("p", vec!["?x"]), SymAtom::new("q", vec!["?x"])],
    };
    let mut rules = convert_rules(&[spec]);
    let model = compute_model(&mut rules, &facts);
    assert!(model
        .iter()
        .any(|a| a.predicate == "r" && matches!(&a.args[0], Arg::Const(s) if s=="a")));
}

#[test]
fn product_rule_crosses_bindings() {
    // r(X,Y) :- p(X), q(Y) with p(a), q(b) -> r(a,b)
    let facts = vec![const_atom("p", &["a"]), const_atom("q", &["b"])];
    let spec = RuleSpec {
        rtype: "product".to_string(),
        effect: SymAtom::new("r", vec!["?x", "?y"]),
        conditions: vec![SymAtom::new("p", vec!["?x"]), SymAtom::new("q", vec!["?y"])],
    };
    let mut rules = convert_rules(&[spec]);
    let model = compute_model(&mut rules, &facts);
    assert!(model.iter().any(|a| a.predicate == "r"
        && matches!((&a.args[0], &a.args[1]), (Arg::Const(x), Arg::Const(y)) if x=="a" && y=="b")));
}

#[test]
fn project_rule_projects() {
    // r(X) :- p(X, ?z) with p(a,b) -> r(a)
    let facts = vec![const_atom("p", &["a", "b"])];
    let spec = RuleSpec {
        rtype: "project".to_string(),
        effect: SymAtom::new("r", vec!["?x"]),
        conditions: vec![SymAtom::new("p", vec!["?x", "?z"])],
    };
    let mut rules = convert_rules(&[spec]);
    let model = compute_model(&mut rules, &facts);
    assert!(model
        .iter()
        .any(|a| a.predicate == "r" && matches!(&a.args[0], Arg::Const(s) if s=="a")));
}
// End of build_model.rs
