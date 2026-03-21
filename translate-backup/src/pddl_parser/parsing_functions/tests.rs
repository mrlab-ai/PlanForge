use super::*;

#[test]
fn parse_typed_list_groups_items_by_dash_type() {
    let input = vec![
        "?x".to_string(),
        "?y".to_string(),
        "-".to_string(),
        "satellite".to_string(),
        "?z".to_string(),
    ];
    let result = parse_typed_list(&input, true, "object");
    assert_eq!(
        result,
        vec![
            ("?x".to_string(), "satellite".to_string()),
            ("?y".to_string(), "satellite".to_string()),
            ("?z".to_string(), "object".to_string()),
        ]
    );
}

#[test]
fn set_supertypes_builds_transitive_chain() {
    let mut types = vec![
        Type {
            name: "object".to_string(),
            basetype_name: None,
            supertype_names: vec![],
        },
        Type {
            name: "vehicle".to_string(),
            basetype_name: Some("object".to_string()),
            supertype_names: vec![],
        },
        Type {
            name: "truck".to_string(),
            basetype_name: Some("vehicle".to_string()),
            supertype_names: vec![],
        },
    ];
    set_supertypes(&mut types);
    let truck = types.iter().find(|item| item.name == "truck").unwrap();
    assert!(truck.supertype_names.contains(&"vehicle".to_string()));
    assert!(truck.supertype_names.contains(&"object".to_string()));
}

#[test]
fn parse_predicate_builds_typed_arguments() {
    let predicate = parse_predicate(&[
        SExpr::Atom("calibrated".to_string()),
        SExpr::Atom("?i".to_string()),
        SExpr::Atom("-".to_string()),
        SExpr::Atom("instrument".to_string()),
    ])
    .expect("predicate parse should succeed");
    assert_eq!(predicate.name, "calibrated");
    assert_eq!(predicate.get_arity(), 1);
    assert_eq!(predicate.arguments[0].type_name, "instrument");
}

#[test]
fn parse_function_builds_typed_arguments() {
    let function = parse_function(
        &[
            SExpr::Atom("fuel".to_string()),
            SExpr::Atom("?s".to_string()),
            SExpr::Atom("-".to_string()),
            SExpr::Atom("satellite".to_string()),
        ],
        "number",
    )
    .expect("function parse should succeed");
    assert_eq!(function.name, "fuel");
    assert_eq!(function.arguments.len(), 1);
    assert_eq!(function.arguments[0].type_name, "satellite");
    assert_eq!(function.type_name.as_deref(), Some("number"));
}

#[test]
fn parse_action_reads_parameters_precondition_and_effect() {
    let action = parse_action(&[
        SExpr::Atom(":action".to_string()),
        SExpr::Atom("turn_to".to_string()),
        SExpr::Atom(":parameters".to_string()),
        SExpr::List(vec![
            SExpr::Atom("?s".to_string()),
            SExpr::Atom("-".to_string()),
            SExpr::Atom("satellite".to_string()),
            SExpr::Atom("?d_new".to_string()),
            SExpr::Atom("-".to_string()),
            SExpr::Atom("direction".to_string()),
        ]),
        SExpr::Atom(":precondition".to_string()),
        SExpr::List(vec![
            SExpr::Atom("and".to_string()),
            SExpr::List(vec![
                SExpr::Atom("pointing".to_string()),
                SExpr::Atom("?s".to_string()),
                SExpr::Atom("?d_new".to_string()),
            ]),
        ]),
        SExpr::Atom(":effect".to_string()),
        SExpr::List(vec![
            SExpr::Atom("and".to_string()),
            SExpr::List(vec![
                SExpr::Atom("increase".to_string()),
                SExpr::Atom("total-cost".to_string()),
                SExpr::Atom("1".to_string()),
            ]),
        ]),
    ])
    .expect("action parse should succeed");

    assert_eq!(action.name, "turn_to");
    assert_eq!(action.parameters.len(), 2);
    assert_eq!(
        action.parameters[0],
        ("?s".to_string(), Some("satellite".to_string()))
    );
    assert!(action.precond.is_some());
    assert!(action.effect.is_some());
}
