use super::*;
use std::fs;

#[test]
fn parse_satellite_domain_sexprs_smoke() {
    let s = fs::read_to_string("others/satellite/domain.pddl").expect("read pddl file");
    let sexprs = parse_sexprs(&s).expect("parse should succeed");
    assert!(!sexprs.is_empty());
    match &sexprs[0] {
        SExpr::List(items) => match &items[0] {
            SExpr::Atom(a) => assert_eq!(a.to_lowercase(), "define"),
            _ => panic!("expected atom define"),
        },
        _ => panic!("expected list"),
    }
}

#[test]
fn parse_nested_list_accepts_comments() {
    let input = "(define ; comment\n  (problem test))\n";
    let parsed = parse_nested_list(input).expect("nested list parse should succeed");
    assert!(!parsed.is_empty());
}

#[test]
fn tokenize_rejects_non_ascii_outside_comments() {
    let err = tokenize("(define ä)").expect_err("tokenizer should reject non-ascii");
    assert!(err.to_string().contains("Non-ASCII"));
}
