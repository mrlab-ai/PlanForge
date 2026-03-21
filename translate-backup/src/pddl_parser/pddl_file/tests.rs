use super::*;
use crate::translate::pddl::{Domain, Problem};

#[test]
fn parse_satellite_files_smoke() {
    let task = PddlTask::from_files(
        std::path::Path::new("others/satellite/domain.pddl"),
        std::path::Path::new("others/satellite/pfile1.pddl"),
    )
    .expect("pddl task should load");
    assert!(!task.domain_forms.is_empty());
    assert!(!task.problem_forms.is_empty());
    assert!(task.summary().contains("forms"));
}

#[test]
fn parse_pddl_file_returns_toplevel_define_form() {
    let forms = parse_pddl_file(
        "domain",
        std::path::Path::new("others/satellite/domain.pddl"),
    )
    .expect("domain parse should succeed");
    assert!(!forms.is_empty());
    match &forms[0] {
        SExpr::List(items) => match &items[0] {
            SExpr::Atom(atom) => assert_eq!(atom, "define"),
            SExpr::List(_) => panic!("expected define atom"),
        },
        SExpr::Atom(_) => panic!("expected top-level define list"),
    }
}

#[test]
fn parse_satellite_ast_smoke() {
    let task = PddlTask::from_files(
        std::path::Path::new("others/satellite/domain.pddl"),
        std::path::Path::new("others/satellite/pfile1.pddl"),
    )
    .expect("pddl task should load");
    let domain = Domain::from_sexprs(&task.domain_forms).expect("domain should parse");
    let problem = Problem::from_sexprs(&task.problem_forms).expect("problem should parse");
    assert_eq!(domain.name, "satellite");
    assert!(!problem.name.is_empty());
}
