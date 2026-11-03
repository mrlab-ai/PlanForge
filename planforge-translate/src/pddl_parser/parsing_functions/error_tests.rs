use super::parse_task;
use crate::pddl_parser::lisp_parser::{ParseErrorKind, parse_nested_list_string};

fn task_error(domain: &str, problem: &str) -> crate::pddl_parser::lisp_parser::ParseError {
    let domain = parse_nested_list_string(domain).unwrap();
    let problem = parse_nested_list_string(problem).unwrap();
    parse_task(&domain, &problem).unwrap_err()
}

const DOMAIN: &str = "(define (domain d)\n (:predicates (p) (q ?x))\n (:action a :parameters () :precondition (p) :effect (q)))";

#[test]
fn malformed_input_is_an_error_with_its_line_not_a_panic() {
    for (problem, line, message) in [
        (
            "(define (problem p) (:domain d)\n (:init (p))\n (:goal))",
            3,
            "(:goal CONDITION)",
        ),
        (
            "(define (problem p) (:domain d)\n (:init (= (p)))\n (:goal (p)))",
            2,
            "(= FLUENT VALUE)",
        ),
        (
            "(define (problem p) (:domain d)\n (:init (p) (not (p)))\n (:goal (p)))",
            2,
            "asserts and denies",
        ),
        ("(define problem)", 1, "(problem NAME)"),
    ] {
        let error = task_error(DOMAIN, problem);
        assert_eq!(error.kind, ParseErrorKind::Malformed, "{error}");
        assert_eq!(error.line, Some(line), "{error}");
        assert!(error.message.contains(message), "{error}");
    }
}

#[test]
/// The innermost list still open at the end of the file is reported.
fn an_unbalanced_file_names_the_unclosed_list() {
    let error = parse_nested_list_string("(define (domain d)\n (:predicates (p)").unwrap_err();
    assert_eq!(error.line, Some(2));
    assert!(error.message.contains("no closing parenthesis"), "{error}");
}
