use super::{TypeHierarchy, parse_action, parse_domain_pddl, parse_task_pddl};
use crate::pddl::f_expression::FunctionalExpression;
use crate::pddl::tasks::{DomainDefinition, ProblemDefinition};
use crate::pddl_parser::lisp_parser::{ParseError, ParseErrorKind, parse_nested_list_string};

fn try_parse_problem(problem: &str) -> Result<ProblemDefinition, ParseError> {
    let form = parse_nested_list_string(problem).expect("test problem is valid S-expression");
    parse_task_pddl(&form, &object_types())
}

fn parse_problem(problem: &str) -> ProblemDefinition {
    try_parse_problem(problem).unwrap()
}

/// The error parsing `problem` reports, which must be of `kind` and say
/// `message`.
fn problem_error(problem: &str, kind: ParseErrorKind, message: &str) -> ParseError {
    let error = try_parse_problem(problem).unwrap_err();
    assert_eq!(error.kind, kind, "{error}");
    assert!(error.message.contains(message), "{error}");
    error
}

fn try_parse_domain(domain: &str) -> Result<DomainDefinition, ParseError> {
    let form = parse_nested_list_string(domain).expect("test domain is valid S-expression");
    parse_domain_pddl(&form)
}

fn domain_error(domain: &str, kind: ParseErrorKind, message: &str) -> ParseError {
    let error = try_parse_domain(domain).unwrap_err();
    assert_eq!(error.kind, kind, "{error}");
    assert!(error.message.contains(message), "{error}");
    error
}

fn object_types() -> TypeHierarchy {
    TypeHierarchy::new(&[]).expect("the empty hierarchy is well-formed")
}

#[test]
fn rejects_unknown_problem_section() {
    problem_error(
        "(define (problem misspelled-goal)
                (:domain typo-test)
                (:init)
                (:goals (and)))",
        ParseErrorKind::Malformed,
        "unknown problem section :goals",
    );
}

#[test]
fn accepts_problem_requirements() {
    parse_problem(
        "(define (problem typed-problem)
                (:domain typing-test)
                (:requirements :typing)
                (:init)
                (:goal (and)))",
    );
}

#[test]
fn validates_problem_requirements() {
    problem_error(
        "(define (problem temporal-problem)
                (:domain temporal-test)
                (:requirements :durative-actions)
                (:init)
                (:goal (and)))",
        ParseErrorKind::Unsupported,
        "requirement :durative-actions",
    );
}

#[test]
fn rejects_unsupported_valid_problem_section_distinctly() {
    problem_error(
        "(define (problem constrained-problem)
                (:domain constraint-test)
                (:init)
                (:goal (and))
                (:constraints (always (p))))",
        ParseErrorKind::Unsupported,
        "problem section :constraints",
    );
}

#[test]
fn rejects_unknown_action_key() {
    let form = parse_nested_list_string(
        "(:action move
                :parameters ()
                :preconditions (and)
                :effect (and))",
    )
    .expect("test action is valid S-expression");
    let action = form.expect_list("an action").unwrap().rest(1);
    let error = parse_action(action, &object_types()).unwrap_err();
    assert_eq!(error.kind, ParseErrorKind::Malformed);
    assert!(
        error
            .message
            .contains("action move: unknown key :preconditions"),
        "{error}"
    );
    assert_eq!(error.line, Some(3));
}

#[test]
fn rejects_unknown_domain_section() {
    domain_error(
        "(define (domain typo-test)
                (:requirements :strips)
                (:predicate (p)))",
        ParseErrorKind::Malformed,
        "unknown domain section :predicate",
    );
}

#[test]
fn rejects_unsupported_valid_domain_section_distinctly() {
    domain_error(
        "(define (domain temporal-test)
                (:requirements :strips)
                (:durative-action wait
                    :parameters ()
                    :duration (= ?duration 1)
                    :condition (and)
                    :effect (and)))",
        ParseErrorKind::Unsupported,
        "domain section :durative-action",
    );
}

#[test]
fn rejects_unrecognized_metric_direction() {
    problem_error(
        "(define (problem bad-direction)
                (:domain metric-test)
                (:init (= (total-cost) 0))
                (:goal (and))
                (:metric minimise (total-cost)))",
        ParseErrorKind::Malformed,
        "unrecognized metric direction `minimise`",
    );
}

#[test]
fn accepts_maximize_metric_direction() {
    let problem = parse_problem(
        "(define (problem maximize-direction)
                (:domain metric-test)
                (:init (= (total-cost) 0))
                (:goal (and))
                (:metric maximize (total-cost)))",
    );

    assert_eq!(problem.metric.expect("metric was parsed").0, ">");
}

#[test]
fn accepts_compound_metric_expression() {
    let problem = parse_problem(
        "(define (problem compound-metric)
                (:domain metric-test)
                (:init (= (fuel) 1) (= (time) 2))
                (:goal (and))
                (:metric minimize (+ (fuel) (time))))",
    );

    assert!(matches!(
        problem.metric,
        Some((direction, FunctionalExpression::ArithmeticExpression(_)))
            if direction == "<"
    ));
}

fn parse_domain(domain: &str) -> DomainDefinition {
    try_parse_domain(domain).unwrap()
}

/// `settlersnumeric` declares `(:types place vehicle - store resource)`:
/// `store` is only ever used as a supertype, which declares it.
#[test]
fn a_type_used_only_as_a_supertype_is_declared_below_object() {
    let supertypes = parse_domain(
        "(define (domain implicit-supertype)
                (:requirements :typing)
                (:types place vehicle - store resource)
                (:predicates (at ?v - vehicle ?p - place) (holds ?s - store))
                (:action a :parameters (?s - store ?v - vehicle)
                    :precondition (holds ?s) :effect (not (holds ?s))))",
    );
    assert_eq!(supertypes.types.ancestors("store"), vec!["store", "object"]);
    assert_eq!(
        supertypes.types.ancestors("place"),
        vec!["place", "store", "object"]
    );
    assert_eq!(
        supertypes.types.ancestors("vehicle"),
        vec!["vehicle", "store", "object"]
    );
    assert_eq!(
        supertypes.types.ancestors("resource"),
        vec!["resource", "object"]
    );
}

#[test]
fn an_either_type_declares_a_predicate_argument() {
    let domain = parse_domain(
        "(define (domain either-declaration)
                (:requirements :typing)
                (:types a b)
                (:predicates (p ?x - (either a b))))",
    );
    assert_eq!(domain.predicates.len(), 1);
}

#[test]
fn an_either_type_on_an_action_parameter_is_rejected() {
    domain_error(
        "(define (domain either-parameter)
                (:requirements :typing)
                (:types a b)
                (:predicates (p ?x - a))
                (:action act :parameters (?x - (either a b))
                    :precondition () :effect (p ?x)))",
        ParseErrorKind::Unsupported,
        "(either ...) types are supported in predicate and function declarations only",
    );
}

#[test]
fn a_cyclic_type_hierarchy_is_rejected() {
    let form = parse_nested_list_string(
        "(define (domain cyclic)
                (:requirements :typing)
                (:types a - b b - a))",
    )
    .expect("test domain is valid S-expression");
    let error = parse_domain_pddl(&form).unwrap_err().to_string();
    assert!(error.contains("is its own supertype"), "{error}");
}

#[test]
fn an_explicit_declaration_of_a_supertype_is_kept() {
    let supertypes = parse_domain(
        "(define (domain explicit-supertype)
                (:requirements :typing)
                (:types place - store store - thing thing)
                (:predicates (p ?x - place)))",
    );
    assert_eq!(
        supertypes.types.ancestors("place"),
        vec!["place", "store", "thing", "object"]
    );
    assert_eq!(
        supertypes.types.ancestors("store"),
        vec!["store", "thing", "object"]
    );
}

#[test]
fn rejects_undeclared_predicate_parameter_type() {
    domain_error(
        "(define (domain bad-predicate-type)
                (:requirements :typing)
                (:types place)
                (:predicates (at ?x - typoo)))",
        ParseErrorKind::Malformed,
        "undeclared type typoo",
    );
}

#[test]
fn rejects_undeclared_action_parameter_type() {
    domain_error(
        "(define (domain bad-action-type)
                (:requirements :typing)
                (:types place)
                (:predicates (p))
                (:action a :parameters (?x - typoo) :effect (p)))",
        ParseErrorKind::Malformed,
        "undeclared type typoo",
    );
}

#[test]
fn rejects_undeclared_universal_parameter_type() {
    domain_error(
        "(define (domain bad-forall-type)
                (:requirements :adl)
                (:types place)
                (:predicates (p ?x - place))
                (:action a :parameters ()
                    :precondition (forall (?x - typoo) (p ?x))
                    :effect (p nowhere)))",
        ParseErrorKind::Malformed,
        "undeclared type typoo",
    );
}

#[test]
fn rejects_undeclared_existential_parameter_type() {
    domain_error(
        "(define (domain bad-exists-type)
                (:requirements :adl)
                (:types place)
                (:predicates (p ?x - place))
                (:action a :parameters ()
                    :precondition (exists (?x - typoo) (p ?x))
                    :effect (p nowhere)))",
        ParseErrorKind::Malformed,
        "undeclared type typoo",
    );
}
