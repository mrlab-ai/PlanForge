//! The PDDL parser proper: S-expressions into the PDDL AST. This module
//! parses requirements, predicates, functions and the domain and problem
//! sections; [`types`], [`conditions`] and [`actions`] the parts they
//! contain.
mod actions;
mod conditions;
mod numeric_literals;
mod types;

use actions::*;
use conditions::*;
use types::*;

use super::lisp_parser::{Form, ParseError, ParseResult, SExpr};
use crate::pddl::actions::Action;
use crate::pddl::axioms::Axiom;
use crate::pddl::conditions::*;
use crate::pddl::effects::*;
use crate::pddl::f_expression::*;
use crate::pddl::functions::Function;
use crate::pddl::pddl_types::{Type, TypeHierarchy, TypedObject};
use crate::pddl::predicates::Predicate;
use crate::pddl::tasks::{DomainDefinition, ProblemDefinition};

/// The `:requirements` a task may declare.
///
/// A planner that silently ignores a declared requirement, such as
/// `:durative-actions`, answers a question nobody asked, so any other
/// requirement is refused.
const SUPPORTED_REQUIREMENTS: &[&str] = &[
    // Everything the Fast Downward translator accepts.
    ":strips",
    ":typing",
    ":negative-preconditions",
    ":disjunctive-preconditions",
    ":equality",
    ":existential-preconditions",
    ":universal-preconditions",
    ":quantified-preconditions",
    ":conditional-effects",
    ":adl",
    ":derived-predicates",
    ":action-costs",
    // Plus the numeric fragment, which is what this planner is for.
    ":numeric-fluents",
    ":fluents",
];

/// Rejects a task declaring a requirement this translator does not implement.
///
/// A declared requirement it *does* implement needs no action: support is a
/// property of the code, not of the list, and every construct is handled
/// wherever it appears whether or not the domain announced it.
fn check_requirements(requirements: Form<'_>) -> ParseResult<()> {
    for requirement in requirements.items {
        let name = requirement.expect_atom("a requirement")?;
        if !SUPPORTED_REQUIREMENTS.contains(&name) {
            return Err(ParseError::unsupported(
                requirement.line(),
                format!(
                    "requirement {name}; this translator implements {}",
                    SUPPORTED_REQUIREMENTS.join(" ")
                ),
            ));
        }
    }
    Ok(())
}

pub fn parse_predicate(form: Form<'_>, types: &TypeHierarchy) -> ParseResult<Predicate> {
    let name = form.head("a predicate name")?.to_string();
    let arguments = parse_typed_list(form.rest(1), TypedList::Declaration, types)?;
    Ok(Predicate::new(name, arguments))
}

pub fn parse_function(
    form: Form<'_>,
    type_name: &str,
    types: &TypeHierarchy,
) -> ParseResult<Function> {
    let name = form.head("a function name")?.to_string();
    let arguments = parse_typed_list(form.rest(1), TypedList::Declaration, types)?;
    Ok(Function::new(name, arguments, type_name.to_string()))
}

/// The `:key value` pairs of an action-like block, with `allowed` the keys it
/// may use.
fn keyed_values<'a>(
    form: Form<'a>,
    owner: &str,
    allowed: &[&str],
) -> ParseResult<Vec<(&'a str, &'a SExpr)>> {
    let mut pairs = Vec::new();
    let mut i = 0;
    while i < form.len() {
        let key = form.items[i].expect_atom("a keyword")?;
        if !allowed.contains(&key) {
            return Err(ParseError::malformed(
                form.items[i].line(),
                format!(
                    "{owner}: unknown key {key}, expected {}",
                    allowed.join(", ")
                ),
            ));
        }
        pairs.push((key, form.get(i + 1, &format!("a value for {key}"))?));
        i += 2;
    }
    Ok(pairs)
}

/// Combines parsed domain and problem S-expressions into a Task.
#[cfg(test)]
pub fn parse_task(domain_pddl: &SExpr, task_pddl: &SExpr) -> ParseResult<crate::pddl::tasks::Task> {
    let domain = parse_domain(domain_pddl)?;
    let problem = parse_problem(task_pddl, &domain.types)?;
    Ok(crate::pddl::tasks::Task::new(domain, problem))
}

/// Parses a `(define (domain ...))` form.
pub fn parse_domain(domain_pddl: &SExpr) -> ParseResult<DomainDefinition> {
    numeric_literals::validate(domain_pddl, false)?;
    parse_domain_pddl(domain_pddl)
}

/// Parses a `(define (problem ...))` form over the domain's `types`.
pub fn parse_problem(task_pddl: &SExpr, types: &TypeHierarchy) -> ParseResult<ProblemDefinition> {
    numeric_literals::validate(task_pddl, false)?;
    parse_task_pddl(task_pddl, types)
}

/// The sections of a `(define (KIND NAME) SECTION...)` form.
fn define_sections<'a>(expr: &'a SExpr, kind: &str) -> ParseResult<Form<'a>> {
    let form = expr.expect_list(&format!("(define ({kind} ...) ...)"))?;
    if form.head("define")? != "define" {
        return Err(ParseError::malformed(
            form.line,
            format!("expected (define ({kind} ...) ...)"),
        ));
    }
    let header = form
        .get(1, &format!("({kind} NAME)"))?
        .expect_list(&format!("({kind} NAME)"))?;
    if header.head(kind)? != kind {
        return Err(ParseError::malformed(
            header.line,
            format!("expected ({kind} NAME), got {}", form.items[1]),
        ));
    }
    Ok(form.rest(2))
}

/// Parses a `(define (domain ...))` form.
fn parse_domain_pddl(expr: &SExpr) -> ParseResult<DomainDefinition> {
    let sections = define_sections(expr, "domain")?;
    let mut type_list: Vec<Type> = vec![];
    let mut constants: Vec<TypedObject> = vec![];
    let mut predicates: Vec<Predicate> = vec![];
    let mut functions: Vec<Function> = vec![];
    let mut actions: Vec<Action> = vec![];
    let mut axioms: Vec<Axiom> = vec![];

    // Domain sections are not required to put `:types` before declarations
    // that use them, so build the complete type table before parsing any such
    // declaration.
    for item in sections.items {
        let section = item.expect_list("a domain section")?;
        if section.items.first().and_then(SExpr::atom) == Some(":types") {
            type_list = parse_type_list(section.rest(1))?;
        }
    }
    let types = TypeHierarchy::new(&type_list)
        .map_err(|error| ParseError::malformed(sections.line, error))?;

    for item in sections.items {
        let section = item.expect_list("a domain section")?;
        if section.is_empty() {
            continue;
        }
        let tag = section.head("a section keyword")?;
        let body = section.rest(1);
        match tag {
            ":requirements" => check_requirements(body)?,
            ":types" => {}
            ":constants" => constants = parse_typed_list(body, TypedList::Objects, &types)?,
            ":predicates" => {
                predicates = body
                    .items
                    .iter()
                    .map(|predicate| parse_predicate(predicate.expect_list("a predicate")?, &types))
                    .collect::<ParseResult<_>>()?;
            }
            ":functions" => functions = parse_function_list(body, &types)?,
            ":action" => actions.push(parse_action(body, &types)?),
            ":derived" => axioms.push(parse_axiom(body, &types)?),
            ":global-constraint" => axioms.push(parse_global_constraint(body, &types)?),
            ":durative-action" | ":constraints" | ":safety" => {
                return Err(ParseError::unsupported(
                    section.line,
                    format!("domain section {tag}"),
                ));
            }
            // PDDL 2.1's `(:axiom :vars ... :context ... :implies ...)` is a
            // different block shape, not a spelling of `:derived`.
            ":axiom" => {
                return Err(ParseError::unsupported(
                    section.line,
                    "domain section :axiom; write the derived predicate as \
                     `(:derived (NAME ?x - t) CONDITION)`",
                ));
            }
            unknown => {
                return Err(ParseError::malformed(
                    section.line,
                    format!("unknown domain section {unknown}"),
                ));
            }
        }
    }

    Ok(DomainDefinition {
        types,
        constants,
        predicates,
        functions,
        actions,
        axioms,
    })
}

/// Parses function declarations, each group optionally followed by
/// `- TYPE`; the default type is `number`.
fn parse_function_list(items: Form<'_>, types: &TypeHierarchy) -> ParseResult<Vec<Function>> {
    let mut result = vec![];
    let mut pending: Vec<&SExpr> = vec![];
    let mut i = 0;
    while i < items.len() {
        let item = &items.items[i];
        if item.atom() == Some("-") {
            let type_name = items
                .get(i + 1, "a function type after `-`")?
                .expect_atom("a function type")?;
            for function in pending.drain(..) {
                result.push(parse_function(
                    function.expect_list("a function")?,
                    type_name,
                    types,
                )?);
            }
            i += 2;
        } else {
            pending.push(item);
            i += 1;
        }
    }
    for function in pending {
        result.push(parse_function(
            function.expect_list("a function")?,
            "number",
            types,
        )?);
    }
    Ok(result)
}

fn parse_task_pddl(expr: &SExpr, types: &TypeHierarchy) -> ParseResult<ProblemDefinition> {
    let sections = define_sections(expr, "problem")?;
    let mut objects: Vec<TypedObject> = vec![];
    let mut init: Vec<Atom> = vec![];
    let mut denied: Vec<(Atom, u32)> = vec![];
    let mut num_init: Vec<FunctionAssignment> = vec![];
    let mut goal = Condition::Truth;
    let mut metric: Option<(String, FunctionalExpression)> = None;

    for item in sections.items {
        let section = item.expect_list("a problem section")?;
        if section.is_empty() {
            continue;
        }
        let tag = section.head("a section keyword")?;
        let body = section.rest(1);
        match tag {
            // The problem names the domain it belongs to; nothing checks
            // that against the domain file, which the caller chose.
            ":domain" => {}
            ":requirements" => check_requirements(body)?,
            ":objects" => objects = parse_typed_list(body, TypedList::Objects, types)?,
            ":init" => {
                for fact in body.items {
                    let fact_form = fact.expect_list("an initial fact")?;
                    if fact_form.is_empty() {
                        continue;
                    }
                    match fact_form.head("a predicate, `=` or `not`")? {
                        "=" => {
                            fact_form
                                .expect_len(3, "a numeric initial value is (= FLUENT VALUE)")?;
                            let fluent = parse_fluent(&fact_form.items[1])?;
                            let value = parse_expression(&fact_form.items[2])?;
                            num_init.push(FunctionAssignment::new("=".to_string(), fluent, value));
                        }
                        // A negative fact repeats what the closed world says,
                        // unless the initial state also asserts it.
                        "not" => {
                            fact_form.expect_len(2, "not takes exactly one atom")?;
                            let atom = fact_form.items[1].expect_list("a denied atom")?;
                            denied.push((
                                Atom::new(
                                    atom.head("a predicate")?.to_string(),
                                    parse_terms(atom.rest(1))?,
                                ),
                                atom.line,
                            ));
                        }
                        predicate => init.push(Atom::new(
                            predicate.to_string(),
                            parse_terms(fact_form.rest(1))?,
                        )),
                    }
                }
            }
            ":goal" => {
                section.expect_len(2, "(:goal CONDITION) has one condition")?;
                goal = parse_condition(&section.items[1], types)?;
            }
            ":metric" => {
                section.expect_len(3, "(:metric DIRECTION EXPRESSION)")?;
                let direction = section.items[1].expect_atom("a metric direction")?;
                let symbol = match direction {
                    "minimize" => "<",
                    "maximize" => ">",
                    unknown => {
                        return Err(ParseError::malformed(
                            section.items[1].line(),
                            format!(
                                "unrecognized metric direction `{unknown}`, expected `minimize` or `maximize`"
                            ),
                        ));
                    }
                };
                metric = Some((symbol.to_string(), parse_expression(&section.items[2])?));
            }
            ":constraints" | ":length" | ":situation" | ":expansion" => {
                return Err(ParseError::unsupported(
                    section.line,
                    format!("problem section {tag}"),
                ));
            }
            unknown => {
                return Err(ParseError::malformed(
                    section.line,
                    format!("unknown problem section {unknown}"),
                ));
            }
        }
    }
    if let Some((atom, line)) = denied.into_iter().find(|(atom, _)| init.contains(atom)) {
        return Err(ParseError::malformed(
            line,
            format!("the initial state both asserts and denies {atom}"),
        ));
    }

    Ok(ProblemDefinition {
        objects,
        init,
        num_init,
        goal,
        metric,
    })
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod error_tests;
