/// Main PDDL parsing functions that convert S-expressions into PDDL AST.
mod numeric_literals;

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

/// What a typed list declares, which decides what it may contain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypedList {
    /// The arguments of a predicate or function: variables whose types only
    /// document the declaration, so `(either ...)` is allowed.
    Declaration,
    /// The parameters of an action, axiom or quantifier: variables whose types
    /// decide what grounding binds them to.
    Parameters,
    /// Constants or objects.
    Objects,
}

/// Parses a list of typed items, `?x ?y - type ?z - type2`; an item without
/// a type is an `object`.
///
/// `(either t1 t2)` is a type only in a declaration. On a parameter it would
/// need a disjunction of type atoms in grounding, and on an object it names no
/// type, so both are rejected rather than grounded as a type nothing has.
pub fn parse_typed_list(
    items: Form<'_>,
    list: TypedList,
    types: &TypeHierarchy,
) -> ParseResult<Vec<TypedObject>> {
    let mut result = vec![];
    let mut untyped_items: Vec<&SExpr> = vec![];
    let mut i = 0;
    while i < items.len() {
        let item = &items.items[i];
        if item.atom() != Some("-") {
            untyped_items.push(item);
            i += 1;
            continue;
        }
        let type_expr = items.get(i + 1, "a type after `-`")?;
        let type_name = match type_expr {
            SExpr::Atom(name, line) => {
                check_type_declared(types, name, *line)?;
                name.clone()
            }
            SExpr::List(..) => {
                let parts = type_expr.expect_list("a type")?;
                if parts.head("`either`")? != "either" {
                    return Err(ParseError::malformed(
                        parts.line,
                        format!("a type is a single word or (either WORD...), got {type_expr}"),
                    ));
                }
                if list != TypedList::Declaration {
                    return Err(ParseError::unsupported(
                        parts.line,
                        format!(
                            "(either ...) types are supported in predicate and function declarations only, got {type_expr}"
                        ),
                    ));
                }
                if parts.len() < 2 {
                    return Err(ParseError::malformed(
                        parts.line,
                        "an either type must name at least one type",
                    ));
                }
                let mut names = Vec::new();
                for part in parts.rest(1).items {
                    let name = part.expect_atom("a type name")?;
                    check_type_declared(types, name, part.line())?;
                    names.push(name);
                }
                format!("either-{}", names.join("-"))
            }
        };
        for item in untyped_items.drain(..) {
            result.push(typed_item(item, &type_name, list)?);
        }
        i += 2;
    }
    for item in untyped_items {
        result.push(typed_item(item, "object", list)?);
    }
    Ok(result)
}

fn check_type_declared(types: &TypeHierarchy, name: &str, line: u32) -> ParseResult<()> {
    if types.contains(name) {
        Ok(())
    } else {
        Err(ParseError::malformed(
            line,
            format!("undeclared type {name}"),
        ))
    }
}

fn typed_item(item: &SExpr, type_name: &str, list: TypedList) -> ParseResult<TypedObject> {
    let name = item.expect_atom("a name")?;
    if list != TypedList::Objects && !name.starts_with('?') {
        return Err(ParseError::malformed(
            item.line(),
            format!("expected a variable, got {name}"),
        ));
    }
    Ok(TypedObject::new(name, type_name))
}

pub fn parse_type_list(items: Form<'_>) -> ParseResult<Vec<Type>> {
    let mut result = vec![];
    let mut untyped_items: Vec<&str> = vec![];
    let mut i = 0;
    while i < items.len() {
        let item = items.items[i].expect_atom("a type name")?;
        if item == "-" {
            let basetype_name = items
                .get(i + 1, "a supertype after `-`")?
                .expect_atom("a supertype")?;
            for name in untyped_items.drain(..) {
                result.push(Type::new(name, Some(basetype_name)));
            }
            i += 2;
        } else {
            untyped_items.push(item);
            i += 1;
        }
    }
    for name in untyped_items {
        result.push(Type::new(name, Some("object")));
    }

    // A type that only appears as a supertype, as `store` in
    // `place vehicle - store`, is declared by that use: it is a type in its own
    // right, directly below `object`.
    let mut implicit: Vec<String> = Vec::new();
    for supertype in result.iter().filter_map(|t| t.basetype_name.as_deref()) {
        if supertype != "object"
            && !result.iter().any(|t| t.name == supertype)
            && !implicit.iter().any(|name| name == supertype)
        {
            implicit.push(supertype.to_string());
        }
    }
    for name in &implicit {
        result.push(Type::new(name, Some("object")));
    }
    Ok(result)
}

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

/// The terms `items` lists: object names and variables.
fn parse_terms(items: Form<'_>) -> ParseResult<Vec<String>> {
    items
        .items
        .iter()
        .map(|item| item.expect_atom("a term").map(str::to_string))
        .collect()
}

pub fn parse_condition(expr: &SExpr, types: &TypeHierarchy) -> ParseResult<Condition> {
    // A condition is a parenthesised form in PDDL, so a bare atom is
    // malformed input; reading it as no condition would make the operator
    // holding a misspelled condition unconditionally applicable.
    if let SExpr::Atom(atom, line) = expr {
        return Err(ParseError::malformed(
            *line,
            format!(
                "expected a parenthesised condition, got the bare atom {atom}; \
                 a nullary predicate is written ({atom})"
            ),
        ));
    }
    let form = expr.expect_list("a condition")?;
    if form.is_empty() {
        return Ok(Condition::Truth);
    }
    let tag = form.head("a condition keyword or predicate")?;
    let parts = |types| -> ParseResult<Vec<Condition>> {
        form.rest(1)
            .items
            .iter()
            .map(|item| parse_condition(item, types))
            .collect()
    };
    Ok(match tag {
        "and" => Condition::Conjunction(Conjunction::new(parts(types)?)),
        "or" => Condition::Disjunction(Disjunction::new(parts(types)?)),
        "not" => {
            form.expect_len(2, "not takes exactly one argument")?;
            let inner = form.get(1, "a negated condition")?;
            let inner_form = inner.expect_list("a negated condition")?;
            if is_function_comparison(inner_form.items) {
                match parse_function_comparison(inner_form)? {
                    Condition::FunctionComparison(comparison) => {
                        Condition::NegatedFunctionComparison(comparison)
                    }
                    other => unreachable!("a function comparison parses to one, got {other}"),
                }
            } else if matches!(
                inner_form.items.first().and_then(SExpr::atom),
                Some("and" | "or" | "not" | "imply" | "forall" | "exists")
            ) {
                // `not` around a compound condition, which ADL domains use
                // freely: `(not (exists (?a - airplane) ...))`.
                // `Condition::negate` puts any condition into negation normal
                // form.
                parse_condition(inner, types)?.negate()
            } else {
                let predicate = inner_form.head("a predicate")?.to_string();
                Condition::NegatedAtom(NegatedAtom::new(
                    predicate,
                    parse_terms(inner_form.rest(1))?,
                ))
            }
        }
        "imply" => {
            form.expect_len(3, "imply takes exactly two arguments")?;
            let left = parse_condition(form.get(1, "an antecedent")?, types)?;
            let right = parse_condition(form.get(2, "a consequent")?, types)?;
            // `(imply a b)` is `(or (not a) b)`, for any antecedent.
            Condition::Disjunction(Disjunction::new(vec![left.negate(), right]))
        }
        "forall" | "exists" => {
            form.expect_len(3, &format!("{tag} takes a parameter list and a condition"))?;
            let parameters = parse_typed_list(
                form.get(1, "a parameter list")?
                    .expect_list("a parameter list")?,
                TypedList::Parameters,
                types,
            )?;
            let body = parse_condition(form.get(2, "a quantified condition")?, types)?;
            if tag == "forall" {
                Condition::UniversalCondition(UniversalCondition::new(parameters, vec![body]))
            } else {
                Condition::ExistentialCondition(ExistentialCondition::new(parameters, vec![body]))
            }
        }
        // `=` is two different operators in PDDL: between numeric expressions
        // it is a comparison; between terms it is object equality, which this
        // translator encodes as the predicate `=` and supports by adding
        // `(= o o)` to the initial state for every object.
        "=" if !is_function_comparison(form.items) => {
            form.expect_len(3, "object equality takes exactly two terms")?;
            Condition::Atom(Atom::new("=".to_string(), parse_terms(form.rest(1))?))
        }
        "<" | "<=" | "=" | ">=" | ">" => {
            if !is_function_comparison(form.items) {
                return Err(ParseError::malformed(
                    form.line,
                    format!("{tag} compares numeric expressions, got {expr}"),
                ));
            }
            parse_function_comparison(form)?
        }
        _ => Condition::Atom(Atom::new(tag.to_string(), parse_terms(form.rest(1))?)),
    })
}

fn is_function_comparison(items: &[SExpr]) -> bool {
    fn expression_looks_numeric(expr: &SExpr) -> bool {
        match expr {
            SExpr::Atom(atom, _) => atom.parse::<f64>().is_ok(),
            SExpr::List(items, _) => items.first().is_some_and(|head| head.atom().is_some()),
        }
    }

    match items.first().and_then(SExpr::atom) {
        Some("<" | "<=" | ">=" | ">") => true,
        Some("=") => {
            items.len() == 3
                && (expression_looks_numeric(&items[1]) || expression_looks_numeric(&items[2]))
        }
        _ => false,
    }
}

/// Parses a function comparison like `(< (f x) 5)` into `(f x) - 5 < 0`.
fn parse_function_comparison(form: Form<'_>) -> ParseResult<Condition> {
    let comparator = form.head("a comparator")?.to_string();
    form.expect_len(3, &format!("{comparator} compares exactly two expressions"))?;
    let parts = form
        .rest(1)
        .items
        .iter()
        .map(parse_expression)
        .collect::<ParseResult<Vec<_>>>()?;
    let difference = FunctionalExpression::ArithmeticExpression(ArithmeticExpression::new(
        "-".to_string(),
        parts,
    ));
    let zero = FunctionalExpression::NumericConstant(NumericConstant::new(0.0));
    Ok(Condition::FunctionComparison(Comparison::new(
        comparator,
        vec![difference, zero],
    )))
}

pub fn parse_expression(expr: &SExpr) -> ParseResult<FunctionalExpression> {
    fn classify_pne(symbol: String, args: Vec<String>) -> PrimitiveNumericExpression {
        if symbol == "total-cost" && args.is_empty() {
            PrimitiveNumericExpression::with_type(symbol, args, 'I')
        } else {
            PrimitiveNumericExpression::new(symbol, args)
        }
    }

    match expr {
        SExpr::Atom(text, _) => Ok(match text.parse::<f64>() {
            Ok(value) => FunctionalExpression::NumericConstant(NumericConstant::new(value)),
            // A function symbol with no arguments.
            Err(_) => {
                FunctionalExpression::PrimitiveNumericExpression(classify_pne(text.clone(), vec![]))
            }
        }),
        SExpr::List(..) => {
            let form = expr.expect_list("a numeric expression")?;
            let tag = form.head("an arithmetic operator or function")?;
            Ok(match tag {
                "-" if form.len() == 2 => {
                    FunctionalExpression::AdditiveInverse(AdditiveInverse::new(vec![
                        parse_expression(&form.items[1])?,
                    ]))
                }
                "+" | "-" | "*" | "/" => {
                    let parts = form
                        .rest(1)
                        .items
                        .iter()
                        .map(parse_expression)
                        .collect::<ParseResult<Vec<_>>>()?;
                    FunctionalExpression::ArithmeticExpression(ArithmeticExpression::new(
                        tag.to_string(),
                        parts,
                    ))
                }
                _ => FunctionalExpression::PrimitiveNumericExpression(classify_pne(
                    tag.to_string(),
                    parse_terms(form.rest(1))?,
                )),
            })
        }
    }
}

/// The fluent `expr` names, as the target of an assignment or numeric init.
fn parse_fluent(expr: &SExpr) -> ParseResult<PrimitiveNumericExpression> {
    match parse_expression(expr)? {
        FunctionalExpression::PrimitiveNumericExpression(fluent) => Ok(fluent),
        other => Err(ParseError::malformed(
            expr.line(),
            format!("expected a fluent, got the expression {other}"),
        )),
    }
}

pub fn parse_assignment(form: Form<'_>) -> ParseResult<FunctionAssignment> {
    let tag = form.head("an assignment operator")?;
    let symbol = match tag {
        "assign" => "=",
        "increase" => "+",
        "decrease" => "-",
        "scale-up" => "*",
        "scale-down" => "/",
        _ => {
            return Err(ParseError::malformed(
                form.line,
                format!("unknown assignment operator {tag}"),
            ));
        }
    };
    form.expect_len(3, &format!("{tag} takes a fluent and an expression"))?;
    let fluent = parse_fluent(&form.items[1])?;
    let expression = parse_expression(&form.items[2])?;
    Ok(FunctionAssignment::new(
        symbol.to_string(),
        fluent,
        expression,
    ))
}

/// Parses the effects section.
pub fn parse_effects(expr: &SExpr, types: &TypeHierarchy) -> ParseResult<EffectType> {
    if expr.expect_list("an effect")?.is_empty() {
        return Ok(EffectType::Conjunctive(ConjunctiveEffect::new(vec![])));
    }
    parse_effect(expr, types)
}

fn parse_effect(expr: &SExpr, types: &TypeHierarchy) -> ParseResult<EffectType> {
    let form = expr.expect_list("an effect")?;
    let tag = form.head("an effect keyword or predicate")?;
    Ok(match tag {
        "and" => EffectType::Conjunctive(ConjunctiveEffect::new(
            form.rest(1)
                .items
                .iter()
                .map(|item| parse_effect(item, types))
                .collect::<ParseResult<_>>()?,
        )),
        "not" => {
            form.expect_len(2, "not takes exactly one atom")?;
            let inner = form.items[1].expect_list("a negated atom")?;
            EffectType::Simple(SimpleEffect::new(Condition::NegatedAtom(NegatedAtom::new(
                inner.head("a predicate")?.to_string(),
                parse_terms(inner.rest(1))?,
            ))))
        }
        "when" => {
            form.expect_len(3, "when takes a condition and an effect")?;
            let condition = parse_condition(&form.items[1], types)?;
            let effect = parse_effect(&form.items[2], types)?;
            EffectType::Conditional(ConditionalEffect::new(condition, effect))
        }
        "forall" => {
            form.expect_len(3, "forall takes a parameter list and an effect")?;
            let parameters = parse_typed_list(
                form.items[1].expect_list("a parameter list")?,
                TypedList::Parameters,
                types,
            )?;
            let effect = parse_effect(&form.items[2], types)?;
            EffectType::Universal(UniversalEffect::new(parameters, effect))
        }
        "assign" | "increase" | "decrease" | "scale-up" | "scale-down" => {
            EffectType::Numeric(NumericEffect::new(parse_assignment(form)?))
        }
        // Anything else is an add effect, whose arguments are terms.
        _ => EffectType::Simple(SimpleEffect::new(Condition::Atom(Atom::new(
            tag.to_string(),
            parse_terms(form.rest(1))?,
        )))),
    })
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

pub fn parse_action(form: Form<'_>, types: &TypeHierarchy) -> ParseResult<Action> {
    let name = form.head("an action name")?.to_string();
    let mut parameters = vec![];
    let mut precondition = Condition::Truth;
    let mut effect_type: Option<EffectType> = None;
    let mut cost: Option<FunctionAssignment> = None;

    for (key, value) in keyed_values(
        form.rest(1),
        &format!("action {name}"),
        &[":parameters", ":precondition", ":effect"],
    )? {
        match key {
            ":parameters" => {
                parameters = parse_typed_list(
                    value.expect_list("a parameter list")?,
                    TypedList::Parameters,
                    types,
                )?;
            }
            ":precondition" => precondition = parse_condition(value, types)?,
            ":effect" => {
                let (remaining, action_cost) = parse_effects(value, types)?.extract_cost();
                effect_type = Some(remaining);
                cost = action_cost;
            }
            _ => unreachable!("keyed_values admits only the listed keys"),
        }
    }

    let num_external = parameters.len();
    let mut action = Action::new(name, parameters, num_external, precondition, vec![], cost);
    // An action may declare no `:effect` at all: a legal, if pointless, no-op.
    if let Some(effect) = &effect_type {
        for (params, condition, kind) in effect.normalize() {
            match kind {
                EffectKind::Literal(literal) => {
                    action.effects.push(Effect::new(params, condition, literal));
                }
                EffectKind::Numeric(assignment) => {
                    action.assign_effects.push((params, condition, assignment));
                }
            }
        }
    }
    Ok(action)
}

pub fn parse_global_constraint(form: Form<'_>, types: &TypeHierarchy) -> ParseResult<Axiom> {
    let name = form.head("a global constraint name")?.to_string();
    let mut parameters = vec![];
    let mut condition = Condition::Truth;
    for (key, value) in keyed_values(
        form.rest(1),
        &format!("global constraint {name}"),
        &[":parameters", ":condition"],
    )? {
        match key {
            ":parameters" => {
                parameters = parse_typed_list(
                    value.expect_list("a parameter list")?,
                    TypedList::Parameters,
                    types,
                )?;
            }
            ":condition" => condition = parse_condition(value, types)?,
            _ => unreachable!("keyed_values admits only the listed keys"),
        }
    }
    let num_external = parameters.len();
    Ok(Axiom::new_global_constraint(
        name,
        parameters,
        num_external,
        condition,
    ))
}

/// Parses the body of a `(:derived (HEAD ?x - t ...) CONDITION)` block, i.e.
/// `form` is the block with `:derived` already stripped.
///
/// A derived predicate quantifies over exactly its head's arguments, so all of
/// them are external and the body may not introduce free variables of its own.
pub fn parse_axiom(form: Form<'_>, types: &TypeHierarchy) -> ParseResult<Axiom> {
    form.expect_len(2, "(:derived HEAD CONDITION) takes exactly two elements")?;
    let head = parse_predicate(
        form.items[0].expect_list("the head of a derived predicate, (NAME ?x - t ...),")?,
        types,
    )?;
    let condition = parse_condition(&form.items[1], types)?;
    let num_external = head.arguments.len();
    Ok(Axiom::new(
        head.name,
        head.arguments,
        num_external,
        condition,
    ))
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
mod tests {
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
}

#[cfg(test)]
mod error_tests {
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
}
