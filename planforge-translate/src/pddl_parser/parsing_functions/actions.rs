//! Actions, their effects and numeric assignments, axioms and the global
//! constraint.

use super::*;

pub(super) fn parse_assignment(form: Form<'_>) -> ParseResult<FunctionAssignment> {
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
pub(super) fn parse_effects(expr: &SExpr, types: &TypeHierarchy) -> ParseResult<EffectType> {
    if expr.expect_list("an effect")?.is_empty() {
        return Ok(EffectType::Conjunctive(ConjunctiveEffect::new(vec![])));
    }
    parse_effect(expr, types)
}

pub(super) fn parse_effect(expr: &SExpr, types: &TypeHierarchy) -> ParseResult<EffectType> {
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

pub(super) fn parse_action(form: Form<'_>, types: &TypeHierarchy) -> ParseResult<Action> {
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

pub(super) fn parse_global_constraint(form: Form<'_>, types: &TypeHierarchy) -> ParseResult<Axiom> {
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
pub(super) fn parse_axiom(form: Form<'_>, types: &TypeHierarchy) -> ParseResult<Axiom> {
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
