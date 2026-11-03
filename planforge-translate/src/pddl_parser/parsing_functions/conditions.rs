//! Conditions, numeric comparisons and the arithmetic expressions they
//! compare.

use super::*;

/// The terms `items` lists: object names and variables.
pub(super) fn parse_terms(items: Form<'_>) -> ParseResult<Vec<String>> {
    items
        .items
        .iter()
        .map(|item| item.expect_atom("a term").map(str::to_string))
        .collect()
}

pub(super) fn parse_condition(expr: &SExpr, types: &TypeHierarchy) -> ParseResult<Condition> {
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

pub(super) fn is_function_comparison(items: &[SExpr]) -> bool {
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
pub(super) fn parse_function_comparison(form: Form<'_>) -> ParseResult<Condition> {
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

pub(super) fn parse_expression(expr: &SExpr) -> ParseResult<FunctionalExpression> {
    fn classify_pne(symbol: String, args: Vec<String>) -> PrimitiveNumericExpression {
        if symbol == TOTAL_COST && args.is_empty() {
            PrimitiveNumericExpression::total_cost()
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
pub(super) fn parse_fluent(expr: &SExpr) -> ParseResult<PrimitiveNumericExpression> {
    match parse_expression(expr)? {
        FunctionalExpression::PrimitiveNumericExpression(fluent) => Ok(fluent),
        other => Err(ParseError::malformed(
            expr.line(),
            format!("expected a fluent, got the expression {other}"),
        )),
    }
}
