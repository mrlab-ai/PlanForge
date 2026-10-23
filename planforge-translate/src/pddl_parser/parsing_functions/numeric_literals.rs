use super::{SExpr, is_function_comparison};
use planforge_sas::numeric_task::InvalidNumericValue;

/// Validate only numeric expression positions: NaN remains a legal identifier
/// in object/function arguments and metadata, but never a numeric literal.
fn validate_numeric_expression(form: &SExpr) -> Result<(), InvalidNumericValue> {
    match form {
        SExpr::Atom(atom) => {
            if atom.parse::<f64>().is_ok_and(f64::is_nan) {
                return Err(InvalidNumericValue);
            }
        }
        SExpr::List(items) if items.first().is_some_and(|head| matches!(head, SExpr::Atom(operator) if matches!(operator.as_str(), "+" | "-" | "*" | "/"))) => {
            for argument in &items[1..] { validate_numeric_expression(argument)?; }
        }
        SExpr::List(_) => {} // A fluent's arguments are object identifiers.
    }
    Ok(())
}

pub(super) fn validate(form: &SExpr, effect: bool) -> Result<(), InvalidNumericValue> {
    let SExpr::List(items) = form else {
        return Ok(());
    };
    let head = items.first().and_then(|head| match head {
        SExpr::Atom(atom) => Some(atom.as_str()),
        SExpr::List(_) => None,
    });
    if is_function_comparison(items) {
        for argument in &items[1..] {
            validate_numeric_expression(argument)?;
        }
        return Ok(());
    }
    if head == Some(":metric") {
        if let Some(expression) = items.get(2) {
            validate_numeric_expression(expression)?;
        }
        return Ok(());
    }
    if effect {
        match head {
            Some("assign" | "increase" | "decrease" | "scale-up" | "scale-down") => {
                for argument in &items[1..] {
                    validate_numeric_expression(argument)?;
                }
                return Ok(());
            }
            Some("not") => return Ok(()), // A negative propositional effect.
            Some("when") => {
                if let Some(condition) = items.get(1) {
                    validate(condition, false)?;
                }
                if let Some(effect) = items.get(2) {
                    validate(effect, true)?;
                }
                return Ok(());
            }
            _ => {}
        }
    }
    let mut next_is_effect = false;
    for child in items {
        validate(child, effect || next_is_effect)?;
        next_is_effect = matches!(child, SExpr::Atom(tag) if tag == ":effect");
    }
    Ok(())
}

#[cfg(test)]
mod numeric_literal_tests {
    use super::super::parse_task;
    use super::*;
    use crate::pddl_parser::lisp_parser::parse_nested_list_string;

    #[test]
    fn numeric_nan_is_rejected_with_its_typed_error() {
        let domain = parse_nested_list_string("(define (domain d) (:functions (x)))").unwrap();
        let problem = parse_nested_list_string(
            "(define (problem p) (:domain d) (:init (= (x) NaN)) (:goal (> (x) 0)))",
        )
        .unwrap();
        assert!(
            parse_task(&domain, &problem)
                .unwrap_err()
                .is::<planforge_sas::numeric_task::InvalidNumericValue>()
        );
    }

    #[test]
    fn identifiers_are_not_validated_as_numeric_literals() {
        for text in [
            "(define (domain nan))",
            "(> (f nan) 0)",
            "(:action a :precondition (increase nan nan) :effect (not (increase nan nan)))",
        ] {
            let form = parse_nested_list_string(text).unwrap();
            assert!(validate(&form, false).is_ok());
        }
        let form =
            parse_nested_list_string("(:action a :effect (when (p nan) (increase (f nan) NaN)))")
                .unwrap();
        assert!(validate(&form, false).is_err());
    }
}
