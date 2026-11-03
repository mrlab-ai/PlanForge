//! Typed lists, as in `?x ?y - block`, and the type declarations they name.

use super::*;

/// What a typed list declares, which decides what it may contain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TypedList {
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
pub(super) fn parse_typed_list(
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

pub(super) fn check_type_declared(types: &TypeHierarchy, name: &str, line: u32) -> ParseResult<()> {
    if types.contains(name) {
        Ok(())
    } else {
        Err(ParseError::malformed(
            line,
            format!("undeclared type {name}"),
        ))
    }
}

pub(super) fn typed_item(
    item: &SExpr,
    type_name: &str,
    list: TypedList,
) -> ParseResult<TypedObject> {
    let name = item.expect_atom("a name")?;
    if list != TypedList::Objects && !name.starts_with('?') {
        return Err(ParseError::malformed(
            item.line(),
            format!("expected a variable, got {name}"),
        ));
    }
    Ok(TypedObject::new(name, type_name))
}

pub(super) fn parse_type_list(items: Form<'_>) -> ParseResult<Vec<Type>> {
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
