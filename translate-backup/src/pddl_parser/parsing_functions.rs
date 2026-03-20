use std::sync::atomic::{AtomicBool, Ordering};

use crate::translate::pddl::{Action, Atom, Function, Predicate, Problem, Type, TypedObject};
use crate::translate::pddl_parser::SExpr;

pub const DEBUG: bool = false;
pub static SEEN_WARNING_TYPE_PREDICATE_NAME_CLASH: AtomicBool = AtomicBool::new(false);

fn atom_text(expr: &SExpr) -> anyhow::Result<String> {
    match expr {
        SExpr::Atom(text) => Ok(text.clone()),
        SExpr::List(_) => anyhow::bail!("expected atom, found list"),
    }
}

fn atom_texts(alist: &[SExpr]) -> anyhow::Result<Vec<String>> {
    alist.iter().map(atom_text).collect()
}

pub fn parse_typed_list(alist: &[String], only_variables: bool, default_type: &str) -> Vec<(String, String)> {
    let mut result = Vec::new();
    let mut index = 0;
    while index < alist.len() {
        let mut separator = index;
        while separator < alist.len() && alist[separator] != "-" {
            separator += 1;
        }
        let item_type = if separator + 1 < alist.len() && separator < alist.len() && alist[separator] == "-" {
            alist[separator + 1].clone()
        } else {
            default_type.to_string()
        };
        let end = if separator < alist.len() && alist[separator] == "-" { separator } else { alist.len() };
        for item in &alist[index..end] {
            if only_variables {
                assert!(item.starts_with('?'), "Expected variable, got {item}");
            }
            result.push((item.clone(), item_type.clone()));
        }
        index = if separator < alist.len() && alist[separator] == "-" { separator + 2 } else { alist.len() };
    }
    result
}

pub fn set_supertypes(type_list: &mut [Type]) {
    let pairs: Vec<(String, String)> = type_list
        .iter()
        .filter_map(|pddl_type| {
            pddl_type
                .basetype_name
                .as_ref()
                .map(|base| (pddl_type.name.clone(), base.clone()))
        })
        .collect();

    for pddl_type in type_list.iter_mut() {
        pddl_type.supertype_names.clear();
        let mut pending = vec![pddl_type.name.clone()];
        while let Some(current) = pending.pop() {
            for (descendant, ancestor) in &pairs {
                if descendant == &current && !pddl_type.supertype_names.contains(ancestor) {
                    pddl_type.supertype_names.push(ancestor.clone());
                    pending.push(ancestor.clone());
                }
            }
        }
    }
}

pub fn parse_predicate(alist: &[SExpr]) -> anyhow::Result<Predicate> {
    let items = atom_texts(alist)?;
    let (name, args) = items
        .split_first()
        .ok_or_else(|| anyhow::anyhow!("predicate list must not be empty"))?;
    let arguments = parse_typed_list(args, true, "object")
        .into_iter()
        .map(|(name, type_name)| TypedObject::new(name, type_name))
        .collect();
    Ok(Predicate::new(name.clone(), arguments))
}

pub fn parse_function(alist: &[SExpr], type_name: &str) -> anyhow::Result<Function> {
    let items = atom_texts(alist)?;
    let (name, args) = items
        .split_first()
        .ok_or_else(|| anyhow::anyhow!("function list must not be empty"))?;
    let arguments = parse_typed_list(args, false, "object")
        .into_iter()
        .map(|(name, type_name)| TypedObject::new(name, type_name))
        .collect();
    Ok(Function::new(
        name.clone(),
        arguments,
        Some(type_name.to_string()),
    ))
}

pub fn parse_condition(_alist: &[SExpr]) -> anyhow::Result<SExpr> {
    anyhow::bail!("parse_condition is not ported yet")
}

pub fn parse_condition_aux(_alist: &[SExpr], _negated: bool) -> anyhow::Result<SExpr> {
    anyhow::bail!("parse_condition_aux is not ported yet")
}

pub fn is_function_comparison(alist: &[SExpr]) -> bool {
    match alist.first() {
        Some(SExpr::Atom(tag)) => matches!(tag.as_str(), ">" | "<" | ">=" | "<="),
        _ => false,
    }
}

pub fn is_object_comparison(alist: &[SExpr]) -> bool {
    matches!(alist.first(), Some(SExpr::Atom(tag)) if tag == "=") && alist.len() == 3
}

pub fn parse_literal(_alist: &[SExpr], _negated: bool) -> anyhow::Result<Atom> {
    anyhow::bail!("parse_literal is not ported yet")
}

pub fn _get_predicate_id_and_arity(
    _text: &str,
) -> anyhow::Result<(String, usize)> {
    if !SEEN_WARNING_TYPE_PREDICATE_NAME_CLASH.load(Ordering::Relaxed) {
        SEEN_WARNING_TYPE_PREDICATE_NAME_CLASH.store(true, Ordering::Relaxed);
    }
    anyhow::bail!("_get_predicate_id_and_arity is not ported yet")
}

pub fn parse_effects(_alist: &[SExpr]) -> anyhow::Result<()> {
    anyhow::bail!("parse_effects is not ported yet")
}

pub fn add_effect(_tmp_effect: &SExpr) -> anyhow::Result<()> {
    anyhow::bail!("add_effect is not ported yet")
}

pub fn parse_effect(_alist: &[SExpr]) -> anyhow::Result<SExpr> {
    anyhow::bail!("parse_effect is not ported yet")
}

pub fn parse_expression(_exp: &SExpr) -> anyhow::Result<SExpr> {
    anyhow::bail!("parse_expression is not ported yet")
}

pub fn parse_assignment(_alist: &[SExpr]) -> anyhow::Result<SExpr> {
    anyhow::bail!("parse_assignment is not ported yet")
}

pub fn parse_action(alist: &[SExpr]) -> anyhow::Result<Action> {
    let mut iterator = alist.iter();
    let action_tag = atom_text(iterator.next().ok_or_else(|| anyhow::anyhow!("missing :action tag"))?)?;
    anyhow::ensure!(action_tag == ":action", "expected :action, found {action_tag}");

    let name = atom_text(iterator.next().ok_or_else(|| anyhow::anyhow!("missing action name"))?)?;

    let mut parameters = Vec::new();
    let mut precond = None;
    let mut effect = None;

    while let Some(entry) = iterator.next() {
        let tag = atom_text(entry)?;
        match tag.as_str() {
            ":parameters" => {
                let parameter_list = iterator
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("missing parameter list"))?;
                let parameter_atoms = match parameter_list {
                    SExpr::List(items) => atom_texts(items)?,
                    SExpr::Atom(_) => anyhow::bail!("expected parameter list"),
                };
                parameters = parse_typed_list(&parameter_atoms, true, "object")
                    .into_iter()
                    .map(|(parameter_name, type_name)| (parameter_name, Some(type_name)))
                    .collect();
            }
            ":precondition" => {
                precond = Some(
                    iterator
                        .next()
                        .ok_or_else(|| anyhow::anyhow!("missing precondition expression"))?
                        .clone(),
                );
            }
            ":effect" => {
                effect = Some(
                    iterator
                        .next()
                        .ok_or_else(|| anyhow::anyhow!("missing effect expression"))?
                        .clone(),
                );
            }
            other => anyhow::bail!("unexpected action field {other}"),
        }
    }

    Ok(Action {
        name,
        parameters,
        precond,
        effect,
    })
}

pub fn parse_global_constraint(_alist: &[SExpr]) -> anyhow::Result<()> {
    anyhow::bail!("parse_global_constraint is not ported yet")
}

pub fn parse_axiom(_alist: &[SExpr]) -> anyhow::Result<()> {
    anyhow::bail!("parse_axiom is not ported yet")
}

pub fn parse_task(_domain_pddl: &[SExpr], _task_pddl: &[SExpr]) -> anyhow::Result<(Vec<SExpr>, Vec<SExpr>)> {
    anyhow::bail!("parse_task is not ported yet")
}

pub fn parse_domain_pddl(_domain_pddl: &[SExpr]) -> anyhow::Result<Vec<SExpr>> {
    anyhow::bail!("parse_domain_pddl is not ported yet")
}

pub fn parse_task_pddl(_task_pddl: &[SExpr]) -> anyhow::Result<Problem> {
    anyhow::bail!("parse_task_pddl is not ported yet")
}

#[allow(non_snake_case)]
pub fn isFloat(astring: &str) -> bool {
    astring.parse::<f64>().is_ok()
}

pub fn check_atom_consistency<T: PartialEq>(
    atom: &T,
    same_truth_value: &[T],
    other_truth_value: &[T],
    atom_is_true: bool,
) -> anyhow::Result<()> {
    let in_same = same_truth_value.iter().any(|item| item == atom);
    let in_other = other_truth_value.iter().any(|item| item == atom);
    if in_same && in_other {
        anyhow::bail!("atom occurs with inconsistent truth value: {}", atom_is_true)
    }
    Ok(())
}

pub fn check_for_duplicates<T: Eq + std::hash::Hash + std::fmt::Debug>(
    elements: &[T],
    errmsg: &str,
    finalmsg: &str,
) -> anyhow::Result<()> {
    use std::collections::HashSet;

    let mut seen = HashSet::new();
    for element in elements {
        if !seen.insert(element) {
            anyhow::bail!("{}: {:?}. {}", errmsg, element, finalmsg)
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_typed_list_groups_items_by_dash_type() {
        let input = vec![
            "?x".to_string(),
            "?y".to_string(),
            "-".to_string(),
            "satellite".to_string(),
            "?z".to_string(),
        ];
        let result = parse_typed_list(&input, true, "object");
        assert_eq!(
            result,
            vec![
                ("?x".to_string(), "satellite".to_string()),
                ("?y".to_string(), "satellite".to_string()),
                ("?z".to_string(), "object".to_string()),
            ]
        );
    }

    #[test]
    fn set_supertypes_builds_transitive_chain() {
        let mut types = vec![
            Type {
                name: "object".to_string(),
                basetype_name: None,
                supertype_names: vec![],
            },
            Type {
                name: "vehicle".to_string(),
                basetype_name: Some("object".to_string()),
                supertype_names: vec![],
            },
            Type {
                name: "truck".to_string(),
                basetype_name: Some("vehicle".to_string()),
                supertype_names: vec![],
            },
        ];
        set_supertypes(&mut types);
        let truck = types.iter().find(|item| item.name == "truck").unwrap();
        assert!(truck.supertype_names.contains(&"vehicle".to_string()));
        assert!(truck.supertype_names.contains(&"object".to_string()));
    }

    #[test]
    fn parse_predicate_builds_typed_arguments() {
        let predicate = parse_predicate(&[
            SExpr::Atom("calibrated".to_string()),
            SExpr::Atom("?i".to_string()),
            SExpr::Atom("-".to_string()),
            SExpr::Atom("instrument".to_string()),
        ])
        .expect("predicate parse should succeed");
        assert_eq!(predicate.name, "calibrated");
        assert_eq!(predicate.get_arity(), 1);
        assert_eq!(predicate.arguments[0].type_name, "instrument");
    }

    #[test]
    fn parse_function_builds_typed_arguments() {
        let function = parse_function(
            &[
                SExpr::Atom("fuel".to_string()),
                SExpr::Atom("?s".to_string()),
                SExpr::Atom("-".to_string()),
                SExpr::Atom("satellite".to_string()),
            ],
            "number",
        )
        .expect("function parse should succeed");
        assert_eq!(function.name, "fuel");
        assert_eq!(function.arguments.len(), 1);
        assert_eq!(function.arguments[0].type_name, "satellite");
        assert_eq!(function.type_name.as_deref(), Some("number"));
    }

    #[test]
    fn parse_action_reads_parameters_precondition_and_effect() {
        let action = parse_action(&[
            SExpr::Atom(":action".to_string()),
            SExpr::Atom("turn_to".to_string()),
            SExpr::Atom(":parameters".to_string()),
            SExpr::List(vec![
                SExpr::Atom("?s".to_string()),
                SExpr::Atom("-".to_string()),
                SExpr::Atom("satellite".to_string()),
                SExpr::Atom("?d_new".to_string()),
                SExpr::Atom("-".to_string()),
                SExpr::Atom("direction".to_string()),
            ]),
            SExpr::Atom(":precondition".to_string()),
            SExpr::List(vec![
                SExpr::Atom("and".to_string()),
                SExpr::List(vec![
                    SExpr::Atom("pointing".to_string()),
                    SExpr::Atom("?s".to_string()),
                    SExpr::Atom("?d_new".to_string()),
                ]),
            ]),
            SExpr::Atom(":effect".to_string()),
            SExpr::List(vec![
                SExpr::Atom("and".to_string()),
                SExpr::List(vec![
                    SExpr::Atom("increase".to_string()),
                    SExpr::Atom("total-cost".to_string()),
                    SExpr::Atom("1".to_string()),
                ]),
            ]),
        ])
        .expect("action parse should succeed");

        assert_eq!(action.name, "turn_to");
        assert_eq!(action.parameters.len(), 2);
        assert_eq!(action.parameters[0], ("?s".to_string(), Some("satellite".to_string())));
        assert!(action.precond.is_some());
        assert!(action.effect.is_some());
    }
}