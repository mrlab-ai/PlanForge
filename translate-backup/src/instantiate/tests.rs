use super::{
    extract_constant_numeric_facts, extract_init_function_values, extract_init_predicate_facts,
};
use crate::translate::build_model;
use crate::translate::normalize::NormalizableTask;
use crate::translate::pddl::{Condition, Domain, Problem};
use crate::translate::pddl_parser::SExpr;
use std::collections::HashMap;

fn empty_task_with_init(init: Vec<SExpr>) -> NormalizableTask {
    let domain = Domain {
        name: "d".to_string(),
        predicates: vec![],
        functions: vec![],
        types: vec![],
        actions: vec![],
        axioms: vec![],
    };
    let problem = Problem {
        name: "p".to_string(),
        objects: vec![],
        init,
        goal: Some(SExpr::Atom("truth".to_string())),
        metric: None,
    };

    let mut task = NormalizableTask::from_ast(&domain, &problem);
    task.goal = Condition::True;
    task
}

#[test]
fn init_function_values_only_include_problem_assignments() {
    let task = empty_task_with_init(vec![SExpr::List(vec![
        SExpr::Atom("=".to_string()),
        SExpr::List(vec![
            SExpr::Atom("fuel".to_string()),
            SExpr::Atom("satellite0".to_string()),
        ]),
        SExpr::Atom("17.5".to_string()),
    ])]);

    let values = extract_init_function_values(&task);

    assert_eq!(values.len(), 1);
    assert_eq!(
        values.get(&("fuel".to_string(), vec!["satellite0".to_string()])),
        Some(&17.5)
    );
    assert!(!values.contains_key(&("derived!0.0".to_string(), vec![])));
}

#[test]
fn init_predicate_facts_ignore_numeric_assignments() {
    let task = empty_task_with_init(vec![
        SExpr::List(vec![
            SExpr::Atom("power_avail".to_string()),
            SExpr::Atom("satellite0".to_string()),
        ]),
        SExpr::List(vec![
            SExpr::Atom("=".to_string()),
            SExpr::List(vec![
                SExpr::Atom("fuel".to_string()),
                SExpr::Atom("satellite0".to_string()),
            ]),
            SExpr::Atom("3.0".to_string()),
        ]),
    ]);

    let facts = extract_init_predicate_facts(&task);

    assert_eq!(
        facts,
        vec![build_model::Atom {
            predicate: "power_avail".to_string(),
            args: vec![build_model::Arg::Const("satellite0".to_string())],
        }]
    );
}

#[test]
fn constant_numeric_facts_exclude_fluent_functions() {
    let init_values = HashMap::from([
        (("fuel".to_string(), vec!["satellite0".to_string()]), 10.0),
        (
            (
                "slew_time".to_string(),
                vec!["a".to_string(), "b".to_string()],
            ),
            2.0,
        ),
    ]);

    let constants = extract_constant_numeric_facts(&init_values, &["fuel(satellite0)".to_string()]);

    assert_eq!(constants.len(), 1);
    assert_eq!(
        constants.get(&(
            "slew_time".to_string(),
            vec!["a".to_string(), "b".to_string()]
        )),
        Some(&2.0)
    );
}
