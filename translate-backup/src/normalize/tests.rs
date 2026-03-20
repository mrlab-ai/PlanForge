use super::*;
use crate::translate::pddl::{Domain, Problem};
use crate::translate::pddl_parser::PddlTask;
use std::path::Path;

fn rule(effect: (&str, Vec<&str>), conds: Vec<(&str, Vec<&str>)>, rtype: &str) -> bm::RuleSpec {
    bm::RuleSpec {
        rtype: rtype.to_string(),
        effect: sym_atom(effect.0, effect.1),
        conditions: conds
            .into_iter()
            .map(|(p, args)| sym_atom(p, args))
            .collect(),
    }
}

fn sym_atom(pred: &str, args: Vec<&str>) -> bm::SymAtom {
    bm::SymAtom::new(
        pred.to_string(),
        args.into_iter().map(|a| a.to_string()).collect(),
    )
}

#[test]
fn adds_object_condition_for_free_head_var() {
    let mut rules = vec![rule(("move", vec!["?x"]), vec![], "project")];
    let inserted = remove_free_effect_variables(&mut rules);
    assert!(inserted);
    assert_eq!(rules[0].conditions.len(), 1);
    assert_eq!(rules[0].conditions[0].predicate, OBJECT_PREDICATE);
    assert_eq!(rules[0].conditions[0].args, vec!["?x".to_string()]);
}

#[test]
fn duplicate_conditions_are_removed() {
    let mut rules = vec![rule(
        ("move", vec!["?x"]),
        vec![("at", vec!["?x"]), ("at", vec!["?x"])],
        "project",
    )];
    split_duplicate_arguments(&mut rules);
    assert_eq!(rules[0].conditions.len(), 1);
}

#[test]
fn trivial_constant_rule_becomes_fact() {
    let mut rules = vec![rule(("ready", vec!["a1"]), vec![], "project")];
    let facts = convert_trivial_rules(&mut rules);
    assert!(rules.is_empty());
    assert_eq!(facts.len(), 1);
    assert_eq!(facts[0].predicate, "ready");
    assert_eq!(facts[0].args, vec![bm::Arg::Const("a1".to_string())]);
}

#[test]
fn normalization_pipeline_runs_steps() {
    let mut rules = vec![rule(("move", vec!["?x"]), vec![], "project")];
    let outcome = normalize_rules(&mut rules);
    assert!(outcome.object_predicate_required);
    assert!(outcome.new_facts.is_empty());
    assert_eq!(rules.len(), 1);
    assert_eq!(rules[0].conditions.len(), 1);
}

#[test]
#[ignore]
fn dump_normalized_plant_watering_task() {
    let domain_path = Path::new("misc/plant-watering/domain.pddl");
    let problem_path = Path::new("misc/plant-watering/prob_4_1_1.pddl");
    let task = PddlTask::from_files(domain_path, problem_path).expect("pddl loaded");
    let domain = Domain::from_sexprs(&task.domain_forms).expect("domain parsed");
    let problem = Problem::from_sexprs(&task.problem_forms).expect("problem parsed");

    let mut normalizable = NormalizableTask::from_ast(&domain, &problem);
    normalize(&mut normalizable).expect("normalize succeeded");

    println!("{}", normalizable.dump());
}

#[test]
fn numeric_axiom_predicate_uses_parameter_list_once() {
    use crate::translate::function_expression::{FunctionalExpression, PrimitiveNumericExpression};
    use crate::translate::normalization_function_admin::NumericAxiom;

    let axiom = NumericAxiom::new(
        "derived!difference_fuel_slew".to_string(),
        vec!["?v0".to_string(), "?v1".to_string(), "?v2".to_string()],
        Some("-".to_string()),
        vec![
            FunctionalExpression::Primitive(PrimitiveNumericExpression::new(
                "fuel".to_string(),
                vec!["?v0".to_string()],
                'S',
            )),
            FunctionalExpression::Primitive(PrimitiveNumericExpression::new(
                "slew_time".to_string(),
                vec!["?v1".to_string(), "?v2".to_string()],
                'S',
            )),
        ],
    );

    let predicate = get_numeric_axiom_predicate(&axiom);
    assert_eq!(predicate.args, vec!["?v0", "?v1", "?v2"]);
}
