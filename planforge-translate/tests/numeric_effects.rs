//! Grounding folds the additive effects an action has on one fluent.

use std::collections::HashSet;
use std::path::PathBuf;

use planforge_sas::numeric_task::{
    AbstractNumericTask, AssignmentEffect, AssignmentOperation, NumericType, NumericValue,
};

const DOMAIN: &str = "(define (domain transfer)
  (:requirements :typing :numeric-fluents)
  (:types place)
  (:functions (x ?p - place))
  (:action move-one
    :parameters (?a ?b - place)
    :precondition (>= (x ?a) 1)
    :effect (and (decrease (x ?a) 1) (increase (x ?b) 1)))
  (:action move-two
    :parameters (?a ?b - place)
    :precondition (>= (x ?a) 2)
    :effect (and (decrease (x ?a) 2) (increase (x ?b) 1))))";

const PROBLEM: &str = "(define (problem transfer-1)
  (:domain transfer)
  (:objects p q - place)
  (:init (= (x p) 3) (= (x q) 0))
  (:goal (>= (x q) 2)))";

fn write_fixture() -> (PathBuf, PathBuf) {
    let dir = std::env::temp_dir().join(format!("planforge-transfer-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create fixture directory");
    let domain = dir.join("domain.pddl");
    let problem = dir.join("problem.pddl");
    std::fs::write(&domain, DOMAIN).expect("write domain");
    std::fs::write(&problem, PROBLEM).expect("write problem");
    (domain, problem)
}

#[test]
fn a_transfer_onto_itself_has_one_effect_per_fluent() {
    let (domain, problem) = write_fixture();
    let task = planforge_translate::translate_to_task(
        domain.to_str().expect("UTF-8 path"),
        problem.to_str().expect("UTF-8 path"),
    )
    .expect("translation failed");
    std::fs::remove_dir_all(domain.parent().expect("fixture directory")).expect("remove fixture");

    let initial = task.get_initial_numeric_state_values();
    // Effects on the fluent `x`, not the `total-cost` bookkeeping.
    let fluent_effects = |effects: &[AssignmentEffect]| -> Vec<AssignmentEffect> {
        effects
            .iter()
            .filter(|effect| {
                task.numeric_variables()[effect.affected_var_id().index()].get_type()
                    != &NumericType::Cost
            })
            .cloned()
            .collect()
    };
    let mut checked = HashSet::new();
    for operator in task.get_operators() {
        let effects = fluent_effects(operator.assignment_effects());
        let mut affected = HashSet::new();
        for effect in &effects {
            assert!(
                affected.insert(effect.affected_var_id()),
                "{} has two effects on numeric variable {}",
                operator.name(),
                effect.affected_var_id()
            );
        }
        match operator.name() {
            // -1 + 1 cancels out.
            "move-one p p" | "move-one q q" => {
                assert!(
                    operator.assignment_effects().is_empty(),
                    "{}",
                    operator.name()
                );
                checked.insert(operator.name().to_string());
            }
            // -2 + 1 folds into a decrease by 1.
            "move-two p p" => {
                let [effect] = effects.as_slice() else {
                    panic!("move-two p p must have one effect on x: {effects:?}");
                };
                assert_eq!(effect.operation(), &AssignmentOperation::Minus);
                assert_eq!(initial[effect.var_id().index()], NumericValue::new(1.0));
                checked.insert(operator.name().to_string());
            }
            _ => {}
        }
    }
    assert!(checked.contains("move-two p p"), "operators: {checked:?}");
}
