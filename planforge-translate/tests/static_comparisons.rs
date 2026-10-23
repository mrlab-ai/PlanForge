//! Translation decides a comparison of two constants, as instantiation decides
//! a static atom: a true one is dropped from its condition, a false one makes
//! its operator, or the goal, unsatisfiable.

use std::path::PathBuf;

const DOMAIN: &str = "(define (domain market)
  (:requirements :typing :numeric-fluents)
  (:types goods)
  (:functions (on-sale ?g - goods) (bought))
  (:action buy
    :parameters (?g - goods)
    :precondition (> (on-sale ?g) 0)
    :effect (increase (bought) 1)))";

fn translate(name: &str, problem: &str) -> String {
    let dir: PathBuf = std::env::temp_dir().join(format!(
        "planforge-static-comparisons-{name}-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("create fixture directory");
    let domain = dir.join("domain.pddl");
    let problem_path = dir.join("problem.pddl");
    std::fs::write(&domain, DOMAIN).expect("write domain");
    std::fs::write(&problem_path, problem).expect("write problem");
    let sas = planforge_translate::translate_to_sas_string(
        domain.to_str().expect("UTF-8 path"),
        problem_path.to_str().expect("UTF-8 path"),
    )
    .expect("translation failed");
    std::fs::remove_dir_all(&dir).expect("remove fixture");
    sas
}

fn section<'a>(sas: &'a str, name: &str) -> Vec<&'a str> {
    let begin = format!("begin_{name}");
    let end = format!("end_{name}");
    sas.lines()
        .skip_while(|line| *line != begin)
        .skip(1)
        .take_while(|line| *line != end)
        .collect()
}

#[test]
fn a_constant_precondition_keeps_or_drops_its_operator() {
    let sas = translate(
        "precondition",
        "(define (problem market-1)
  (:domain market)
  (:objects apples pears - goods)
  (:init (= (on-sale apples) 2) (= (on-sale pears) 0) (= (bought) 0))
  (:goal (>= (bought) 2)))",
    );
    let operators: Vec<&str> = sas
        .lines()
        .zip(sas.lines().skip(1))
        .filter(|(line, _)| *line == "begin_operator")
        .map(|(_, name)| name)
        .collect();
    assert_eq!(operators, ["buy apples"]);
    // Only the goal comparison `bought >= 2` is left to evaluate.
    assert_eq!(section(&sas, "comparison_axioms").len(), 1, "{sas}");
}

#[test]
fn a_false_constant_goal_makes_the_task_unsolvable() {
    let sas = translate(
        "goal",
        "(define (problem market-2)
  (:domain market)
  (:objects apples - goods)
  (:init (= (on-sale apples) 2) (= (bought) 0))
  (:goal (and (>= (bought) 1) (> (on-sale apples) 3))))",
    );
    assert_eq!(section(&sas, "goal"), ["1", "0 1"], "{sas}");
    assert_eq!(section(&sas, "state"), ["0"], "{sas}");
}
