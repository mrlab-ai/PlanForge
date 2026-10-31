//! An action whose precondition is a disjunction is split into one operator
//! per disjunct, and every copy keeps the PDDL name the plan file shows.

use std::path::PathBuf;

const DOMAIN: &str = "(define (domain door)
  (:requirements :strips :disjunctive-preconditions)
  (:predicates (key) (card) (open))
  (:action open-door
    :parameters ()
    :precondition (or (key) (card))
    :effect (open)))";

const PROBLEM: &str = "(define (problem door-1)
  (:domain door)
  (:init (key) (card))
  (:goal (open)))";

#[test]
fn split_copies_keep_the_pddl_action_name() {
    let dir: PathBuf = std::env::temp_dir().join(format!(
        "planforge-disjunctive-preconditions-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("create fixture directory");
    let domain = dir.join("domain.pddl");
    let problem = dir.join("problem.pddl");
    std::fs::write(&domain, DOMAIN).expect("write domain");
    std::fs::write(&problem, PROBLEM).expect("write problem");
    let sas = planforge_translate::translate_to_sas_string(
        domain.to_str().expect("UTF-8 path"),
        problem.to_str().expect("UTF-8 path"),
    )
    .expect("translation failed");
    std::fs::remove_dir_all(&dir).expect("remove fixture");

    let names: Vec<&str> = sas
        .lines()
        .zip(sas.lines().skip(1))
        .filter(|(line, _)| *line == "begin_operator")
        .map(|(_, name)| name)
        .collect();
    assert_eq!(names, ["open-door", "open-door"]);
}
