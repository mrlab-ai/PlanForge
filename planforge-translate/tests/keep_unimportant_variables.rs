//! `TranslateOptions::keep_unimportant_variables` keeps the variables and
//! operators the goal does not depend on, and the parse entry points return
//! the lifted task as the files state it.

use planforge_translate::{TranslateOptions, parse_pddl_domain, parse_pddl_task};

const DOMAIN: &str = "(define (domain errands)
  (:requirements :strips)
  (:predicates (paid) (whistled))
  (:action pay
    :parameters ()
    :precondition ()
    :effect (paid))
  (:action whistle
    :parameters ()
    :precondition ()
    :effect (whistled)))";

const PROBLEM: &str = "(define (problem errands-1)
  (:domain errands)
  (:init)
  (:goal (paid)))";

/// The domain and problem files of a fixture named `name`, removed on drop.
struct Fixture {
    dir: std::path::PathBuf,
}

impl Fixture {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "planforge-keep-unimportant-{name}-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("create fixture directory");
        std::fs::write(dir.join("domain.pddl"), DOMAIN).expect("write domain");
        std::fs::write(dir.join("problem.pddl"), PROBLEM).expect("write problem");
        Self { dir }
    }

    fn path(&self, file: &str) -> String {
        self.dir
            .join(file)
            .to_str()
            .expect("UTF-8 path")
            .to_string()
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn operator_names(keep_unimportant_variables: bool) -> Vec<String> {
    let fixture = Fixture::new(if keep_unimportant_variables {
        "keep"
    } else {
        "prune"
    });
    let options = TranslateOptions {
        keep_unimportant_variables,
        ..TranslateOptions::default()
    };
    let task = planforge_translate::translate_to_task_with_options(
        &fixture.path("domain.pddl"),
        &fixture.path("problem.pddl"),
        &options,
    )
    .expect("translation failed");
    let mut names: Vec<String> = task
        .operators()
        .iter()
        .map(|operator| operator.name().to_string())
        .collect();
    names.sort();
    let files = task
        .pddl_files()
        .expect("a translated task records its files");
    assert_eq!(
        files.problem,
        std::path::Path::new(&fixture.path("problem.pddl"))
    );
    names
}

#[test]
fn by_default_an_operator_on_an_unimportant_variable_is_pruned() {
    assert_eq!(operator_names(false), ["pay"]);
}

#[test]
fn keep_unimportant_variables_keeps_it() {
    assert_eq!(operator_names(true), ["pay", "whistle"]);
}

#[test]
fn parsing_returns_the_lifted_task() {
    let fixture = Fixture::new("parse");
    let task = parse_pddl_task(&fixture.path("domain.pddl"), &fixture.path("problem.pddl"))
        .expect("parsing failed");
    let mut actions: Vec<&str> = task.actions.iter().map(|a| a.name.as_str()).collect();
    actions.sort_unstable();
    assert_eq!(actions, ["pay", "whistle"]);

    let domain = parse_pddl_domain(&fixture.path("domain.pddl")).expect("parsing failed");
    assert_eq!(domain.actions.len(), 2);
    assert!(domain.predicates.iter().any(|p| p.name == "whistled"));
}
