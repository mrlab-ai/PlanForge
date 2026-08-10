//! The planner must be a function of its input alone.
//!
//! Everything between a PDDL pair and a plan runs over hash maps -- the
//! translator groups atoms, the search registers states -- and every `HashMap`
//! gets its own hash key, so two pipelines in the same process already visit
//! those containers in two different orders. Anything that leaks such an order
//! into its output therefore shows up here as a difference between the runs.
//!
//! What is compared is the whole run, not just the optimal cost. A cost is
//! insensitive: several optimal plans usually exist, and which one A* returns and
//! how many states it looks at on the way are decided by the order successors
//! arrive in, which is the order a variable numbering induces. Those are the
//! numbers that move first, so those are the numbers pinned.
//!
//! Two neighbours of this file cover the ends of the same pipeline:
//! `planforge-translate/tests/determinism.rs` pins that the SAS+ *text* of a
//! translation is reproducible, and `planforge/tests/determinism.rs` pins that a
//! whole *process* is, which is what catches a dependence on something seeded
//! once per process rather than once per map.

use std::path::PathBuf;

use crate::corpus::{
    Scratch, SearchRun, assets, blind_astar_run, problem_file, translate_in_memory,
    translate_to_disk,
};

/// Fixtures planned from scratch several times over. Small enough for a debug
/// build to plan repeatedly, and between them they cover what stands between the
/// PDDL and the search: multi-valued variables and mutex groups (depots),
/// numeric conditions on several variables (plant-watering, sailing), and
/// derived variables closed by axioms (sailing).
const REPLANNED_FIXTURES: &[&str] = &["depots", "plant-watering", "zenotravel", "sailing"];

/// How often each fixture is planned. Three runs rather than two: a difference
/// between two runs can be read as either one being the odd one out, and the
/// third says which.
const RUNS: usize = 3;

fn fixture_dir(name: &str) -> PathBuf {
    assets().join("numeric-pddl-files").join(name)
}

fn plan_in_memory(name: &str) -> SearchRun {
    let dir = fixture_dir(name);
    let task = translate_in_memory(&dir.join("domain.pddl"), &problem_file(&dir));
    blind_astar_run(&task).unwrap_or_else(|| panic!("{name}: blind A* found no plan"))
}

fn plan_through_a_file(name: &str) -> SearchRun {
    let dir = fixture_dir(name);
    let scratch = Scratch::new("determinism");
    let task = translate_to_disk(&dir.join("domain.pddl"), &problem_file(&dir), &scratch);
    blind_astar_run(&task).unwrap_or_else(|| panic!("{name}: blind A* found no plan"))
}

fn assert_runs_agree(name: &str, what: &str, run: impl Fn(&str) -> SearchRun) {
    let first = run(name);
    for run_no in 2..=RUNS {
        let again = run(name);
        assert_eq!(
            first, again,
            "{name} via {what} is not reproducible: run 1 and run {run_no} differ"
        );
    }
}

/// Translating and planning the same PDDL pair repeatedly gives the same plan and
/// the same search, down to the state counters.
#[test]
fn replanning_a_fixture_in_memory_gives_the_same_search() {
    for &name in REPLANNED_FIXTURES {
        assert_runs_agree(name, "the in-memory path", plan_in_memory);
    }
}

/// The same, for the path that goes through a SAS+ file: the writer and the
/// parser sit between the translation and the search there, and both have their
/// own opportunity to depend on an iteration order.
#[test]
fn replanning_a_fixture_through_a_file_gives_the_same_search() {
    for &name in REPLANNED_FIXTURES {
        assert_runs_agree(name, "the SAS+ file path", plan_through_a_file);
    }
}

/// Translating the same PDDL pair twice gives the same task.
///
/// Stricter than the two tests above and it localizes a failure: a task differs in
/// full -- every variable's name and domain, every operator, the namespace tag of
/// every fact, both halves of the axiom-closed initial state -- whereas a search
/// only shows a difference that changes what it does. Naming variables out of a
/// `HashMap` is the failure this catches, and it is one the pipeline has had.
#[test]
fn translating_a_fixture_twice_gives_the_same_task() {
    for &name in REPLANNED_FIXTURES {
        let dir = fixture_dir(name);
        let problem = problem_file(&dir);
        let first = translate_in_memory(&dir.join("domain.pddl"), &problem);
        for run_no in 2..=RUNS {
            let again = translate_in_memory(&dir.join("domain.pddl"), &problem);
            assert!(
                first == again,
                "translating {name} is not reproducible: run 1 and run {run_no} differ"
            );
        }
    }
}

/// Searching the *same* task twice gives the same search.
///
/// The two tests above rebuild the task each time, so a difference in them can
/// come from either half of the pipeline. This one holds the task fixed, which
/// leaves the search's own containers -- the state registry above all -- as the
/// only thing that can differ between the runs.
#[test]
fn searching_one_task_twice_gives_the_same_search() {
    for &name in REPLANNED_FIXTURES {
        let dir = fixture_dir(name);
        let task = translate_in_memory(&dir.join("domain.pddl"), &problem_file(&dir));

        let first = blind_astar_run(&task).unwrap_or_else(|| panic!("{name}: no plan"));
        for run_no in 2..=RUNS {
            let again = blind_astar_run(&task).unwrap_or_else(|| panic!("{name}: no plan"));
            assert_eq!(
                first, again,
                "searching {name} is not reproducible: run 1 and run {run_no} differ"
            );
        }
    }
}
