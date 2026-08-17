//! What actually reaches a goal, counted over every fixture corpus on disk.
//!
//! A goal fact names one of exactly three kinds of variable, and which one it is
//! decides what the planner can do with the task:
//!
//! * an **ordinary propositional** variable — an operator writes it, so every
//!   heuristic can reason about reaching it;
//! * a **condition** variable, carrying the truth of a numeric comparison. The
//!   comparison axioms compute it, and the abstraction machinery reasons about the
//!   comparison itself: refining its operands is what the numeric CEGAR loop does.
//!   This is what a numeric goal compiles to;
//! * a **propositional-axiom-derived** variable. Nothing writes it but the axioms,
//!   so the abstraction families refuse the task
//!   (`planforge_search::evaluation::validate_abstractable_goal`) while blind A*,
//!   `ff` and `lmcutnumeric` still solve it.
//!
//! The third kind is therefore a capability boundary, and the counts below are the
//! evidence for where it falls: **no benchmark has one**. Only two hand-written
//! `derived-predicates` fixtures do, both by design — `goal-condition` puts a
//! `:derived` predicate in its goal and `negated-goal` puts the negation of one
//! there.
//!
//! A numeric goal used to be compiled into a `@goal-reachable` derived predicate,
//! which put ten of the 21 benchmarks in the third column with one goal fact each —
//! Drone's four propositional goals included, since they were conjuncts of the same
//! axiom body. That is what these tables are pinned against regressing to.
//!
//! Both translation paths are censused, because the on-disk one is the format
//! another planner reads: a condition goal fact has to survive being written and
//! read back, and the counts agreeing is what says it does.

use std::path::{Path, PathBuf};

use planforge_sas::numeric_task::{AbstractNumericTask, FactNamespace, NumericRootTask};

use crate::corpus::{
    Scratch, assert_fixture_set_is_pinned, assets, problem_file_names, subdirectory_names,
    translate_in_memory, translate_to_disk,
};
use crate::numeric_corpus_tests::TOO_SLOW_TO_TRANSLATE_IN_TEST_BUILDS;

/// How many goal facts of a fixture name each kind of variable.
#[derive(Debug, Default, PartialEq, Eq, Clone, Copy)]
struct Census {
    /// Ordinary propositional variables, written by operators.
    propositional: usize,
    /// Condition variables, carrying a numeric comparison's truth value.
    condition: usize,
    /// Propositional variables only the axioms write.
    axiom_derived: usize,
}

impl Census {
    fn add(&mut self, other: Census) {
        self.propositional += other.propositional;
        self.condition += other.condition;
        self.axiom_derived += other.axiom_derived;
    }
}

/// One pinned row: fixture directory, problem file, and the three counts.
type Row = (&'static str, &'static str, usize, usize, usize);

/// The 21 IPC-derived numeric benchmarks. Ten of them have a numeric goal, and it
/// is in the condition column where it belongs; none is in the derived column.
///
/// `minecraft-pogo-advanced` and `minecraft-sword-advanced` are absent because
/// grounding them takes about seven seconds each in a debug build — see
/// [`TOO_SLOW_TO_TRANSLATE_IN_TEST_BUILDS`]. Both were censused through the release
/// binary instead: one ordinary propositional goal fact each,
/// `Atom have_pogo_stick()` and `Atom have_wooden_sword()`, at axiom layer -1.
const NUMERIC_BENCHMARKS: &[Row] = &[
    ("counters-sym", "problem_2.pddl", 0, 1, 0),
    ("delivery", "pfile1.pddl", 4, 0, 0),
    ("depots", "pfile1.pddl", 2, 0, 0),
    ("depots-sym", "pfile1.pddl", 2, 0, 0),
    ("drone", "pfile2.pddl", 4, 3, 0),
    ("expedition", "pfile1.pddl", 2, 0, 0),
    ("farmland", "prob_2_100_1229_scale.pddl", 0, 3, 0),
    ("farmland2", "prob_2_100_1229_scale.pddl", 0, 3, 0),
    ("fn-counters-small_instances", "problem_2.pddl", 0, 1, 0),
    ("forestfire", "prob01.pddl", 0, 1, 0),
    ("hydropower", "pfile4.pddl", 0, 1, 0),
    ("mprime", "pfile1.pddl", 1, 0, 0),
    ("onlycraft-opt", "P01_opt.pddl", 0, 1, 0),
    ("pathwaysmetric", "pfile1.pddl", 0, 1, 0),
    ("plant-watering", "prob_4_1_1.pddl", 0, 2, 0),
    ("rover-unit", "pfile1.pddl", 3, 0, 0),
    ("sailing", "prob_1_1_1229.pddl", 1, 0, 0),
    ("satellite", "pfile1.pddl", 3, 0, 0),
    ("zenotravel", "pfile1.pddl", 3, 0, 0),
];

/// The hand-written sailing corpus: propositional `saved` goals throughout, one per
/// person, and no numeric goal at all.
const SAILING_SIMPLE: &[Row] = &[
    ("", "prob_1b1p_diag.pddl", 1, 0, 0),
    ("", "prob_1b1p_far.pddl", 1, 0, 0),
    ("", "prob_1b1p_x.pddl", 1, 0, 0),
    ("", "prob_1b2p_diag.pddl", 2, 0, 0),
    ("", "prob_1b2p_x.pddl", 2, 0, 0),
    ("", "prob_1b4p_axes.pddl", 4, 0, 0),
    ("", "prob_2b1p.pddl", 1, 0, 0),
    ("", "prob_2b2p_assign.pddl", 2, 0, 0),
    ("", "prob_2b2p_x.pddl", 2, 0, 0),
];

/// The `:derived` fixtures, and the only two goal facts anywhere on disk that name
/// an axiom-derived variable: `goal-condition` asks for a derived predicate,
/// `negated-goal` for the negation of one. Both are the abstraction refusal's
/// fixtures — see `derived_predicate_tests::abstractions_refuse_a_negated_derived_goal`.
///
/// The other seven use derived predicates in *preconditions*, which stays
/// supported everywhere, so their goals are ordinary facts.
const DERIVED_PREDICATES: &[Row] = &[
    ("conjunctive-chain", "problem.pddl", 1, 0, 0),
    ("cyclic-negation", "problem.pddl", 2, 0, 0),
    ("disjunctive-support", "problem.pddl", 2, 0, 0),
    ("goal-condition", "problem.pddl", 1, 0, 1),
    ("layered-chain", "problem.pddl", 1, 0, 0),
    ("negated-dependency", "problem.pddl", 1, 0, 0),
    ("negated-goal", "problem.pddl", 1, 0, 1),
    ("numeric-body", "problem.pddl", 1, 0, 0),
    ("recursive-closure", "problem.pddl", 2, 0, 0),
];

/// The numeric-condition fixtures keep a propositional goal. The first two put
/// a comparison in a precondition; the metric fixture instead exercises a
/// compound numeric objective.
const NUMERIC_CONDITIONS: &[Row] = &[
    ("conditional-numeric-effect", "problem.pddl", 1, 0, 0),
    ("strict-comparison", "problem.pddl", 1, 0, 0),
    ("weighted-sum-metric", "problem.pddl", 1, 0, 0),
];

/// Censused as a corpus of its own, the way `sailing_simple_tests` treats it, even
/// though it lives under the benchmark root.
const SEPARATE_CORPUS: &[&str] = &["sailing-simple"];

/// Blocksworld: no numbers, no axioms, so every goal fact is an ordinary one.
const STRIPS: &[Row] = &[
    ("blocks-4-0", "probBLOCKS-4-0.pddl", 3, 0, 0),
    ("blocks-5-0", "probBLOCKS-5-0.pddl", 4, 0, 0),
    ("blocks-8-0", "probBLOCKS-8-0.pddl", 7, 0, 0),
    ("blocks-minimal", "probBLOCKS-2-reverse.pddl", 1, 0, 0),
    (
        "blocks-minimal",
        "probBLOCKS-3-preserve-middle.pddl",
        2,
        0,
        0,
    ),
    ("blocks-minimal", "probBLOCKS-3-reverse.pddl", 2, 0, 0),
    (
        "blocks-minimal",
        "probBLOCKS-4-preserve-middle.pddl",
        3,
        0,
        0,
    ),
    ("blocks-minimal", "probBLOCKS-4-reverse.pddl", 3, 0, 0),
];

/// Classifies every goal fact of `task`.
///
/// The namespace is what separates a condition variable from an axiom-derived one:
/// both have an axiom layer, and telling them apart by walking the comparison
/// axioms is what [`FactNamespace`] exists to spare every caller.
fn census(task: &NumericRootTask, what: &str) -> Census {
    let mut census = Census::default();
    for index in 0..task.get_num_goals() {
        let fact = task.get_goal_fact(index);
        match fact.namespace() {
            FactNamespace::Condition => census.condition += 1,
            FactNamespace::Propositional => {
                if task.variables()[fact.var()].axiom_layer().is_some() {
                    census.axiom_derived += 1;
                } else {
                    census.propositional += 1;
                }
            }
            FactNamespace::NumericVariable => panic!(
                "{what}: goal fact {fact:?} names a domain abstraction's private numeric id space"
            ),
        }
    }
    census
}

/// Censuses one corpus through both translation paths, printing a per-fixture line
/// and returning the rows in the pinned tables' format.
fn census_corpus(root: &Path, rows: &[Row]) -> (Vec<String>, Census) {
    let mut discovered = Vec::new();
    let mut totals = Census::default();

    for (directory, problem) in fixtures(root) {
        let dir = root.join(&directory);
        let what = format!("{directory}/{problem}");

        let in_memory = census(
            &translate_in_memory(&dir.join("domain.pddl"), &dir.join(&problem)),
            &what,
        );
        let scratch = Scratch::new("goal_census");
        let on_disk = census(
            &translate_to_disk(&dir.join("domain.pddl"), &dir.join(&problem), &scratch),
            &what,
        );
        assert_eq!(
            in_memory, on_disk,
            "{what}: the SAS+ file has to carry the same goal facts the in-memory task has; \
             writing or reading one of them lost its kind"
        );

        eprintln!(
            "  {what}: {} propositional, {} condition, {} axiom-derived",
            in_memory.propositional, in_memory.condition, in_memory.axiom_derived
        );
        discovered.push(row(
            &directory,
            &problem,
            in_memory.propositional,
            in_memory.condition,
            in_memory.axiom_derived,
        ));
        totals.add(in_memory);
    }

    let pinned: Vec<String> = rows
        .iter()
        .map(|&(directory, problem, propositional, condition, derived)| {
            row(directory, problem, propositional, condition, derived)
        })
        .collect();
    assert_fixture_set_is_pinned(
        &format!("goal census of {}", root.display()),
        &discovered,
        &pinned.iter().map(String::as_str).collect::<Vec<_>>(),
    );

    (discovered, totals)
}

fn row(
    directory: &str,
    problem: &str,
    propositional: usize,
    condition: usize,
    derived: usize,
) -> String {
    format!("{directory}/{problem} = {propositional}/{condition}/{derived}")
}

/// Every `(directory, problem)` pair under `root`, sorted. A corpus of problems
/// with no sub-directory (`sailing-simple`) reports an empty directory name, which
/// is what the tables above spell as `""`.
fn fixtures(root: &Path) -> Vec<(String, String)> {
    let mut pairs: Vec<(String, String)> = Vec::new();
    if root.join("domain.pddl").is_file() {
        pairs.extend(
            problem_file_names(root)
                .into_iter()
                .map(|problem| (String::new(), problem)),
        );
        return pairs;
    }
    for directory in subdirectory_names(root) {
        let dir = root.join(&directory);
        let name = directory.as_str();
        if !dir.join("domain.pddl").is_file()
            || TOO_SLOW_TO_TRANSLATE_IN_TEST_BUILDS.contains(&name)
            || SEPARATE_CORPUS.contains(&name)
        {
            continue;
        }
        pairs.extend(
            problem_file_names(&dir)
                .into_iter()
                .map(|problem| (directory.clone(), problem)),
        );
    }
    pairs
}

fn benchmarks_root() -> PathBuf {
    assets().join("numeric-pddl-files")
}

#[test]
fn every_goal_fact_is_classified_and_pinned() {
    let corpora: [(PathBuf, &[Row]); 5] = [
        (benchmarks_root(), NUMERIC_BENCHMARKS),
        (benchmarks_root().join("sailing-simple"), SAILING_SIMPLE),
        (assets().join("derived-predicates"), DERIVED_PREDICATES),
        (assets().join("numeric-conditions"), NUMERIC_CONDITIONS),
        (assets().join("strips-pddl-files"), STRIPS),
    ];

    let mut derived_goal_fixtures: Vec<String> = Vec::new();
    for (root, rows) in &corpora {
        eprintln!("{}:", root.display());
        let (discovered, totals) = census_corpus(root, rows);
        eprintln!(
            "  TOTAL: {} propositional, {} condition, {} axiom-derived",
            totals.propositional, totals.condition, totals.axiom_derived
        );
        derived_goal_fixtures.extend(
            discovered
                .into_iter()
                .filter(|row| !row.ends_with("/0"))
                .map(|row| format!("{}: {row}", root.display())),
        );
    }

    // The loud half. A benchmark in this list would have *lost* abstraction
    // support rather than been simplified, so the two entries here are named
    // individually: both are hand-written fixtures whose whole purpose is to put a
    // `:derived` predicate in a goal.
    let expected: Vec<String> = ["goal-condition", "negated-goal"]
        .iter()
        .map(|name| {
            format!(
                "{}: {name}/problem.pddl = 1/0/1",
                assets().join("derived-predicates").display()
            )
        })
        .collect();
    assert_fixture_set_is_pinned(
        "fixtures whose goal names an axiom-derived variable",
        &derived_goal_fixtures,
        &expected.iter().map(String::as_str).collect::<Vec<_>>(),
    );
}
