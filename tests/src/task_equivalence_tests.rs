//! One translation, two ways of getting the task out of it.
//!
//! The translation can hand its task straight to the search, or write it as the
//! SAS+ file another planner reads and have that read back. The first is the
//! default; the second is what the format is for. They are two producers of one
//! thing, so they can drift apart — and the drift would be silent, because both
//! answers are well-formed tasks that merely disagree.
//!
//! Every fixture in the corpus is therefore built both ways here and the two
//! results compared in full: variables and their axiom layers and defaults,
//! numeric variables and their types, operators with their preconditions,
//! effects and assignment effects, propositional, comparison and assignment
//! axioms, goals, mutexes, the global constraint, the metric, the numeric
//! conditions, and both halves of the axiom-closed initial state.
//!
//! The whole-task `assert_eq!` at the end of [`assert_tasks_are_equivalent`] is
//! what makes the list above exhaustive rather than a list someone maintained:
//! a field neither this file nor `NumericRootTask` forgets is a field the test
//! covers.

use std::path::Path;

use planforge_sas::numeric_task::{
    AbstractNumericTask, ExplicitFact, NumericRootTask, assert_fact_namespaces,
};
use planforge_translator::{translate_to_sas_string, translate_to_task};

use crate::corpus::{self, assert_fixture_set_is_pinned, problem_file_names, subdirectory_names};

/// The fixture corpora every group under `assets` that starts from PDDL belongs
/// to. `numeric_sas` is deliberately absent: it holds SAS files, so there is no
/// translation to compare two ways of reading.
const PDDL_CORPORA: &[&str] = &[
    "derived-predicates",
    "numeric-conditions",
    "numeric-pddl-files",
    "strips-pddl-files",
];

/// Fixtures whose grounding is too slow for a test build. They are translated by
/// `numeric_corpus_tests` under the same exclusion, and the equivalence the rest
/// of the corpus establishes is a property of the two paths, not of a fixture.
const TOO_SLOW_TO_TRANSLATE_IN_TEST_BUILDS: &[&str] =
    &["minecraft-pogo-advanced", "minecraft-sword-advanced"];

/// Every `(domain, problem)` pair under `assets` that a translation starts from.
fn pddl_fixtures() -> Vec<(String, std::path::PathBuf, std::path::PathBuf)> {
    assert_fixture_set_is_pinned(
        "the PDDL corpora",
        &subdirectory_names(&corpus::assets())
            .into_iter()
            .filter(|name| name != "numeric_sas")
            .collect::<Vec<_>>(),
        PDDL_CORPORA,
    );

    let mut fixtures = Vec::new();
    for corpus_name in PDDL_CORPORA {
        let root = corpus::assets().join(corpus_name);
        for fixture in subdirectory_names(&root) {
            if TOO_SLOW_TO_TRANSLATE_IN_TEST_BUILDS.contains(&fixture.as_str()) {
                continue;
            }
            let dir = root.join(&fixture);
            let domain = dir.join("domain.pddl");
            assert!(
                domain.is_file(),
                "missing domain for {corpus_name}/{fixture}"
            );
            for problem in problem_file_names(&dir) {
                fixtures.push((
                    format!("{corpus_name}/{fixture}/{problem}"),
                    domain.clone(),
                    dir.join(problem),
                ));
            }
        }
    }
    assert!(!fixtures.is_empty(), "no PDDL fixture was discovered");
    fixtures
}

/// The task the translation hands over, and the task the SAS+ file it writes
/// reads back as.
fn both_ways(domain: &Path, problem: &Path, what: &str) -> (NumericRootTask, NumericRootTask) {
    let path_of = |path: &Path| {
        path.to_str()
            .unwrap_or_else(|| panic!("non-utf8 fixture path: {path:?}"))
            .to_owned()
    };
    let direct = translate_to_task(&path_of(domain), &path_of(problem))
        .unwrap_or_else(|error| panic!("translating {what} failed: {error}"));

    let sas_text = translate_to_sas_string(&path_of(domain), &path_of(problem))
        .unwrap_or_else(|error| panic!("writing the SAS text of {what} failed: {error}"));
    let parsed = NumericRootTask::try_from_str(&sas_text)
        .unwrap_or_else(|error| panic!("reading the SAS text of {what} back failed: {error}"));

    (direct, parsed)
}

/// A fact as this comparison sees it: the namespace tag, and the pair the tag is
/// deliberately not part of.
type TaggedFact = (u32, usize, usize);

/// The facts of one part of a task, named by the part they came from.
type TaggedSection = (String, Vec<TaggedFact>);

/// Every fact the task exposes, with the namespace tag it carries.
///
/// The tag is not part of a fact's identity — [`ExplicitFact`] compares over
/// `(variable, value)` on purpose — so the whole-task comparison cannot see it,
/// and a task whose facts were all tagged propositional would compare equal to
/// one that tagged its condition variables correctly.
fn tagged_facts(task: &NumericRootTask) -> Vec<TaggedSection> {
    let tagged = |facts: &[ExplicitFact]| -> Vec<TaggedFact> {
        facts
            .iter()
            .map(|fact| (fact.namespace() as u32, fact.var(), fact.value()))
            .collect()
    };

    let mut sections = vec![(
        "goals".to_owned(),
        tagged(
            &(0..task.get_num_goals())
                .map(|index| *task.get_goal_fact(index))
                .collect::<Vec<_>>(),
        ),
    )];
    sections.push((
        "global constraint".to_owned(),
        tagged(&[*task.global_constraint()]),
    ));
    for (mutex_id, mutex) in task.mutexes().iter().enumerate() {
        sections.push((format!("mutex group {mutex_id}"), tagged(mutex)));
    }
    for (operator_id, operator) in task.get_operators().iter().enumerate() {
        sections.push((
            format!("operator {operator_id} preconditions"),
            tagged(operator.preconditions()),
        ));
        for (effect_id, effect) in operator.effects().iter().enumerate() {
            sections.push((
                format!("operator {operator_id} effect {effect_id} conditions"),
                tagged(effect.conditions()),
            ));
        }
        for (effect_id, effect) in operator.assignment_effects().iter().enumerate() {
            sections.push((
                format!("operator {operator_id} assignment effect {effect_id} conditions"),
                tagged(effect.conditions()),
            ));
        }
    }
    for (axiom_id, axiom) in task.axioms().iter().enumerate() {
        sections.push((
            format!("axiom {axiom_id} conditions"),
            tagged(axiom.conditions()),
        ));
    }
    sections
}

/// Asserts that two tasks are the same task.
///
/// Section by section first, so that a failure names the section that disagrees
/// rather than dumping two whole tasks, and then in full.
fn assert_tasks_are_equivalent(direct: &NumericRootTask, parsed: &NumericRootTask, what: &str) {
    assert_eq!(direct.metric(), parsed.metric(), "{what}: metric");

    assert_eq!(
        direct.variables().len(),
        parsed.variables().len(),
        "{what}: number of variables"
    );
    for var_id in 0..direct.variables().len() {
        assert_eq!(
            direct.variables()[var_id],
            parsed.variables()[var_id],
            "{what}: variable {var_id}"
        );
        assert_eq!(
            direct.get_variable_default_axiom_value(var_id),
            parsed.get_variable_default_axiom_value(var_id),
            "{what}: axiom default of variable {var_id}"
        );
    }
    assert_eq!(
        direct.numeric_variables(),
        parsed.numeric_variables(),
        "{what}: numeric variables"
    );

    assert_eq!(
        direct.get_operators().len(),
        parsed.get_operators().len(),
        "{what}: number of operators"
    );
    for (operator_id, (direct_operator, parsed_operator)) in direct
        .get_operators()
        .iter()
        .zip(parsed.get_operators())
        .enumerate()
    {
        assert_eq!(
            direct_operator, parsed_operator,
            "{what}: operator {operator_id}"
        );
    }

    assert_eq!(direct.axioms(), parsed.axioms(), "{what}: axioms");
    assert_eq!(
        direct.comparison_axioms(),
        parsed.comparison_axioms(),
        "{what}: comparison axioms"
    );
    assert_eq!(
        direct.assignment_axioms(),
        parsed.assignment_axioms(),
        "{what}: assignment axioms"
    );
    assert_eq!(
        direct.numeric_conditions(),
        parsed.numeric_conditions(),
        "{what}: numeric conditions"
    );

    assert_eq!(direct.mutexes(), parsed.mutexes(), "{what}: mutex groups");
    assert_eq!(
        direct.global_constraint(),
        parsed.global_constraint(),
        "{what}: global constraint"
    );
    assert_eq!(
        direct.get_num_goals(),
        parsed.get_num_goals(),
        "{what}: number of goals"
    );
    for index in 0..direct.get_num_goals() {
        assert_eq!(
            direct.get_goal_fact(index),
            parsed.get_goal_fact(index),
            "{what}: goal {index}"
        );
    }

    // Both halves of the initial state, already closed under the axioms, so this
    // compares what the axiom closure made of the two tasks as well.
    assert_eq!(
        direct.get_initial_propositional_state_values(),
        parsed.get_initial_propositional_state_values(),
        "{what}: initial propositional state"
    );
    assert_eq!(
        direct.get_initial_numeric_state_values(),
        parsed.get_initial_numeric_state_values(),
        "{what}: initial numeric state"
    );

    assert_eq!(
        tagged_facts(direct),
        tagged_facts(parsed),
        "{what}: fact namespaces"
    );
    // Every fact of each task names a variable of the kind the task's own
    // numeric conditions put it in, which is what the tags above are compared
    // against being wrong in the same way twice.
    assert_fact_namespaces(direct);
    assert_fact_namespaces(parsed);

    assert_eq!(direct, parsed, "{what}: the tasks differ");
}

/// The default path and the file path produce the same task, for every fixture
/// the corpus holds.
#[test]
fn building_a_task_directly_matches_reading_the_sas_file_it_writes() {
    for (what, domain, problem) in pddl_fixtures() {
        let (direct, parsed) = both_ways(&domain, &problem, &what);
        assert_tasks_are_equivalent(&direct, &parsed, &what);
    }
}

/// The comparison above only means something if it can fail, and a comparison
/// of two tasks nobody claims are equal is the cheapest way to show that it can.
#[test]
#[should_panic(expected = "two different fixtures")]
fn the_comparison_notices_a_task_that_is_not_the_same_task() {
    let derived = corpus::assets().join("derived-predicates");
    let one = derived.join("layered-chain");
    let (one, _) = both_ways(&one.join("domain.pddl"), &one.join("problem.pddl"), "one");
    let other = derived.join("conjunctive-chain");
    let (other, _) = both_ways(
        &other.join("domain.pddl"),
        &other.join("problem.pddl"),
        "other",
    );

    assert_tasks_are_equivalent(&one, &other, "two different fixtures");
}
