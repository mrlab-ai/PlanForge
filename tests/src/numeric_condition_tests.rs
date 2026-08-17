//! Hand-written fixtures for numeric behaviour the IPC corpus does not cover.
//!
//! `numeric_effect_shapes_are_pinned_per_benchmark` records that not one
//! benchmark under `assets/numeric-pddl-files` has a numeric effect guarded by
//! an effect condition, so nothing there exercises the guarded path end to end.
//! The unit tests in `planforge-translate` and `planforge-sas` cover the SAS
//! stream and the parser; these fixtures cover the whole pipeline, and they are
//! built so that the *optimal cost changes* if the behaviour is lost. A test
//! that only checks structure would still pass if a guard were parsed and then
//! ignored during search.

use std::path::PathBuf;

use planforge_sas::axioms::ComparisonOperator;
use planforge_sas::numeric_task::{AbstractNumericTask, NumericRootTask};

use crate::corpus::{
    self, Solution, assert_fixture_set_is_pinned, blind_astar, subdirectory_names,
    translate_in_memory,
};
use crate::numeric_corpus_tests::assert_axiom_layering_contract;

/// Optima of the hand-written fixtures, derived by hand in each fixture's
/// header comment and machine-checked below.
const FIXTURE_OPTIMA: &[(&str, f64, u64)] = &[
    ("conditional-numeric-effect", 3.0, 3),
    ("strict-comparison", 2.0, 2),
    ("weighted-sum-metric", 2.0, 1),
];

fn fixtures_root() -> PathBuf {
    corpus::assets().join("numeric-conditions")
}

fn fixture_task(name: &str) -> NumericRootTask {
    let dir = fixtures_root().join(name);
    translate_in_memory(&dir.join("domain.pddl"), &dir.join("problem.pddl"))
}

#[test]
fn hand_written_fixtures_keep_their_optima_and_layering() {
    let discovered = subdirectory_names(&fixtures_root());
    let pinned: Vec<&str> = FIXTURE_OPTIMA.iter().map(|(name, _, _)| *name).collect();
    assert_fixture_set_is_pinned("numeric-conditions", &discovered, &pinned);

    for &(name, cost, length) in FIXTURE_OPTIMA {
        let task = fixture_task(name);
        assert_axiom_layering_contract(name, &task);

        let expected = Solution { cost, length };
        let actual = blind_astar(&task).unwrap_or_else(|| panic!("{name}: blind A* found no plan"));
        assert!(
            actual.matches(&expected),
            "{name}: blind A* returned {actual:?}, expected {expected:?}"
        );
    }
}

/// The guard on the second `decrease` must survive translation *and* be honoured
/// by successor generation.
///
/// The fixture is built so the two failure modes have different costs: dropping
/// the guard makes both moves cheap and the optimum falls to 2, while applying
/// it unconditionally makes the task unsolvable. `hand_written_fixtures_...`
/// pins the cost; this test pins the structure that produces it, so a failure
/// says which of the two broke.
#[test]
fn a_guarded_numeric_effect_survives_the_whole_pipeline() {
    let task = fixture_task("conditional-numeric-effect");

    let moves: Vec<&planforge_sas::numeric_task::Operator> = task
        .get_operators()
        .iter()
        .filter(|op| op.name().starts_with("move "))
        .collect();
    assert_eq!(
        moves.len(),
        2,
        "expected `move l0 l1` and `move l1 l2`, got {:?}",
        task.get_operators()
            .iter()
            .map(|op| op.name())
            .collect::<Vec<_>>()
    );

    for operator in moves {
        let fuel_effects: Vec<_> = operator
            .assignment_effects()
            .iter()
            .filter(|effect| {
                task.numeric_variables()[effect.affected_var_id()]
                    .name()
                    .contains("fuel")
            })
            .collect();
        assert_eq!(
            fuel_effects.len(),
            2,
            "{}: the unconditional and the guarded `decrease (fuel)` must both survive, got {:?}",
            operator.name(),
            operator.assignment_effects()
        );

        let guarded: Vec<_> = fuel_effects
            .iter()
            .filter(|effect| effect.is_conditional())
            .collect();
        assert_eq!(
            guarded.len(),
            1,
            "{}: exactly one of the two fuel effects is guarded by `(when (boosted) ...)`",
            operator.name()
        );
        assert_eq!(
            guarded[0].conditions().len(),
            1,
            "{}: the guard is a single `boosted` fact, got {:?}",
            operator.name(),
            guarded[0].conditions()
        );
    }
}

/// `(> (charge) 2)` must compile to a strict comparison. Compiling it as `>=`
/// makes the initial state launchable and the optimum drops from 2 to 1.
#[test]
fn a_strict_numeric_precondition_stays_strict() {
    let task = fixture_task("strict-comparison");

    let operators: Vec<ComparisonOperator> = task
        .comparison_axioms()
        .iter()
        .map(|axiom| axiom.get_operator().clone())
        .collect();
    assert_eq!(
        operators.len(),
        1,
        "the fixture has exactly one numeric precondition, got {operators:?}"
    );
    assert!(
        matches!(operators[0], ComparisonOperator::GreaterThan),
        "`(> (charge) 2)` must compile to a strict comparison, got {:?}",
        operators[0]
    );
}
