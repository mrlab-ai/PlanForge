//! The STRIPS corpus under `assets/strips-pddl-files`.
//!
//! Two things are pinned here. First, the optimum of every problem in the
//! corpus, discovered from disk so an unpinned fixture fails. Second, plan
//! verification and the task shape the gradient engine relies on: the unit
//! tests in `planforge-sas` cover the replay logic on hand-built tasks, while
//! these run it on the output of the real translator pipeline, so they also pin
//! that a pure `:strips` task (no `:functions`, no `:action-costs`) still
//! carries a `Cost` numeric variable, a per-operator `total-cost` assignment
//! effect, and a derived global-constraint atom.

use std::path::PathBuf;
use std::sync::Arc;

use planforge_sas::numeric_task::{AbstractNumericTask, NumericRootTask, NumericType, Operator};
use planforge_sas::plan_verification::{PlanRejection, ReplayOutcome, replay_plan};
use planforge_sas::state_registry::StateRegistry;

use crate::corpus::{
    self, Solution, assert_fixture_set_is_pinned, blind_astar, problem_file_names,
    subdirectory_names, translate_in_memory,
};

/// Optima of every problem in the corpus, measured with blind A*, which is
/// optimal. Keyed by `<folder>/<problem file>`.
const STRIPS_OPTIMA: &[(&str, f64, u64)] = &[
    ("blocks-4-0/probBLOCKS-4-0.pddl", 6.0, 6),
    ("blocks-5-0/probBLOCKS-5-0.pddl", 12.0, 12),
    ("blocks-8-0/probBLOCKS-8-0.pddl", 18.0, 18),
    ("blocks-minimal/probBLOCKS-2-reverse.pddl", 4.0, 4),
    ("blocks-minimal/probBLOCKS-3-preserve-middle.pddl", 8.0, 8),
    ("blocks-minimal/probBLOCKS-3-reverse.pddl", 6.0, 6),
    ("blocks-minimal/probBLOCKS-4-preserve-middle.pddl", 8.0, 8),
    ("blocks-minimal/probBLOCKS-4-reverse.pddl", 8.0, 8),
];

fn corpus_root() -> PathBuf {
    corpus::assets().join("strips-pddl-files")
}

/// Translates a fixture the way the `planforge` binary does for a two-argument
/// PDDL invocation.
///
/// Deliberately *not* the `..._fast` path, which requests singleton fact groups
/// and would collapse every variable to domain 2. The real pipeline builds
/// genuine multi-valued variables, and those are what these tests are about.
fn translated_task(fixture: &str, problem_file: &str) -> NumericRootTask {
    let dir = corpus_root().join(fixture);
    translate_in_memory(&dir.join("domain.pddl"), &dir.join(problem_file))
}

fn blocks_4_0() -> NumericRootTask {
    translated_task("blocks-4-0", "probBLOCKS-4-0.pddl")
}

fn operators_named<'t>(task: &'t NumericRootTask, names: &[&str]) -> Vec<&'t Operator> {
    names
        .iter()
        .map(|name| {
            task.get_operators()
                .iter()
                .find(|op| op.name() == *name)
                .unwrap_or_else(|| panic!("no operator named {name:?} in the translated task"))
        })
        .collect()
}

/// The optimal 6-step plan for `probBLOCKS-4-0`, as found by `astar(blind())`.
const BLOCKS_4_0_PLAN: [&str; 6] = [
    "pick-up b",
    "stack b a",
    "pick-up c",
    "stack c b",
    "pick-up d",
    "stack d c",
];

#[test]
fn blind_astar_reproduces_every_pinned_strips_optimum() {
    let discovered: Vec<String> = subdirectory_names(&corpus_root())
        .iter()
        .flat_map(|folder| {
            problem_file_names(&corpus_root().join(folder))
                .into_iter()
                .map(move |problem| format!("{folder}/{problem}"))
        })
        .collect();
    let pinned: Vec<&str> = STRIPS_OPTIMA.iter().map(|(name, _, _)| *name).collect();
    assert_fixture_set_is_pinned("strips-pddl-files", &discovered, &pinned);

    let mut mismatches: Vec<String> = Vec::new();
    for &(key, cost, length) in STRIPS_OPTIMA {
        let (folder, problem) = key
            .split_once('/')
            .unwrap_or_else(|| panic!("STRIPS_OPTIMA key {key:?} is not `<folder>/<problem>`"));
        let expected = Solution { cost, length };
        match blind_astar(&translated_task(folder, problem)) {
            Some(actual) if actual.matches(&expected) => {}
            Some(actual) => {
                mismatches.push(format!("{key}: expected {expected:?}, got {actual:?}"))
            }
            None => mismatches.push(format!("{key}: expected {expected:?}, got no plan")),
        }
    }
    assert!(
        mismatches.is_empty(),
        "blind A* did not reproduce the known STRIPS optima:\n{}",
        mismatches.join("\n")
    );
}

#[test]
fn optimal_blocks_plan_verifies_with_the_expected_cost() {
    let arc = Arc::new(blocks_4_0());
    let mut registry = StateRegistry::for_task(arc.clone());
    let operators = operators_named(&arc, &BLOCKS_4_0_PLAN);

    let replay = replay_plan(&*arc, &mut registry, arc.global_constraint(), &operators)
        .expect("replay machinery failed");

    match replay.outcome {
        ReplayOutcome::Solved(plan) => {
            assert_eq!(plan.prefix_len, 6, "the whole plan is needed");
            assert_eq!(plan.cost, 6.0, "unit-cost blocks plan of length 6");
        }
        other => panic!("expected the optimal plan to verify, got {other:?}"),
    }
    assert_eq!(
        replay.states.len(),
        7,
        "initial state plus one per operator"
    );
}

#[test]
fn plan_padded_past_the_goal_still_verifies_at_the_goal_prefix() {
    let arc = Arc::new(blocks_4_0());
    let mut registry = StateRegistry::for_task(arc.clone());

    // Append an operator that is applicable in the goal state but leaves it.
    let mut names = BLOCKS_4_0_PLAN.to_vec();
    names.push("unstack d c");
    let operators = operators_named(&arc, &names);

    let replay = replay_plan(&*arc, &mut registry, arc.global_constraint(), &operators)
        .expect("replay machinery failed");

    assert!(replay.is_solved(), "expected the goal prefix to verify");
    assert_eq!(replay.verified().expect("solved").prefix_len, 6);
    assert_eq!(
        replay.applied, 6,
        "verification must stop at the goal and not apply the trailing operator"
    );
}

#[test]
fn reordered_blocks_plan_is_rejected_at_the_first_inapplicable_operator() {
    let arc = Arc::new(blocks_4_0());
    let mut registry = StateRegistry::for_task(arc.clone());

    // Swap the first two operators: `stack b a` needs `holding b`, which
    // `pick-up b` has not yet established.
    let operators = operators_named(&arc, &["stack b a", "pick-up b"]);

    let replay = replay_plan(&*arc, &mut registry, arc.global_constraint(), &operators)
        .expect("replay machinery failed");

    match replay.outcome {
        ReplayOutcome::Rejected(PlanRejection::InapplicableOperator { step, operator, .. }) => {
            assert_eq!(step, 0);
            assert_eq!(operator, "stack b a");
        }
        other => panic!("expected an applicability rejection, got {other:?}"),
    }
    assert_eq!(replay.applied, 0);
}

#[test]
fn truncated_blocks_plan_is_rejected_for_missing_goals() {
    let arc = Arc::new(blocks_4_0());
    let mut registry = StateRegistry::for_task(arc.clone());
    let operators = operators_named(&arc, &BLOCKS_4_0_PLAN[..4]);

    let replay = replay_plan(&*arc, &mut registry, arc.global_constraint(), &operators)
        .expect("replay machinery failed");

    match replay.outcome {
        ReplayOutcome::Rejected(PlanRejection::GoalNotReached { unsatisfied }) => {
            assert!(
                !unsatisfied.is_empty(),
                "a truncated plan must leave a goal open"
            );
        }
        other => panic!("expected a goal rejection, got {other:?}"),
    }
    assert_eq!(replay.applied, 4, "all four operators are applicable");
}

/// Pins the task shape the `sgd` engine's supported-task check relies on. If the
/// translator ever changes what a pure `:strips` task looks like, this fails
/// here rather than as a confusing rejection inside the engine.
#[test]
fn pure_strips_task_shape_is_as_the_sgd_engine_expects() {
    let task = blocks_4_0();

    // Only Constant and Cost numeric variables: no real numeric planning.
    for numeric in task.numeric_variables() {
        assert!(
            matches!(
                numeric.get_type(),
                NumericType::Constant | NumericType::Cost
            ),
            "unexpected numeric variable {:?} of type {:?}",
            numeric.name(),
            numeric.get_type()
        );
    }

    // No compiled numeric conditions, and no numeric arithmetic.
    assert!(
        task.comparison_axioms().is_empty(),
        "a pure STRIPS task must have no comparison axioms"
    );
    assert!(
        task.assignment_axioms().is_empty(),
        "a pure STRIPS task must have no assignment axioms"
    );

    // Every operator increments total-cost, even without `:action-costs`.
    // A blanket rejection of assignment effects would therefore reject blocks.
    let cost_var_ids: Vec<usize> = task
        .numeric_variables()
        .iter()
        .enumerate()
        .filter(|(_, v)| matches!(v.get_type(), NumericType::Cost))
        .map(|(id, _)| id)
        .collect();
    assert!(!cost_var_ids.is_empty(), "expected a total-cost variable");
    for op in task.get_operators() {
        for effect in op.assignment_effects() {
            assert!(
                cost_var_ids.contains(&effect.affected_var_id()),
                "operator {:?} writes non-cost numeric variable {}",
                op.name(),
                effect.affected_var_id()
            );
        }
    }

    // Propositional axioms exist but are unconditional, and the derived
    // variables they define are never written by an operator effect.
    assert!(
        !task.axioms().is_empty(),
        "the translator always injects a global-constraint axiom"
    );
    for axiom in task.axioms() {
        assert!(
            axiom.conditions().is_empty(),
            "a pure STRIPS task must not have conditioned axioms"
        );
    }
    for op in task.get_operators() {
        for effect in op.effects() {
            assert!(
                task.get_variable_axiom_layer(effect.var_id())
                    .expect("effect variable is in range")
                    .is_none(),
                "operator {:?} writes derived variable {}",
                op.name(),
                effect.var_id()
            );
        }
    }

    // The multi-valued structure the transcription depends on: each block gets
    // one variable covering holding / on / ontable, so "a block on two supports
    // at once" is not representable in a per-variable simplex.
    let domain_sizes: Vec<usize> = (0..task.get_num_variables())
        .map(|v| {
            task.get_variable_domain_size(v)
                .expect("variable is in range")
        })
        .collect();
    assert!(
        domain_sizes.iter().any(|&size| size > 2),
        "expected genuine multi-valued variables, got {domain_sizes:?}"
    );
}
