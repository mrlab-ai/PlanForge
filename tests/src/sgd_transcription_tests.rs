//! The transcription on real translated tasks.
//!
//! `planforge-sgd`'s own tests establish exactness on generated tiny tasks. These
//! tests check the two things only a real task can show: that a genuine STRIPS
//! instance transcribes to the shape the method expects, and that a genuine
//! numeric instance is refused with a usable reason.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use planforge_sas::numeric_task::{AbstractNumericTask, NumericRootTask, Operator};
use planforge_sas::plan_verification::{ReplayOutcome, replay_plan};
use planforge_sas::state_registry::StateRegistry;
use planforge_sgd::classical::NotClassical;
use planforge_sgd::residuals::{Assignment, evaluate};
use planforge_sgd::transcription::{Transcription, TranscriptionError};

fn translated(fixture_root: &str, fixture: &str, problem_file: &str) -> NumericRootTask {
    let dir: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(fixture_root)
        .join(fixture);
    let domain = dir.join("domain.pddl");
    let problem = dir.join(problem_file);
    assert!(domain.is_file(), "missing {domain:?}");
    assert!(problem.is_file(), "missing {problem:?}");

    let sas_text = planforge_translator::translate_to_sas_string(
        domain.to_str().expect("path is valid UTF-8"),
        problem.to_str().expect("path is valid UTF-8"),
    )
    .expect("translation failed");
    let preprocessed = planforge_translate::preprocess::run_preprocess_to_string(&sas_text);
    NumericRootTask::try_from_str(&preprocessed).expect("parsing failed")
}

fn blocks_4_0() -> NumericRootTask {
    translated(
        "assets/strips-pddl-files",
        "blocks-4-0",
        "probBLOCKS-4-0.pddl",
    )
}

const BLOCKS_4_0_PLAN: [&str; 6] = [
    "pick-up b",
    "stack b a",
    "pick-up c",
    "stack c b",
    "pick-up d",
    "stack d c",
];

#[test]
fn blocks_transcribes_to_the_expected_shape() {
    let task = blocks_4_0();
    let transcription = Transcription::build(&task).expect("blocks is a classical task");

    // Nine primary variables: clear(a..d), handempty, and one "where is block x"
    // variable per block. The tenth task variable is the derived
    // global-constraint atom, which carries no row.
    assert_eq!(transcription.num_variables(), 9);
    assert_eq!(
        transcription.num_facts(),
        30,
        "4 binary clear + 1 binary handempty + 4 five-valued position variables"
    );

    // 32 grounded operators plus the appended no-op, which is last.
    assert_eq!(transcription.num_actions(), 33);
    assert_eq!(transcription.noop_action(), 32);
    assert!(
        transcription
            .action_source(transcription.noop_action())
            .is_none(),
        "the no-op has no source operator"
    );

    // Multi-valued variables are what make split worlds unrepresentable.
    let domains = transcription.var_domain();
    assert_eq!(
        domains.iter().filter(|&&size| size == 5).count(),
        4,
        "one five-valued position variable per block, got {domains:?}"
    );

    // Blocks has no conditional effects, so the last-write-wins arithmetic
    // degenerates and every group holds exactly one effect.
    assert!(
        transcription.cond_effect().is_empty(),
        "blocks has no conditional effects"
    );
    assert_eq!(
        transcription.max_group_size(),
        1,
        "no operator writes the same variable twice"
    );
    assert!(
        transcription.dropped_operators().is_empty(),
        "no blocks operator should be structurally inapplicable"
    );
}

/// The exactness proposition on a real task and a real plan: the integral
/// assignment built from a verified plan has zero residual in every family,
/// including the goal family.
#[test]
fn integral_assignment_of_a_real_plan_has_zero_residual() {
    let task = blocks_4_0();
    let transcription = Transcription::build(&task).expect("blocks is a classical task");
    let arc = Arc::new(task);
    let mut registry = StateRegistry::for_task(arc.clone());

    let operators: Vec<&Operator> = BLOCKS_4_0_PLAN
        .iter()
        .map(|name| {
            arc.get_operators()
                .iter()
                .find(|op| op.name() == *name)
                .expect("plan operator exists")
        })
        .collect();

    let replay = replay_plan(&*arc, &mut registry, arc.global_constraint(), &operators)
        .expect("replay failed");
    assert!(matches!(replay.outcome, ReplayOutcome::Solved(_)));

    // Pad the horizon past the plan so the no-op slots are exercised too.
    let horizon = BLOCKS_4_0_PLAN.len() + 3;
    let mut assignment = Assignment::zeros(&transcription, horizon);

    // Actions: the plan, then no-ops.
    for (t, name) in BLOCKS_4_0_PLAN.iter().enumerate() {
        let action = (0..transcription.num_actions())
            .find(|&a| {
                transcription
                    .action_source(a)
                    .is_some_and(|op| arc.get_operators()[op].name() == *name)
            })
            .expect("plan operator is in the transcription");
        assignment.set_action_one_hot(t, action);
    }
    for t in BLOCKS_4_0_PLAN.len()..horizon {
        assignment.set_action_one_hot(t, transcription.noop_action());
    }

    // States: the exact replay states, then the goal state repeated, since a
    // no-op leaves the state unchanged.
    let values_of = |state: &planforge_sas::state_registry::ConcreteState,
                     registry: &StateRegistry| {
        transcription
            .primary_vars()
            .iter()
            .map(|&task_var| {
                state
                    .get_propositional_value(registry, task_var)
                    .expect("variable in range")
            })
            .collect::<Vec<usize>>()
    };
    let goal_values = values_of(replay.states.last().expect("non-empty"), &registry);
    for (t, state) in replay.states.iter().enumerate() {
        let values = values_of(state, &registry);
        assignment.set_state_one_hot(&transcription, t, &values);
    }
    for t in replay.states.len()..=horizon {
        assignment.set_state_one_hot(&transcription, t, &goal_values);
    }

    let residuals = evaluate(&transcription, &assignment);
    assert!(
        residuals.is_zero(1e-12),
        "a verified plan must have zero residual, got max {}",
        residuals.max()
    );
}

#[test]
fn perturbing_a_valid_assignment_makes_a_residual_positive() {
    let task = blocks_4_0();
    let transcription = Transcription::build(&task).expect("blocks is a classical task");

    // Start from a trivially consistent assignment: no-op everywhere, and the
    // initial state repeated. That satisfies preconditions and transitions; only
    // the goal family is violated, because no-ops never reach the goal.
    let horizon = 4;
    let mut assignment = Assignment::zeros(&transcription, horizon);
    let initial: Vec<usize> = transcription
        .initial_fact()
        .iter()
        .enumerate()
        .map(|(var, &fact)| (fact - transcription.var_offset()[var]) as usize)
        .collect();
    for t in 0..horizon {
        assignment.set_action_one_hot(t, transcription.noop_action());
    }
    for t in 0..=horizon {
        assignment.set_state_one_hot(&transcription, t, &initial);
    }

    let baseline = evaluate(&transcription, &assignment);
    assert!(
        baseline.precondition.iter().all(|&r| r <= 1e-12),
        "no-ops have no preconditions to violate"
    );
    assert!(
        baseline
            .transition
            .iter()
            .all(|f| f.iter().all(|&r| r <= 1e-12)),
        "repeating a state under no-ops satisfies every transition constraint"
    );

    // Now break exactly one transition: change one intermediate state value
    // without any action producing it.
    let mut broken = assignment.clone();
    let position_var = transcription
        .var_domain()
        .iter()
        .position(|&size| size == 5)
        .expect("blocks has five-valued variables");
    let other_value = (initial[position_var] + 1) % 5;
    let mut values = initial.clone();
    values[position_var] = other_value;
    broken.set_state_one_hot(&transcription, 2, &values);

    let perturbed = evaluate(&transcription, &broken);
    assert!(
        perturbed
            .transition
            .iter()
            .any(|f| f.iter().any(|&r| r > 1e-9)),
        "an unexplained state change must violate a transition constraint"
    );
}

#[test]
fn numeric_tasks_are_rejected_with_a_usable_reason() {
    // sailing-simple is a genuine numeric domain: its conditions compile to
    // comparison axioms, which is exactly what the engine cannot handle.
    let task = translated(
        "assets/numeric-pddl-files",
        "sailing-simple",
        "prob_1b1p_x.pddl",
    );

    match Transcription::build(&task) {
        Err(TranscriptionError::NotClassical(problems)) => {
            assert!(
                problems
                    .iter()
                    .any(|p| matches!(p, NotClassical::NumericConditions { .. })),
                "expected numeric conditions to be the reported reason, got {problems:?}"
            );
            // The message must name the problem, not just fail.
            let rendered = TranscriptionError::NotClassical(problems).to_string();
            assert!(
                rendered.contains("numeric"),
                "rejection message should mention numeric: {rendered}"
            );
        }
        other => panic!("expected a numeric task to be rejected, got {other:?}"),
    }
}
