//! End-to-end solves with the gradient engine.
//!
//! Every plan reported here goes through the exact verifier, so a passing test
//! means a genuinely valid plan was synthesized by optimization — not that the
//! loss got small.

#![cfg(feature = "sgd")]

use std::sync::Arc;

use planforge_sas::numeric_task::{AbstractNumericTask, NumericRootTask, Operator, TaskRef};
use planforge_sas::plan_verification::{ReplayOutcome, replay_plan};
use planforge_sas::state_registry::StateRegistry;
use planforge_sgd::config::{HorizonPolicy, SgdConfig};
use planforge_sgd::engine::{SgdStatus, solve};

use crate::corpus::{assets, translate_in_memory};

fn translated(fixture_root: &str, fixture: &str, problem_file: &str) -> NumericRootTask {
    let dir = assets().join(fixture_root).join(fixture);
    translate_in_memory(&dir.join("domain.pddl"), &dir.join(problem_file))
}

fn blocks_4_0() -> NumericRootTask {
    translated("strips-pddl-files", "blocks-4-0", "probBLOCKS-4-0.pddl")
}

/// Independently re-verify a returned plan, rather than trusting the engine's
/// own report.
fn revalidate(task: &Arc<NumericRootTask>, plan: &[usize]) -> f64 {
    let mut registry = StateRegistry::for_task(Arc::clone(task) as TaskRef<'_>);
    let operators: Vec<&Operator> = plan
        .iter()
        .map(|&index| &task.get_operators()[index])
        .collect();
    let replay = replay_plan(&**task, &mut registry, task.global_constraint(), &operators)
        .expect("replay machinery failed");
    match replay.outcome {
        ReplayOutcome::Solved(verified) => {
            assert_eq!(
                verified.prefix_len,
                plan.len(),
                "the returned plan should contain no redundant trailing operators"
            );
            verified.cost
        }
        other => panic!("engine returned a plan the verifier rejects: {other:?}"),
    }
}

#[test]
fn solves_blocks_4_0_and_the_plan_verifies() {
    let task = Arc::new(blocks_4_0());

    let config = SgdConfig {
        // Fixed horizon so the test is a controlled experiment rather than a
        // race between the optimizer and the dovetail schedule.
        horizon: HorizonPolicy::Fixed(12),
        // The method is stochastic, so this budget was chosen by measurement
        // rather than by finding a lucky seed: at 24 particles and 8000 updates
        // all of seeds 1-8 solve this instance (median ~2500 updates), whereas
        // at 8 particles only about half of them do. Picking a seed that happens
        // to work at a budget that usually fails would make this test
        // meaningless.
        particles: 24,
        updates: 8_000,
        seed: 7,
        // Refreshes off: a solve here must be attributable to gradients.
        refresh: false,
        ..SgdConfig::default()
    };

    let outcome = solve(
        Arc::clone(&task) as TaskRef<'_>,
        *task.global_constraint(),
        &config,
    )
    .expect("the engine must run");

    eprintln!(
        "status={:?} updates={} verifier_calls={} best_goals={}/{} longest_prefix={} best_residual={:.6}",
        outcome.status,
        outcome.updates,
        outcome.verifier_calls,
        outcome.best_goals_reached,
        outcome.num_goals,
        outcome.longest_applicable_prefix,
        outcome.best_total_residual
    );
    eprintln!("final: {}", outcome.final_diagnostics);

    assert_eq!(
        outcome.status,
        SgdStatus::Solved,
        "blocks-4-0 should be solvable within the budget"
    );
    let plan = outcome.plan.expect("a solved outcome carries a plan");
    let cost = revalidate(&task, &plan);
    eprintln!("plan length {} cost {cost}", plan.len());
    assert!(
        plan.len() >= 6,
        "the optimal plan is 6 steps, so a shorter one would be a bug"
    );
    // Plan quality is explicitly not part of the objective, so any valid plan is
    // acceptable here; only validity is asserted, by `revalidate` above.
}

/// The engine is deterministic given a seed: the same seed must reproduce the
/// same plan exactly, or none of the ablations mean anything.
#[test]
fn identical_seeds_produce_identical_plans() {
    let task = Arc::new(blocks_4_0());
    let config = SgdConfig {
        horizon: HorizonPolicy::Fixed(10),
        particles: 4,
        updates: 400,
        seed: 99,
        ..SgdConfig::default()
    };

    let first = solve(
        Arc::clone(&task) as TaskRef<'_>,
        *task.global_constraint(),
        &config,
    )
    .expect("run one");
    let second = solve(
        Arc::clone(&task) as TaskRef<'_>,
        *task.global_constraint(),
        &config,
    )
    .expect("run two");

    assert_eq!(first.status, second.status);
    assert_eq!(first.plan, second.plan);
    assert_eq!(first.updates, second.updates);
    assert_eq!(
        first.best_total_residual, second.best_total_residual,
        "residual traces must match bit for bit"
    );
}

/// A numeric task must be refused by the engine, with the reason surfaced.
#[test]
fn numeric_tasks_are_refused_by_the_engine() {
    let task = Arc::new(translated(
        "numeric-pddl-files",
        "sailing-simple",
        "prob_1b1p_x.pddl",
    ));
    let config = SgdConfig {
        horizon: HorizonPolicy::Fixed(4),
        particles: 1,
        updates: 1,
        ..SgdConfig::default()
    };

    let error = solve(
        Arc::clone(&task) as TaskRef<'_>,
        *task.global_constraint(),
        &config,
    )
    .expect_err("a numeric task must be refused");
    let message = error.to_string();
    assert!(
        message.contains("numeric"),
        "the refusal should say what is wrong: {message}"
    );
}

/// An invalid configuration must fail before any work happens.
#[test]
fn invalid_configuration_fails_before_optimizing() {
    let task = Arc::new(blocks_4_0());
    let config = SgdConfig {
        horizon: HorizonPolicy::Fixed(8),
        particles: 0,
        ..SgdConfig::default()
    };
    let error = solve(
        Arc::clone(&task) as TaskRef<'_>,
        *task.global_constraint(),
        &config,
    )
    .expect_err("zero particles must be refused");
    assert!(error.to_string().contains("particles"), "{error}");
}

/// The ordinary direct lane must not require the quadratic causal-link ticket
/// when both losses that consume it are disabled. Keep refresh enabled here as
/// well, so its optional-link plumbing is exercised by the same run.
#[test]
fn inert_causal_links_are_omitted_and_report_zero_diagnostics() {
    let task = Arc::new(blocks_4_0());
    let config = SgdConfig {
        horizon: HorizonPolicy::Fixed(6),
        particles: 1,
        updates: 2,
        verify_period: 1,
        refresh: true,
        refresh_period: 1,
        refresh_particles: 1,
        causal_link_weight: 0.0,
        causal_link_integrality_final: 0.0,
        ..SgdConfig::default()
    };
    assert!(
        !config.causal_links_enabled(),
        "the test must exercise the allocation-free direct lane"
    );

    let outcome = solve(
        Arc::clone(&task) as TaskRef<'_>,
        *task.global_constraint(),
        &config,
    )
    .expect("the direct recurrent lane must run without causal-link tensors");

    assert!(outcome.refreshes > 0, "the optional refresh path must run");
    assert_eq!(outcome.final_diagnostics.causal_link_source, 0.0);
    assert_eq!(outcome.final_diagnostics.causal_link_threat, 0.0);
    assert_eq!(outcome.final_diagnostics.causal_link_integrality, 0.0);
}

/// The dovetail schedule must reach a horizon that admits a plan, starting from
/// one that does not.
#[test]
fn dovetail_grows_the_horizon_until_a_plan_fits() {
    let task = Arc::new(blocks_4_0());
    let config = SgdConfig {
        // Round one has horizon 4, which is shorter than the 6-step optimum, so
        // no plan exists there at all and the schedule has to grow.
        horizon: HorizonPolicy::Dovetail {
            start: 4,
            growth: 2.0,
            max: 16,
        },
        particles: 8,
        updates: 3_000,
        seed: 11,
        ..SgdConfig::default()
    };

    let outcome = solve(
        Arc::clone(&task) as TaskRef<'_>,
        *task.global_constraint(),
        &config,
    )
    .expect("the engine must run");

    assert_eq!(outcome.status, SgdStatus::Solved, "{outcome:?}");
    assert!(
        outcome.horizon_rounds >= 2,
        "horizon 4 cannot contain a 6-step plan, so the schedule must have grown: {outcome:?}"
    );
    let plan = outcome.plan.expect("solved");
    revalidate(&task, &plan);
}

/// The control the research note insists on: with the learning rate at zero,
/// nothing is learned, so anything solved is solved by the *sampling* machinery
/// alone. Reported so that a solve at a real learning rate can be attributed to
/// gradients rather than to luck.
///
/// With refreshes off there is no full-support resampling at all, so this must
/// not solve the task.
#[test]
fn learning_rate_zero_without_refresh_does_not_solve() {
    let task = Arc::new(blocks_4_0());
    let config = SgdConfig {
        horizon: HorizonPolicy::Fixed(12),
        particles: 8,
        updates: 2_000,
        learning_rate: 0.0,
        // No noise either: with both off the logits never move at all.
        noise: (0.0, 0.0),
        remelt_noise: 0.0,
        refresh: false,
        seed: 7,
        ..SgdConfig::default()
    };

    let outcome = solve(
        Arc::clone(&task) as TaskRef<'_>,
        *task.global_constraint(),
        &config,
    )
    .expect("the engine must run");

    assert_eq!(
        outcome.status,
        SgdStatus::BudgetExhausted,
        "a frozen optimizer must not find a plan; if it does, the solve above was luck"
    );
    assert!(outcome.plan.is_none());
}

/// Refreshes are the note's probabilistic-completeness mechanism, and must
/// actually fire when enabled.
#[test]
fn refresh_is_off_by_default_and_fires_when_enabled() {
    let task = Arc::new(blocks_4_0());
    assert!(
        !SgdConfig::default().refresh,
        "refresh must default to off so results stay attributable to gradients"
    );

    let config = SgdConfig {
        horizon: HorizonPolicy::Fixed(6),
        particles: 2,
        updates: 600,
        // Freeze the optimizer so refreshes are the only thing that can move
        // anything, which is what makes this a test of the refresh mechanism.
        learning_rate: 0.0,
        noise: (0.0, 0.0),
        refresh: true,
        refresh_period: 2,
        refresh_particles: 1,
        seed: 3,
        ..SgdConfig::default()
    };
    let outcome = solve(
        Arc::clone(&task) as TaskRef<'_>,
        *task.global_constraint(),
        &config,
    )
    .expect("the engine must run");
    assert!(
        outcome.refreshes > 0,
        "refreshes were enabled but none happened: {outcome:?}"
    );
}
