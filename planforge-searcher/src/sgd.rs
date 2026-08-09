//! `--search 'sgd(...)'`: option parsing and the bridge to the gradient engine.
//!
//! The engine is not a search algorithm. It rides on `--search` purely to reuse
//! the translation pipeline, the resource limits and exit codes, and the plan
//! writer, none of which have anything to do with searching.
//!
//! Options are applied by hand rather than through `#[derive(ApplyOptions)]`:
//! that trait is sealed inside `planforge-search`, and `SgdConfig` lives in
//! `planforge-sgd`, which must not depend on `planforge-search` — that
//! dependency edge is what makes "this engine cannot search" a structural fact
//! rather than a promise. `apply_da_options` in `recursive_config` is
//! hand-written for the same reason.

use std::str::FromStr;

use planforge_sas::numeric_task::{ExplicitFact, Operator, TaskRef};
use planforge_sas::plan_verification::{ReplayOutcome, replay_plan};
use planforge_sas::state_registry::StateRegistry;
use planforge_search::search::{SearchResult, SearchStatus};
use planforge_sgd::config::{CausalCopyMode, HorizonPolicy, SgdConfig};
use planforge_sgd::engine::{SgdStatus, solve};
use tracing::info;

use crate::recursive_config::{ConfigArg, ConfigValue};

/// Read an atom argument as `T`, naming the option when it does not parse.
fn atom<T: FromStr>(key: &str, value: &ConfigValue) -> Result<T, String>
where
    T::Err: std::fmt::Display,
{
    match value {
        ConfigValue::Atom(text) => text
            .parse::<T>()
            .map_err(|error| format!("option `{key}`: cannot parse `{text}`: {error}")),
        ConfigValue::Call(call) => Err(format!(
            "option `{key}` expects a value, got the call `{}(...)`",
            call.name()
        )),
    }
}

fn boolean(key: &str, value: &ConfigValue) -> Result<bool, String> {
    match value {
        ConfigValue::Atom(text) => match text.as_str() {
            "true" | "1" | "yes" => Ok(true),
            "false" | "0" | "no" => Ok(false),
            other => Err(format!("option `{key}`: expected a boolean, got `{other}`")),
        },
        ConfigValue::Call(call) => Err(format!(
            "option `{key}` expects a boolean, got the call `{}(...)`",
            call.name()
        )),
    }
}

/// `horizon=12` or `horizon=dovetail` or `horizon=dovetail(start, growth, max)`.
fn horizon(value: &ConfigValue) -> Result<HorizonPolicy, String> {
    match value {
        ConfigValue::Atom(text) if text == "dovetail" => Ok(SgdConfig::default().horizon),
        ConfigValue::Atom(text) => {
            let fixed = text.parse::<usize>().map_err(|error| {
                format!("option `horizon`: expected a number or `dovetail`, got `{text}`: {error}")
            })?;
            Ok(HorizonPolicy::Fixed(fixed))
        }
        ConfigValue::Call(call) if call.name() == "dovetail" => {
            let args = call.args();
            if args.len() != 3 {
                return Err(format!(
                    "option `horizon`: `dovetail(start, growth, max)` expects 3 arguments, got {}",
                    args.len()
                ));
            }
            Ok(HorizonPolicy::Dovetail {
                start: atom("horizon.start", args[0].value())?,
                growth: atom("horizon.growth", args[1].value())?,
                max: atom("horizon.max", args[2].value())?,
            })
        }
        ConfigValue::Call(call) => Err(format!(
            "option `horizon`: unknown form `{}(...)`; use a number, `dovetail`, \
             or `dovetail(start, growth, max)`",
            call.name()
        )),
    }
}

/// Apply `sgd(...)` options onto `config`.
///
/// Unknown keys are an error: silently ignoring a misspelled option would mean
/// running a different configuration than the one asked for, and in an
/// experiment that is worse than crashing.
pub fn apply_sgd_options(config: &mut SgdConfig, args: &[ConfigArg]) -> Result<(), String> {
    for arg in args {
        let Some(key) = arg.key() else {
            return Err(format!(
                "`sgd(...)` takes only named options, got the positional value `{}`",
                match arg.value() {
                    ConfigValue::Atom(text) => text.clone(),
                    ConfigValue::Call(call) => format!("{}(...)", call.name()),
                }
            ));
        };
        let value = arg.value();
        match key {
            "horizon" => config.horizon = horizon(value)?,
            "particles" => config.particles = atom(key, value)?,
            "updates" => config.updates = atom(key, value)?,
            "learning_rate" | "lr" => config.learning_rate = atom(key, value)?,
            "grad_clip" => config.grad_clip = atom(key, value)?,
            "action_logit_clip" => config.action_logit_clip = atom(key, value)?,
            "state_logit_clip" => config.state_logit_clip = atom(key, value)?,
            "initial_noop_logit_gap" => config.initial_noop_logit_gap = atom(key, value)?,
            "temporal_tokens" => config.temporal_tokens = boolean(key, value)?,
            "temporal_reserved_slots" => config.temporal_reserved_slots = atom(key, value)?,
            "temporal_reservation_weight" => config.temporal_reservation_weight = atom(key, value)?,
            "temporal_restart_patience" => config.temporal_restart_patience = atom(key, value)?,
            "schedule_temperature_start" => config.schedule_temperature.0 = atom(key, value)?,
            "schedule_temperature_end" => config.schedule_temperature.1 = atom(key, value)?,
            "schedule_sinkhorn_iterations" => {
                config.schedule_sinkhorn_iterations = atom(key, value)?
            }
            "schedule_identity_bias" => config.schedule_identity_bias = atom(key, value)?,
            "schedule_integrality_final" => config.schedule_integrality_final = atom(key, value)?,
            "slot_slack_window" => config.slot_slack_window = atom(key, value)?,
            "slot_slack_logit_gap" => config.slot_slack_logit_gap = atom(key, value)?,
            "slot_slack_weight" => config.slot_slack_weight = atom(key, value)?,
            "insertion_repair_weight" => config.insertion_repair_weight = atom(key, value)?,
            "insertion_min_prefix_fraction" => {
                config.insertion_min_prefix_fraction = atom(key, value)?
            }
            "anchor_trust_weight" => config.anchor_trust_weight = atom(key, value)?,
            "goal_survival_weight" => config.goal_survival_weight = atom(key, value)?,
            "backward_bridge_weight" => config.backward_bridge_weight = atom(key, value)?,
            "action_temperature_start" => config.action_temperature.0 = atom(key, value)?,
            "action_temperature_mid" => config.action_temperature.1 = atom(key, value)?,
            "action_temperature_end" => config.action_temperature.2 = atom(key, value)?,
            "state_temperature_start" => config.state_temperature.0 = atom(key, value)?,
            "state_temperature_end" => config.state_temperature.1 = atom(key, value)?,
            "crystallization_start" => config.crystallization_start = atom(key, value)?,
            "rho_precondition" => config.rho_precondition = atom(key, value)?,
            "rho_transition" => config.rho_transition = atom(key, value)?,
            "rho_goal" => config.rho_goal = atom(key, value)?,
            "dual_growth" => config.dual_growth = atom(key, value)?,
            "dual_decay" => config.dual_decay = atom(key, value)?,
            "dual_cap" => config.dual_cap = atom(key, value)?,
            "dual_period" => config.dual_period = atom(key, value)?,
            "top_residual_fraction" => config.top_residual_fraction = atom(key, value)?,
            "action_integrality_final" => config.action_integrality_final = atom(key, value)?,
            "state_integrality_final" => config.state_integrality_final = atom(key, value)?,
            "worst_integrality_final" => config.worst_integrality_final = atom(key, value)?,
            "causal_copy" => config.causal_copy = atom::<CausalCopyMode>(key, value)?,
            "causal_shadow_end" => config.causal_shadow_end = atom(key, value)?,
            "causal_discovery_end" => config.causal_discovery_end = atom(key, value)?,
            "causal_proof_end" => config.causal_proof_end = atom(key, value)?,
            "causal_transfer_end" => config.causal_transfer_end = atom(key, value)?,
            "causal_takeover_end" => config.causal_takeover_end = atom(key, value)?,
            "q_action_temperature_start" => config.q_action_temperature.0 = atom(key, value)?,
            "q_action_temperature_end" => config.q_action_temperature.1 = atom(key, value)?,
            "q_logit_perturbation" => config.q_logit_perturbation = atom(key, value)?,
            "teacher_tolerance" => config.teacher_tolerance = atom(key, value)?,
            "teacher_weight" => config.teacher_weight = atom(key, value)?,
            "applicability_barrier_margin" => {
                config.applicability_barrier_margin = atom(key, value)?
            }
            "applicability_mass_weight" => config.applicability_mass_weight = atom(key, value)?,
            "remelt_cooldown_updates" => config.remelt_cooldown_updates = atom(key, value)?,
            "polish_p_norm" => config.polish_p_norm = atom(key, value)?,
            "noop_suffix_weight" => config.noop_suffix_weight = atom(key, value)?,
            "causal_link_weight" => config.causal_link_weight = atom(key, value)?,
            "causal_link_learning_rate" => config.causal_link_learning_rate = atom(key, value)?,
            "causal_link_integrality_final" => {
                config.causal_link_integrality_final = atom(key, value)?
            }
            "causal_link_temperature_start" => config.causal_link_temperature.0 = atom(key, value)?,
            "causal_link_temperature_end" => config.causal_link_temperature.1 = atom(key, value)?,
            "causal_link_initial_bias" => config.causal_link_initial_bias = atom(key, value)?,
            "residual_tolerance" => config.residual_tolerance = atom(key, value)?,
            "verify_period" => config.verify_period = atom(key, value)?,
            "trace_period" => config.trace_period = atom(key, value)?,
            "trace_particle" => config.trace_particle = atom(key, value)?,
            "focus_growth" => config.focus_growth = atom(key, value)?,
            "focus_cap" => config.focus_cap = atom(key, value)?,
            "cycles" => config.cycles = atom(key, value)?,
            "noise_start" => config.noise.0 = atom(key, value)?,
            "noise_end" => config.noise.1 = atom(key, value)?,
            "remelt_patience" => config.remelt_patience = atom(key, value)?,
            "remelt_noise" => config.remelt_noise = atom(key, value)?,
            "remelt_shrink" => config.remelt_shrink = atom(key, value)?,
            "remelt_stop_progress" => config.remelt_stop_progress = atom(key, value)?,
            "refresh" => config.refresh = boolean(key, value)?,
            "refresh_period" => config.refresh_period = atom(key, value)?,
            "refresh_particles" => config.refresh_particles = atom(key, value)?,
            "seed" => config.seed = atom(key, value)?,
            other => return Err(format!("unknown option `{other}` for `sgd(...)`")),
        }
    }
    Ok(())
}

/// Run the gradient engine and shape its outcome like a `SearchResult`.
///
/// The node counters stay at zero. That is not a placeholder: no state was
/// expanded, evaluated or generated, and reporting anything else would be
/// false.
pub fn run_sgd(
    task: TaskRef<'_>,
    global_constraint: ExplicitFact,
    args: &[ConfigArg],
) -> std::io::Result<SearchResult> {
    let mut config = SgdConfig::default();
    apply_sgd_options(&mut config, args).map_err(std::io::Error::other)?;

    info!("=== Gradient plan synthesis (no search) ===");
    info!(
        "particles={} updates={} seed={} refresh={}",
        config.particles, config.updates, config.seed, config.refresh
    );

    let started = std::time::Instant::now();
    let outcome = solve(task.clone(), global_constraint, &config).map_err(std::io::Error::other)?;
    let elapsed = started.elapsed();

    if !outcome.trace.is_empty() {
        let mut action_names = task
            .get_operators()
            .iter()
            .map(|operator| operator.name().to_string())
            .collect::<Vec<_>>();
        action_names.push("<noop>".to_string());
        let action_map = action_names
            .iter()
            .enumerate()
            .map(|(index, name)| format!("{index}={name}"))
            .collect::<Vec<_>>()
            .join("|");
        info!("sgd_trace_actions {action_map}");
        let fact_map = outcome
            .trace_fact_names
            .iter()
            .enumerate()
            .map(|(index, name)| format!("{index}={name}"))
            .collect::<Vec<_>>()
            .join("|");
        info!("sgd_trace_facts {fact_map}");
        for point in &outcome.trace {
            info!(
                "sgd_trace_meta round={} update={} particle={} phase={:?} \
                 goal_repair_start={} goal_weights={:?} missing_goals={:?}",
                point.round,
                point.update,
                point.particle,
                point.phase,
                point.goal_repair_start,
                point.goal_weights,
                point.missing_goals
            );
            if point.temporal_obligations.iter().any(Option::is_some) {
                info!(
                    "sgd_trace_obligations update={} facts_by_token={:?} achievers_by_token={:?}",
                    point.update, point.temporal_obligations, point.temporal_obligation_achievers
                );
                let roles = point
                    .temporal_obligations
                    .iter()
                    .enumerate()
                    .filter_map(|(token, obligation)| {
                        let fact = obligation.as_ref()?;
                        let probabilities = &point.token_action_probabilities[token];
                        let decoded = probabilities
                            .iter()
                            .copied()
                            .enumerate()
                            .max_by(|left, right| left.1.total_cmp(&right.1))
                            .expect("a temporal token has an action distribution")
                            .0;
                        let achievers = point.temporal_obligation_achievers[token]
                            .iter()
                            .map(|&action| action_names[action].as_str())
                            .collect::<Vec<_>>()
                            .join(",");
                        Some(format!(
                            "token{token}:fact{fact}:identity_focus={:.6}:applicability_focus={:.6}:decoded={}:achievers=[{achievers}]",
                            point.temporal_obligation_focus[token],
                            point.temporal_applicability_focus[token],
                            action_names[decoded]
                        ))
                    })
                    .collect::<Vec<_>>()
                    .join("|");
                info!("sgd_trace_roles update={} {roles}", point.update);
            }
            let decoded_rows = point
                .action_probabilities
                .iter()
                .map(|probabilities| {
                    let decoded = probabilities
                        .iter()
                        .copied()
                        .enumerate()
                        .max_by(|left, right| left.1.total_cmp(&right.1))
                        .expect("an execution row has an action distribution")
                        .0;
                    action_names[decoded].as_str()
                })
                .collect::<Vec<_>>();
            info!(
                "sgd_trace_decoded update={} rows={:?}",
                point.update, decoded_rows
            );
            info!(
                "sgd_trace_loss update={} precondition_by_row={:?} \
                 transition_by_row={:?} goal_residuals={:?} \
                 recurrent_precondition_by_row={:?} recurrent_terminal_goals={:?} \
                 recurrent_producer_goals={:?} action_integrality_by_row={:?}",
                point.update,
                point.precondition_by_row,
                point.transition_by_row,
                point.goal_residuals,
                point.recurrent_precondition_by_row,
                point.recurrent_terminal_goals,
                point.recurrent_producer_goals,
                point.action_integrality_by_row,
            );
            assert_eq!(
                point.action_probabilities.len(),
                point.action_temperatures.len(),
                "trace stores one temperature per action row"
            );
            for (token, assignment) in point.temporal_assignment.iter().enumerate() {
                info!(
                    "sgd_trace_assignment update={} token={} hard={:?} soft={:?}",
                    point.update, token, assignment, point.temporal_soft_assignment[token]
                );
            }
            for (consumed_scaffold, gradients) in point.schedule_logit_gradients.iter().enumerate()
            {
                info!(
                    "sgd_trace_schedule_gate update={} consumed_scaffold={} gradients_by_consumed_repair={:?}",
                    point.update, consumed_scaffold, gradients
                );
            }
            for (token, probabilities) in point.token_action_probabilities.iter().enumerate() {
                let (decoded, _) = probabilities
                    .iter()
                    .copied()
                    .enumerate()
                    .max_by(|left, right| left.1.total_cmp(&right.1))
                    .expect("every temporal token has the explicit no-op action");
                info!(
                    "sgd_trace_token update={} token={} decoded={} decoded_name={} probabilities={:?} gradients={:?}",
                    point.update,
                    token,
                    decoded,
                    action_names[decoded],
                    probabilities,
                    point.action_logit_gradients[token],
                );
            }
            assert_eq!(
                point.action_logit_gradients.len(),
                point.action_probabilities.len(),
                "trace stores one scheduled-loss gradient per action row"
            );
            for (row, (probabilities, temperature)) in point
                .action_probabilities
                .iter()
                .zip(&point.action_temperatures)
                .enumerate()
            {
                let (decoded, _) = probabilities
                    .iter()
                    .copied()
                    .enumerate()
                    .max_by(|left, right| left.1.total_cmp(&right.1))
                    .expect("every trace row has the explicit no-op action");
                let values = probabilities
                    .iter()
                    .map(|probability| format!("{probability:.9}"))
                    .collect::<Vec<_>>()
                    .join(",");
                if point.token_action_probabilities.is_empty() {
                    info!(
                        "sgd_trace_row update={} row={} temperature={:.9} decoded={} \
                         decoded_name={} probabilities=[{}] gradients={:?}",
                        point.update,
                        row,
                        temperature,
                        decoded,
                        action_names[decoded],
                        values,
                        point.action_logit_gradients[row],
                    );
                } else {
                    info!(
                        "sgd_trace_row update={} row={} temperature={:.9} decoded={} \
                         decoded_name={} probabilities=[{}]",
                        point.update, row, temperature, decoded, action_names[decoded], values,
                    );
                }
            }
        }
    }

    info!(
        "sgd updates {} / verifier_calls {} / horizon_rounds {} / final_horizon {}",
        outcome.updates, outcome.verifier_calls, outcome.horizon_rounds, outcome.final_horizon
    );
    info!(
        "sgd best_goals {}/{} / longest_applicable_prefix {} / best_residual {:.6} / \
         remelts {} / temporal_restarts {} / temporal_order_conflicts {} / \
         temporal_causal_cycles {} / temporal_cycle_interventions {} / \
         temporal_scaffold_repairs {} / refreshes {} / \
         bridge_updates {} / max_bridge_loss {:.6}",
        outcome.best_goals_reached,
        outcome.num_goals,
        outcome.longest_applicable_prefix,
        outcome.best_total_residual,
        outcome.remelts,
        outcome.temporal_restarts,
        outcome.temporal_order_conflicts,
        outcome.temporal_causal_cycles,
        outcome.temporal_cycle_interventions,
        outcome.temporal_scaffold_repairs,
        outcome.refreshes,
        outcome.backward_bridge_updates,
        outcome.max_backward_bridge_loss,
    );
    info!("sgd final relaxation: {}", outcome.final_diagnostics);
    if let Some(checkpoint) = &outcome.best_exact_checkpoint {
        info!("sgd exact_checkpoint: {checkpoint}");
        let operator_names = checkpoint
            .decoded_plan
            .iter()
            .map(|&operator| task.get_operators()[operator].name())
            .collect::<Vec<_>>();
        let slot_names = checkpoint
            .decoded_slots
            .iter()
            .map(|operator| operator.map_or("<noop>", |index| task.get_operators()[index].name()))
            .collect::<Vec<_>>();
        info!(
            "sgd exact_checkpoint_plan: operators={operator_names:?} slots={slot_names:?} \
             missing_goals={:?}",
            checkpoint.missing_goals
        );
    }

    let mut result = SearchResult {
        status: SearchStatus::Failed,
        plan: None,
        solution_cost: None,
        nodes_expanded: 0,
        nodes_reopened: 0,
        nodes_evaluated: 0,
        evaluations: 0,
        nodes_generated: 0,
        dead_ends: 0,
        nodes_expanded_until_last_jump: 0,
        nodes_reopened_until_last_jump: 0,
        nodes_evaluated_until_last_jump: 0,
        nodes_generated_until_last_jump: 0,
        registered_states: 0,
        search_time: elapsed,
    };

    match outcome.status {
        SgdStatus::Solved => {
            let indices = outcome
                .plan
                .expect("a solved outcome always carries a plan");
            // Re-verify here rather than trusting the engine, and take the goal
            // state's id from that replay so the reported status refers to a
            // real state.
            let mut registry = StateRegistry::for_task(task.clone());
            let operators: Vec<&Operator> = indices
                .iter()
                .map(|&index| &task.get_operators()[index])
                .collect();
            let replay = replay_plan(&*task, &mut registry, &global_constraint, &operators)
                .map_err(|error| std::io::Error::other(error.message))?;
            match replay.outcome {
                ReplayOutcome::Solved(verified) => {
                    let goal_state = replay
                        .states
                        .last()
                        .expect("a verified replay visits at least the initial state");
                    result.status = SearchStatus::Solved(goal_state.get_id());
                    result.solution_cost = Some(verified.cost);
                    result.plan = Some(operators.into_iter().cloned().collect());
                    result.registered_states = replay.states.len();
                }
                other => {
                    // The engine claimed a plan the verifier rejects. That is a
                    // bug in the engine, not an unsolved task, and it must not
                    // be reported as "no solution".
                    return Err(std::io::Error::other(format!(
                        "the sgd engine returned a plan that fails verification: {other:?}"
                    )));
                }
            }
        }
        SgdStatus::Unsolvable => {
            info!("sgd: the task is structurally unsolvable");
            result.status = SearchStatus::Failed;
        }
        SgdStatus::BudgetExhausted => {
            info!(
                "sgd: budget exhausted without finding a plan. This does NOT mean the task is \
                 unsolvable -- the gradient optimizer is incomplete by construction."
            );
            result.status = SearchStatus::Failed;
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recursive_config::{SearchSpec, parse_search_spec};

    #[test]
    fn parses_staged_causal_copy_surface() {
        let SearchSpec::Sgd(args) = parse_search_spec(
            "sgd(causal_copy=staged, causal_shadow_end=0.08, causal_discovery_end=0.28, \
             causal_proof_end=0.48, causal_transfer_end=0.68, causal_takeover_end=0.88, \
             q_action_temperature_start=3.0, q_action_temperature_end=0.4, \
             q_logit_perturbation=0.02, teacher_tolerance=0.03, teacher_weight=9.5, \
             applicability_barrier_margin=0.9, remelt_cooldown_updates=100, \
             polish_p_norm=10, noop_suffix_weight=1.5, remelt_stop_progress=0.97, \
             slot_slack_window=4, slot_slack_logit_gap=2.5, slot_slack_weight=0.75, \
             insertion_repair_weight=1.25, insertion_min_prefix_fraction=0.6, \
             anchor_trust_weight=0.5, \
             goal_survival_weight=0.625, backward_bridge_weight=1.75)",
        )
        .expect("staged causal-copy options parse") else {
            panic!("expected sgd search specification");
        };
        let mut config = SgdConfig::default();
        apply_sgd_options(&mut config, &args).expect("staged causal-copy options apply");
        assert_eq!(config.causal_copy, CausalCopyMode::Staged);
        assert_eq!(
            [
                config.causal_shadow_end,
                config.causal_discovery_end,
                config.causal_proof_end,
                config.causal_transfer_end,
                config.causal_takeover_end,
            ],
            [0.08, 0.28, 0.48, 0.68, 0.88]
        );
        assert_eq!(config.q_action_temperature, (3.0, 0.4));
        assert_eq!(config.q_logit_perturbation, 0.02);
        assert_eq!(config.teacher_tolerance, 0.03);
        assert_eq!(config.teacher_weight, 9.5);
        assert_eq!(config.applicability_barrier_margin, 0.9);
        assert_eq!(config.remelt_cooldown_updates, 100);
        assert_eq!(config.polish_p_norm, 10.0);
        assert_eq!(config.noop_suffix_weight, 1.5);
        assert_eq!(config.remelt_stop_progress, 0.97);
        assert_eq!(config.slot_slack_window, 4);
        assert_eq!(config.slot_slack_logit_gap, 2.5);
        assert_eq!(config.slot_slack_weight, 0.75);
        assert_eq!(config.insertion_repair_weight, 1.25);
        assert_eq!(config.insertion_min_prefix_fraction, 0.6);
        assert_eq!(config.anchor_trust_weight, 0.5);
        assert_eq!(config.goal_survival_weight, 0.625);
        assert_eq!(config.backward_bridge_weight, 1.75);
        config
            .validate()
            .expect("parsed staged configuration is valid");
    }

    #[test]
    fn rejects_removed_cyclic_consensus_options() {
        let SearchSpec::Sgd(args) =
            parse_search_spec("sgd(causal_consensus_weight_start=0.25)").unwrap()
        else {
            panic!("expected sgd search specification");
        };
        let error = apply_sgd_options(&mut SgdConfig::default(), &args)
            .expect_err("removed cyclic consensus option must not be accepted");
        assert!(error.contains("unknown option `causal_consensus_weight_start`"));
    }

    #[test]
    fn rejects_unknown_causal_copy_mode() {
        let SearchSpec::Sgd(args) = parse_search_spec("sgd(causal_copy=independent)").unwrap()
        else {
            panic!("expected sgd search specification");
        };
        let error = apply_sgd_options(&mut SgdConfig::default(), &args)
            .expect_err("unknown causal-copy mode must be rejected");
        assert!(error.contains("expected one of `off`, `shadow`, or `staged`"));
    }
}
