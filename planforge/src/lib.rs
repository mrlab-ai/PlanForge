#[cfg(test)]
mod tests;

use clap::Parser;
use ordered_float::NotNan;
use planforge_cli_utils::*;
use planforge_sas::numeric_task::{
    AbstractNumericTask, NumericRootTask, NumericType, Operator, TaskRef,
};
use planforge_sas::state_registry::{ConcreteState, StateRegistry};
use planforge_search::evaluation::domain_abstractions::cegar::{Cegar, CegarConfig};
use planforge_search::evaluation::domain_abstractions::domain_abstraction_generator::{
    DomainAbstraction, DomainAbstractionGenerator, compute_hash_multipliers,
};
use planforge_search::evaluation::domain_abstractions::domain_abstraction_heuristic::DomainAbstractionHeuristic;
use planforge_search::evaluation::evaluator::EvaluationState;
use planforge_search::evaluation::g_evaluator::{GEvaluator, SumEvaluator};
use planforge_search::evaluation::{EvaluationResult, Evaluator};
use planforge_search::open_lists::{OpenList, SearchNode, TieBreakingOpenList};
use planforge_search::search::{AStarSearch, SearchEngine};
use planforge_search::search::{SearchResult, SearchStatus, compute_effective_operator_costs};
use planforge_search::successor_generator::SuccessorTree;
use planforge_search::task_restriction::build_restricted_task;
use planforge_searcher::*;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::ffi::OsString;
use std::num::NonZero;
use std::process::Command;
use std::sync::Arc;
use std::time::Duration;
use time::format_description::well_known::iso8601::{Config, TimePrecision};
use tracing::{debug, info};
use tracing_subscriber::fmt::time::UtcTime;

#[cfg(unix)]
use std::os::unix::process::CommandExt;

use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::prelude::*;

pub fn init_logger(level: LevelFilter) {
    let timer = UtcTime::new(
        time::format_description::well_known::Iso8601::<
            {
                Config::DEFAULT
                    .set_time_precision(TimePrecision::Second {
                        decimal_digits: NonZero::new(3),
                    })
                    .encode()
            },
        >,
    );
    // Layer for stdout (info + debug + trace)
    let stdout_layer = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stdout)
        .with_target(false)
        .with_timer(timer)
        .with_filter(level);

    // Layer for stderr (error + warn only)
    let stderr_layer = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stderr)
        .with_target(false)
        .with_filter(LevelFilter::WARN);

    tracing_subscriber::registry()
        .with(stdout_layer)
        .with(stderr_layer)
        .init();
}

type OpenListElements = (Reverse<NotNan<f64>>, Reverse<NotNan<f64>>, usize);

#[derive(Parser, Debug, Clone)]
#[command(author, version, about = "Numeric planner")]
pub struct PlannersCli {
    #[arg(long = "max-memory", value_name = "SIZE", value_parser = parse_memory_limit)]
    pub max_memory: Option<u64>,

    #[arg(long = "max-time", value_name = "DURATION", value_parser = parse_time_limit)]
    pub max_time: Option<Duration>,

    #[arg(long = "log-level")]
    pub log_level: Option<tracing_subscriber::filter::LevelFilter>,

    #[arg(long, hide = true)]
    pub internal_run: bool,

    #[arg(long = "restrict-task")]
    pub restrict_task: bool,

    /// Recursive search configuration.
    /// Examples: `astar(blind())`, `astar(domain_abstraction())`, `da_debug()`.
    #[arg(
        long,
        value_name = "SPEC",
        default_value = "astar(blind())",
        value_parser = planforge_searcher::parse_search_spec
    )]
    pub search: planforge_searcher::SearchSpec,

    #[arg(value_name = "INPUT", required = true, num_args = 1..=2)]
    pub inputs: Vec<String>,
}

#[cfg(unix)]
pub fn run_wrapped_process(cli: &PlannersCli) -> std::io::Result<()> {
    let current_executable = std::env::current_exe()?;
    let mut child_args = vec![OsString::from("--internal-run")];
    if let Some(level) = cli.log_level {
        child_args.push(OsString::from("--log-level"));
        child_args.push(OsString::from(level.to_string()));
    }
    if cli.restrict_task {
        child_args.push(OsString::from("--restrict-task"));
    }
    child_args.push(OsString::from("--search"));
    child_args.push(OsString::from(cli.search.to_string()));
    child_args.extend(cli.inputs.iter().cloned().map(OsString::from));

    let time_limit = cli.max_time;
    let memory_limit = cli.max_memory;

    let mut command = Command::new(current_executable);
    command.args(child_args);
    command.stdin(std::process::Stdio::inherit());
    command.stdout(std::process::Stdio::inherit());
    command.stderr(std::process::Stdio::inherit());

    unsafe {
        command.pre_exec(move || apply_process_limits(time_limit, memory_limit));
    }

    let status = command.status()?;
    let exit_code = normalize_wrapped_exit(status, time_limit, memory_limit);

    std::process::exit(exit_code)
}

/// Run search for an already-parsed task and return the result. Contains no
/// CLI, signal-handler, logging-setup, or file-writing side effects -- those
/// stay in `run_internal`. Handles the astar / gbfs / astar_fs specs; the
/// debug-only specs are handled directly in `run_internal`.
pub fn solve_task(
    task: TaskRef<'_>,
    spec: &planforge_searcher::SearchSpec,
    time_limit: Option<Duration>,
    memory_limit: Option<u64>,
) -> std::io::Result<SearchResult> {
    let state_registry = StateRegistry::for_task(task.clone());
    match spec {
        planforge_searcher::SearchSpec::Astar(heuristic)
        | planforge_searcher::SearchSpec::Gbfs(heuristic) => {
            // GBFS and A* share heuristic construction; only the open-list
            // priority differs (h vs g+h).
            let gbfs_priority = matches!(spec, planforge_searcher::SearchSpec::Gbfs(_));
            let heuristic_override = build_heuristic_from_spec(heuristic, &*task)?;
            let mut search = if gbfs_priority {
                AStarSearch::new_gbfs(
                    task.clone(),
                    state_registry,
                    heuristic_override,
                    time_limit,
                    memory_limit,
                )
            } else {
                AStarSearch::new(
                    task.clone(),
                    state_registry,
                    heuristic_override,
                    time_limit,
                    memory_limit,
                )
            };

            info!(
                "Starting {} search with {:?}...",
                if gbfs_priority { "GBFS" } else { "A*" },
                heuristic,
            );
            Ok(search.search())
        }
        planforge_searcher::SearchSpec::AstarFs(fast_spec, slow_spec) => {
            // A* with two admissible heuristics: a fast one for ordering
            // and a slow one evaluated lazily on second-pop. Treats the
            // user's `blind` choice as a placeholder by materializing a
            // real `BlindHeuristic` with the task's min-action-cost.
            let task_ref: &dyn AbstractNumericTask = &*task;
            let original_costs: Vec<f64> = task
                .get_operators()
                .iter()
                .map(|op| {
                    planforge_sas::numeric_task::metric_operator_cost_from_initial_values(
                        task_ref, op,
                    )
                })
                .collect();
            let min_cost = original_costs
                .iter()
                .copied()
                .fold(f64::INFINITY, |a, b| a.min(b));
            let min_action_cost = if min_cost.is_finite() {
                min_cost.max(0.0)
            } else {
                1.0
            };
            let make_blind = || {
                Box::new(
                    planforge_search::evaluation::heuristic::BlindHeuristic::with_min_action_cost(
                        min_action_cost,
                        None,
                    ),
                ) as Box<dyn planforge_search::evaluation::Heuristic + '_>
            };
            let fast_h = build_heuristic_from_spec(fast_spec, task_ref)?.unwrap_or_else(make_blind);
            let slow_h = build_heuristic_from_spec(slow_spec, task_ref)?.unwrap_or_else(make_blind);
            let mut search = AStarSearch::new_fast_slow(
                task.clone(),
                state_registry,
                fast_h,
                slow_h,
                time_limit,
                memory_limit,
            );
            info!("Starting A* fast/slow search with fast={fast_spec:?} slow={slow_spec:?}...");
            Ok(search.search())
        }
        planforge_searcher::SearchSpec::DaDebug | planforge_searcher::SearchSpec::AstarDaDebug => {
            Err(std::io::Error::other(
                "solve_task does not handle debug search specs",
            ))
        }
    }
}

#[allow(clippy::field_reassign_with_default)]
pub fn run_internal(cli: &PlannersCli) -> std::io::Result<SearchResult> {
    register_event_handlers();

    let start_time = std::time::Instant::now();
    let (mut task, sas_label) = if cli.inputs.len() == 2 {
        let domain = &cli.inputs[0];
        let problem = &cli.inputs[1];
        // In-memory pipeline: translate → preprocess → parse, no disk I/O.
        let sas_text = planforge_translator::translate_to_sas_string(domain, problem)
            .map_err(|err| std::io::Error::other(err.to_string()))?;
        let preprocessed = planforge_translate::preprocess::run_preprocess_to_string(&sas_text);
        (
            NumericRootTask::from_str(&preprocessed),
            format!("{domain} + {problem} (in-memory)"),
        )
    } else {
        let path = cli.inputs[0].clone();
        (NumericRootTask::from_file(&path), path)
    };
    if cli.restrict_task {
        let original_numeric_count = task.numeric_variables().len();
        if let Some(restricted_task) = build_restricted_task(&task).map_err(|err| {
            std::io::Error::other(format!("failed to build restricted task: {err:#}"))
        })? {
            task = restricted_task.into_task();
            info!(
                "restricted task: numeric variables {} -> {}",
                original_numeric_count,
                task.numeric_variables().len()
            );
        }
    }
    let task: TaskRef<'static> = Arc::new(task);
    let parse_time = start_time.elapsed();
    info!("Parsed numeric SAS output in: {:?}", parse_time);

    info!("=== Search Engine ===");
    info!("File: {}", sas_label);
    info!(
        "Variables: {} regular, {} numeric",
        task.variables().len(),
        task.numeric_variables().len()
    );

    let time_limit = if cli.internal_run { None } else { cli.max_time };
    let memory_limit = if cli.internal_run {
        None
    } else {
        cli.max_memory
    };
    let result = match &cli.search {
        planforge_searcher::SearchSpec::Astar(_)
        | planforge_searcher::SearchSpec::Gbfs(_)
        | planforge_searcher::SearchSpec::AstarFs(_, _) => {
            solve_task(task.clone(), &cli.search, time_limit, memory_limit)?
        }
        planforge_searcher::SearchSpec::DaDebug => run_da_debug(
            &*task,
            StateRegistry::for_task(task.clone()),
            if cli.internal_run { None } else { cli.max_time },
        )?,
        planforge_searcher::SearchSpec::AstarDaDebug => run_astar_da_debug(
            &*task,
            StateRegistry::for_task(task.clone()),
            if cli.internal_run { None } else { cli.max_time },
        )?,
    };

    print_search_result(&result);

    Ok(result)
}

fn build_heuristic_from_spec<'a>(
    spec: &planforge_searcher::HeuristicSpec,
    task_ref: &'a dyn AbstractNumericTask,
) -> std::io::Result<Option<Box<dyn planforge_search::evaluation::Heuristic + 'a>>> {
    planforge_searcher::build_heuristic_from_spec(spec, task_ref).map_err(std::io::Error::other)
}

fn build_successor_generator(task: &dyn AbstractNumericTask) -> SuccessorTree {
    SuccessorTree::new(task)
}

fn state_is_goal(
    task: &dyn AbstractNumericTask,
    state_registry: &StateRegistry<'_>,
    state: &ConcreteState,
) -> bool {
    for i in 0..task.get_num_goals() {
        let goal_fact = task.get_goal_fact(i);
        if !goal_fact.is_hold(state, state_registry) {
            return false;
        }
    }
    true
}

fn operator_cost(operator_costs: &[f64], operator_id: usize, operator: &Operator) -> f64 {
    operator_costs
        .get(operator_id)
        .copied()
        .unwrap_or(operator.cost() as f64)
}

fn evaluate_da_heuristic(
    task: &dyn AbstractNumericTask,
    state_registry: &StateRegistry<'_>,
    heuristic: &DomainAbstractionHeuristic,
    state: &ConcreteState,
    g_value: f64,
) -> std::io::Result<f64> {
    let mut eval_state =
        EvaluationState::new_with_registry(state, g_value, false, task, state_registry);
    eval_state.set_is_goal(state_is_goal(task, state_registry, state));
    eval_state
        .get_or_compute_heuristic(heuristic)
        .map_err(|e| std::io::Error::other(format!("failed to evaluate DA heuristic: {e}")))
}

fn evaluate_da_state(
    task: &dyn AbstractNumericTask,
    state_registry: &StateRegistry<'_>,
    heuristic: &DomainAbstractionHeuristic,
    g_evaluator: &GEvaluator,
    f_evaluator: &SumEvaluator,
    state: &ConcreteState,
    g_value: f64,
) -> std::io::Result<EvaluationResult> {
    let mut eval_state =
        EvaluationState::new_with_registry(state, g_value, false, task, state_registry);
    eval_state.set_is_goal(state_is_goal(task, state_registry, state));
    g_evaluator
        .evaluate_state(&mut eval_state)
        .map_err(|e| std::io::Error::other(format!("failed to evaluate g-value: {e}")))?;
    heuristic
        .evaluate_state(&mut eval_state)
        .map_err(|e| std::io::Error::other(format!("failed to evaluate DA heuristic: {e}")))?;
    f_evaluator
        .evaluate_state(&mut eval_state)
        .map_err(|e| std::io::Error::other(format!("failed to evaluate f-value: {e}")))?;
    Ok(eval_state.into_result())
}

fn build_da_heuristic(
    task: &dyn AbstractNumericTask,
    name: Option<String>,
) -> std::io::Result<DomainAbstractionHeuristic> {
    info!("Building domain abstraction (CEGAR)...");
    let config = CegarConfig {
        debug: true,
        ..Default::default()
    };

    let generator = DomainAbstractionGenerator::new(config).map_err(|e| {
        std::io::Error::other(format!(
            "failed to construct DomainAbstractionGenerator: {e:#}"
        ))
    })?;
    let abstraction = generator
        .generate(task)
        .map_err(|e| std::io::Error::other(format!("failed to build domain abstraction: {e:#}")))?;
    Ok(DomainAbstractionHeuristic::new(name, abstraction))
}

#[derive(Debug, Clone)]
struct DebugSearchNodeInfo {
    parent_state: Option<usize>,
    parent_operator_id: Option<usize>,
    g_value: f64,
}

#[derive(Debug, Clone)]
struct AdmissibilityWitness {
    phase: &'static str,
    state_id: usize,
    g_value: f64,
    h_value: f64,
    f_value: f64,
    blind_remaining: f64,
}

fn first_witness_line(prefix: &str, witness: &Option<AdmissibilityWitness>) {
    match witness {
        Some(witness) => debug!(
            "[{prefix}] first {} witness: sid={} g={:.3} h={:.3} f={:.3} blind_remaining={:.3} delta={:.3}",
            witness.phase,
            witness.state_id,
            witness.g_value,
            witness.h_value,
            witness.f_value,
            witness.blind_remaining,
            witness.h_value - witness.blind_remaining,
        ),
        None => debug!("[{prefix}] no inadmissible states found."),
    }
}

fn blind_min_action_cost(operator_costs: &[f64]) -> f64 {
    let min_cost = operator_costs
        .iter()
        .copied()
        .fold(f64::INFINITY, |left, right| left.min(right));

    if min_cost.is_finite() {
        min_cost.max(0.0)
    } else {
        1.0
    }
}

fn blind_remaining_distance(
    task: &dyn AbstractNumericTask,
    state_registry: &mut StateRegistry<'_>,
    start_state: &ConcreteState,
    operator_costs: &[f64],
) -> std::io::Result<f64> {
    if state_is_goal(task, state_registry, start_state) {
        return Ok(0.0);
    }

    let min_action_cost = blind_min_action_cost(operator_costs);
    let mut best_g: HashMap<usize, f64> = HashMap::new();
    let mut open: BinaryHeap<OpenListElements> = BinaryHeap::new();
    let successor_generator = build_successor_generator(task);
    let mut state_values_buffer: Vec<usize> = Vec::new();
    let mut applicable_operators: Vec<u32> = Vec::new();
    let mut successor_numeric_values: Vec<f64> = Vec::new();
    let mut successor_cost_values: Vec<f64> = Vec::new();

    best_g.insert(start_state.get_id(), 0.0);
    open.push((
        Reverse(NotNan::new(min_action_cost).map_err(|_| {
            std::io::Error::other("blind remaining distance encountered NaN initial f-value")
        })?),
        Reverse(NotNan::new(0.0).unwrap()),
        start_state.get_id(),
    ));

    while let Some((Reverse(_f_value), Reverse(g), state_id)) = open.pop() {
        let g = g.into_inner();
        let Some(&known_best) = best_g.get(&state_id) else {
            continue;
        };
        if g > known_best + 1e-12 {
            continue;
        }

        let state = state_registry.lookup_state(state_id).map_err(|e| {
            std::io::Error::other(format!("failed to look up state {state_id}: {e:?}"))
        })?;
        if state_is_goal(task, state_registry, &state) {
            return Ok(g);
        }

        state.fill_state(state_registry, &mut state_values_buffer);
        applicable_operators.clear();
        successor_generator
            .get_applicable_operators(&state_values_buffer, &mut applicable_operators);

        for &op_id in applicable_operators.iter() {
            let operator_id = op_id as usize;
            let operator = &task.get_operators()[operator_id];
            let successor = state_registry
                .get_successor_state_with_buffers(
                    &state,
                    operator,
                    &mut successor_numeric_values,
                    &mut successor_cost_values,
                )
                .map_err(|e| {
                    std::io::Error::other(format!(
                        "failed to generate successor for {}: {e:?}",
                        operator.name()
                    ))
                })?;
            let step_cost = operator_cost(operator_costs, operator_id, operator);
            let new_g = g + step_cost;
            let successor_id = successor.get_id();
            if new_g + 1e-12 >= *best_g.get(&successor_id).unwrap_or(&f64::INFINITY) {
                continue;
            }

            best_g.insert(successor_id, new_g);
            let successor_h = if state_is_goal(task, state_registry, &successor) {
                0.0
            } else {
                min_action_cost
            };
            let successor_f = new_g + successor_h;
            open.push((
                Reverse(NotNan::new(successor_f).map_err(|_| {
                    std::io::Error::other("blind remaining distance encountered NaN f-value")
                })?),
                Reverse(NotNan::new(new_g).map_err(|_| {
                    std::io::Error::other("blind remaining distance encountered NaN g-value")
                })?),
                successor_id,
            ));
        }
    }

    Ok(f64::INFINITY)
}

fn blind_remaining_distance_cached(
    task: &dyn AbstractNumericTask,
    state_registry: &mut StateRegistry<'_>,
    cache: &mut HashMap<usize, f64>,
    state: &ConcreteState,
    operator_costs: &[f64],
) -> std::io::Result<f64> {
    if let Some(value) = cache.get(&state.get_id()).copied() {
        return Ok(value);
    }
    let value = blind_remaining_distance(task, state_registry, state, operator_costs)?;
    cache.insert(state.get_id(), value);
    Ok(value)
}

#[allow(clippy::too_many_arguments)]
fn record_admissibility_check(
    prefix: &str,
    phase: &'static str,
    task: &dyn AbstractNumericTask,
    state_registry: &mut StateRegistry<'_>,
    blind_cache: &mut HashMap<usize, f64>,
    state: &ConcreteState,
    operator_costs: &[f64],
    evaluation: &EvaluationResult,
    first_witness: &mut Option<AdmissibilityWitness>,
    checked_count: &mut usize,
) -> std::io::Result<()> {
    *checked_count += 1;
    let h_name = format!("f_{}", prefix.to_lowercase());
    let h_value = evaluation
        .get_heuristic_value_optional(prefix)
        .unwrap_or_else(|| evaluation.get_heuristic_value(prefix));
    let f_value = evaluation
        .get_heuristic_value_optional(&h_name)
        .unwrap_or_else(|| evaluation.get_f_value(prefix));
    let blind_remaining =
        blind_remaining_distance_cached(task, state_registry, blind_cache, state, operator_costs)?;

    if first_witness.is_none() && h_value > blind_remaining + 1e-9 {
        *first_witness = Some(AdmissibilityWitness {
            phase,
            state_id: state.get_id(),
            g_value: evaluation.g_value,
            h_value,
            f_value,
            blind_remaining,
        });
    }
    Ok(())
}

fn extract_debug_plan(
    task: &dyn AbstractNumericTask,
    search_nodes: &HashMap<usize, DebugSearchNodeInfo>,
    goal_state: usize,
) -> Vec<Operator> {
    let mut plan = Vec::new();
    let mut current_state = goal_state;

    while let Some(node_info) = search_nodes.get(&current_state) {
        if let (Some(parent_state), Some(operator_id)) =
            (node_info.parent_state, node_info.parent_operator_id)
        {
            plan.push(task.get_operators()[operator_id].clone());
            current_state = parent_state;
        } else {
            break;
        }
    }

    plan.reverse();
    plan
}

fn format_operator_sequence(plan: &[Operator]) -> String {
    if plan.is_empty() {
        "<empty>".to_string()
    } else {
        plan.iter()
            .map(|operator| operator.name().to_string())
            .collect::<Vec<_>>()
            .join(" -> ")
    }
}

fn format_state_snapshot(
    task: &dyn AbstractNumericTask,
    state_registry: &StateRegistry<'_>,
    state: &ConcreteState,
) -> std::io::Result<(String, String)> {
    let propositional_values = state.get_state(state_registry);
    let proposition_names = propositional_values
        .iter()
        .enumerate()
        .map(|(var_id, value)| {
            let var_name = task.get_variable_name(var_id).unwrap_or("<unknown-var>");
            format!("{}={}", var_name, value)
        })
        .collect::<Vec<_>>()
        .join(" | ");

    let numeric_values = state_registry
        .get_numeric_vars(state)
        .map_err(|e| std::io::Error::other(format!("failed to extract numeric values: {e:?}")))?;
    let numeric_summary = task
        .numeric_variables()
        .iter()
        .enumerate()
        .filter(|(_, numeric_var)| numeric_var.get_type() == &NumericType::Regular)
        .map(|(var_id, numeric_var)| {
            format!("{}={:.3}", numeric_var.name(), numeric_values[var_id])
        })
        .collect::<Vec<_>>()
        .join(", ");

    Ok((proposition_names, numeric_summary))
}

fn exact_remaining_plan(
    task: &dyn AbstractNumericTask,
    state_registry: &mut StateRegistry<'_>,
    start_state: &ConcreteState,
    operator_costs: &[f64],
) -> std::io::Result<(f64, Vec<Operator>)> {
    if state_is_goal(task, state_registry, start_state) {
        return Ok((0.0, Vec::new()));
    }

    let mut best_g: HashMap<usize, f64> = HashMap::new();
    let mut parent: HashMap<usize, (usize, usize)> = HashMap::new();
    let mut open: BinaryHeap<(Reverse<NotNan<f64>>, usize)> = BinaryHeap::new();
    let successor_generator = build_successor_generator(task);
    let mut state_values_buffer: Vec<usize> = Vec::new();
    let mut applicable_operators: Vec<u32> = Vec::new();
    let mut successor_numeric_values: Vec<f64> = Vec::new();
    let mut successor_cost_values: Vec<f64> = Vec::new();

    best_g.insert(start_state.get_id(), 0.0);
    open.push((Reverse(NotNan::new(0.0).unwrap()), start_state.get_id()));

    while let Some((Reverse(g), state_id)) = open.pop() {
        let g = g.into_inner();
        let Some(&known_best) = best_g.get(&state_id) else {
            continue;
        };
        if g > known_best + 1e-12 {
            continue;
        }

        let state = state_registry.lookup_state(state_id).map_err(|e| {
            std::io::Error::other(format!("failed to look up state {state_id}: {e:?}"))
        })?;
        if state_is_goal(task, state_registry, &state) {
            let mut plan = Vec::new();
            let mut current_state = state_id;
            while let Some((parent_state, operator_id)) = parent.get(&current_state).copied() {
                plan.push(task.get_operators()[operator_id].clone());
                current_state = parent_state;
            }
            plan.reverse();
            return Ok((g, plan));
        }

        state.fill_state(state_registry, &mut state_values_buffer);
        applicable_operators.clear();
        successor_generator
            .get_applicable_operators(&state_values_buffer, &mut applicable_operators);

        for &op_id in applicable_operators.iter() {
            let operator_id = op_id as usize;
            let operator = &task.get_operators()[operator_id];
            let successor = state_registry
                .get_successor_state_with_buffers(
                    &state,
                    operator,
                    &mut successor_numeric_values,
                    &mut successor_cost_values,
                )
                .map_err(|e| {
                    std::io::Error::other(format!(
                        "failed to generate successor for {}: {e:?}",
                        operator.name()
                    ))
                })?;
            let step_cost = operator_cost(operator_costs, operator_id, operator);
            let new_g = g + step_cost;
            let successor_id = successor.get_id();
            if new_g + 1e-12 < *best_g.get(&successor_id).unwrap_or(&f64::INFINITY) {
                best_g.insert(successor_id, new_g);
                parent.insert(successor_id, (state_id, operator_id));
                open.push((
                    Reverse(NotNan::new(new_g).map_err(|_| {
                        std::io::Error::other("exact remaining plan encountered NaN cost")
                    })?),
                    successor_id,
                ));
            }
        }
    }

    Ok((f64::INFINITY, Vec::new()))
}

fn print_witness_details(
    label: &str,
    witness: &Option<AdmissibilityWitness>,
    task: &dyn AbstractNumericTask,
    state_registry: &mut StateRegistry<'_>,
    search_nodes: &HashMap<usize, DebugSearchNodeInfo>,
    operator_costs: &[f64],
) -> std::io::Result<()> {
    let Some(witness) = witness else {
        return Ok(());
    };

    let witness_state = state_registry.lookup_state(witness.state_id).map_err(|e| {
        std::io::Error::other(format!(
            "failed to look up witness state {}: {e:?}",
            witness.state_id
        ))
    })?;
    let prefix_plan = extract_debug_plan(task, search_nodes, witness.state_id);
    let (exact_suffix_cost, exact_suffix_plan) =
        exact_remaining_plan(task, state_registry, &witness_state, operator_costs)?;
    let (props, nums) = format_state_snapshot(task, state_registry, &witness_state)?;

    debug!(
        "[{label}] {} witness details: sid={} g={:.3} h={:.3} f={:.3} blind_remaining={:.3} delta={:.3}",
        witness.phase,
        witness.state_id,
        witness.g_value,
        witness.h_value,
        witness.f_value,
        witness.blind_remaining,
        witness.h_value - witness.blind_remaining,
    );
    info!(
        "[{label}] prefix_len={} prefix={}",
        prefix_plan.len(),
        format_operator_sequence(&prefix_plan)
    );
    info!("[{label}] state props: {props}");
    info!("[{label}] state nums: {nums}");
    info!(
        "[{label}] exact_suffix_len={} exact_suffix_cost={:.3} exact_suffix={}",
        exact_suffix_plan.len(),
        exact_suffix_cost,
        format_operator_sequence(&exact_suffix_plan)
    );

    Ok(())
}

fn exact_remaining_distance(
    task: &dyn AbstractNumericTask,
    state_registry: &mut StateRegistry<'_>,
    start_state: &ConcreteState,
    operator_costs: &[f64],
) -> std::io::Result<f64> {
    if state_is_goal(task, state_registry, start_state) {
        return Ok(0.0);
    }

    let mut best_g: HashMap<usize, f64> = HashMap::new();
    let mut open: BinaryHeap<(Reverse<NotNan<f64>>, usize)> = BinaryHeap::new();
    let successor_generator = build_successor_generator(task);
    let mut state_values_buffer: Vec<usize> = Vec::new();
    let mut applicable_operators: Vec<u32> = Vec::new();
    let mut successor_numeric_values: Vec<f64> = Vec::new();
    let mut successor_cost_values: Vec<f64> = Vec::new();

    best_g.insert(start_state.get_id(), 0.0);
    open.push((Reverse(NotNan::new(0.0).unwrap()), start_state.get_id()));

    while let Some((Reverse(g), state_id)) = open.pop() {
        let g = g.into_inner();
        let Some(&known_best) = best_g.get(&state_id) else {
            continue;
        };
        if g > known_best + 1e-12 {
            continue;
        }

        let state = state_registry.lookup_state(state_id).map_err(|e| {
            std::io::Error::other(format!("failed to look up state {state_id}: {e:?}"))
        })?;
        if state_is_goal(task, state_registry, &state) {
            return Ok(g);
        }

        state.fill_state(state_registry, &mut state_values_buffer);
        applicable_operators.clear();
        successor_generator
            .get_applicable_operators(&state_values_buffer, &mut applicable_operators);

        for &op_id in applicable_operators.iter() {
            let operator_id = op_id as usize;
            let operator = &task.get_operators()[operator_id];
            let successor = state_registry
                .get_successor_state_with_buffers(
                    &state,
                    operator,
                    &mut successor_numeric_values,
                    &mut successor_cost_values,
                )
                .map_err(|e| {
                    std::io::Error::other(format!(
                        "failed to generate successor for {}: {e:?}",
                        operator.name()
                    ))
                })?;
            let step_cost = operator_cost(operator_costs, operator_id, operator);
            let new_g = g + step_cost;
            let successor_id = successor.get_id();
            if new_g + 1e-12 < *best_g.get(&successor_id).unwrap_or(&f64::INFINITY) {
                best_g.insert(successor_id, new_g);
                open.push((
                    Reverse(NotNan::new(new_g).map_err(|_| {
                        std::io::Error::other("exact remaining distance encountered NaN cost")
                    })?),
                    successor_id,
                ));
            }
        }
    }

    Ok(f64::INFINITY)
}

fn run_da_debug(
    task: &dyn AbstractNumericTask,
    mut state_registry: StateRegistry<'_>,
    _time_limit: Option<Duration>,
) -> std::io::Result<SearchResult> {
    debug!(
        "Running da_debug(): build terminal domain abstraction, replay wildcard plan, and compare h(s) to exact remaining distance."
    );

    let config = CegarConfig {
        debug: true,
        ..Default::default()
    };

    let cegar = Cegar::new(config.clone())
        .map_err(|e| std::io::Error::other(format!("failed to construct CEGAR: {e:#}")))?;
    let outcome = cegar
        .build_abstraction(task)
        .map_err(|e| std::io::Error::other(format!("failed to build abstraction: {e:#}")))?;

    let wildcard_plan = outcome.last_step.wildcard_plan.clone().ok_or_else(|| {
        std::io::Error::other("da_debug requires a final wildcard plan, but CEGAR returned none")
    })?;
    let factory = outcome.final_state.factory.clone();
    let distance_table = factory
        .build_abstract_distance_table(task, config.combine_labels, false)
        .map_err(|e| std::io::Error::other(format!("failed to build distance table: {e:#}")))?;
    let hash_multipliers =
        compute_hash_multipliers(factory.domain_sizes(), factory.numeric_domain_sizes()).map_err(
            |e| std::io::Error::other(format!("failed to compute hash multipliers: {e:#}")),
        )?;
    let abstraction = DomainAbstraction {
        factory,
        distance_table,
        hash_multipliers,
        combine_labels: config.combine_labels,
        relevant_operator_ids: Vec::new(),
        abstract_operators: Vec::new(),
        abstract_operator_footprints: Vec::new(),
        metadata: Default::default(),
    };
    let heuristic = DomainAbstractionHeuristic::new(Some("da_debug".to_string()), abstraction);

    let mut concrete_plan: Vec<Operator> = Vec::new();
    let mut current_state = state_registry.get_initial_state();
    let operator_costs = compute_effective_operator_costs(task, &state_registry, &current_state);
    let mut total_cost = 0.0;
    let mut witness: Option<(usize, f64, f64)> = None;

    debug!(
        "[DA_DEBUG] wildcard steps={}",
        wildcard_plan.wildcard_plan.len()
    );

    let initial_h = evaluate_da_heuristic(task, &state_registry, &heuristic, &current_state, 0.0)?;
    let initial_exact =
        exact_remaining_distance(task, &mut state_registry, &current_state, &operator_costs)?;
    debug!(
        "[DA_DEBUG] state=0 g=0.000 h={initial_h:.3} exact_remaining={initial_exact:.3} delta={:.3}",
        initial_h - initial_exact
    );
    if initial_h > initial_exact + 1e-9 {
        witness = Some((0, initial_h, initial_exact));
    }

    let mut state_values_buffer: Vec<usize> = Vec::new();
    let mut applicable_operators: Vec<u32> = Vec::new();

    for (step_idx, candidate_ids) in wildcard_plan.wildcard_plan.iter().enumerate() {
        let successor_generator = build_successor_generator(task);
        current_state.fill_state(&state_registry, &mut state_values_buffer);
        applicable_operators.clear();
        successor_generator
            .get_applicable_operators(&state_values_buffer, &mut applicable_operators);

        let chosen = candidate_ids.iter().find_map(|candidate_id| {
            applicable_operators
                .iter()
                .copied()
                .find(|&applicable_id| applicable_id as usize == *candidate_id)
        });
        let operator_id = chosen.ok_or_else(|| {
            std::io::Error::other(format!(
                "no applicable concrete operator found for wildcard step {} candidates {:?}",
                step_idx + 1,
                candidate_ids
            ))
        })? as usize;
        let operator = &task.get_operators()[operator_id];

        let successor = state_registry
            .get_successor_state(&current_state, operator)
            .map_err(|e| {
                std::io::Error::other(format!(
                    "failed to apply operator {} at step {}: {e:?}",
                    operator.name(),
                    step_idx + 1
                ))
            })?;
        let step_cost = operator_cost(&operator_costs, operator_id, operator);
        total_cost += step_cost;
        concrete_plan.push(operator.clone());
        current_state = successor;

        let h_value = evaluate_da_heuristic(
            task,
            &state_registry,
            &heuristic,
            &current_state,
            total_cost,
        )?;
        let exact_remaining =
            exact_remaining_distance(task, &mut state_registry, &current_state, &operator_costs)?;
        debug!(
            "[DA_DEBUG] state={} g={:.3} step={} op=[{}:{}] h={:.3} exact_remaining={:.3} delta={:.3}",
            step_idx + 1,
            total_cost,
            step_idx + 1,
            operator_id,
            operator.name(),
            h_value,
            exact_remaining,
            h_value - exact_remaining
        );
        if witness.is_none() && h_value > exact_remaining + 1e-9 {
            witness = Some((step_idx + 1, h_value, exact_remaining));
        }
    }

    if let Some((state_index, h_value, exact_remaining)) = witness {
        debug!(
            "[DA_DEBUG][WITNESS] first inadmissible state={} h={:.3} exact_remaining={:.3} delta={:.3}",
            state_index,
            h_value,
            exact_remaining,
            h_value - exact_remaining
        );
    } else {
        debug!(
            "[DA_DEBUG] no inadmissibility witness found along the final wildcard-plan execution."
        );
    }

    let solved = state_is_goal(task, &state_registry, &current_state);
    if solved {
        debug!("[DA_DEBUG] replayed concrete plan reaches the goal.");
    } else {
        debug!("[DA_DEBUG] replayed concrete plan does not reach the goal.");
    }

    Ok(SearchResult {
        status: if solved {
            SearchStatus::Solved(current_state.get_id())
        } else {
            SearchStatus::Failed
        },
        plan: Some(concrete_plan),
        solution_cost: Some(total_cost),
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
        registered_states: state_registry.num_registered_states(),
        search_time: Duration::ZERO,
    })
}

fn run_astar_da_debug(
    task: &dyn AbstractNumericTask,
    mut state_registry: StateRegistry<'_>,
    _time_limit: Option<Duration>,
) -> std::io::Result<SearchResult> {
    debug!(
        "Running astar_da_debug(): execute DA-guided A* and compare h(s) to exact remaining distance on the states A* actually touches."
    );

    let heuristic = build_da_heuristic(task, Some("astar_da_debug".to_string()))?;
    let g_evaluator = GEvaluator::new(None);
    let f_evaluator = SumEvaluator::f_evaluator(heuristic.name());
    let mut open_list = TieBreakingOpenList::new(vec![f_evaluator.name(), heuristic.name()], true)
        .expect("A* tie-breaking open list must have at least one evaluator");
    let successor_generator = build_successor_generator(task);

    let mut search_nodes: HashMap<usize, DebugSearchNodeInfo> = HashMap::new();
    let mut closed_set: HashSet<usize> = HashSet::new();
    let mut blind_cache: HashMap<usize, f64> = HashMap::new();

    let mut nodes_evaluated = 0usize;
    let mut nodes_expanded = 0usize;
    let mut nodes_reopened = 0usize;
    let mut nodes_generated = 0usize;
    let mut dead_ends = 0usize;
    let mut evaluated_checks = 0usize;
    let mut expanded_checks = 0usize;
    let mut first_evaluated_witness: Option<AdmissibilityWitness> = None;
    let mut first_expanded_witness: Option<AdmissibilityWitness> = None;

    let mut state_values_buffer: Vec<usize> = Vec::new();
    let mut applicable_operators: Vec<u32> = Vec::new();
    let mut successor_numeric_values: Vec<f64> = Vec::new();
    let mut successor_cost_values: Vec<f64> = Vec::new();

    let initial_state = state_registry.get_initial_state();
    let operator_costs = compute_effective_operator_costs(task, &state_registry, &initial_state);
    let initial_evaluation = evaluate_da_state(
        task,
        &state_registry,
        &heuristic,
        &g_evaluator,
        &f_evaluator,
        &initial_state,
        0.0,
    )?;
    nodes_evaluated += 1;
    if initial_evaluation.is_dead_end {
        dead_ends += 1;
    }
    record_admissibility_check(
        &heuristic.name(),
        "evaluated",
        task,
        &mut state_registry,
        &mut blind_cache,
        &initial_state,
        &operator_costs,
        &initial_evaluation,
        &mut first_evaluated_witness,
        &mut evaluated_checks,
    )?;
    if first_evaluated_witness.is_some() {
        print_witness_details(
            "ASTAR_DA_DEBUG",
            &first_evaluated_witness,
            task,
            &mut state_registry,
            &search_nodes,
            &operator_costs,
        )?;
        return Err(std::io::Error::other(
            "blind-A* admissibility assertion failed on an evaluated state",
        ));
    }
    if !initial_evaluation.is_dead_end || open_list.accepts_dead_ends() {
        open_list.insert(SearchNode::root(initial_state.clone(), initial_evaluation));
    }
    search_nodes.insert(
        initial_state.get_id(),
        DebugSearchNodeInfo {
            parent_state: None,
            parent_operator_id: None,
            g_value: 0.0,
        },
    );

    let result = loop {
        if open_list.is_empty() {
            break SearchResult {
                status: SearchStatus::Failed,
                plan: None,
                solution_cost: None,
                nodes_expanded,
                nodes_reopened,
                nodes_evaluated,
                evaluations: nodes_evaluated,
                nodes_generated,
                dead_ends,
                nodes_expanded_until_last_jump: 0,
                nodes_reopened_until_last_jump: 0,
                nodes_evaluated_until_last_jump: 0,
                nodes_generated_until_last_jump: 0,
                registered_states: state_registry.num_registered_states(),
                search_time: Duration::ZERO,
            };
        }

        let node = match open_list.pop() {
            Some(node) => node,
            None => continue,
        };
        let state_id = node.state.get_id();

        if closed_set.contains(&state_id) {
            continue;
        }
        if let Some(current_info) = search_nodes.get(&state_id)
            && current_info.g_value < node.g_value()
        {
            continue;
        }

        nodes_expanded += 1;
        record_admissibility_check(
            &heuristic.name(),
            "expanded",
            task,
            &mut state_registry,
            &mut blind_cache,
            &node.state,
            &operator_costs,
            &node.evaluation,
            &mut first_expanded_witness,
            &mut expanded_checks,
        )?;

        closed_set.insert(state_id);
        if state_is_goal(task, &state_registry, &node.state) {
            let plan = extract_debug_plan(task, &search_nodes, state_id);
            let solution_cost = search_nodes.get(&state_id).map(|info| info.g_value);
            break SearchResult {
                status: SearchStatus::Solved(state_id),
                plan: Some(plan),
                solution_cost,
                nodes_expanded,
                nodes_reopened,
                nodes_evaluated,
                evaluations: nodes_evaluated,
                nodes_generated,
                dead_ends,
                nodes_expanded_until_last_jump: 0,
                nodes_reopened_until_last_jump: 0,
                nodes_evaluated_until_last_jump: 0,
                nodes_generated_until_last_jump: 0,
                registered_states: state_registry.num_registered_states(),
                search_time: Duration::ZERO,
            };
        }

        let current_g = search_nodes
            .get(&state_id)
            .map(|info| info.g_value)
            .unwrap_or(0.0);

        node.state
            .fill_state(&state_registry, &mut state_values_buffer);
        applicable_operators.clear();
        successor_generator
            .get_applicable_operators(&state_values_buffer, &mut applicable_operators);

        for &op_id in applicable_operators.iter() {
            let operator_id = op_id as usize;
            let operator = &task.get_operators()[operator_id];
            let succ_state = match state_registry.get_successor_state_with_buffers(
                &node.state,
                operator,
                &mut successor_numeric_values,
                &mut successor_cost_values,
            ) {
                Ok(succ_state) => succ_state,
                Err(_) => continue,
            };
            let succ_state_id = succ_state.get_id();
            nodes_generated += 1;
            let was_closed = closed_set.contains(&succ_state_id);

            let op_cost = operator_cost(&operator_costs, operator_id, operator);
            let new_g_value = current_g + op_cost;

            if let Some(existing_info) = search_nodes.get(&succ_state_id)
                && existing_info.g_value <= new_g_value
            {
                continue;
            }

            if was_closed {
                closed_set.remove(&succ_state_id);
                nodes_reopened += 1;
            }

            search_nodes.insert(
                succ_state_id,
                DebugSearchNodeInfo {
                    parent_state: Some(state_id),
                    parent_operator_id: Some(operator_id),
                    g_value: new_g_value,
                },
            );

            let evaluation = evaluate_da_state(
                task,
                &state_registry,
                &heuristic,
                &g_evaluator,
                &f_evaluator,
                &succ_state,
                new_g_value,
            )?;
            nodes_evaluated += 1;
            if evaluation.is_dead_end {
                dead_ends += 1;
                if !open_list.accepts_dead_ends() {
                    let had_witness = first_evaluated_witness.is_some();
                    record_admissibility_check(
                        &heuristic.name(),
                        "evaluated",
                        task,
                        &mut state_registry,
                        &mut blind_cache,
                        &succ_state,
                        &operator_costs,
                        &evaluation,
                        &mut first_evaluated_witness,
                        &mut evaluated_checks,
                    )?;
                    if !had_witness && first_evaluated_witness.is_some() {
                        print_witness_details(
                            "ASTAR_DA_DEBUG",
                            &first_evaluated_witness,
                            task,
                            &mut state_registry,
                            &search_nodes,
                            &operator_costs,
                        )?;
                        return Err(std::io::Error::other(
                            "blind-A* admissibility assertion failed on an evaluated state",
                        ));
                    }
                    continue;
                }
            }

            let had_witness = first_evaluated_witness.is_some();
            record_admissibility_check(
                &heuristic.name(),
                "evaluated",
                task,
                &mut state_registry,
                &mut blind_cache,
                &succ_state,
                &operator_costs,
                &evaluation,
                &mut first_evaluated_witness,
                &mut evaluated_checks,
            )?;
            if !had_witness && first_evaluated_witness.is_some() {
                print_witness_details(
                    "ASTAR_DA_DEBUG",
                    &first_evaluated_witness,
                    task,
                    &mut state_registry,
                    &search_nodes,
                    &operator_costs,
                )?;
                return Err(std::io::Error::other(
                    "blind-A* admissibility assertion failed on an evaluated state",
                ));
            }

            open_list.insert(SearchNode::root(succ_state, evaluation));
        }
    };

    debug!(
        "[ASTAR_DA_DEBUG] checked {} evaluated state(s) and {} expanded state(s).",
        evaluated_checks, expanded_checks
    );
    first_witness_line("ASTAR_DA_DEBUG", &first_evaluated_witness);
    print_witness_details(
        "ASTAR_DA_DEBUG",
        &first_evaluated_witness,
        task,
        &mut state_registry,
        &search_nodes,
        &operator_costs,
    )?;
    match &first_expanded_witness {
        Some(witness) => debug!(
            "[ASTAR_DA_DEBUG] first expanded witness: sid={} g={:.3} h={:.3} f={:.3} blind_remaining={:.3} delta={:.3}",
            witness.state_id,
            witness.g_value,
            witness.h_value,
            witness.f_value,
            witness.blind_remaining,
            witness.h_value - witness.blind_remaining,
        ),
        None => debug!("[ASTAR_DA_DEBUG] no inadmissible expanded states found."),
    }
    print_witness_details(
        "ASTAR_DA_DEBUG_EXPANDED",
        &first_expanded_witness,
        task,
        &mut state_registry,
        &search_nodes,
        &operator_costs,
    )?;

    Ok(result)
}
