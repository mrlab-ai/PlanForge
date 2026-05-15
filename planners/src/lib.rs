#[cfg(test)]
mod tests;

use clap::Parser;
use ordered_float::NotNan;
use time::format_description::well_known::iso8601::{Config, TimePrecision};
use tracing::{debug, info};
use planners_cli_utils::*;
use planners_preprocess::run_preprocess;
use planners_sas::numeric::axioms::AxiomEvaluator;
use planners_sas::numeric::numeric_task::{AbstractNumericTask, NumericRootTask, NumericType, Operator};
use planners_sas::numeric::state_registry::{ConcreteState, StateRegistry};
use planners_sas::numeric::utils::int_packer::IntDoublePacker;
use planners_search::numeric::evaluation::g_evaluator::{GEvaluator, SumEvaluator};
use planners_search::numeric::evaluation::domain_abstractions::canonical_domain_abstraction_heuristic::CanonicalDomainAbstractionHeuristic;
use planners_search::numeric::evaluation::domain_abstractions::cegar::{Cegar, CegarConfig};
use planners_search::numeric::evaluation::domain_abstractions::domain_abstraction_collection_generator_multiple_cegar::{
    DomainAbstractionCollectionGeneratorMultipleCegar,
};
use planners_search::numeric::evaluation::domain_abstractions::domain_abstraction_generator::{
    DomainAbstraction, DomainAbstractionGenerator, compute_hash_multipliers,
};
use planners_search::numeric::evaluation::domain_abstractions::domain_abstraction_heuristic::DomainAbstractionHeuristic;
use planners_search::numeric::evaluation::domain_abstractions::max_domain_abstraction_heuristic::MaxDomainAbstractionHeuristic;
use planners_search::numeric::evaluation::domain_abstractions::saturated_cost_partitioning_online_heuristic::{
    FillScpHeuristic, SaturatedCostPartitioningOnlineHeuristic,
};
use planners_search::numeric::evaluation::evaluator::EvaluationState;
use planners_search::numeric::evaluation::numeric_landmarks::lm_cut_numeric_heuristic::LandmarkCutNumericHeuristic;
use planners_search::numeric::evaluation::pattern_databases::canonical_pdb_heuristic::CanonicalNumericPdbHeuristic;
use planners_search::numeric::evaluation::pattern_databases::pattern_generator_systematic::{
    SystematicPatternGeneratorConfig, generate_systematic_patterns,
};
use planners_search::numeric::evaluation::pattern_databases::pdb_collection::PdbCollection;
use planners_search::numeric::evaluation::pattern_databases::pdb_heuristic::GreedyNumericPdbHeuristic;
use planners_search::numeric::evaluation::{EvaluationResult, Evaluator};
use planners_search::numeric::open_lists::{OpenList, SearchNode, TieBreakingOpenList};
use planners_search::numeric::search_engine::{compute_effective_operator_costs, SearchResult, SearchStatus};
use planners_search::numeric::search_engine::{AStarSearch, SearchEngine};
use planners_search::numeric::successor_generator::{ApplicableOperator, SuccessorTree};
use planners_searcher::*;
use planners_translator::*;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::ffi::OsString;
use std::num::NonZero;
use std::process::Command;
use std::time::Duration;
use tracing_subscriber::fmt::{time::UtcTime};

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

    /// Recursive search configuration.
    /// Examples: `astar(blind())`, `astar(domain_abstraction())`, `da_debug()`.
    #[arg(
        long,
        value_name = "SPEC",
        default_value = "astar(blind())",
        value_parser = planners_searcher::parse_search_spec
    )]
    pub search: planners_searcher::SearchSpec,

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

#[allow(clippy::field_reassign_with_default)]
pub fn run_internal(cli: &PlannersCli) -> std::io::Result<SearchResult> {
    register_event_handlers();

    let sas_file = if cli.inputs.len() == 2 {
        let domain = &cli.inputs[0];
        let problem = &cli.inputs[1];
        translate_to_sas(domain, problem).map_err(|err| std::io::Error::other(err.to_string()))?;

        run_preprocess(&["preprocess".to_string(), "output.sas".to_string()]);
        "output"
    } else {
        &cli.inputs[0]
    };

    let start_time = std::time::Instant::now();
    let task = NumericRootTask::from_file(sas_file);
    let parse_time = start_time.elapsed();
    info!("Parsed numeric SAS output in: {:?}", parse_time);

    info!("=== Search Engine ===");
    info!("File: {}", sas_file);
    info!(
        "Variables: {} regular, {} numeric",
        task.variables().len(),
        task.numeric_variables().len()
    );

    let state_packer = IntDoublePacker::from_task(&task);
    let axiom_evaluator = AxiomEvaluator::new(&task, &state_packer);
    let state_registry = StateRegistry::new(&task, &state_packer, &axiom_evaluator);

    let result = match &cli.search {
        planners_searcher::SearchSpec::Astar(heuristic) => {
            let task_ref: &dyn AbstractNumericTask = &task;
            let heuristic_override = match heuristic {
                planners_searcher::HeuristicSpec::Blind => None,
                planners_searcher::HeuristicSpec::CanonicalDomainAbstractions(config) => {
                    // Canonical reads only the per-abstraction distance table;
                    // skip footprint construction to avoid ~12 GB of
                    // per-concrete-op `StateRegion` storage on tasks with
                    // hundreds of thousands of concrete operators.
                    let mut config = config.clone();
                    config.compute_operator_footprints = false;
                    let generator =
                        DomainAbstractionCollectionGeneratorMultipleCegar::new(config);
                    info!("Building canonical domain abstractions (CEGAR)...");
                    let abstractions = generator.generate_collection(task_ref).map_err(|e| {
                        std::io::Error::other(format!(
                            "failed to build canonical domain abstractions: {e:#}"
                        ))
                    })?;
                    Some(Box::new(
                        CanonicalDomainAbstractionHeuristic::new(None, task_ref, abstractions)
                            .map_err(|e| {
                                std::io::Error::other(format!(
                                    "failed to construct canonical domain abstraction heuristic: {e}"
                                ))
                            })?,
                    ) as Box<dyn planners_search::numeric::evaluation::Heuristic + '_>)
                }
                planners_searcher::HeuristicSpec::DomainAbstraction(domain_config) => {
                    info!("Building domain abstraction (CEGAR)...");
                    let mut config = CegarConfig::default();
                    config.max_abstraction_size = domain_config.max_abstraction_size;
                    config.max_iterations = domain_config.max_iterations;
                    config.use_wildcard_plans = domain_config.use_wildcard_plans;
                    config.combine_labels = domain_config.combine_labels;
                    config.random_seed = domain_config.random_seed;
                    config.flaw_kind = domain_config.flaw_kind;
                    config.flaw_treatment = domain_config.flaw_treatment;
                    config.init_split_method = domain_config.init_split_method;
                    config.transform_linear_task = domain_config.transform_linear_task;
                    // Single DA reads only the distance table; footprints are
                    // SCP-specific. Skip the per-concrete-op StateRegion cost.
                    config.compute_operator_footprints = false;

                    let generator = DomainAbstractionGenerator::new(config).map_err(|e| {
                        std::io::Error::other(format!(
                            "failed to construct DomainAbstractionGenerator: {e:#}"
                        ))
                    })?;
                    let abstraction = generator.generate(task_ref).map_err(|e| {
                        std::io::Error::other(format!("failed to build domain abstraction: {e:#}"))
                    })?;
                    Some(Box::new(DomainAbstractionHeuristic::new(None, abstraction))
                        as Box<
                            dyn planners_search::numeric::evaluation::Heuristic + '_,
                        >)
                }
                planners_searcher::HeuristicSpec::CanonicalNumericPdb(config) => Some(Box::new(
                    CanonicalNumericPdbHeuristic::from_config(task_ref, *config).map_err(|e| {
                        std::io::Error::other(format!(
                            "failed to build canonical numeric pdb heuristic: {e}"
                        ))
                    })?,
                )
                    as Box<dyn planners_search::numeric::evaluation::Heuristic + '_>),
                planners_searcher::HeuristicSpec::GreedyNumericPdb(config) => Some(Box::new(
                    GreedyNumericPdbHeuristic::new(task_ref, *config).map_err(|e| {
                        std::io::Error::other(format!(
                            "failed to build greedy numeric pdb heuristic: {e}"
                        ))
                    })?,
                )
                    as Box<dyn planners_search::numeric::evaluation::Heuristic + '_>),
                planners_searcher::HeuristicSpec::Lmcutnumeric(config) => Some(Box::new(
                    LandmarkCutNumericHeuristic::from_config(task_ref, *config).map_err(|e| {
                        std::io::Error::other(format!(
                            "failed to build lmcutnumeric heuristic: {e}"
                        ))
                    })?,
                )
                    as Box<dyn planners_search::numeric::evaluation::Heuristic + '_>),
                planners_searcher::HeuristicSpec::MultiDomainAbstractions(config) => {
                    // Max-of-abstractions reads only the distance tables;
                    // footprints are SCP-only.
                    let mut config = config.clone();
                    config.compute_operator_footprints = false;
                    let generator =
                        DomainAbstractionCollectionGeneratorMultipleCegar::new(config);
                    info!("Building multiple domain abstractions (CEGAR)...");
                    let abstractions = generator.generate_collection(task_ref).map_err(|e| {
                        std::io::Error::other(format!(
                            "failed to build multi domain abstractions: {e:#}"
                        ))
                    })?;
                    Some(
                        Box::new(MaxDomainAbstractionHeuristic::new(None, abstractions))
                            as Box<dyn planners_search::numeric::evaluation::Heuristic + '_>,
                    )
                }
                planners_searcher::HeuristicSpec::ScpOnline(config) => {
                    let generator = DomainAbstractionCollectionGeneratorMultipleCegar::new(
                        config.collection_config.clone(),
                    );
                    info!("Building scp_online domain abstractions (CEGAR)...");
                    let abstractions = generator.generate_collection(task_ref).map_err(|e| {
                        std::io::Error::other(format!(
                            "failed to build scp_online domain abstractions: {e:#}"
                        ))
                    })?;
                    let pdbs = if config.use_numeric_pdbs {
                        info!("Building scp_online systematic numeric PDBs...");
                        let patterns = generate_systematic_patterns(
                            task_ref,
                            SystematicPatternGeneratorConfig {
                                max_pdb_states: config.max_pdb_states,
                                max_pattern_size: config.max_pattern_size,
                                only_interesting_patterns: config.only_interesting_patterns,
                            },
                        );
                        PdbCollection::with_heuristic_config(
                            task_ref,
                            patterns,
                            config.max_pdb_states,
                            config.pdb_heuristic_config(),
                        )
                        .map_err(|e| {
                            std::io::Error::other(format!(
                                "failed to build scp_online numeric PDBs: {e}"
                            ))
                        })?
                        .into_pdbs()
                    } else {
                        Vec::new()
                    };
                    let h = SaturatedCostPartitioningOnlineHeuristic::new(
                        None,
                        abstractions,
                        pdbs,
                        config.clone(),
                        task_ref,
                    )
                    .map_err(|e| {
                        std::io::Error::other(format!(
                            "failed to construct scp_online heuristic: {e}"
                        ))
                    })?;
                    Some(Box::new(h)
                        as Box<
                            dyn planners_search::numeric::evaluation::Heuristic + '_,
                        >)
                }
                planners_searcher::HeuristicSpec::FillScp(config) => {
                    let mut fill_config = config.clone();
                    fill_config.force_full_goal_tasks();
                    let generator = DomainAbstractionCollectionGeneratorMultipleCegar::new(
                        fill_config.collection_config.clone(),
                    );
                    info!("Building fillSCP domain abstractions (CEGAR)...");
                    let abstractions = generator.generate_collection(task_ref).map_err(|e| {
                        std::io::Error::other(format!(
                            "failed to build fillSCP domain abstractions: {e:#}"
                        ))
                    })?;
                    let h = FillScpHeuristic::new(None, abstractions, fill_config, task_ref)
                        .map_err(|e| {
                            std::io::Error::other(format!(
                                "failed to construct fillSCP heuristic: {e}"
                            ))
                        })?;
                    Some(Box::new(h)
                        as Box<
                            dyn planners_search::numeric::evaluation::Heuristic + '_,
                        >)
                }
            };

            let mut search = AStarSearch::new(
                task_ref,
                state_registry,
                heuristic_override,
                if cli.internal_run { None } else { cli.max_time },
                if cli.internal_run {
                    None
                } else {
                    cli.max_memory
                },
            );

            info!("Starting A* search with {:?}...", heuristic);
            search.search()
        }
        planners_searcher::SearchSpec::DaDebug => run_da_debug(
            &task,
            state_registry,
            if cli.internal_run { None } else { cli.max_time },
        )?,
        planners_searcher::SearchSpec::AstarDaDebug => run_astar_da_debug(
            &task,
            state_registry,
            if cli.internal_run { None } else { cli.max_time },
        )?,
    };

    print_search_result(&result);

    Ok(result)
}

fn build_successor_generator<'a>(task: &'a dyn AbstractNumericTask) -> SuccessorTree<'a> {
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
    let mut applicable_operators: Vec<ApplicableOperator<'_>> = Vec::new();
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

        for (operator, operator_id) in applicable_operators.iter().copied() {
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
    let mut applicable_operators: Vec<ApplicableOperator<'_>> = Vec::new();
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

        for (operator, operator_id) in applicable_operators.iter().copied() {
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
    let mut applicable_operators: Vec<ApplicableOperator<'_>> = Vec::new();
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

        for (operator, operator_id) in applicable_operators.iter().copied() {
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
        task_projection: None,
        transformed_task: None,
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
    let mut applicable_operators: Vec<ApplicableOperator<'_>> = Vec::new();

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
                .find(|(_, applicable_id)| applicable_id == candidate_id)
        });
        let (operator, operator_id) = chosen.ok_or_else(|| {
            std::io::Error::other(format!(
                "no applicable concrete operator found for wildcard step {} candidates {:?}",
                step_idx + 1,
                candidate_ids
            ))
        })?;

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
    let mut applicable_operators: Vec<ApplicableOperator<'_>> = Vec::new();
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

        for (operator, operator_id) in applicable_operators.iter().copied() {
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
