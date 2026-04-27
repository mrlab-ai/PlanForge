use clap::Parser;
use tracing::{error, info};
use planners_cli_utils::*;
use planners_sas::numeric::axioms::AxiomEvaluator;
use planners_sas::numeric::numeric_task::{AbstractNumericTask, NumericRootTask};
use planners_sas::numeric::state_registry::StateRegistry;
use planners_sas::numeric::utils::int_packer::IntDoublePacker;
use planners_search::numeric::evaluation::domain_abstractions::cegar::CegarConfig;
use planners_search::numeric::evaluation::domain_abstractions::canonical_domain_abstraction_heuristic::CanonicalDomainAbstractionHeuristic;
use planners_search::numeric::evaluation::domain_abstractions::domain_abstraction_collection_generator_multiple_cegar::{
    DomainAbstractionCollectionGeneratorMultipleCegar,
};
use planners_search::numeric::evaluation::domain_abstractions::domain_abstraction_generator::DomainAbstractionGenerator;
use planners_search::numeric::evaluation::domain_abstractions::domain_abstraction_heuristic::DomainAbstractionHeuristic;
use planners_search::numeric::evaluation::domain_abstractions::max_domain_abstraction_heuristic::MaxDomainAbstractionHeuristic;
use planners_search::numeric::evaluation::domain_abstractions::saturated_cost_partitioning_online_heuristic::SaturatedCostPartitioningOnlineHeuristic;
use planners_search::numeric::evaluation::numeric_landmarks::lm_cut_numeric_heuristic::LandmarkCutNumericHeuristic;
use planners_search::numeric::evaluation::pattern_databases::canonical_pdb_heuristic::CanonicalNumericPdbHeuristic;
use planners_search::numeric::evaluation::pattern_databases::pdb_heuristic::GreedyNumericPdbHeuristic;
use planners_search::numeric::search_engine::{
    AStarSearch, SearchEngine, SearchResult, SearchStatus,
};
use std::ffi::OsString;
use std::fs;
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::process::Command;
use std::time::Duration;
use std::num::NonZero;
use time::format_description::well_known::iso8601::{Config, TimePrecision};
use tracing_subscriber::fmt::time::UtcTime;
use tracing_subscriber::prelude::*;

pub mod recursive_config;

pub use recursive_config::{HeuristicSpec, SearchSpec, parse_search_spec};

use tracing_subscriber::filter::LevelFilter;

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
    // Layer for stdout (info + deubg + trace)
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

#[derive(Parser, Debug, Clone)]
#[command(author, version, about = "Numeric planner")]
pub struct PlannersSearcherCli {
    #[arg(long = "max-memory", value_name = "SIZE", value_parser = parse_memory_limit)]
    pub max_memory: Option<u64>,

    #[arg(long = "max-time", value_name = "DURATION", value_parser = parse_time_limit)]
    pub max_time: Option<Duration>,

    #[arg(long = "log-level")]
    pub log_level: Option<LevelFilter>,

    #[arg(long, hide = true)]
    pub internal_run: bool,

    /// Recursive search configuration.
    /// Examples: `astar(blind())`, `astar(domain_abstraction())`, `da_debug()`.
    #[arg(
        long,
        value_name = "SPEC",
        default_value = "astar(blind())",
        value_parser = crate::recursive_config::parse_search_spec
    )]
    pub search: crate::recursive_config::SearchSpec,

    #[arg(value_name = "SAS_FILE", required = true)]
    pub sas_file: String,
}

#[cfg(unix)]
pub fn run_wrapped_process(cli: &PlannersSearcherCli) -> std::io::Result<()> {
    let current_executable = std::env::current_exe()?;
    let mut child_args = vec![OsString::from("--internal-run")];
    // Preserve the selected search configuration when re-executing ourselves.
    if let Some(level) = cli.log_level {
        child_args.push(OsString::from("--log-level"));
        child_args.push(OsString::from(level.to_string()));
    }
    child_args.push(OsString::from("--search"));
    child_args.push(OsString::from(cli.search.to_string()));
    child_args.extend([cli.sas_file.clone()].iter().map(OsString::from));

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
pub fn run_internal(cli: &PlannersSearcherCli) -> std::io::Result<SearchResult> {
    register_event_handlers();

    let sas_file = &cli.sas_file;

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
        crate::recursive_config::SearchSpec::Astar(heuristic) => {
            let task_ref: &dyn AbstractNumericTask = &task;
            let heuristic_override = match heuristic {
                crate::recursive_config::HeuristicSpec::Blind => None,
                crate::recursive_config::HeuristicSpec::CanonicalDomainAbstractions(config) => {
                    let generator =
                        DomainAbstractionCollectionGeneratorMultipleCegar::new(config.clone());
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
                crate::recursive_config::HeuristicSpec::DomainAbstraction(domain_config) => {
                    info!("Building domain abstraction (CEGAR)...");
                    let mut config = CegarConfig::default();
                    config.max_abstraction_size = domain_config.max_abstraction_size;
                    config.max_iterations = domain_config.max_iterations;
                    config.use_wildcard_plans = domain_config.use_wildcard_plans;
                    config.combine_labels = domain_config.combine_labels;
                    config.random_seed = if domain_config.random_seed >= 0 {
                        Some(domain_config.random_seed as u64)
                    } else {
                        None
                    };
                    config.flaw_kind = domain_config.flaw_kind;
                    config.flaw_treatment = domain_config.flaw_treatment;
                    config.init_split_method = domain_config.init_split_method;
                    config.exec_entire_plan = domain_config.exec_entire_plan;

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
                crate::recursive_config::HeuristicSpec::CanonicalNumericPdb(config) => Some(
                    Box::new(
                        CanonicalNumericPdbHeuristic::from_config(task_ref, *config).map_err(
                            |e| {
                                std::io::Error::other(format!(
                                    "failed to build canonical numeric pdb heuristic: {e}"
                                ))
                            },
                        )?,
                    )
                        as Box<dyn planners_search::numeric::evaluation::Heuristic + '_>,
                ),
                crate::recursive_config::HeuristicSpec::GreedyNumericPdb(config) => Some(Box::new(
                    GreedyNumericPdbHeuristic::new(task_ref, *config).map_err(|e| {
                        std::io::Error::other(format!(
                            "failed to build greedy numeric pdb heuristic: {e}"
                        ))
                    })?,
                )
                    as Box<dyn planners_search::numeric::evaluation::Heuristic + '_>),
                crate::recursive_config::HeuristicSpec::Lmcutnumeric(config) => Some(Box::new(
                    LandmarkCutNumericHeuristic::from_config(task_ref, *config).map_err(|e| {
                        std::io::Error::other(format!(
                            "failed to build lmcutnumeric heuristic: {e}"
                        ))
                    })?,
                )
                    as Box<dyn planners_search::numeric::evaluation::Heuristic + '_>),
                crate::recursive_config::HeuristicSpec::MultiDomainAbstractions(config) => {
                    let generator =
                        DomainAbstractionCollectionGeneratorMultipleCegar::new(config.clone());
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
                crate::recursive_config::HeuristicSpec::ScpOnline(config) => {
                    let generator = DomainAbstractionCollectionGeneratorMultipleCegar::new(
                        config.collection_config.clone(),
                    );
                    info!("Building scp_online domain abstractions (CEGAR)...");
                    let abstractions = generator.generate_collection(task_ref).map_err(|e| {
                        std::io::Error::other(format!(
                            "failed to build scp_online domain abstractions: {e:#}"
                        ))
                    })?;
                    Some(Box::new(SaturatedCostPartitioningOnlineHeuristic::new(
                        None,
                        abstractions,
                        config.clone(),
                    ))
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
        crate::recursive_config::SearchSpec::DaDebug => {
            return Err(std::io::Error::other(
                "`da_debug()` is implemented in the `planners` binary path, not `planners-searcher`",
            ));
        }
        crate::recursive_config::SearchSpec::AstarDaDebug => {
            return Err(std::io::Error::other(
                "`astar_da_debug()` is implemented in the `planners` binary path, not `planners-searcher`",
            ));
        }
    };

    print_search_result(&result);

    Ok(result)
}

pub fn exit_code_for_search_status(status: &SearchStatus) -> i32 {
    match status {
        SearchStatus::Timeout => EXIT_TIMEOUT,
        SearchStatus::MemoryLimitReached => EXIT_OUT_OF_MEMORY,
        SearchStatus::InProgress | SearchStatus::Solved(_) | SearchStatus::Failed => EXIT_SUCCESS,
    }
}

pub fn print_search_result(result: &SearchResult) {
    match result.status {
        SearchStatus::Solved(_) => {
            info!("Solution found!");
            if let Some(plan) = result.plan.as_ref() {
                let plan_cost = result
                    .solution_cost
                    .unwrap_or_else(|| plan.iter().map(|op| op.cost() as f64).sum());

                let mut plan_content = String::new();
                for op in plan.iter() {
                    plan_content.push_str(&format!("({})\n", op.name()));
                }

                match fs::write("sas_plan", plan_content) {
                    Ok(()) => {}
                    Err(e) => error!("Error writing plan file: {}", e),
                }

                for (i, op) in plan.iter().enumerate() {
                    info!("  {}: {}", i + 1, op.name());
                }

                info!("Plan length: {} step(s).", plan.len());
                info!("Plan cost: {:.6}", plan_cost);
            }
        }
        SearchStatus::Failed => {
            info!("No solution found");
        }
        SearchStatus::Timeout => {
            info!("Search timed out");
        }
        SearchStatus::MemoryLimitReached => {
            info!("Search stopped after reaching the memory limit");
        }
        SearchStatus::InProgress => {
            info!("Search ended in progress");
        }
    }

    // Fast Downward-style statistics block.
    info!("Expanded {} state(s).", result.nodes_expanded);
    info!("Reopened {} state(s).", result.nodes_reopened);
    info!("Evaluated {} state(s).", result.nodes_evaluated);
    info!("Evaluations: {}", result.evaluations);
    info!("Generated {} state(s).", result.nodes_generated);
    info!("Dead ends: {} state(s).", result.dead_ends);
    info!(
        "Expanded until last jump: {} state(s).",
        result.nodes_expanded_until_last_jump
    );
    info!(
        "Reopened until last jump: {} state(s).",
        result.nodes_reopened_until_last_jump
    );
    info!(
        "Evaluated until last jump: {} state(s).",
        result.nodes_evaluated_until_last_jump
    );
    info!(
        "Generated until last jump: {} state(s).",
        result.nodes_generated_until_last_jump
    );
    info!("Number of registered states: {}", result.registered_states);
    info!("Search time: {:.6}s", result.search_time.as_secs_f64());
}
