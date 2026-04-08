use clap::Parser;
use planners_cli_utils::*;
use planners_sas::numeric::axioms::AxiomEvaluator;
use planners_sas::numeric::numeric_task::{AbstractNumericTask, NumericRootTask};
use planners_sas::numeric::state_registry::StateRegistry;
use planners_sas::numeric::utils::int_packer::IntDoublePacker;
use planners_search::numeric::evaluation::domain_abstractions::cegar::CegarConfig;
use planners_search::numeric::evaluation::domain_abstractions::domain_abstraction_collection_generator_multiple_cegar::DomainAbstractionCollectionGeneratorMultipleCegar;
use planners_search::numeric::evaluation::domain_abstractions::domain_abstraction_generator::DomainAbstractionGenerator;
use planners_search::numeric::evaluation::domain_abstractions::domain_abstraction_heuristic::DomainAbstractionHeuristic;
use planners_search::numeric::evaluation::domain_abstractions::max_domain_abstraction_heuristic::MaxDomainAbstractionHeuristic;
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

pub mod recursive_config;

pub use recursive_config::{HeuristicSpec, SearchSpec, parse_search_spec};

#[derive(Parser, Debug, Clone)]
#[command(author, version, about = "Numeric planner")]
pub struct PlannersSearcherCli {
    #[arg(long = "max-memory", value_name = "SIZE", value_parser = parse_memory_limit)]
    pub max_memory: Option<u64>,

    #[arg(long = "max-time", value_name = "DURATION", value_parser = parse_time_limit)]
    pub max_time: Option<Duration>,

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

pub fn run_internal(cli: &PlannersSearcherCli) -> std::io::Result<SearchResult> {
    register_event_handlers();

    let sas_file = &cli.sas_file;

    let start_time = std::time::Instant::now();
    let task = NumericRootTask::from_file(sas_file);
    let parse_time = start_time.elapsed();
    println!("Parsed numeric SAS output in: {:?}", parse_time);

    println!("=== Search Engine ===");
    println!("File: {}", sas_file);
    println!(
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
                crate::recursive_config::HeuristicSpec::DomainAbstraction => {
                    println!("Building domain abstraction (CEGAR)...");
                    let mut config = CegarConfig::default();
                    config.debug = true;

                    let generator = DomainAbstractionGenerator::new(config).map_err(|e| {
                        std::io::Error::new(
                            std::io::ErrorKind::Other,
                            format!("failed to construct DomainAbstractionGenerator: {e:#}"),
                        )
                    })?;
                    let abstraction = generator.generate(task_ref).map_err(|e| {
                        std::io::Error::new(
                            std::io::ErrorKind::Other,
                            format!("failed to build domain abstraction: {e:#}"),
                        )
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
                crate::recursive_config::HeuristicSpec::MultiDomainAbstractions(config) => {
                    let generator =
                        DomainAbstractionCollectionGeneratorMultipleCegar::new(config.clone());
                    println!("Building multiple domain abstractions (CEGAR)...");
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

            println!("Starting A* search with {:?}...", heuristic);
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
            println!("Solution found!");
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
                    Err(e) => eprintln!("Error writing plan file: {}", e),
                }

                for (i, op) in plan.iter().enumerate() {
                    println!("  {}: {}", i + 1, op.name());
                }

                println!("Plan length: {} step(s).", plan.len());
                println!("Plan cost: {:.6}", plan_cost);
            }
        }
        SearchStatus::Failed => {
            println!("No solution found");
        }
        SearchStatus::Timeout => {
            println!("Search timed out");
        }
        SearchStatus::MemoryLimitReached => {
            println!("Search stopped after reaching the memory limit");
        }
        SearchStatus::InProgress => {
            println!("Search ended in progress");
        }
    }

    // Fast Downward-style statistics block.
    println!("Expanded {} state(s).", result.nodes_expanded);
    println!("Reopened {} state(s).", result.nodes_reopened);
    println!("Evaluated {} state(s).", result.nodes_evaluated);
    println!("Evaluations: {}", result.evaluations);
    println!("Generated {} state(s).", result.nodes_generated);
    println!("Dead ends: {} state(s).", result.dead_ends);
    println!(
        "Expanded until last jump: {} state(s).",
        result.nodes_expanded_until_last_jump
    );
    println!(
        "Reopened until last jump: {} state(s).",
        result.nodes_reopened_until_last_jump
    );
    println!(
        "Evaluated until last jump: {} state(s).",
        result.nodes_evaluated_until_last_jump
    );
    println!(
        "Generated until last jump: {} state(s).",
        result.nodes_generated_until_last_jump
    );
    println!("Number of registered states: {}", result.registered_states);
    println!("Search time: {:.6}s", result.search_time.as_secs_f64());
}
