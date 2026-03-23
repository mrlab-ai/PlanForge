#[cfg(test)]
mod tests;

use clap::Parser;
use planners_cli_utils::*;
use planners_preprocess::run_preprocess;
use planners_sas::numeric::axioms::AxiomEvaluator;
use planners_sas::numeric::numeric_task::{AbstractNumericTask, NumericRootTask};
use planners_sas::numeric::state_registry::StateRegistry;
use planners_sas::numeric::utils::int_packer::IntDoublePacker;
use planners_search::numeric::evaluation::domain_abstractions::cegar::CegarConfig;
use planners_search::numeric::evaluation::domain_abstractions::domain_abstraction_generator::DomainAbstractionGenerator;
use planners_search::numeric::evaluation::domain_abstractions::domain_abstraction_heuristic::DomainAbstractionHeuristic;
use planners_search::numeric::search_engine::SearchResult;
use planners_search::numeric::search_engine::{AStarSearch, SearchEngine};
use planners_searcher::*;
use planners_translator::*;
use std::ffi::OsString;
use std::process::Command;
use std::time::Duration;

#[cfg(unix)]
use std::os::unix::process::CommandExt;

#[derive(Parser, Debug, Clone)]
#[command(author, version, about = "Numeric planner")]
pub struct PlannersCli {
    #[arg(long = "max-memory", value_name = "SIZE", value_parser = parse_memory_limit)]
    pub max_memory: Option<u64>,

    #[arg(long = "max-time", value_name = "DURATION", value_parser = parse_time_limit)]
    pub max_time: Option<Duration>,

    #[arg(long, hide = true)]
    pub internal_run: bool,

    /// Recursive search configuration.
    /// Examples: "astar(blind())", "astar(domain_abstraction())".
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
    println!("Parsed numeric SAS output in: {:?}", parse_time);

    println!("=== A* Search Engine ===");
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
        planners_searcher::SearchSpec::Astar(heuristic) => {
            let task_ref: &dyn AbstractNumericTask = &task;
            let heuristic_override = match heuristic {
                planners_searcher::HeuristicSpec::Blind => None,
                planners_searcher::HeuristicSpec::DomainAbstraction => {
                    println!("Building domain abstraction (CEGAR)...");
                    let mut config = CegarConfig::default();
                    config.enable_refinement = true;
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
                    Some(
                        Box::new(DomainAbstractionHeuristic::new(None, abstraction))
                            as Box<dyn planners_search::numeric::evaluation::Heuristic>,
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
    };

    print_search_result(&result);

    Ok(result)
}
