use clap::Parser;
use planners_cli_utils::*;
use planners_sas::numeric::axioms::AxiomEvaluator;
use planners_sas::numeric::numeric_parser::parse_numeric_sas_output;
use planners_sas::numeric::numeric_task::{AbstractNumericTask, NumericRootTask};
use planners_sas::numeric::utils::int_packer::IntDoublePacker;
use planners_sas::numeric::{numeric_task::NumericType, state_registry::StateRegistry};
use planners_search::numeric::search_engine::{
    AStarSearch, SearchEngine, SearchResult, SearchStatus,
};
use planners_search::numeric::successor_generator::{GroundedSuccessorGenerator, Node};
use std::collections::VecDeque;
use std::ffi::OsString;
use std::fs;
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::process::Command;
use std::time::Duration;

#[derive(Parser, Debug, Clone)]
#[command(author, version, about = "Numeric planner")]
pub struct PlannersSearcherCli {
    #[arg(long = "max-memory", value_name = "SIZE", value_parser = parse_memory_limit)]
    pub max_memory: Option<u64>,

    #[arg(long = "max-time", value_name = "DURATION", value_parser = parse_time_limit)]
    pub max_time: Option<Duration>,

    #[arg(long, hide = true)]
    pub internal_run: bool,

    #[arg(value_name = "SAS_FILE", required = true)]
    pub sas_file: String,
}

#[cfg(unix)]
pub fn run_wrapped_process(cli: &PlannersSearcherCli) -> std::io::Result<()> {
    let current_executable = std::env::current_exe()?;
    let mut child_args = vec![OsString::from("--internal-run")];
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

    let result = {
        let task_ref: &dyn AbstractNumericTask = &task;
        let mut search = AStarSearch::new(
            task_ref,
            state_registry,
            None,
            if cli.internal_run { None } else { cli.max_time },
            if cli.internal_run {
                None
            } else {
                cli.max_memory
            },
        );

        println!("Starting search...");
        search.search()
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
            println!("SOLVED!");
            if let Some(plan) = result.plan.as_ref() {
                let plan_cost = result
                    .solution_cost
                    .unwrap_or_else(|| plan.iter().map(|op| op.cost() as f64).sum());
                println!(
                    "Solution plan ({} steps, cost {:.6}):",
                    plan.len(),
                    plan_cost
                );

                let mut plan_content = String::new();
                for op in plan.iter() {
                    plan_content.push_str(&format!("({})\n", op.name()));
                }

                match fs::write("sas_plan", plan_content) {
                    Ok(()) => println!("Plan written to sas_plan file"),
                    Err(e) => eprintln!("Error writing plan file: {}", e),
                }

                for (i, op) in plan.iter().enumerate() {
                    println!("  {}: {}", i + 1, op.name());
                }
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

    println!(
        "Statistics: {} expanded, {} generated, {:?}",
        result.nodes_expanded, result.nodes_generated, result.search_time
    );
}

