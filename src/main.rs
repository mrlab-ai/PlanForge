#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]

mod preprocess;
mod search;

use clap::Parser;
use preprocess::numeric_parser::parse_numeric_sas_output;
use planners::preprocess_port::planner::run_preprocess;
use planners::translate::normalize;
use planners::translate::pddl_parser::PddlTask;
use std::collections::VecDeque;
use std::fs;
use std::time::Duration;

use crate::search::numeric::axioms::AxiomEvaluator;
use crate::search::numeric::numeric_task::AbstractNumericTask;
use crate::search::numeric::numeric_task::NumericRootTask;
use crate::search::numeric::numeric_task::NumericType;
use crate::search::numeric::search_engine::{SearchResult, SearchStatus};
use crate::search::numeric::state_registry::StateRegistry;
use crate::search::numeric::successor_generator;
use crate::search::numeric::successor_generator::GroundedSuccessorGenerator;
use crate::search::numeric::successor_generator::Node;
use crate::search::numeric::utils::int_packer::IntDoublePacker;
use search::numeric::search_engine::{AStarSearch, SearchEngine};

#[derive(Parser, Debug)]
#[command(author, version, about = "Numeric planner")]
struct Cli {
    #[arg(long = "max-memory", value_name = "SIZE", value_parser = parse_memory_limit)]
    max_memory: Option<u64>,

    #[arg(long = "max-time", value_name = "DURATION", value_parser = parse_time_limit)]
    max_time: Option<Duration>,

    #[arg(value_name = "INPUT", required = true, num_args = 1..=2)]
    inputs: Vec<String>,
}

fn parse_suffixed_value(
    input: &str,
    default_multiplier: u64,
    units: &[(&str, u64)],
    kind: &str,
) -> Result<u64, String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(format!("{} cannot be empty", kind));
    }

    let suffix_start = trimmed
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(trimmed.len());
    if suffix_start == 0 {
        return Err(format!("{} must start with a number: {}", kind, input));
    }

    let value = trimmed[..suffix_start]
        .parse::<u64>()
        .map_err(|_| format!("invalid {} value: {}", kind, input))?;
    let suffix = trimmed[suffix_start..].trim().to_ascii_lowercase();

    let multiplier = if suffix.is_empty() {
        default_multiplier
    } else {
        units
            .iter()
            .find_map(|(unit, factor)| (*unit == suffix).then_some(*factor))
            .ok_or_else(|| format!("invalid {} suffix '{}': {}", kind, suffix, input))?
    };

    value
        .checked_mul(multiplier)
        .ok_or_else(|| format!("{} is too large: {}", kind, input))
}

fn parse_memory_limit(input: &str) -> Result<u64, String> {
    parse_suffixed_value(
        input,
        1,
        &[
            ("b", 1),
            ("k", 1024),
            ("kb", 1024),
            ("m", 1024 * 1024),
            ("mb", 1024 * 1024),
            ("g", 1024 * 1024 * 1024),
            ("gb", 1024 * 1024 * 1024),
            ("t", 1024_u64.pow(4)),
            ("tb", 1024_u64.pow(4)),
        ][..],
        "memory limit",
    )
}

fn parse_time_limit(input: &str) -> Result<Duration, String> {
    let seconds = parse_suffixed_value(
        input,
        1,
        &[("ms", 0), ("s", 1), ("m", 60), ("h", 60 * 60)][..],
        "time limit",
    )?;

    if input.trim().to_ascii_lowercase().ends_with("ms") {
        let millis = input.trim()[..input.trim().len() - 2]
            .trim()
            .parse::<u64>()
            .map_err(|_| format!("invalid time limit value: {}", input))?;
        Ok(Duration::from_millis(millis))
    } else {
        Ok(Duration::from_secs(seconds))
    }
}

fn setup_state_registry<'a>(
    problem: &'a NumericRootTask,
    state_packer: &'a IntDoublePacker,
    axiom_evaluator: &'a AxiomEvaluator<'a>,
) -> StateRegistry<'a> {
    StateRegistry::new(problem, state_packer, axiom_evaluator)
}

fn setup_axiom_evaluator<'a>(
    problem: &'a NumericRootTask,
    state_packer: &'a IntDoublePacker,
) -> AxiomEvaluator<'a> {
    let task: &'a dyn AbstractNumericTask = problem;
    let axiom_evaluator = AxiomEvaluator::new(task, &state_packer);
    axiom_evaluator
}

fn setup_state_packer<'a>(problem: &'a NumericRootTask) -> IntDoublePacker {
    let mut domain_sizes = vec![];
    for var in problem.variables().iter() {
        domain_sizes.push(var.domain_size() as u64);
    }
    for numeric_var in problem.numeric_variables().iter() {
        if numeric_var.get_type() == &NumericType::Regular {
            domain_sizes.push(u64::MAX);
        }
    }
    IntDoublePacker::new(&domain_sizes)
}

fn setup_numeric_task(file_name: &str) -> NumericRootTask {
    // This function should create a NumericRootTask with the necessary setup for testing
    // For now, we return an empty task as a placeholder
    let file_content = std::fs::read_to_string(file_name).unwrap();
    parse_numeric_sas_output(&file_content)
        .unwrap() //TODO: Handle errors properly
        .1
}

fn setup_successor_generator<'a>(task: &'a dyn AbstractNumericTask) -> Box<dyn Node<'a> + 'a> {
    let mut queue = VecDeque::new();
    for (op_id, operator) in task.get_operators().iter().enumerate() {
        queue.push_back((operator, op_id));
    }

    let mut generator = GroundedSuccessorGenerator::new(task);

    let node = generator.construct(&mut 0, &mut queue).unwrap();

    node
}

fn translate_to_sas(domain: &str, problem: &str) -> anyhow::Result<()> {
    let task = PddlTask::from_files(std::path::Path::new(domain), std::path::Path::new(problem)).map_err(|e| anyhow::anyhow!(e))?;
    let parsed_task = task.to_task();

    let mut norm_task = normalize::NormalizableTask::from_task(parsed_task);
    norm_task.add_global_constraints();
    normalize::normalize(&mut norm_task).expect("normalization failed");

    let result = planners::translate::instantiate::explore_normalized(&norm_task).map_err(|e| anyhow::anyhow!(e))?;

    let instantiated_num_axioms = result.numeric_axioms;
    let py_groups: Option<Vec<Vec<String>>> = None;
    let mut sastask = planners::translate::translate::translate_task_from_grounded_internal(
        &result.atoms,
        &result.grounded_ops,
        &task.domain_forms,
        &task.problem_forms,
        &result.num_fluents,
        &instantiated_num_axioms,
        py_groups,
        &result.grounded_axioms,
        &result.reachable_action_params,
        &norm_task.goal,
        &norm_task,
    )
    .map_err(|err| anyhow::anyhow!(err))?;

    match planners::translate::simplify::filter_unreachable_propositions(&mut sastask) {
        Ok(()) => {}
        Err(planners::translate::simplify::SimplifyError::Impossible) => {
            sastask = planners::translate::simplify::trivial_task(false);
        }
        Err(planners::translate::simplify::SimplifyError::TriviallySolvable) => {
            sastask = planners::translate::simplify::trivial_task(true);
        }
        Err(planners::translate::simplify::SimplifyError::DoesNothing) => {
            // Task unchanged
        }
    }

    let py_task = planners::translate::sas_tasks::from_internal(&sastask);
    let mut out_file = std::fs::File::create("output.sas")?;
    py_task.output(&mut out_file)?;
    Ok(())
}

fn main() -> std::io::Result<()> {
    let cli = Cli::parse();
    let sas_file = if cli.inputs.len() == 2 {
        let domain = &cli.inputs[0];
        let problem = &cli.inputs[1];
        translate_to_sas(domain, problem).map_err(|err| {
            std::io::Error::new(std::io::ErrorKind::Other, err.to_string())
        })?;

        run_preprocess(&vec!["preprocess".to_string(), "output.sas".to_string()]);
        "output"
    } else {
        &cli.inputs[0]
    };

    let start_time = std::time::Instant::now();
    let task = setup_numeric_task(sas_file);
    let parse_time = start_time.elapsed();
    println!("Parsed numeric SAS output in: {:?}", parse_time);

    println!("=== A* Search Engine ===");
    println!("File: {}", sas_file);
    if let Some(max_time) = cli.max_time {
        println!("Max time: {:?}", max_time);
    }
    if let Some(max_memory) = cli.max_memory {
        println!("Max memory: {} bytes", max_memory);
    }
    println!(
        "Variables: {} regular, {} numeric",
        task.variables().len(),
        task.numeric_variables().len()
    );

    // Create all components in main() scope where lifetimes can be properly managed
    let state_packer = setup_state_packer(&task);
    let axiom_evaluator = setup_axiom_evaluator(&task, &state_packer);
    let mut state_registry = setup_state_registry(&task, &state_packer, &axiom_evaluator);

    // Create search engine and get result, then explicitly drop the search engine
    let result = {
        // Move ownership of state_registry into the search engine to avoid lifetime issues
        let task_ref: &dyn AbstractNumericTask = &task;
        let search = AStarSearch::new(
            task_ref,
            state_registry,
            None, // BlindHeuristic (will be created by default)
            cli.max_time,
            cli.max_memory,
        );

        println!("Starting search...");

        // Mutable binding so we can call search()
        let mut search = search;
        let search_result = search.search();
        search_result
    };

    match result.status {
        SearchStatus::Solved(_) => {
            println!("SOLVED!");
            if let Some(plan) = result.plan {
                println!("Solution plan ({} steps):", plan.len());

                // Create the sas_plan file content
                let mut plan_content = String::new();
                for op in plan.iter() {
                    plan_content.push_str(&format!("({})\n", op.name()));
                }

                // Write the plan to sas_plan file
                match fs::write("sas_plan", plan_content) {
                    Ok(()) => println!("Plan written to sas_plan file"),
                    Err(e) => eprintln!("Error writing plan file: {}", e),
                }

                // Also print the plan to console
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

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_memory_limit_suffixes() {
        assert_eq!(parse_memory_limit("500M").unwrap(), 500 * 1024 * 1024);
        assert_eq!(parse_memory_limit("8g").unwrap(), 8 * 1024 * 1024 * 1024);
        assert_eq!(parse_memory_limit("1024").unwrap(), 1024);
    }

    #[test]
    fn parses_time_limit_suffixes() {
        assert_eq!(parse_time_limit("60s").unwrap(), Duration::from_secs(60));
        assert_eq!(parse_time_limit("5m").unwrap(), Duration::from_secs(300));
        assert_eq!(parse_time_limit("250ms").unwrap(), Duration::from_millis(250));
    }
}
