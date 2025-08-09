mod parser;
mod search;

use parser::numeric_parser::parse_numeric_sas_output;
use std::env;
use std::fs;

use crate::search::numeric::axioms::AxiomEvaluator;
use crate::search::numeric::numeric_task::AbstractNumericTask;
use crate::search::numeric::numeric_task::NumericRootTask;
use crate::search::numeric::numeric_task::NumericType;
use crate::search::numeric::state_registry::StateRegistry;
use crate::search::numeric::utils::int_packer::IntDoublePacker;

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
    let axiom_evaluator = AxiomEvaluator::new(problem as &dyn AbstractNumericTask, &state_packer);
    let state_registry = StateRegistry::new(problem, &state_packer, &axiom_evaluator);

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

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: {} [sas_file]", args[0]);
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "No SAS file provided",
        ));
    }
    let sas_file = &args[1];
    let content = fs::read_to_string(sas_file).expect("Could not read file");
    match parse_numeric_sas_output(&content) {
        Ok((_, sas_output)) => {
            println!("Successfully parsed SAS file");
            let task: &dyn AbstractNumericTask = &sas_output;
        }
        Err(e) => println!("Failed to parse file: {:?}", e),
    }
    Ok(())
}
