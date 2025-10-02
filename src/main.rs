#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]

mod parser;
mod search;

use parser::numeric_parser::parse_numeric_sas_output;
use std::collections::VecDeque;
use std::env;
use std::fs;

use crate::search::numeric::axioms::AxiomEvaluator;
use crate::search::numeric::numeric_task::AbstractNumericTask;
use crate::search::numeric::numeric_task::NumericRootTask;
use crate::search::numeric::numeric_task::NumericType;
use crate::search::numeric::state_registry::StateRegistry;
use crate::search::numeric::successor_generator;
use crate::search::numeric::successor_generator::GroundedSuccessorGenerator;
use crate::search::numeric::successor_generator::Node;
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
        queue.push_back((operator, op_id as u32));
    }

    let mut generator = GroundedSuccessorGenerator::new(task);

    let node = generator.construct(&mut 0, &mut queue).unwrap();
    
    node
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

    let task = setup_numeric_task(sas_file);
    let state_packer = setup_state_packer(&task);
    let axiom_evaluator = setup_axiom_evaluator(&task, &state_packer);
    let mut state_registry = setup_state_registry(&task, &state_packer, &axiom_evaluator);
    let suc_gen = setup_successor_generator(&task);
    Ok(())
}
