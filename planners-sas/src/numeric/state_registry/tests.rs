use std::collections::VecDeque;

use crate::numeric::{
    axioms::AxiomEvaluator,
    numeric_task::{
        AbstractNumericTask, Effect, ExplicitVariable, Fact, Metric, NumericRootTask, NumericType,
        NumericVariable, Operator,
    },
    state_registry::StateRegistry,
    utils::int_packer::IntDoublePacker,
};

use crate::numeric::tests::*;

#[test]
fn test_state_registry_initial_state() {
    let problem = get_root_task();
    let state_packer = setup_state_packer(&problem);
    let axiom_evaluator = setup_axiom_evaluator(&problem, &state_packer);
    let mut state_registry = setup_state_registry(&problem, &state_packer, &axiom_evaluator);
    let initial_state = state_registry.get_initial_state();
    print!(
        "Initial state: {}",
        initial_state.debug_with_registry(&state_registry)
    );
}

#[test]
fn test_cost_information_storage() {
    let task = get_root_task();
    let state_packer = setup_state_packer(&task);
    let axiom_evaluator = setup_axiom_evaluator(&task, &state_packer);
    let mut state_registry = setup_state_registry(&task, &state_packer, &axiom_evaluator);

    let initial_state = state_registry.get_initial_state();

    // Check that cost information is stored
    let cost_info = state_registry.get_cost_information(&initial_state);
    println!("Initial state cost information: {:?}", cost_info);

    // The cost information should be accessible (empty vector if no cost variables)
    println!("Cost information length: {}", cost_info.len());
}
