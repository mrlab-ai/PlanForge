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
    let task = get_root_task();
    let state_packer = IntDoublePacker::from_task(&task);
    let axiom_evaluator = AxiomEvaluator::new(&task, &state_packer);
    let mut state_registry = StateRegistry::new(&task, &state_packer, &axiom_evaluator);
    let initial_state = state_registry.get_initial_state();
    print!(
        "Initial state: {}",
        initial_state.debug_with_registry(&state_registry)
    );
}

#[test]
fn test_cost_information_storage() {
    let task = get_root_task();
    let state_packer = IntDoublePacker::from_task(&task);
    let axiom_evaluator = AxiomEvaluator::new(&task, &state_packer);
    let mut state_registry = StateRegistry::new(&task, &state_packer, &axiom_evaluator);

    let initial_state = state_registry.get_initial_state();

    // Check that cost information is stored
    let cost_info = state_registry.get_cost_information(&initial_state);
    println!("Initial state cost information: {:?}", cost_info);

    // The cost information should be accessible (empty vector if no cost variables)
    println!("Cost information length: {}", cost_info.len());
}
