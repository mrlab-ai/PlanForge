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
    assert_eq!(initial_state.get_state(&state_registry), [1, 0]);
}

#[test]
fn test_cost_information_storage() {
    let task = get_root_task();
    let state_packer = IntDoublePacker::from_task(&task);
    let axiom_evaluator = AxiomEvaluator::new(&task, &state_packer);
    let mut state_registry = StateRegistry::new(&task, &state_packer, &axiom_evaluator);

    let initial_state = state_registry.get_initial_state();

    let cost_info = state_registry.get_cost_information(&initial_state);
    assert_eq!(cost_info, [0.0]);
    assert_eq!(cost_info.len(), 1);
}
