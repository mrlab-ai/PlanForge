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

#[test]
fn duplicate_state_keeps_better_metric_cost_information() {
    let variables = vec![ExplicitVariable::new(
        2,
        "v0".to_string(),
        vec!["off".to_string(), "on".to_string()],
        -1,
        0,
    )];
    let numeric_variables = vec![
        NumericVariable::new("total_cost()".to_string(), NumericType::Cost, -1),
        NumericVariable::new("cheap".to_string(), NumericType::Constant, -1),
        NumericVariable::new("expensive".to_string(), NumericType::Constant, -1),
    ];

    let expensive_op = Operator::new(
        "expensive".to_string(),
        vec![Fact::new(0, 0)],
        vec![Effect::new(Vec::new(), 0, 0, 1)],
        vec![crate::numeric::numeric_task::AssignmentEffect::new(
            0,
            crate::numeric::numeric_task::AssignmentOperation::Plus,
            2,
            false,
            vec![],
        )],
        1,
    );
    let cheap_op = Operator::new(
        "cheap".to_string(),
        vec![Fact::new(0, 0)],
        vec![Effect::new(Vec::new(), 0, 0, 1)],
        vec![crate::numeric::numeric_task::AssignmentEffect::new(
            0,
            crate::numeric::numeric_task::AssignmentOperation::Plus,
            1,
            false,
            vec![],
        )],
        1,
    );

    let task = NumericRootTask::new(
        4,
        Metric::new(true, 0),
        variables,
        numeric_variables,
        vec![Fact::new(0, 1)],
        vec![],
        vec![0],
        vec![0.0, 1.0, 5.0],
        vec![expensive_op, cheap_op],
        vec![],
        vec![],
        vec![],
        (0, 0),
    );

    let state_packer = IntDoublePacker::from_task(&task);
    let axiom_evaluator = AxiomEvaluator::new(&task, &state_packer);
    let mut state_registry = StateRegistry::new(&task, &state_packer, &axiom_evaluator);

    let initial_state = state_registry.get_initial_state();
    let expensive_successor = state_registry
        .get_successor_state(&initial_state, &task.get_operators()[0])
        .unwrap();
    let cheap_successor = state_registry
        .get_successor_state(&initial_state, &task.get_operators()[1])
        .unwrap();

    assert_eq!(expensive_successor.get_id(), cheap_successor.get_id());
    assert_eq!(state_registry.get_cost_information(&cheap_successor), [1.0]);
    assert_eq!(state_registry.transition_cost(&initial_state, &cheap_successor).unwrap(), 1.0);
}
