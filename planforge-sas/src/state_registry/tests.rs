use crate::{
    axioms::PropositionalAxiom,
    numeric_task::{
        Effect, ExplicitFact, ExplicitVariable, Metric, NumericRootTask, NumericType,
        NumericVariable, Operator, TaskRef,
    },
    state_registry::StateRegistry,
};
use std::sync::Arc;

use crate::tests::*;

#[test]
fn test_state_registry_initial_state() {
    let task: TaskRef = Arc::new(get_root_task());
    let mut state_registry = StateRegistry::for_task(task);
    let initial_state = state_registry.get_initial_state();
    assert_eq!(initial_state.get_state(&state_registry), [1, 0]);
}

#[test]
fn initial_state_registration_does_not_mutate_shared_task() {
    let task: TaskRef = Arc::new(NumericRootTask::new(
        4,
        Metric::new(false, None),
        vec![ExplicitVariable::new(
            2,
            "derived".into(),
            vec!["false".into(), "true".into()],
            Some(0),
            0,
        )],
        vec![],
        vec![ExplicitFact::new(0, 1)],
        vec![],
        vec![0],
        vec![],
        vec![],
        vec![PropositionalAxiom::new(vec![], 0, 0, 1)],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    ));
    let original_propositions = task.get_initial_propositional_state_values().to_vec();
    let original_numeric = task.get_initial_numeric_state_values().to_vec();
    let mut registry = StateRegistry::for_task(task.clone());

    let initial = registry.get_initial_state();

    assert_eq!(initial.get_state(&registry), [1]);
    assert_eq!(
        task.get_initial_propositional_state_values().as_slice(),
        original_propositions
    );
    assert_eq!(
        task.get_initial_numeric_state_values().as_slice(),
        original_numeric
    );
}

#[test]
fn test_cost_information_storage() {
    let task: TaskRef = Arc::new(get_root_task());
    let mut state_registry = StateRegistry::for_task(task);

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
        None,
        0,
    )];
    let numeric_variables = vec![
        NumericVariable::new("total_cost()".to_string(), NumericType::Cost, None),
        NumericVariable::new("cheap".to_string(), NumericType::Constant, None),
        NumericVariable::new("expensive".to_string(), NumericType::Constant, None),
    ];

    let expensive_op = Operator::new(
        "expensive".to_string(),
        vec![ExplicitFact::new(0, 0)],
        vec![Effect::new(Vec::new(), 0, Some(0), 1)],
        vec![crate::numeric_task::AssignmentEffect::new(
            0,
            crate::numeric_task::AssignmentOperation::Plus,
            2,
            false,
            vec![],
        )],
        1,
    );
    let cheap_op = Operator::new(
        "cheap".to_string(),
        vec![ExplicitFact::new(0, 0)],
        vec![Effect::new(Vec::new(), 0, Some(0), 1)],
        vec![crate::numeric_task::AssignmentEffect::new(
            0,
            crate::numeric_task::AssignmentOperation::Plus,
            1,
            false,
            vec![],
        )],
        1,
    );

    let task: TaskRef = Arc::new(NumericRootTask::new(
        4,
        Metric::new(true, Some(0)),
        variables,
        numeric_variables,
        vec![ExplicitFact::new(0, 1)],
        vec![],
        vec![0],
        vec![0.0, 1.0, 5.0],
        vec![expensive_op, cheap_op],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    ));

    let mut state_registry = StateRegistry::for_task(task.clone());

    let initial_state = state_registry.get_initial_state();
    let expensive_successor = state_registry
        .get_successor_state(&initial_state, &task.get_operators()[0])
        .unwrap();
    let cheap_successor = state_registry
        .get_successor_state(&initial_state, &task.get_operators()[1])
        .unwrap();

    assert_eq!(expensive_successor.get_id(), cheap_successor.get_id());
    assert_eq!(state_registry.get_cost_information(&cheap_successor), [1.0]);
    assert_eq!(
        state_registry
            .transition_cost(&initial_state, &cheap_successor)
            .unwrap(),
        1.0
    );
}

#[test]
fn register_state_deduplicates_canonicalized_numeric_values() {
    let variables = vec![ExplicitVariable::new(
        2,
        "v0".to_string(),
        vec!["off".to_string(), "on".to_string()],
        None,
        0,
    )];
    let numeric_variables = vec![
        NumericVariable::new("x".to_string(), NumericType::Regular, None),
        NumericVariable::new("c1".to_string(), NumericType::Constant, None),
        NumericVariable::new("c2".to_string(), NumericType::Constant, None),
    ];
    let add_c1 = Operator::new(
        "add-c1".to_string(),
        vec![],
        vec![],
        vec![crate::numeric_task::AssignmentEffect::new(
            0,
            crate::numeric_task::AssignmentOperation::Plus,
            1,
            false,
            vec![],
        )],
        1,
    );
    let add_c2 = Operator::new(
        "add-c2".to_string(),
        vec![],
        vec![],
        vec![crate::numeric_task::AssignmentEffect::new(
            0,
            crate::numeric_task::AssignmentOperation::Plus,
            2,
            false,
            vec![],
        )],
        1,
    );
    let task: TaskRef = Arc::new(NumericRootTask::new(
        4,
        Metric::new(false, None),
        variables,
        numeric_variables,
        vec![],
        vec![],
        vec![0],
        vec![0.0, 0.1 + 0.2, 0.3],
        vec![add_c1, add_c2],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    ));

    for compact in [false, true] {
        let mut state_registry =
            StateRegistry::for_task_with_compact_numeric(task.clone(), compact);

        let initial = state_registry.get_initial_state();
        let first = state_registry
            .get_successor_state(&initial, &task.get_operators()[0])
            .unwrap();
        let second = state_registry
            .get_successor_state(&initial, &task.get_operators()[1])
            .unwrap();

        assert_eq!(first.get_id(), second.get_id());
        assert_eq!(
            state_registry
                .get_numeric_var_value_unevaluated(&first, 0)
                .unwrap(),
            0.3
        );
    }
}

#[test]
fn compact_numeric_states_pack_exact_value_ids_with_propositions() {
    let task: TaskRef = Arc::new(NumericRootTask::new(
        2,
        Metric::new(false, None),
        vec![ExplicitVariable::new(
            2,
            "v0".into(),
            vec!["off".into(), "on".into()],
            None,
            0,
        )],
        vec![
            NumericVariable::new("x".into(), NumericType::Regular, None),
            NumericVariable::new("y".into(), NumericType::Regular, None),
        ],
        vec![],
        vec![],
        vec![0],
        vec![1.25, -7.5],
        vec![],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    ));
    let mut regular = StateRegistry::for_task(task.clone());
    let mut compact = StateRegistry::for_task_with_compact_numeric(task, true);
    let regular_initial = regular.get_initial_state();
    let compact_initial = compact.get_initial_state();

    assert_eq!(regular_initial.buffer(&regular).len(), 3);
    assert_eq!(compact_initial.buffer(&compact).len(), 1);
    assert_eq!(compact_initial.get_numeric_state(&compact), [1.25, -7.5]);
}

#[test]
fn distinct_states_with_the_same_hash_remain_distinguishable() {
    let task: TaskRef = Arc::new(get_root_task());
    let mut registry = StateRegistry::for_task(task.clone());
    let initial = registry.get_initial_state();
    let successor = registry
        .get_successor_state(&initial, &task.get_operators()[0])
        .unwrap();
    assert_ne!(initial.get_id(), successor.get_id());

    let initial_bins = initial.buffer(&registry).to_vec();
    let successor_bins = successor.buffer(&registry).to_vec();
    registry.registered_states.clear();

    let forced_hash = 7;
    registry.insert_registered_state_id(forced_hash, initial.get_id());
    registry.insert_registered_state_id(forced_hash, successor.get_id());

    assert_eq!(
        registry.find_registered_state_id(forced_hash, &initial_bins),
        Some(initial.get_id())
    );
    assert_eq!(
        registry.find_registered_state_id(forced_hash, &successor_bins),
        Some(successor.get_id())
    );
    assert_eq!(registry.registered_states.len(), 2);
}
