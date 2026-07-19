use super::*;
use planforge_sas::{
    numeric_task::{
        AbstractNumericTask, Effect, ExplicitFact, ExplicitVariable, Metric, NumericRootTask,
        NumericType, NumericVariable, Operator,
    },
    state_registry::StateRegistry,
};
use std::sync::Arc;

fn get_root_task() -> NumericRootTask {
    let version = 4;
    let metric = Metric::new(true, Some(1));
    let variables = vec![
        ExplicitVariable::new(
            2,
            String::from("var13"),
            vec![String::from("new-axiom"), String::from("not-new-axiom")],
            Some(1),
            0,
        ),
        ExplicitVariable::new(
            7,
            String::from("var10"),
            vec![
                String::from("on(d, a)"),
                String::from("on(d, b)"),
                String::from("on(d, c)"),
                String::from("on(d, e)"),
                String::from("on(d, f)"),
                String::from("ontable(d)"),
            ],
            None,
            0,
        ),
    ];
    let numeric_variables = vec![
        NumericVariable::new(String::from("derived!1.0()"), NumericType::Constant, None),
        NumericVariable::new(String::from("total_cost()"), NumericType::Cost, None),
    ];
    let goals = vec![
        ExplicitFact::new(9, 4),
        ExplicitFact::new(10, 1),
        ExplicitFact::new(11, 2),
        ExplicitFact::new(12, 5),
        ExplicitFact::new(13, 4),
    ];
    let mutexes = Vec::new();
    let state = vec![1, 1];
    let numeric_state = vec![1f64, 0f64];
    let operators = vec![Operator::new(
        String::from("drop"),
        vec![ExplicitFact::new(1, 1)],
        vec![Effect::new(Vec::new(), 1, Some(1), 5)],
        Vec::new(),
        1,
    )];
    let axioms = Vec::new();
    let comparison_axioms = Vec::new();
    let assignment_axioms = Vec::new();
    let global_constraint = ExplicitFact::new(0, 0);
    NumericRootTask::new(
        version,
        metric,
        variables,
        numeric_variables,
        goals,
        mutexes,
        state,
        numeric_state,
        operators,
        axioms,
        comparison_axioms,
        assignment_axioms,
        global_constraint,
    )
}

#[test]
fn test_grounded_successor_generator() {
    let task = get_root_task();

    let mut generator = GroundedSuccessorGenerator::new(&task);

    let mut queue: VecDeque<u32> = (0..task.get_operators().len() as u32).collect();

    let mut state_registry = StateRegistry::for_task(Arc::new(&task));

    let state = state_registry.get_initial_state();
    let state_values = state.get_state(&state_registry);
    assert_eq!(state_values, [1, 1]);

    let root = generator.construct(&mut 0, &mut queue).unwrap();
    let tree = generator.into_tree(root);

    let mut applicable_operators: Vec<u32> = Vec::new();
    tree.get_applicable_operators(&state_values[..], &mut applicable_operators);

    // Only operator id 0 ("drop") is applicable in the initial state.
    assert_eq!(applicable_operators, vec![0]);
}

#[test]
fn test_generate_immediate_successor_of_init_state() {
    let task = get_root_task();
    let mut state_registry = StateRegistry::for_task(Arc::new(&task));
    let initial_state = state_registry.get_initial_state();

    let state = initial_state.get_state(&state_registry);
    let suc_gen = GroundedSuccessorGenerator::construct_node_from_task(&task);

    let mut applicable_operators = Vec::new();
    suc_gen.get_applicable_operators(&state, &mut applicable_operators);

    let op = &task.get_operators()[applicable_operators[0] as usize];

    let successor = state_registry
        .get_successor_state(&initial_state, op)
        .expect("Failed to get successor state");
    assert_eq!(successor.get_state(&state_registry), [1, 5]);
    assert_eq!(state_registry.get_numeric_indices(), [Some(0), Some(0)]);
}

#[test]
fn test_per_state_info_subscription() {
    let task = get_root_task();
    let state_registry = StateRegistry::for_task(Arc::new(&task));

    // Create a PerStateInformation instance
    let mut custom_per_state_info =
        planforge_sas::utils::per_state_info::PerStateInformation::<i32>::new();

    // Subscribe it to the registry
    state_registry.subscribe_per_state_info(&mut custom_per_state_info);

    // Verify subscription
    assert!(custom_per_state_info.is_subscribed_to(state_registry.id()));

    // Test unsubscription
    state_registry.unsubscribe_per_state_info(&mut custom_per_state_info);
    assert!(!custom_per_state_info.is_subscribed_to(state_registry.id()));

    // Re-subscribe for cleanup test
    state_registry.subscribe_per_state_info(&mut custom_per_state_info);
    assert!(custom_per_state_info.is_subscribed_to(state_registry.id()));

    // Manually cleanup (simulating registry destruction)
    custom_per_state_info.cleanup_registry(state_registry.id());
    assert!(!custom_per_state_info.is_subscribed_to(state_registry.id()));
}

#[test]
fn test_duplicate_successor_should_not_generate_new_id() {
    let task = get_root_task();
    let mut state_registry = StateRegistry::for_task(Arc::new(&task));
    let initial_state = state_registry.get_initial_state();

    let state = initial_state.get_state(&state_registry);
    let suc_gen = GroundedSuccessorGenerator::construct_node_from_task(&task);

    let mut applicable_operators = Vec::new();
    suc_gen.get_applicable_operators(&state, &mut applicable_operators);

    // Get the first applicable operator
    let op = &task.get_operators()[applicable_operators[0] as usize];

    assert_eq!(op.name(), "drop");
    assert_eq!(initial_state.get_id(), 0);
    assert_eq!(state_registry.num_registered_states(), 1);

    // Generate the successor state twice
    let successor1 = state_registry
        .get_successor_state(&initial_state, op)
        .expect("Failed to get first successor state");

    assert_eq!(state_registry.num_registered_states(), 2);
    assert_eq!(successor1.get_id(), 1);

    let successor2 = state_registry
        .get_successor_state(&initial_state, op)
        .expect("Failed to get second successor state");

    assert_eq!(state_registry.num_registered_states(), 2);
    assert_eq!(successor2.get_id(), 1);

    // They should have the same ID if duplicate detection is working
    assert_eq!(
        successor1.get_id(),
        successor2.get_id(),
        "Generating the same successor twice should yield the same state ID"
    );

    // Ensure only two unique states exist (initial + 1 successor)
    assert_eq!(
        state_registry.get_state_data_pool().len(),
        2,
        "There should be exactly 2 unique states in the pool: initial + 1 successor"
    );
}
