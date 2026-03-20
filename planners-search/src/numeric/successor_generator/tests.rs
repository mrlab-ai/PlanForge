use super::*;
use planners_sas::numeric::axioms::AxiomEvaluator;
use planners_sas::numeric::numeric_parser::parse_numeric_sas_output;
use planners_sas::numeric::{
    numeric_task::{
        AbstractNumericTask, Effect, ExplicitVariable, Fact, Metric, NumericRootTask, NumericType,
        NumericVariable, Operator,
    },
    state_registry::StateRegistry,
    utils::int_packer::IntDoublePacker,
};

fn get_root_task() -> NumericRootTask {
    let version = 4;
    let metric = Metric::new(true, 1);
    let variables = vec![
        ExplicitVariable::new(
            2,
            String::from("var13"),
            vec![String::from("new-axiom"), String::from("not-new-axiom")],
            1,
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
            -1,
            0,
        ),
    ];
    let numeric_variables = vec![
        NumericVariable::new(String::from("derived!1.0()"), NumericType::Constant, -1),
        NumericVariable::new(String::from("total_cost()"), NumericType::Cost, -1),
    ];
    let goals = vec![
        Fact::new(9, 4),
        Fact::new(10, 1),
        Fact::new(11, 2),
        Fact::new(12, 5),
        Fact::new(13, 4),
    ];
    let mutexes = Vec::new();
    let state = vec![1, 1];
    let numeric_state = vec![1f64, 0f64];
    let operators = vec![Operator::new(
        String::from("drop"),
        vec![Fact::new(1, 1)],
        vec![Effect::new(Vec::new(), 1, 1, 5)],
        Vec::new(),
        1,
    )];
    let axioms = Vec::new();
    let comparison_axioms = Vec::new();
    let assignment_axioms = Vec::new();
    let global_constraint = (0, 0);
    let output = NumericRootTask::new(
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
    );

    output
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

fn setup_successor_generator<'a>(task: &'a dyn AbstractNumericTask) -> Box<dyn Node<'a> + 'a> {
    let mut queue = VecDeque::new();
    for (op_id, operator) in task.get_operators().iter().enumerate() {
        queue.push_back((operator, op_id));
    }

    let mut generator = GroundedSuccessorGenerator::new(task);

    let node = generator.construct(&mut 0, &mut queue).unwrap();

    node
}

#[test]
fn test_grounded_successor_generator() {
    let problem = get_root_task();

    let mut generator = GroundedSuccessorGenerator::new(&problem);

    let mut queue = VecDeque::new();
    for (op_id, operator) in problem.get_operators().iter().enumerate() {
        queue.push_back((operator, op_id));
    }

    let state_packer = setup_state_packer(&problem);
    let axiom_evaluator = setup_axiom_evaluator(&problem, &state_packer);
    let mut state_registry = setup_state_registry(&problem, &state_packer, &axiom_evaluator);

    let state = state_registry.get_initial_state();
    let state = state.get_state(&state_registry);
    println!("State values: {:?}", state);

    let node = generator.construct(&mut 0, &mut queue).unwrap();

    let mut applicable_operators: Vec<ApplicableOperator<'_>> = Vec::new();
    node.get_applicable_operators(&state[..], &mut applicable_operators);

    //dbg!("Facts: {:?}", facts_refs);
    dbg!("Applicable operators: {:?}", applicable_operators);
    //dbg!("Node: {:?}", node);
}

#[test]
fn test_generate_immediate_successor_of_init_state() {
    let task = get_root_task();
    let state_packer = setup_state_packer(&task);
    let axiom_evaluator = setup_axiom_evaluator(&task, &state_packer);
    let mut state_registry = setup_state_registry(&task, &state_packer, &axiom_evaluator);
    let initial_state = state_registry.get_initial_state();

    let state = initial_state.get_state(&state_registry);
    let suc_gen = setup_successor_generator(&task);

    let mut applicable_operators = Vec::new();
    suc_gen.get_applicable_operators(&state, &mut applicable_operators);

    let (op, _) = applicable_operators.into_iter().next().unwrap();

    println!(
        "Initial state: {}",
        initial_state.debug_with_registry(&state_registry)
    );
    println!("OP: {:?}", op);

    let successor = state_registry
        .get_successor_state(&initial_state, op)
        .expect("Failed to get successor state");

    println!(
        "Successor state: {}",
        successor.debug_with_registry(&state_registry)
    );
    println!(
        "Numeric indices: {:?}",
        state_registry.get_numeric_indices()
    );
}

#[test]
fn test_per_state_info_subscription() {
    let task = get_root_task();
    let state_packer = setup_state_packer(&task);
    let axiom_evaluator = setup_axiom_evaluator(&task, &state_packer);
    let state_registry = setup_state_registry(&task, &state_packer, &axiom_evaluator);

    // Create a PerStateInformation instance
    let mut custom_per_state_info =
        planners_sas::numeric::utils::per_state_info::PerStateInformation::<i32>::new();

    // Subscribe it to the registry
    state_registry.subscribe_per_state_info(&mut custom_per_state_info);

    // Verify subscription
    assert!(custom_per_state_info.is_subscribed_to(state_registry.id()));
    println!(
        "PerStateInformation subscribed to registry {}",
        state_registry.id()
    );

    // Test unsubscription
    state_registry.unsubscribe_per_state_info(&mut custom_per_state_info);
    assert!(!custom_per_state_info.is_subscribed_to(state_registry.id()));
    println!(
        "PerStateInformation unsubscribed from registry {}",
        state_registry.id()
    );

    // Re-subscribe for cleanup test
    state_registry.subscribe_per_state_info(&mut custom_per_state_info);
    assert!(custom_per_state_info.is_subscribed_to(state_registry.id()));

    // Manually cleanup (simulating registry destruction)
    custom_per_state_info.cleanup_registry(state_registry.id());
    assert!(!custom_per_state_info.is_subscribed_to(state_registry.id()));
    println!(
        "PerStateInformation cleaned up for registry {}",
        state_registry.id()
    );
}

#[test]
fn test_automatic_cleanup_on_drop() {
    let task = get_root_task();
    let state_packer = setup_state_packer(&task);
    let axiom_evaluator = setup_axiom_evaluator(&task, &state_packer);

    let registry_id = {
        let state_registry = setup_state_registry(&task, &state_packer, &axiom_evaluator);
        let id = state_registry.id();

        // Verify the cost_info is automatically subscribed
        assert!(state_registry.get_cost_info().borrow().is_subscribed_to(id));
        println!("StateRegistry {} has auto-subscribed cost_info", id);

        id
    }; // StateRegistry drops here, triggering automatic cleanup

    println!(
        "StateRegistry {} has been dropped with automatic cleanup",
        registry_id
    );
}

#[test]
fn test_duplicate_successor_should_not_generate_new_id() {
    let task = get_root_task();
    let state_packer = setup_state_packer(&task);
    let axiom_evaluator = setup_axiom_evaluator(&task, &state_packer);
    let mut state_registry = setup_state_registry(&task, &state_packer, &axiom_evaluator);
    let initial_state = state_registry.get_initial_state();

    let state = initial_state.get_state(&state_registry);
    let suc_gen = setup_successor_generator(&task);

    let mut applicable_operators = Vec::new();
    suc_gen.get_applicable_operators(&state, &mut applicable_operators);

    // Get the first applicable operator
    let (op, _) = applicable_operators.first().unwrap();

    println!("Testing operator: {:?}", op.name());
    println!("Initial state ID: {}", initial_state.get_id());
    println!(
        "Initial registered_states size: {}",
        state_registry.get_registered_states().len()
    );

    // Generate the successor state twice
    let successor1 = state_registry
        .get_successor_state(&initial_state, op)
        .expect("Failed to get first successor state");

    println!(
        "After first successor - registered_states size: {}",
        state_registry.get_registered_states().len()
    );
    println!("First successor ID: {}", successor1.get_id());

    let successor2 = state_registry
        .get_successor_state(&initial_state, op)
        .expect("Failed to get second successor state");

    println!(
        "After second successor - registered_states size: {}",
        state_registry.get_registered_states().len()
    );
    println!("Second successor ID: {}", successor2.get_id());

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
