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

fn setup_axiom_evaluator<'a>(
    problem: &'a NumericRootTask,
    state_packer: &'a IntDoublePacker,
) -> AxiomEvaluator<'a> {
    let task: &'a dyn AbstractNumericTask = problem;
    let axiom_evaluator = AxiomEvaluator::new(task, &state_packer);
    axiom_evaluator
}

fn setup_state_registry<'a>(
    problem: &'a NumericRootTask,
    state_packer: &'a IntDoublePacker,
    axiom_evaluator: &'a AxiomEvaluator<'a>,
) -> StateRegistry<'a> {
    StateRegistry::new(problem, state_packer, axiom_evaluator)
}

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
