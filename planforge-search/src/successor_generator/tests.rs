use super::*;
use planforge_sas::{
    numeric_task::{
        AbstractNumericTask, Effect, ExplicitFact, ExplicitVariable, Metric, NumericRootTask,
        NumericRootTaskParts, NumericType, NumericVariable, Operator,
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
    let goals = vec![ExplicitFact::propositional(1, 5)];
    let mutexes = Vec::new();
    let state = vec![1, 1];
    let numeric_state = vec![1f64, 0f64];
    let operators = vec![Operator::new(
        String::from("drop"),
        vec![ExplicitFact::propositional(1, 1)],
        vec![Effect::new(Vec::new(), 1, Some(1), 5)],
        Vec::new(),
        1,
    )];
    let axioms = Vec::new();
    let comparison_axioms = Vec::new();
    let assignment_axioms = Vec::new();
    let global_constraint = ExplicitFact::propositional(0, 0);
    NumericRootTask::new(NumericRootTaskParts {
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
    })
}

/// A task whose emission order no ordering chosen at construction time can make
/// ascending in operator id: one variable, and three operators whose ids straddle
/// the branch's shared immediate list — `0` needs `v=0`, `1` needs nothing, `2`
/// needs `v=1`.
fn immediate_list_straddling_task() -> NumericRootTask {
    let variables = vec![ExplicitVariable::new(
        2,
        String::from("v"),
        vec![String::from("v=0"), String::from("v=1")],
        None,
        0,
    )];
    let operator = |name: &str, preconditions: Vec<ExplicitFact>| {
        Operator::new(
            String::from(name),
            preconditions,
            vec![Effect::new(Vec::new(), 0, None, 1)],
            Vec::new(),
            1,
        )
    };
    NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, Some(0)),
        variables,
        numeric_variables: vec![NumericVariable::new(
            String::from("total_cost()"),
            NumericType::Cost,
            None,
        )],
        goals: vec![ExplicitFact::propositional(0, 1)],
        mutexes: Vec::new(),
        state: vec![0],
        numeric_state: vec![0f64],
        operators: vec![
            operator("needs_zero", vec![ExplicitFact::propositional(0, 0)]),
            operator("needs_nothing", Vec::new()),
            operator("needs_one", vec![ExplicitFact::propositional(0, 1)]),
        ],
        axioms: Vec::new(),
        comparison_axioms: Vec::new(),
        assignment_axioms: Vec::new(),
        global_constraint: ExplicitFact::propositional(0, 0),
    })
}

/// Emission is the walk's order — a branch's shared immediate operators, then
/// the child the tested variable's value selects — and not ascending operator id.
///
/// The fixture pins both halves of that: the two orders differ, and no ordering
/// chosen at construction time could reconcile them. `needs_nothing` sits in the
/// root branch's immediate list, so it is emitted either before or after both
/// value children, while ascending id wants it last for state `v=0` and first for
/// state `v=1`, and one tree has to serve both.
#[test]
fn applicable_operators_are_emitted_in_tree_walk_order() {
    let task = immediate_list_straddling_task();
    let tree = SuccessorTree::new(&task);

    let mut applicable: Vec<u32> = Vec::new();
    tree.get_applicable_operators(&[0], &mut applicable);
    assert_eq!(applicable, vec![1, 0]);

    applicable.clear();
    tree.get_applicable_operators(&[1], &mut applicable);
    assert_eq!(applicable, vec![1, 2]);
}

/// `get_applicable_operators` appends, so a caller may collect several states'
/// operators into one buffer and what is already there stays untouched.
#[test]
fn applicable_operators_leave_the_caller_s_prefix_alone() {
    let task = immediate_list_straddling_task();
    let tree = SuccessorTree::new(&task);

    let mut applicable: Vec<u32> = vec![7, 3];
    tree.get_applicable_operators(&[0], &mut applicable);
    assert_eq!(applicable, vec![7, 3, 1, 0]);
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
}
