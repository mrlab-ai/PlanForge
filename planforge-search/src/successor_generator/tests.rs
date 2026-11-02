use super::*;
use planforge_sas::{
    numeric_task::{
        Effect, ExplicitFact, ExplicitVariable, Metric, NumericType, NumericValue, NumericVariable,
        Operator, OperatorCost, OperatorIndex, Task, TaskParts, VariableIndex,
    },
    state_registry::StateRegistry,
};
use std::sync::Arc;

fn get_root_task() -> Task {
    let metric = Metric::new(true, Some(VariableIndex::from_usize(1)));
    let variables = vec![
        ExplicitVariable::new(
            2,
            String::from("var13"),
            vec![String::from("new-axiom"), String::from("not-new-axiom")],
            Some(1),
            ExplicitValueIndex::new(0),
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
            ExplicitValueIndex::new(0),
        ),
    ];
    let numeric_variables = vec![
        NumericVariable::new(String::from("derived!1.0()"), NumericType::Constant, None),
        NumericVariable::new(String::from("total_cost()"), NumericType::Cost, None),
    ];
    let goals = vec![ExplicitFact::propositional(1, 5)];
    let mutexes = Vec::new();
    let state = vec![ExplicitValueIndex::new(1), ExplicitValueIndex::new(1)];
    let numeric_state = vec![NumericValue::new(1f64), NumericValue::new(0f64)];
    let operators = vec![Operator::new(
        String::from("drop"),
        vec![ExplicitFact::propositional(1, 1)],
        vec![Effect::new(
            Vec::new(),
            VariableIndex::from_usize(1),
            Some(ExplicitValueIndex::new(1)),
            ExplicitValueIndex::new(5),
        )],
        Vec::new(),
        OperatorCost::new(1),
    )];
    let axioms = Vec::new();
    let comparison_axioms = Vec::new();
    let assignment_axioms = Vec::new();
    let global_constraint = ExplicitFact::propositional(0, 0);
    Task::new(TaskParts {
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
        global_constraint: Some(global_constraint),
    })
}

/// A task whose emission order no ordering chosen at construction time can make
/// ascending in operator id: one variable, and three operators whose ids straddle
/// the branch's shared immediate list — `0` needs `v=0`, `1` needs nothing, `2`
/// needs `v=1`.
fn immediate_list_straddling_task() -> Task {
    let variables = vec![ExplicitVariable::new(
        2,
        String::from("v"),
        vec![String::from("v=0"), String::from("v=1")],
        None,
        ExplicitValueIndex::new(0),
    )];
    let operator = |name: &str, preconditions: Vec<ExplicitFact>| {
        Operator::new(
            String::from(name),
            preconditions,
            vec![Effect::new(
                Vec::new(),
                VariableIndex::from_usize(0),
                None,
                ExplicitValueIndex::new(1),
            )],
            Vec::new(),
            OperatorCost::new(1),
        )
    };
    Task::new(TaskParts {
        metric: Metric::new(true, Some(VariableIndex::from_usize(0))),
        variables,
        numeric_variables: vec![NumericVariable::new(
            String::from("total_cost()"),
            NumericType::Cost,
            None,
        )],
        goals: vec![ExplicitFact::propositional(0, 1)],
        mutexes: Vec::new(),
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![NumericValue::new(0f64)],
        operators: vec![
            operator("needs_zero", vec![ExplicitFact::propositional(0, 0)]),
            operator("needs_nothing", Vec::new()),
            operator("needs_one", vec![ExplicitFact::propositional(0, 1)]),
        ],
        axioms: Vec::new(),
        comparison_axioms: Vec::new(),
        assignment_axioms: Vec::new(),
        global_constraint: Some(ExplicitFact::propositional(0, 0)),
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
    let tree = StandardGroundedTree::new(&task);

    let mut applicable: Vec<OperatorIndex> = Vec::new();
    tree.applicable_operators(&[ExplicitValueIndex::new(0)], &mut applicable);
    assert_eq!(
        applicable,
        vec![OperatorIndex::new(1), OperatorIndex::new(0)]
    );

    applicable.clear();
    tree.applicable_operators(&[ExplicitValueIndex::new(1)], &mut applicable);
    assert_eq!(
        applicable,
        vec![OperatorIndex::new(1), OperatorIndex::new(2)]
    );
}

/// `applicable_operators` appends, so a caller may collect several states'
/// operators into one buffer and what is already there stays untouched.
#[test]
fn applicable_operators_leave_the_caller_s_prefix_alone() {
    let task = immediate_list_straddling_task();
    let tree = StandardGroundedTree::new(&task);

    let mut applicable: Vec<OperatorIndex> = vec![OperatorIndex::new(7), OperatorIndex::new(3)];
    tree.applicable_operators(&[ExplicitValueIndex::new(0)], &mut applicable);
    assert_eq!(
        applicable,
        vec![
            OperatorIndex::new(7),
            OperatorIndex::new(3),
            OperatorIndex::new(1),
            OperatorIndex::new(0)
        ]
    );
}

#[test]
fn the_successor_tree_lists_the_applicable_operators() {
    let task = get_root_task();

    let mut state_registry = StateRegistry::for_task(Arc::new(task.clone()));

    let state = state_registry.register_initial_state();
    let state_values = state.propositional_values(&state_registry);
    assert_eq!(
        state_values,
        [ExplicitValueIndex::new(1), ExplicitValueIndex::new(1)]
    );

    let tree = StandardGroundedTree::new(&task);

    let mut applicable_operators: Vec<OperatorIndex> = Vec::new();
    tree.applicable_operators(&state_values[..], &mut applicable_operators);

    // Only operator id 0 ("drop") is applicable in the initial state.
    assert_eq!(applicable_operators, vec![OperatorIndex::new(0)]);
}

#[test]
fn test_generate_immediate_successor_of_init_state() {
    let task = get_root_task();
    let mut state_registry = StateRegistry::for_task(Arc::new(task.clone()));
    let initial_state = state_registry.register_initial_state();

    let state = initial_state.propositional_values(&state_registry);
    let suc_gen = StandardGroundedTree::new(&task);

    let mut applicable_operators = Vec::new();
    suc_gen.applicable_operators(&state, &mut applicable_operators);

    let op = &task.operators()[applicable_operators[0].index()];

    let successor = state_registry
        .successor(&initial_state, op)
        .expect("Failed to get successor state")
        .0;
    assert_eq!(
        successor.propositional_values(&state_registry),
        [ExplicitValueIndex::new(1), ExplicitValueIndex::new(5)]
    );
}

#[test]
fn test_duplicate_successor_should_not_generate_new_id() {
    let task = get_root_task();
    let mut state_registry = StateRegistry::for_task(Arc::new(task.clone()));
    let initial_state = state_registry.register_initial_state();

    let state = initial_state.propositional_values(&state_registry);
    let suc_gen = StandardGroundedTree::new(&task);

    let mut applicable_operators = Vec::new();
    suc_gen.applicable_operators(&state, &mut applicable_operators);

    // Get the first applicable operator
    let op = &task.operators()[applicable_operators[0].index()];

    assert_eq!(op.name(), "drop");
    assert_eq!(initial_state.id(), 0);
    assert_eq!(state_registry.num_registered_states(), 1);

    // Generate the successor state twice
    let successor1 = state_registry
        .successor(&initial_state, op)
        .expect("Failed to get first successor state")
        .0;

    assert_eq!(state_registry.num_registered_states(), 2);
    assert_eq!(successor1.id(), 1);

    let successor2 = state_registry
        .successor(&initial_state, op)
        .expect("Failed to get second successor state")
        .0;

    assert_eq!(state_registry.num_registered_states(), 2);
    assert_eq!(successor2.id(), 1);

    // They should have the same ID if duplicate detection is working
    assert_eq!(
        successor1.id(),
        successor2.id(),
        "Generating the same successor twice should yield the same state ID"
    );
}

#[test]
fn registry_expander_matches_registry_successors() {
    let task = get_root_task();
    let mut registry = StateRegistry::for_task(Arc::new(task.clone()));
    let mut reference = StateRegistry::for_task(Arc::new(task.clone()));
    let initial = registry.register_initial_state();
    let reference_initial = reference.register_initial_state();
    let mut expander = StateExpander::new(StandardGroundedTree::new(&task));

    let mut expansion = expander.expand(&registry, &initial).unwrap();
    assert_eq!(
        expansion.propositional(),
        initial.propositional_values(&registry)
    );
    assert_eq!(
        expansion.numeric(),
        registry.numeric_values(&initial).unwrap()
    );
    assert_eq!(expansion.applicable_operators(), [OperatorIndex::new(0)]);
    for &operator_id in expansion.applicable_operators() {
        let operator = &task.operators()[operator_id.index()];
        let (successor, cost) = expansion.successor(&mut registry, operator).unwrap();
        let (reference_successor, reference_cost) =
            reference.successor(&reference_initial, operator).unwrap();
        assert_eq!(cost, reference_cost);
        assert_eq!(
            successor.propositional_values(&registry),
            reference_successor.propositional_values(&reference)
        );
        assert_eq!(
            registry.numeric_values(&successor).unwrap(),
            reference.numeric_values(&reference_successor).unwrap()
        );
    }
}

/// `n` binary variables and one operator per variable, operator `i` needing
/// `v_i = 1`: the root's default chain passes through all `n` variables.
fn one_condition_per_variable_task(n: usize) -> Task {
    let variables = (0..n)
        .map(|i| {
            ExplicitVariable::new(
                2,
                format!("v{i}"),
                vec![format!("v{i}=0"), format!("v{i}=1")],
                None,
                ExplicitValueIndex::new(0),
            )
        })
        .collect();
    let operators = (0..n)
        .map(|i| {
            Operator::new(
                format!("unset{i}"),
                vec![ExplicitFact::propositional(i, 1)],
                vec![Effect::new(
                    Vec::new(),
                    VariableIndex::from_usize(i),
                    None,
                    ExplicitValueIndex::new(0),
                )],
                Vec::new(),
                OperatorCost::new(1),
            )
        })
        .collect();
    Task::new(TaskParts {
        metric: Metric::new(true, Some(VariableIndex::from_usize(0))),
        variables,
        numeric_variables: vec![NumericVariable::new(
            String::from("total_cost()"),
            NumericType::Cost,
            None,
        )],
        goals: vec![ExplicitFact::propositional(0, 0)],
        mutexes: Vec::new(),
        state: vec![ExplicitValueIndex::new(1); n],
        numeric_state: vec![NumericValue::new(0f64)],
        operators,
        axioms: Vec::new(),
        comparison_axioms: Vec::new(),
        assignment_axioms: Vec::new(),
        global_constraint: Some(ExplicitFact::propositional(0, 0)),
    })
}

/// A default chain through every variable of a large task is built and
/// walked without recursing once per variable (which overflowed the stack
/// on grounded tasks with tens of thousands of variables).
#[test]
fn default_chains_through_all_variables_neither_overflow_nor_reorder() {
    let n = 5_000;
    let task = one_condition_per_variable_task(n);
    let tree = StandardGroundedTree::new(&task);

    let mut applicable: Vec<OperatorIndex> = Vec::new();
    tree.applicable_operators(&vec![ExplicitValueIndex::new(1); n], &mut applicable);
    assert_eq!(
        applicable,
        (0..n as u32).map(OperatorIndex::new).collect::<Vec<_>>()
    );

    let mut state = vec![ExplicitValueIndex::new(0); n];
    state[n - 1] = ExplicitValueIndex::new(1);
    applicable.clear();
    tree.applicable_operators(&state, &mut applicable);
    assert_eq!(applicable, vec![OperatorIndex::new(n as u32 - 1)]);
}

#[test]
fn a_search_names_its_successor_generator() {
    let features = crate::registry::Features::builtin();
    let search = features
        .parse_search("astar(blind(), successor_generator=standard_grounded_tree())")
        .unwrap();
    assert_eq!(search.to_string(), "astar(blind())");
    let error = features
        .parse_search("gbfs(blind(), successor_generator=naive())")
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("unknown successor generator `naive`"),
        "{error}"
    );
}
