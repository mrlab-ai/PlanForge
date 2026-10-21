use super::*;
use crate::evaluation::abstractions::collections::generator::AbstractionGenerator;
use crate::evaluation::abstractions::common::cegar::options::CegarOptions;

#[test]
fn unchanged_transition_source_sets_share_state_dimensions() {
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 1,
        metric: Metric::new(true, None),
        variables: vec![ExplicitVariable::new(
            1,
            "position".into(),
            vec!["same".into()],
            None,
            ExplicitValueIndex::new(0),
        )],
        numeric_variables: vec![
            NumericVariable::new("x".into(), NumericType::Regular, None),
            NumericVariable::new("one".into(), NumericType::Constant, None),
        ],
        goals: vec![ExplicitFact::propositional(0, 0)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![NumericValue::new(0.0), NumericValue::new(1.0)],
        operators: vec![Operator::new(
            "increment-x".into(),
            vec![],
            vec![],
            vec![AssignmentEffect::new(
                VariableIndex::from_usize(0),
                AssignmentOperation::Plus,
                VariableIndex::from_usize(1),
                false,
                vec![],
            )],
            OperatorCost::new(1),
        )],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });
    let semantics = CartesianSemantics::new(&task, &CartesianAbstractionConfig::default()).unwrap();
    let source = CartesianSet::with_all_props_constrained(
        vec![vec![ExplicitValueIndex::new(0)]],
        vec![Interval::unbounded(), Interval::singleton(ONE_VALUE)],
    );

    let source_set = semantics
        .regions()
        .transition_source_set(&source, OperatorIndex::new(0), &source)
        .unwrap()
        .unwrap();

    assert!(Arc::ptr_eq(
        source.propositions_arc(),
        source_set.propositions_arc()
    ));
    assert!(Arc::ptr_eq(&source.numeric, &source_set.numeric));
}

#[test]
fn finalized_abstractions_omit_zero_contribution_self_loops() {
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 1,
        metric: Metric::new(true, None),
        variables: vec![ExplicitVariable::new(
            2,
            "goal".into(),
            vec!["true".into(), "false".into()],
            None,
            ExplicitValueIndex::new(1),
        )],
        numeric_variables: vec![],
        goals: vec![ExplicitFact::propositional(0, 0)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![],
        operators: vec![Operator::new(
            "self-loop".into(),
            vec![],
            vec![],
            vec![],
            OperatorCost::new(1),
        )],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 1),
    });
    let abstraction = CartesianAbstractionGenerator::new(CartesianAbstractionConfig {
        cegar: CegarOptions {
            max_states: 1,
            ..CartesianAbstractionConfig::default().cegar
        },
        compute_source_sets: true,
        ..CartesianAbstractionConfig::default()
    })
    .unwrap()
    .generate(&task)
    .unwrap();

    assert!(abstraction.transition_system.transitions.is_empty());
    assert!(abstraction.abstract_source_sets.is_empty());
    assert!(abstraction.relevant_operator_ids.is_empty());
    assert_eq!(abstraction.distance_table.distances, vec![0.0]);
}

#[test]
fn standalone_finalization_reuses_exact_distances_without_materializing_transitions() {
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 1,
        metric: Metric::new(true, None),
        variables: vec![ExplicitVariable::new(
            2,
            "goal".into(),
            vec!["false".into(), "true".into()],
            None,
            ExplicitValueIndex::new(0),
        )],
        numeric_variables: vec![],
        goals: vec![ExplicitFact::propositional(0, 1)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![],
        operators: vec![Operator::new(
            "achieve".into(),
            vec![ExplicitFact::propositional(0, 0)],
            vec![Effect::new(
                vec![],
                VariableIndex::from_usize(0),
                Some(ExplicitValueIndex::new(0)),
                ExplicitValueIndex::new(1),
            )],
            vec![],
            OperatorCost::new(3),
        )],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });
    let generate = |retain_transition_system| {
        CartesianAbstractionGenerator::new(CartesianAbstractionConfig {
            cegar: CegarOptions {
                max_states: 2,
                ..CartesianAbstractionConfig::default().cegar
            },
            compute_source_sets: false,
            retain_transition_system,
            ..CartesianAbstractionConfig::default()
        })
        .unwrap()
        .generate(&task)
        .unwrap()
    };

    let full = generate(true);
    let standalone = generate(false);
    assert_eq!(
        standalone.distance_table.distances,
        full.distance_table.distances
    );
    assert_eq!(standalone.relevant_operator_ids, full.relevant_operator_ids);
    assert_eq!(standalone.transition_count, full.transition_count);
    assert!(!full.transition_system.transitions.is_empty());
    assert!(standalone.transition_system.transitions.is_empty());
    assert!(standalone.transition_system.cartesian_sets.is_empty());
}

#[test]
fn removed_transitions_are_unlinked_and_their_slots_are_reused() {
    let mut working = WorkingAbstraction::new(
        CartesianSet::with_all_props_constrained(Vec::new(), Vec::new()),
        8,
    );
    working
        .states
        .push(CartesianSet::with_all_props_constrained(
            Vec::new(),
            Vec::new(),
        ));
    working.outgoing.push(Vec::new());
    working.incoming.push(Vec::new());
    working
        .self_loop_operator_ids
        .push(OperatorBitSet::empty(8));
    working.add_transition(0, OperatorIndex::new(7), 1);
    assert_eq!(working.transitions.len(), 1);

    let removed = working.remove_incident_transitions(0);
    assert_eq!(removed.len(), 1);
    assert!(working.outgoing[0].is_empty());
    assert!(working.incoming[0].is_empty());
    assert!(working.transitions[0].is_none());

    working.add_transition(0, OperatorIndex::new(7), 1);
    assert_eq!(working.transitions.len(), 1);
    assert_eq!(working.outgoing[0], vec![0]);
    assert_eq!(working.incoming[1], vec![0]);
    assert!(working.transitions[0].is_some());
}

#[test]
fn shortest_path_dependency_positions_survive_swap_removal() {
    let first = TransitionKey {
        source: 0,
        concrete_op_id: OperatorIndex::new(0),
        target: 2,
    };
    let second = TransitionKey {
        source: 1,
        concrete_op_id: OperatorIndex::new(1),
        target: 2,
    };
    let mut shortest_paths = ShortestPaths::for_test(
        vec![1.0, 1.0, 0.0],
        vec![Some(first), Some(second), None],
        vec![vec![], vec![], vec![0, 1]],
        vec![Some(0), Some(1), None],
        vec![false, false, true],
    );

    shortest_paths.remove_generating_transition(0);
    assert_eq!(shortest_paths.dependents(2), [1]);
    assert_eq!(shortest_paths.dependent_position(1), Some(0));
    shortest_paths.remove_generating_transition(1);
    assert!(shortest_paths.dependents(2).is_empty());
}
