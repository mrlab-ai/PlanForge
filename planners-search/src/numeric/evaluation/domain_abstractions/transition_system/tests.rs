use std::collections::HashSet;

use planners_sas::numeric::numeric_task::ExplicitFact;

use crate::numeric::evaluation::domain_abstractions::{
    comparison_expression::Interval,
    domain_abstraction::NumericPartitions,
    transition_system::{Refinement, TransitionSystem},
};

#[test]
fn test_abstract_state_hashes() {
    let ts = TransitionSystem {
        initial_state_prop: vec![0, 0],
        initial_state_numeric: vec![0.0],
        goals: vec![ExplicitFact::new(0, 1)],
        non_looping_transitions: 0,
        n_loops: 0,
        incoming_transitions: vec![],
        outgoing_transitions: vec![],
        loops: vec![],
        initial_abstract_state_hash: 0,
        goal_abstract_states_hashes: HashSet::from_iter([1]),
        domain_mapping: vec![vec![0, 1, 2, 3], vec![0, 1]],
        domain_sizes: vec![4, 2],
        partitions: NumericPartitions::with_partitions(vec![vec![
            Interval::new(-f64::INFINITY, 0.0, false, false),
            Interval::new(0.0, 10.0, true, true),
            Interval::new(10.0, f64::INFINITY, false, false),
        ]]),
        numeric_domain_sizes: vec![3],
        applied_refinements: vec![
            Refinement {
                var: 0,
                value: 0,
                next_value: 1,
                n_states_before_refinement: 1,
                numeric: false,
            },
            Refinement {
                var: 1,
                value: 0,
                next_value: 1,
                n_states_before_refinement: 2,
                numeric: false,
            },
            Refinement {
                var: 0,
                value: 0,
                next_value: 1,
                n_states_before_refinement: 4,
                numeric: true,
            },
            Refinement {
                var: 0,
                value: 0,
                next_value: 2,
                n_states_before_refinement: 8,
                numeric: false,
            },
            Refinement {
                var: 0,
                value: 1,
                next_value: 2,
                n_states_before_refinement: 12,
                numeric: true,
            },
            Refinement {
                var: 0,
                value: 1,
                next_value: 3,
                n_states_before_refinement: 18,
                numeric: false,
            },
        ],
    };

    assert_eq!(ts.abstract_state_hash(&[0, 0], &[0]), 0);
    assert_eq!(ts.abstract_state_hash(&[2, 1], &[0]), 9);
    assert_eq!(ts.abstract_state_hash(&[3, 1], &[0]), 19);
    assert_eq!(ts.abstract_state_hash(&[3, 1], &[2]), 23);
    assert_eq!(ts.abstract_state_hash(&[0, 1], &[1]), 6);
    assert_eq!(ts.abstract_state_hash(&[0, 1], &[0]), 2);
    assert_eq!(ts.abstract_state_hash(&[1, 1], &[0]), 3);
    assert_eq!(ts.abstract_state_hash(&[2, 0], &[2]), 16);
    assert_eq!(ts.abstract_state_hash(&[2, 1], &[1]), 11);
    assert_eq!(ts.abstract_state_hash(&[1, 1], &[2]), 15);

    assert_eq!(
        ts.abstract_states_with_abstract_value(0, 0, false),
        vec![0, 2, 4, 6, 12, 14]
    );
    assert_eq!(
        ts.abstract_states_with_abstract_value(0, 1, false),
        vec![1, 3, 5, 7, 13, 15]
    );
    assert_eq!(
        ts.abstract_states_with_abstract_value(0, 2, false),
        vec![8, 9, 10, 11, 16, 17]
    );
    assert_eq!(
        ts.abstract_states_with_abstract_value(0, 3, false),
        vec![18, 19, 20, 21, 22, 23]
    );
    assert_eq!(
        ts.abstract_states_with_abstract_value(0, 2, true),
        vec![4, 5, 6, 7, 10, 11, 20, 21]
    );
}
