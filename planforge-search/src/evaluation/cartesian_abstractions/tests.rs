use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use planforge_sas::axioms::{
    AssignmentAxiom, CalOperator, ComparisonAxiom, ComparisonOperator, PropositionalAxiom,
};
use planforge_sas::numeric_task::{
    AssignmentEffect, AssignmentOperation, Effect, ExplicitFact, ExplicitVariable, Metric,
    NumericRootTask, NumericType, NumericVariable, Operator,
};

use super::icaps26::{ArtifactMt19937, Icaps26SplitSelection};
use super::{
    CartesianAbstractionCollectionConfig, CartesianAbstractionCollectionGenerator,
    CartesianAbstractionConfig, CartesianAbstractionGenerator, CartesianRefinementDirection,
    CartesianSemantics, CartesianSplitSelection, CartesianStopReason, FlawKind, OperatorBitSet,
    RefinementNode, ShortestPaths, Split, StateRegion, TransitionKey, WorkingAbstraction,
    apply_split, artifact_unwanted_score, numeric_split_choice_key, numeric_split_intervals,
    push_unique_split, retain_min_growth_splits, select_next_cartesian_collection_goal,
    select_refinement_split,
};
use crate::evaluation::abstraction_collections::portfolio::CollectionStrategy;
use crate::evaluation::domain_abstractions::comparison_expression::Interval;

#[test]
fn icaps_rng_matches_std_mt19937_uniform_integer_distribution() {
    let mut rng = ArtifactMt19937::new(2011);
    let sampled = [2, 3, 5, 7, 11, 100, 2, 17].map(|bound| rng.uniform_index(bound));
    assert_eq!(sampled, [1, 0, 2, 5, 4, 50, 1, 8]);
}

#[test]
fn operator_bitsets_preserve_exact_membership_and_intersections() {
    let mut left = OperatorBitSet::empty(130);
    let mut right = OperatorBitSet::empty(130);
    for operator_id in [0, 1, 63, 64, 65, 129] {
        assert!(left.insert(operator_id));
    }
    assert!(!left.insert(64));
    for operator_id in [1, 64, 127, 129] {
        assert!(right.insert(operator_id));
    }

    assert_eq!(
        left.intersection_iter(&right).collect::<Vec<_>>(),
        vec![1, 64, 129]
    );
    let difference = left.clone_without(&right);
    for operator_id in [0, 63, 65] {
        assert!(difference.contains(operator_id));
    }
    for operator_id in [1, 64, 129] {
        assert!(!difference.contains(operator_id));
    }
}

#[test]
fn numeric_split_keys_include_semantic_identity_and_boundary() {
    let key = numeric_split_choice_key("x(b0)", 1.0, true);
    assert_ne!(key, numeric_split_choice_key("x(b1)", 1.0, true));
    assert_ne!(key, numeric_split_choice_key("x(b0)", 2.0, true));
    assert_ne!(key, numeric_split_choice_key("x(b0)", 1.0, false));
}

#[test]
fn icaps_numeric_splits_preserve_integer_lattices_without_dropping_continuous_values() {
    let (integer_lower, integer_upper) =
        numeric_split_intervals(Interval::unbounded(), 0.0, false, true).unwrap();
    assert!(integer_lower.contains(-1.0));
    assert!(integer_upper.contains(0.0));
    assert!(!integer_lower.contains(-0.5));
    assert!(!integer_upper.contains(-0.5));

    let (continuous_lower, continuous_upper) =
        numeric_split_intervals(Interval::unbounded(), 0.0, false, false).unwrap();
    assert!(continuous_lower.contains(-0.5));
    assert!(!continuous_lower.contains(0.0));
    assert!(continuous_upper.contains(0.0));
}

#[test]
fn icaps_prevail_conditions_fix_the_post_value_only_in_artifact_mode() {
    let task = NumericRootTask::new(
        1,
        Metric::new(true, None),
        vec![ExplicitVariable::new(
            2,
            "position".into(),
            vec!["left".into(), "right".into()],
            None,
            0,
        )],
        vec![],
        vec![ExplicitFact::new(0, 0)],
        vec![],
        vec![0],
        vec![],
        vec![Operator::new(
            "prevail-left".into(),
            vec![ExplicitFact::new(0, 0)],
            vec![],
            vec![],
            1,
        )],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    );
    let source = StateRegion {
        propositions: vec![vec![0, 1]].into(),
        numeric: vec![].into(),
    };
    let target = StateRegion {
        propositions: vec![vec![1]].into(),
        numeric: vec![].into(),
    };

    let native = CartesianSemantics::new(&task, &CartesianAbstractionConfig::default()).unwrap();
    assert!(native.may_transition(&source, 0, &target).unwrap());

    let icaps = CartesianSemantics::new(
        &task,
        &CartesianAbstractionConfig {
            split_selection: CartesianSplitSelection::Icaps26(Icaps26SplitSelection::MinUnwanted),
            ..CartesianAbstractionConfig::default()
        },
    )
    .unwrap();
    assert!(!icaps.may_transition(&source, 0, &target).unwrap());
}

#[test]
fn icaps_split_preserves_artifact_loop_and_arc_order() {
    let operators = vec![
        Operator::new("independent".into(), vec![], vec![], vec![], 1),
        Operator::new(
            "set-right".into(),
            vec![],
            vec![Effect::new(vec![], 0, None, 1)],
            vec![],
            1,
        ),
        Operator::new(
            "set-left".into(),
            vec![],
            vec![Effect::new(vec![], 0, None, 0)],
            vec![],
            1,
        ),
        Operator::new(
            "prevail-left".into(),
            vec![ExplicitFact::new(0, 0)],
            vec![],
            vec![],
            1,
        ),
        Operator::new(
            "prevail-right".into(),
            vec![ExplicitFact::new(0, 1)],
            vec![],
            vec![],
            1,
        ),
    ];
    let task = NumericRootTask::new(
        1,
        Metric::new(true, None),
        vec![ExplicitVariable::new(
            2,
            "position".into(),
            vec!["left".into(), "right".into()],
            None,
            0,
        )],
        vec![],
        vec![ExplicitFact::new(0, 1)],
        vec![],
        vec![0],
        vec![],
        operators,
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    );
    let semantics = CartesianSemantics::new(
        &task,
        &CartesianAbstractionConfig {
            split_selection: CartesianSplitSelection::Icaps26(Icaps26SplitSelection::MinUnwanted),
            ..CartesianAbstractionConfig::default()
        },
    )
    .unwrap();
    let mut working = WorkingAbstraction::new_icaps26(
        StateRegion {
            propositions: vec![vec![0, 1]].into(),
            numeric: vec![].into(),
        },
        5,
    );
    for op_id in 0..5 {
        working.add_transition(0, op_id, 0);
    }

    apply_split(
        &mut working,
        &semantics,
        Split::Propositional {
            state_id: 0,
            var_id: 0,
            wanted: vec![1],
            witness_value: 0,
            description: String::new(),
        },
    )
    .unwrap();

    assert_eq!(
        working.icaps_self_loop_order.as_ref().unwrap(),
        &vec![vec![0, 2, 3], vec![0, 1, 4]]
    );
    let arcs = working
        .active_transition_ids()
        .map(|id| {
            let transition = working.transition(id);
            (
                transition.source,
                transition.concrete_op_id,
                transition.target,
            )
        })
        .collect::<HashSet<_>>();
    assert_eq!(arcs, HashSet::from([(0, 1, 1), (1, 2, 0)]));
}

#[test]
fn whole_plan_candidates_deduplicate_identical_refinements() {
    let split = Split::Numeric {
        state_id: 3,
        var_id: 1,
        boundary: 2.0,
        lower_includes_boundary: true,
        witness_value: 1.0,
        desired_contains_witness: false,
        integer_lattice: false,
        description: "first witness".into(),
    };
    let mut candidates = Vec::new();
    let mut identities = HashSet::new();
    push_unique_split(&mut candidates, &mut identities, split.clone());
    let mut duplicate = split;
    let Split::Numeric { description, .. } = &mut duplicate else {
        unreachable!()
    };
    *description = "same refinement seen later".into();
    push_unique_split(&mut candidates, &mut identities, duplicate);

    assert_eq!(candidates.len(), 1);
}

#[test]
fn min_growth_uses_projected_transition_count() {
    let task = NumericRootTask::new(
        1,
        Metric::new(true, None),
        vec![ExplicitVariable::new(
            2,
            "goal".into(),
            vec!["true".into(), "false".into()],
            None,
            1,
        )],
        vec![
            NumericVariable::new("x".into(), NumericType::Regular, None),
            NumericVariable::new("y".into(), NumericType::Regular, None),
            NumericVariable::new("one".into(), NumericType::Constant, None),
        ],
        vec![ExplicitFact::new(0, 0)],
        vec![],
        vec![0],
        vec![0.0, 0.0, 1.0],
        vec![Operator::new(
            "increment-x".into(),
            vec![],
            vec![],
            vec![AssignmentEffect::new(
                0,
                AssignmentOperation::Plus,
                2,
                false,
                vec![],
            )],
            1,
        )],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 1),
    );
    let semantics = CartesianSemantics::new(&task, &CartesianAbstractionConfig::default()).unwrap();
    let mut working = WorkingAbstraction::new(
        StateRegion {
            propositions: vec![vec![0, 1]].into(),
            numeric: vec![
                Interval::unbounded(),
                Interval::unbounded(),
                Interval::singleton(1.0),
            ]
            .into(),
        },
        1,
    );
    working.add_transition(0, 0, 0);
    let split = |var_id| Split::Numeric {
        state_id: 0,
        var_id,
        boundary: 0.0,
        lower_includes_boundary: true,
        witness_value: 0.0,
        desired_contains_witness: false,
        integer_lattice: false,
        description: String::new(),
    };

    let mut candidates = vec![split(0), split(1)];
    retain_min_growth_splits(&working, &semantics, &mut candidates, |candidate| candidate).unwrap();
    assert_eq!(candidates.len(), 1);
    assert!(matches!(candidates[0], Split::Numeric { var_id: 1, .. }));
}

#[test]
fn icaps_transition_storage_matches_indexed_storage_after_refinement() {
    let task = NumericRootTask::new(
        1,
        Metric::new(true, None),
        vec![ExplicitVariable::new(
            2,
            "location".into(),
            vec!["left".into(), "right".into()],
            None,
            0,
        )],
        vec![
            NumericVariable::new("x".into(), NumericType::Regular, None),
            NumericVariable::new("one".into(), NumericType::Constant, None),
        ],
        vec![ExplicitFact::new(0, 0)],
        vec![],
        vec![0],
        vec![0.0, 1.0],
        vec![Operator::new(
            "increment-x".into(),
            vec![],
            vec![],
            vec![AssignmentEffect::new(
                0,
                AssignmentOperation::Plus,
                1,
                false,
                vec![],
            )],
            1,
        )],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    );
    let config = CartesianAbstractionConfig::default();
    let semantics = CartesianSemantics::new(&task, &config).unwrap();
    let region = semantics.trivial_region().unwrap();
    let mut indexed = WorkingAbstraction::new(region.clone(), 1);
    let mut icaps = WorkingAbstraction::new_icaps26(region, 1);
    indexed.add_transition(0, 0, 0);
    icaps.add_transition(0, 0, 0);

    let numeric_split = Split::Numeric {
        state_id: 0,
        var_id: 0,
        boundary: 0.0,
        lower_includes_boundary: true,
        witness_value: 0.0,
        desired_contains_witness: true,
        integer_lattice: false,
        description: String::new(),
    };
    apply_split(&mut indexed, &semantics, numeric_split.clone()).unwrap();
    apply_split(&mut icaps, &semantics, numeric_split).unwrap();

    let propositional_split = Split::Propositional {
        state_id: 0,
        var_id: 0,
        wanted: vec![0],
        witness_value: 0,
        description: String::new(),
    };
    apply_split(&mut indexed, &semantics, propositional_split.clone()).unwrap();
    apply_split(&mut icaps, &semantics, propositional_split).unwrap();

    let snapshot = |working: &WorkingAbstraction| {
        let mut transitions = working
            .active_transition_ids()
            .map(|transition_id| {
                let transition = working.transition(transition_id);
                (
                    transition.source,
                    transition.concrete_op_id,
                    transition.target,
                )
            })
            .collect::<Vec<_>>();
        transitions.sort_unstable();
        let loops = working
            .self_loop_operator_ids
            .iter()
            .map(|operators| operators.intersection_iter(operators).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        (transitions, loops)
    };
    assert_eq!(snapshot(&indexed), snapshot(&icaps));
    assert!(indexed.transition_ids_by_key.is_some());
    assert!(icaps.transition_ids_by_key.is_none());
}

#[test]
fn icaps26_unwanted_score_counts_excluded_values_and_penalizes_open_desired_tails() {
    let working = WorkingAbstraction::new(
        StateRegion {
            propositions: vec![vec![0, 1, 2, 3]].into(),
            numeric: vec![Interval::new(-10.0, 10.0, true, true)].into(),
        },
        0,
    );
    let propositional = Split::Propositional {
        state_id: 0,
        var_id: 0,
        wanted: vec![2, 3],
        witness_value: 0,
        description: String::new(),
    };
    assert_eq!(
        artifact_unwanted_score(&working, &propositional).unwrap(),
        2.0
    );

    let finite_numeric = Split::Numeric {
        state_id: 0,
        var_id: 0,
        boundary: 4.0,
        lower_includes_boundary: false,
        witness_value: 8.0,
        desired_contains_witness: false,
        integer_lattice: false,
        description: String::new(),
    };
    assert_eq!(
        artifact_unwanted_score(&working, &finite_numeric).unwrap(),
        7.0
    );
    let mut witness_is_desired = finite_numeric.clone();
    let Split::Numeric {
        desired_contains_witness,
        ..
    } = &mut witness_is_desired
    else {
        unreachable!()
    };
    *desired_contains_witness = true;
    assert_eq!(
        artifact_unwanted_score(&working, &witness_is_desired).unwrap(),
        14.0
    );

    let open_tail = WorkingAbstraction::new(
        StateRegion {
            propositions: vec![vec![0, 1]].into(),
            numeric: vec![Interval::unbounded()].into(),
        },
        0,
    );
    let desired_open_tail = Split::Numeric {
        state_id: 0,
        var_id: 0,
        boundary: 4.0,
        lower_includes_boundary: false,
        witness_value: 8.0,
        desired_contains_witness: false,
        integer_lattice: false,
        description: String::new(),
    };
    assert!(
        artifact_unwanted_score(&open_tail, &desired_open_tail)
            .unwrap()
            .is_infinite()
    );

    let fractional = WorkingAbstraction::new(
        StateRegion {
            propositions: vec![vec![0, 1]].into(),
            numeric: vec![Interval::new(0.0, 0.5, true, true)].into(),
        },
        0,
    );
    let fractional_split = Split::Numeric {
        state_id: 0,
        var_id: 0,
        boundary: 0.25,
        lower_includes_boundary: true,
        witness_value: 0.0,
        desired_contains_witness: true,
        integer_lattice: false,
        description: String::new(),
    };
    assert_eq!(
        artifact_unwanted_score(&fractional, &fractional_split).unwrap(),
        0.25
    );

    let open_integer_interval = WorkingAbstraction::new(
        StateRegion {
            propositions: vec![vec![0, 1]].into(),
            numeric: vec![Interval::new(5.0, 10.0, false, true)].into(),
        },
        0,
    );
    let open_integer_split = Split::Numeric {
        state_id: 0,
        var_id: 0,
        boundary: 7.0,
        lower_includes_boundary: false,
        witness_value: 6.0,
        desired_contains_witness: false,
        integer_lattice: false,
        description: String::new(),
    };
    assert_eq!(
        artifact_unwanted_score(&open_integer_interval, &open_integer_split).unwrap(),
        1.0
    );
}

#[test]
fn icaps26_selector_uses_unwanted_values_without_native_growth_filtering() {
    let task = NumericRootTask::new(
        1,
        Metric::new(true, None),
        vec![ExplicitVariable::new(
            4,
            "location".into(),
            vec!["a".into(), "b".into(), "c".into(), "d".into()],
            None,
            3,
        )],
        vec![],
        vec![ExplicitFact::new(0, 1)],
        vec![],
        vec![0],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 3),
    );
    let mut config = CartesianAbstractionConfig {
        split_selection: CartesianSplitSelection::Icaps26(Icaps26SplitSelection::MinUnwanted),
        ..CartesianAbstractionConfig::default()
    };
    let mut semantics = CartesianSemantics::new(&task, &config).unwrap();
    let working = WorkingAbstraction::new(
        StateRegion {
            propositions: vec![vec![0, 1, 2, 3]].into(),
            numeric: semantics.trivial_region().unwrap().numeric,
        },
        0,
    );
    let split = |wanted| Split::Propositional {
        state_id: 0,
        var_id: 0,
        wanted,
        witness_value: 0,
        description: String::new(),
    };
    let candidates = vec![split(vec![1]), split(vec![1, 2, 3])];
    let selected = select_refinement_split(&working, &semantics, candidates.clone(), 0).unwrap();
    assert!(matches!(selected, Split::Propositional { wanted, .. } if wanted.len() == 3));

    config.split_selection = CartesianSplitSelection::Icaps26(Icaps26SplitSelection::MaxUnwanted);
    semantics.split_selection = config.split_selection;
    let selected = select_refinement_split(&working, &semantics, candidates, 0).unwrap();
    assert!(matches!(selected, Split::Propositional { wanted, .. } if wanted.len() == 1));

    config.split_selection = CartesianSplitSelection::Icaps26(Icaps26SplitSelection::Random);
    config.random_seed = Some(2011);
    let random_a = CartesianSemantics::new(&task, &config).unwrap();
    let random_b = CartesianSemantics::new(&task, &config).unwrap();
    let draw = |semantics: &CartesianSemantics<'_>| {
        (0..32)
            .map(|_| {
                let selected = select_refinement_split(
                    &working,
                    semantics,
                    vec![split(vec![1]), split(vec![1, 2, 3])],
                    0,
                )
                .unwrap();
                match selected {
                    Split::Propositional { wanted, .. } => wanted.len(),
                    Split::Numeric { .. } => unreachable!(),
                }
            })
            .collect::<Vec<_>>()
    };
    let sequence = draw(&random_a);
    assert_eq!(sequence, draw(&random_b));
    assert!(sequence.contains(&1) && sequence.contains(&3));
}

#[test]
fn unsupported_cartesian_flaw_kinds_are_rejected() {
    let error = CartesianAbstractionGenerator::new(CartesianAbstractionConfig {
        flaw_kind: FlawKind::Regression,
        ..Default::default()
    })
    .err()
    .expect("unsupported flaw kind must fail");
    assert!(error.to_string().contains("flaw_kind=regression"));
}

#[test]
fn unchanged_transition_footprints_share_state_dimensions() {
    let task = NumericRootTask::new(
        1,
        Metric::new(true, None),
        vec![ExplicitVariable::new(
            1,
            "position".into(),
            vec!["same".into()],
            None,
            0,
        )],
        vec![
            NumericVariable::new("x".into(), NumericType::Regular, None),
            NumericVariable::new("one".into(), NumericType::Constant, None),
        ],
        vec![ExplicitFact::new(0, 0)],
        vec![],
        vec![0],
        vec![0.0, 1.0],
        vec![Operator::new(
            "increment-x".into(),
            vec![],
            vec![],
            vec![AssignmentEffect::new(
                0,
                AssignmentOperation::Plus,
                1,
                false,
                vec![],
            )],
            1,
        )],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    );
    let semantics = CartesianSemantics::new(&task, &CartesianAbstractionConfig::default()).unwrap();
    let source = StateRegion {
        propositions: vec![vec![0]].into(),
        numeric: vec![Interval::unbounded(), Interval::singleton(1.0)].into(),
    };

    let footprint = semantics
        .transition_source_footprint(&source, 0, &source)
        .unwrap()
        .unwrap();

    assert!(Arc::ptr_eq(&source.propositions, &footprint.propositions));
    assert!(Arc::ptr_eq(&source.numeric, &footprint.numeric));
}

#[test]
fn finalized_abstractions_omit_zero_contribution_self_loops() {
    let task = NumericRootTask::new(
        1,
        Metric::new(true, None),
        vec![ExplicitVariable::new(
            2,
            "goal".into(),
            vec!["true".into(), "false".into()],
            None,
            1,
        )],
        vec![],
        vec![ExplicitFact::new(0, 0)],
        vec![],
        vec![0],
        vec![],
        vec![Operator::new("self-loop".into(), vec![], vec![], vec![], 1)],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 1),
    );
    let abstraction = CartesianAbstractionGenerator::new(CartesianAbstractionConfig {
        max_states: 1,
        compute_operator_footprints: true,
        ..CartesianAbstractionConfig::default()
    })
    .unwrap()
    .generate(&task)
    .unwrap();

    assert!(abstraction.transition_system.transitions.is_empty());
    assert!(abstraction.abstract_operator_footprints.is_empty());
    assert!(abstraction.relevant_operator_ids.is_empty());
    assert_eq!(abstraction.distance_table.distances, vec![0.0]);
}

#[test]
fn removed_transitions_are_unlinked_and_their_slots_are_reused() {
    let mut working = WorkingAbstraction::new(
        StateRegion {
            propositions: Vec::new().into(),
            numeric: Vec::new().into(),
        },
        8,
    );
    working.states.push(StateRegion {
        propositions: Vec::new().into(),
        numeric: Vec::new().into(),
    });
    working.outgoing.push(Vec::new());
    working.incoming.push(Vec::new());
    working
        .self_loop_operator_ids
        .push(OperatorBitSet::empty(8));
    working.add_transition(0, 7, 1);
    assert_eq!(working.transitions.len(), 1);

    let removed = working.remove_incident_transitions(0);
    assert_eq!(removed.len(), 1);
    assert!(working.outgoing[0].is_empty());
    assert!(working.incoming[0].is_empty());
    assert!(working.transitions[0].is_none());

    working.add_transition(0, 7, 1);
    assert_eq!(working.transitions.len(), 1);
    assert_eq!(working.outgoing[0], vec![0]);
    assert_eq!(working.incoming[1], vec![0]);
    assert!(working.transitions[0].is_some());
}

#[test]
fn shortest_path_dependency_positions_survive_swap_removal() {
    let first = TransitionKey {
        source: 0,
        concrete_op_id: 0,
        target: 2,
    };
    let second = TransitionKey {
        source: 1,
        concrete_op_id: 1,
        target: 2,
    };
    let mut shortest_paths = ShortestPaths {
        distances: vec![1.0, 1.0, 0.0],
        generating_transition: vec![Some(first), Some(second), None],
        dependents: vec![vec![], vec![], vec![0, 1]],
        dependent_positions: vec![Some(0), Some(1), None],
        is_goal: vec![false, false, true],
        invalid: vec![false; 3],
    };

    shortest_paths.remove_generating_transition(0);
    assert_eq!(shortest_paths.dependents[2], vec![1]);
    assert_eq!(shortest_paths.dependent_positions[1], Some(0));
    shortest_paths.remove_generating_transition(1);
    assert!(shortest_paths.dependents[2].is_empty());
}

#[test]
fn refines_through_propositional_axiom_goal_support() {
    let variables = vec![
        ExplicitVariable::new(
            2,
            "x-at-limit".into(),
            vec!["true".into(), "false".into()],
            Some(0),
            1,
        ),
        ExplicitVariable::new(
            2,
            "goal".into(),
            vec!["true".into(), "false".into()],
            Some(1),
            1,
        ),
    ];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("limit".into(), NumericType::Constant, None),
        NumericVariable::new("one".into(), NumericType::Constant, None),
    ];
    let increment = Operator::new(
        "increment".into(),
        vec![],
        vec![],
        vec![AssignmentEffect::new(
            0,
            AssignmentOperation::Plus,
            2,
            false,
            vec![],
        )],
        1,
    );
    let task = NumericRootTask::new(
        1,
        Metric::new(true, None),
        variables,
        numeric_variables,
        vec![ExplicitFact::new(1, 0)],
        vec![],
        vec![1, 1],
        vec![0.0, 2.0, 1.0],
        vec![increment],
        vec![PropositionalAxiom::new(
            vec![ExplicitFact::new(0, 0)],
            1,
            1,
            0,
        )],
        vec![ComparisonAxiom::new(
            0,
            0,
            1,
            ComparisonOperator::GreaterThanOrEqual,
        )],
        vec![],
        ExplicitFact::new(0, 0),
    );

    let abstraction = CartesianAbstractionGenerator::new(CartesianAbstractionConfig {
        max_states: 16,
        ..Default::default()
    })
    .unwrap()
    .generate(&task)
    .unwrap();

    assert_eq!(
        abstraction.distance_table.distances[abstraction.distance_table.initial_state_hash],
        2.0
    );
    assert!(abstraction.metadata.solved_by_self);
    assert_eq!(
        abstraction.metadata.stop_reason,
        CartesianStopReason::ConcretePlan
    );
}

#[test]
fn supports_snp_assignment_axiom_comparisons() {
    let variables = vec![comparison_variable("sum-at-limit")];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("one".into(), NumericType::Constant, None),
        NumericVariable::new("sum".into(), NumericType::Derived, None),
        NumericVariable::new("limit".into(), NumericType::Constant, None),
    ];
    let increment = Operator::new(
        "increment".into(),
        vec![],
        vec![],
        vec![AssignmentEffect::new(
            0,
            AssignmentOperation::Plus,
            1,
            false,
            vec![],
        )],
        1,
    );
    let task = numeric_goal_task(
        variables,
        numeric_variables,
        vec![0.0, 1.0, 0.0, 3.0],
        vec![increment],
        vec![ComparisonAxiom::new(
            0,
            2,
            3,
            ComparisonOperator::GreaterThanOrEqual,
        )],
        vec![AssignmentAxiom::new(2, CalOperator::Sum, 0, 1)],
    );

    assert_solved_with_h(&task, 2.0);
}

#[test]
fn supports_nonlinear_snp_assignment_axiom_comparisons() {
    let variables = vec![comparison_variable("product-at-nine")];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("y".into(), NumericType::Regular, None),
        NumericVariable::new("one".into(), NumericType::Constant, None),
        NumericVariable::new("product".into(), NumericType::Derived, None),
        NumericVariable::new("nine".into(), NumericType::Constant, None),
    ];
    let increment_both = Operator::new(
        "increment-both".into(),
        vec![],
        vec![],
        vec![
            AssignmentEffect::new(0, AssignmentOperation::Plus, 2, false, vec![]),
            AssignmentEffect::new(1, AssignmentOperation::Plus, 2, false, vec![]),
        ],
        1,
    );
    let task = numeric_goal_task(
        variables,
        numeric_variables,
        vec![1.0, 1.0, 1.0, 0.0, 9.0],
        vec![increment_both],
        vec![ComparisonAxiom::new(
            0,
            3,
            4,
            ComparisonOperator::GreaterThanOrEqual,
        )],
        vec![AssignmentAxiom::new(3, CalOperator::Product, 0, 1)],
    );

    assert_solved_with_h(&task, 2.0);
}

#[test]
fn supports_comparisons_between_regular_variables() {
    let variables = vec![comparison_variable("x-at-y")];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("y".into(), NumericType::Regular, None),
        NumericVariable::new("one".into(), NumericType::Constant, None),
    ];
    let increment = Operator::new(
        "increment-x".into(),
        vec![],
        vec![],
        vec![AssignmentEffect::new(
            0,
            AssignmentOperation::Plus,
            2,
            false,
            vec![],
        )],
        1,
    );
    let task = numeric_goal_task(
        variables,
        numeric_variables,
        vec![0.0, 2.0, 1.0],
        vec![increment],
        vec![ComparisonAxiom::new(
            0,
            0,
            1,
            ComparisonOperator::GreaterThanOrEqual,
        )],
        vec![],
    );

    assert_solved_with_h(&task, 2.0);
}

#[test]
fn supports_multiplicative_numeric_effects() {
    let variables = vec![comparison_variable("x-at-four")];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("two".into(), NumericType::Constant, None),
        NumericVariable::new("four".into(), NumericType::Constant, None),
    ];
    let double = Operator::new(
        "double".into(),
        vec![],
        vec![],
        vec![AssignmentEffect::new(
            0,
            AssignmentOperation::Times,
            1,
            false,
            vec![],
        )],
        1,
    );
    let task = numeric_goal_task(
        variables,
        numeric_variables,
        vec![1.0, 2.0, 4.0],
        vec![double],
        vec![ComparisonAxiom::new(
            0,
            0,
            2,
            ComparisonOperator::GreaterThanOrEqual,
        )],
        vec![],
    );

    assert_solved_with_h(&task, 2.0);
}

fn affine_effect_task(operation: AssignmentOperation, rhs: f64) -> NumericRootTask {
    NumericRootTask::new(
        1,
        Metric::new(false, None),
        vec![ExplicitVariable::new(
            1,
            "dummy".into(),
            vec!["value".into()],
            None,
            0,
        )],
        vec![
            NumericVariable::new("x".into(), NumericType::Regular, None),
            NumericVariable::new("rhs".into(), NumericType::Constant, None),
        ],
        vec![],
        vec![],
        vec![0],
        vec![0.0, rhs],
        vec![Operator::new(
            "affine".into(),
            vec![],
            vec![],
            vec![AssignmentEffect::new(0, operation, 1, false, vec![])],
            1,
        )],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    )
}

#[test]
fn exact_affine_preimages_preserve_open_boundaries() {
    let plus = affine_effect_task(AssignmentOperation::Plus, 1.0);
    let plus_semantics =
        CartesianSemantics::new(&plus, &CartesianAbstractionConfig::default()).unwrap();
    assert_eq!(
        plus_semantics
            .numeric_effect_preimage(Interval::new(5.0, 10.0, false, true), 0, 0)
            .unwrap()
            .unwrap(),
        Interval::new(4.0, 9.0, false, true)
    );

    let times = affine_effect_task(AssignmentOperation::Times, -2.0);
    let times_semantics =
        CartesianSemantics::new(&times, &CartesianAbstractionConfig::default()).unwrap();
    assert_eq!(
        times_semantics
            .numeric_effect_preimage(Interval::new(-10.0, -4.0, false, true), 0, 0)
            .unwrap()
            .unwrap(),
        Interval::new(2.0, 5.0, true, false)
    );

    let divide = affine_effect_task(AssignmentOperation::Divide, -2.0);
    let divide_semantics =
        CartesianSemantics::new(&divide, &CartesianAbstractionConfig::default()).unwrap();
    assert_eq!(
        divide_semantics
            .numeric_effect_preimage(Interval::new(-5.0, -2.0, false, true), 0, 0)
            .unwrap()
            .unwrap(),
        Interval::new(4.0, 10.0, true, false)
    );
}

#[test]
fn assignment_preimage_is_universal_exactly_when_target_contains_rhs() {
    let task = affine_effect_task(AssignmentOperation::Assign, 3.0);
    let semantics = CartesianSemantics::new(&task, &CartesianAbstractionConfig::default()).unwrap();
    assert_eq!(
        semantics
            .numeric_effect_preimage(Interval::new(2.0, 3.0, true, true), 0, 0)
            .unwrap()
            .unwrap(),
        Interval::unbounded()
    );
    assert!(
        semantics
            .numeric_effect_preimage(Interval::new(2.0, 3.0, true, false), 0, 0)
            .unwrap()
            .is_none()
    );
}

#[test]
fn assignment_outside_target_is_not_a_cartesian_transition() {
    let task = affine_effect_task(AssignmentOperation::Assign, 3.0);
    let semantics = CartesianSemantics::new(&task, &CartesianAbstractionConfig::default()).unwrap();
    let source = semantics.trivial_region().unwrap();
    let mut target = source.clone();
    Arc::make_mut(&mut target.numeric)[0] = Interval::new(2.0, 3.0, true, false);

    assert!(!semantics.may_transition(&source, 0, &target).unwrap());
    assert!(
        semantics
            .transition_source_footprint(&source, 0, &target)
            .unwrap()
            .is_none()
    );
}

#[test]
fn refines_failed_default_derived_goal() {
    let variables = vec![
        ExplicitVariable::new(2, "base".into(), vec!["on".into(), "off".into()], None, 0),
        ExplicitVariable::new(
            2,
            "derived".into(),
            vec!["active".into(), "default".into()],
            Some(0),
            1,
        ),
    ];
    let turn_off = Operator::new(
        "turn-off".into(),
        vec![],
        vec![Effect::new(vec![], 0, None, 1)],
        vec![],
        1,
    );
    let task = NumericRootTask::new(
        1,
        Metric::new(true, None),
        variables,
        vec![],
        vec![ExplicitFact::new(1, 1)],
        vec![],
        vec![0, 1],
        vec![],
        vec![turn_off],
        vec![PropositionalAxiom::new(
            vec![ExplicitFact::new(0, 0)],
            1,
            1,
            0,
        )],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    );

    assert_solved_with_h(&task, 1.0);
}

#[test]
fn state_limit_returns_an_admissible_partial_snp_abstraction() {
    let variables = vec![comparison_variable("sum-at-limit")];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("one".into(), NumericType::Constant, None),
        NumericVariable::new("sum".into(), NumericType::Derived, None),
        NumericVariable::new("limit".into(), NumericType::Constant, None),
    ];
    let increment = Operator::new(
        "increment".into(),
        vec![],
        vec![],
        vec![AssignmentEffect::new(
            0,
            AssignmentOperation::Plus,
            1,
            false,
            vec![],
        )],
        1,
    );
    let task = numeric_goal_task(
        variables,
        numeric_variables,
        vec![0.0, 1.0, 0.0, 3.0],
        vec![increment],
        vec![ComparisonAxiom::new(
            0,
            2,
            3,
            ComparisonOperator::GreaterThanOrEqual,
        )],
        vec![AssignmentAxiom::new(2, CalOperator::Sum, 0, 1)],
    );

    let abstraction = CartesianAbstractionGenerator::new(CartesianAbstractionConfig {
        max_states: 1,
        compute_operator_footprints: false,
        ..Default::default()
    })
    .unwrap()
    .generate(&task)
    .unwrap();
    let h = abstraction.distance_table.distances[abstraction.distance_table.initial_state_hash];
    assert!(h <= 2.0);
    assert!(!abstraction.metadata.solved_by_self);
    assert_eq!(
        abstraction.metadata.stop_reason,
        CartesianStopReason::StateLimit
    );
    assert!(abstraction.metadata.pending_flaw.is_some());
    assert!(abstraction.abstract_operator_footprints.is_empty());
}

#[test]
fn goal_collection_builds_every_goal_with_operator_footprints() {
    let variables = vec![
        comparison_variable("x-at-two"),
        comparison_variable("y-at-three"),
    ];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("y".into(), NumericType::Regular, None),
        NumericVariable::new("one".into(), NumericType::Constant, None),
        NumericVariable::new("two".into(), NumericType::Constant, None),
        NumericVariable::new("three".into(), NumericType::Constant, None),
    ];
    let operators = vec![
        Operator::new(
            "increment-x".into(),
            vec![],
            vec![],
            vec![AssignmentEffect::new(
                0,
                AssignmentOperation::Plus,
                2,
                false,
                vec![],
            )],
            1,
        ),
        Operator::new(
            "increment-y".into(),
            vec![],
            vec![],
            vec![AssignmentEffect::new(
                1,
                AssignmentOperation::Plus,
                2,
                false,
                vec![],
            )],
            1,
        ),
    ];
    let task = NumericRootTask::new(
        2,
        Metric::new(true, None),
        variables,
        numeric_variables,
        vec![ExplicitFact::new(0, 0), ExplicitFact::new(1, 0)],
        vec![],
        vec![2, 2],
        vec![0.0, 0.0, 1.0, 2.0, 3.0],
        operators,
        vec![],
        vec![
            ComparisonAxiom::new(0, 0, 3, ComparisonOperator::GreaterThanOrEqual),
            ComparisonAxiom::new(1, 1, 4, ComparisonOperator::GreaterThanOrEqual),
        ],
        vec![],
        ExplicitFact::new(0, 0),
    );

    let abstractions =
        CartesianAbstractionCollectionGenerator::new(CartesianAbstractionCollectionConfig {
            abstraction: CartesianAbstractionConfig {
                max_states: 64,
                compute_operator_footprints: true,
                ..Default::default()
            },
            collection_strategy: CollectionStrategy::Complementary,
            variants_per_goal: 3,
            max_collection_states: 384,
            total_max_time: None,
            progressive_goal_roots: false,
        })
        .unwrap()
        .generate(&task)
        .unwrap();

    assert_eq!(abstractions.len(), 6);
    let initial_h = abstractions
        .iter()
        .map(|abstraction| {
            assert_eq!(
                abstraction.abstract_operator_footprints.len(),
                abstraction.transition_system.transitions.len()
            );
            let mut expected_relevant = abstraction
                .transition_system
                .transitions
                .iter()
                .filter(|transition| transition.source_hash != transition.target_hash)
                .flat_map(|transition| transition.concrete_op_ids.iter().copied())
                .collect::<Vec<_>>();
            expected_relevant.sort_unstable();
            expected_relevant.dedup();
            assert_eq!(abstraction.relevant_operator_ids, expected_relevant);
            abstraction.distance_table.distances[abstraction.distance_table.initial_state_hash]
        })
        .collect::<Vec<_>>();
    assert_eq!(initial_h, vec![2.0, 3.0, 2.0, 3.0, 3.0, 2.0]);
    assert_eq!(abstractions[0].metadata.collection_goal_id, Some(0));
    assert_eq!(abstractions[0].metadata.collection_variant_id, Some(0));
    assert_eq!(abstractions[2].metadata.collection_goal_id, Some(0));
    assert_eq!(abstractions[2].metadata.collection_variant_id, Some(1));
    assert_eq!(abstractions[3].metadata.collection_goal_id, Some(1));
    assert_eq!(abstractions[3].metadata.collection_variant_id, Some(1));
    assert_eq!(abstractions[4].metadata.collection_goal_id, Some(1));
    assert_eq!(abstractions[4].metadata.collection_variant_id, Some(2));
    assert_eq!(abstractions[5].metadata.collection_goal_id, Some(0));
    assert_eq!(abstractions[5].metadata.collection_variant_id, Some(2));
    assert_eq!(
        abstractions[0].metadata.refinement_direction,
        CartesianRefinementDirection::Progression
    );
    assert_eq!(abstractions[0].metadata.split_selection_rank, Some(0));
    assert_eq!(
        abstractions[2].metadata.refinement_direction,
        CartesianRefinementDirection::Regression
    );
    assert_eq!(abstractions[2].metadata.split_selection_rank, Some(0));
    assert_eq!(
        abstractions[4].metadata.refinement_direction,
        CartesianRefinementDirection::Progression
    );
    assert_eq!(abstractions[4].metadata.split_selection_rank, Some(1));

    let standard =
        CartesianAbstractionCollectionGenerator::new(CartesianAbstractionCollectionConfig {
            abstraction: CartesianAbstractionConfig {
                max_states: 64,
                ..Default::default()
            },
            collection_strategy: CollectionStrategy::Standard,
            variants_per_goal: 3,
            max_collection_states: 384,
            total_max_time: None,
            progressive_goal_roots: false,
        })
        .unwrap()
        .generate(&task)
        .unwrap();
    assert!(standard.iter().all(|abstraction| {
        abstraction.metadata.refinement_direction == CartesianRefinementDirection::Progression
            && abstraction.metadata.split_selection_rank.is_none()
    }));

    let bounded =
        CartesianAbstractionCollectionGenerator::new(CartesianAbstractionCollectionConfig {
            abstraction: CartesianAbstractionConfig {
                max_states: 64,
                ..Default::default()
            },
            collection_strategy: CollectionStrategy::Complementary,
            variants_per_goal: 3,
            max_collection_states: 4,
            total_max_time: None,
            progressive_goal_roots: false,
        })
        .unwrap()
        .generate(&task)
        .unwrap();
    assert_eq!(bounded.len(), 1);
    assert!(bounded[0].num_states() > 1);
    assert!(bounded[0].num_states() <= 4);
}

#[test]
fn progressive_goal_roots_refine_from_reachable_concrete_checkpoints() {
    let task = NumericRootTask::new(
        2,
        Metric::new(true, None),
        vec![
            comparison_variable("x-at-least-two"),
            comparison_variable("x-at-least-four"),
        ],
        vec![
            NumericVariable::new("x".into(), NumericType::Regular, None),
            NumericVariable::new("one".into(), NumericType::Constant, None),
            NumericVariable::new("two".into(), NumericType::Constant, None),
            NumericVariable::new("four".into(), NumericType::Constant, None),
        ],
        vec![ExplicitFact::new(0, 0), ExplicitFact::new(1, 0)],
        vec![],
        vec![2, 2],
        vec![0.0, 1.0, 2.0, 4.0],
        vec![Operator::new(
            "increment".into(),
            vec![],
            vec![],
            vec![AssignmentEffect::new(
                0,
                AssignmentOperation::Plus,
                1,
                false,
                vec![],
            )],
            1,
        )],
        vec![],
        vec![
            ComparisonAxiom::new(0, 0, 2, ComparisonOperator::GreaterThanOrEqual),
            ComparisonAxiom::new(1, 0, 3, ComparisonOperator::GreaterThanOrEqual),
        ],
        vec![],
        ExplicitFact::new(0, 0),
    );

    let abstractions =
        CartesianAbstractionCollectionGenerator::new(CartesianAbstractionCollectionConfig {
            abstraction: CartesianAbstractionConfig {
                max_states: 64,
                ..Default::default()
            },
            collection_strategy: CollectionStrategy::Standard,
            variants_per_goal: 1,
            max_collection_states: 128,
            total_max_time: None,
            progressive_goal_roots: true,
        })
        .unwrap()
        .generate(&task)
        .unwrap();

    assert_eq!(abstractions.len(), 3);
    assert_eq!(
        abstractions[0]
            .metadata
            .concrete_plan_operator_ids
            .as_ref()
            .unwrap()
            .len(),
        2
    );
    assert_eq!(
        abstractions[1]
            .metadata
            .concrete_plan_operator_ids
            .as_ref()
            .unwrap()
            .len(),
        2,
        "the second CEGAR run must start after the first concrete plan"
    );
    let second_initial_h =
        abstractions[1].distance_table.distances[abstractions[1].distance_table.initial_state_hash];
    assert!(second_initial_h <= 4.0);
    assert_eq!(abstractions[2].metadata.collection_goal_id, Some(1));
    assert_eq!(abstractions[2].metadata.collection_variant_id, Some(1));
    assert!(!abstractions[2].metadata.progressive_refinement_root);
    assert_eq!(
        abstractions[2]
            .metadata
            .concrete_plan_operator_ids
            .as_ref()
            .expect("initial-root specialist must find the concrete goal")
            .len(),
        4,
        "a goal first refined from a checkpoint also needs an initial-root specialist"
    );
    for x in 0..=4 {
        let propositions = vec![usize::from(x < 2), usize::from(x < 4)];
        let numeric = vec![x as f64, 1.0, 2.0, 4.0];
        let true_distance = (4 - x) as f64;
        for (kind, abstraction) in [
            ("checkpoint-rooted", &abstractions[1]),
            ("initial-root specialist", &abstractions[2]),
        ] {
            let state_id = abstraction
                .abstract_state_id(&propositions, &numeric)
                .unwrap();
            let h = abstraction.distance_table.distances[state_id];
            assert!(
                h <= true_distance,
                "{kind} abstraction overestimates at x={x}: h={h}, h*={true_distance}"
            );
        }
    }
}

#[test]
fn progressive_goal_roots_make_a_lane_terminal_after_reaching_the_full_goal() {
    let task = NumericRootTask::new(
        2,
        Metric::new(true, None),
        vec![
            comparison_variable("x-at-least-four"),
            comparison_variable("x-at-least-two"),
        ],
        vec![
            NumericVariable::new("x".into(), NumericType::Regular, None),
            NumericVariable::new("one".into(), NumericType::Constant, None),
            NumericVariable::new("four".into(), NumericType::Constant, None),
            NumericVariable::new("two".into(), NumericType::Constant, None),
        ],
        vec![ExplicitFact::new(0, 0), ExplicitFact::new(1, 0)],
        vec![],
        vec![2, 2],
        vec![0.0, 1.0, 4.0, 2.0],
        vec![Operator::new(
            "increment".into(),
            vec![],
            vec![],
            vec![AssignmentEffect::new(
                0,
                AssignmentOperation::Plus,
                1,
                false,
                vec![],
            )],
            1,
        )],
        vec![],
        vec![
            ComparisonAxiom::new(0, 0, 2, ComparisonOperator::GreaterThanOrEqual),
            ComparisonAxiom::new(1, 0, 3, ComparisonOperator::GreaterThanOrEqual),
        ],
        vec![],
        ExplicitFact::new(0, 0),
    );

    let abstractions =
        CartesianAbstractionCollectionGenerator::new(CartesianAbstractionCollectionConfig {
            abstraction: CartesianAbstractionConfig {
                max_states: 64,
                ..Default::default()
            },
            collection_strategy: CollectionStrategy::Standard,
            variants_per_goal: 1,
            max_collection_states: 192,
            total_max_time: None,
            progressive_goal_roots: true,
        })
        .unwrap()
        .generate(&task)
        .unwrap();

    assert_eq!(
        abstractions
            .iter()
            .map(|abstraction| abstraction.metadata.collection_goal_id)
            .collect::<Vec<_>>(),
        vec![Some(0), Some(1)]
    );
    assert_eq!(
        abstractions
            .iter()
            .map(|abstraction| {
                abstraction
                    .metadata
                    .concrete_plan_operator_ids
                    .as_ref()
                    .expect("unbounded one-dimensional abstraction must find a plan")
                    .len()
            })
            .collect::<Vec<_>>(),
        vec![4, 2],
        "members after a completed progressive lane must start independently from the initial state"
    );
    assert_eq!(
        abstractions
            .iter()
            .map(|abstraction| abstraction.metadata.progressive_refinement_root)
            .collect::<Vec<_>>(),
        vec![false, false],
        "a completed lane must not be progressed or retried again"
    );
}

#[test]
fn progressive_goal_roots_make_a_lane_terminal_after_a_dead_root() {
    let task = NumericRootTask::new(
        2,
        Metric::new(true, None),
        vec![
            comparison_variable("x-at-least-one"),
            comparison_variable("x-at-most-zero"),
        ],
        vec![
            NumericVariable::new("x".into(), NumericType::Regular, None),
            NumericVariable::new("one".into(), NumericType::Constant, None),
            NumericVariable::new("zero".into(), NumericType::Constant, None),
        ],
        vec![ExplicitFact::new(0, 0), ExplicitFact::new(1, 0)],
        vec![],
        vec![2, 2],
        vec![0.0, 1.0, 0.0],
        vec![Operator::new(
            "increment".into(),
            vec![],
            vec![],
            vec![AssignmentEffect::new(
                0,
                AssignmentOperation::Plus,
                1,
                false,
                vec![],
            )],
            1,
        )],
        vec![],
        vec![
            ComparisonAxiom::new(0, 0, 1, ComparisonOperator::GreaterThanOrEqual),
            ComparisonAxiom::new(1, 0, 2, ComparisonOperator::LessThanOrEqual),
        ],
        vec![],
        ExplicitFact::new(0, 0),
    );

    let abstractions =
        CartesianAbstractionCollectionGenerator::new(CartesianAbstractionCollectionConfig {
            abstraction: CartesianAbstractionConfig {
                max_states: 64,
                ..Default::default()
            },
            collection_strategy: CollectionStrategy::Standard,
            variants_per_goal: 1,
            max_collection_states: 192,
            total_max_time: None,
            progressive_goal_roots: true,
        })
        .unwrap()
        .generate(&task)
        .unwrap();

    assert_eq!(
        abstractions.len(),
        2,
        "unexpected members: {:?}",
        abstractions
            .iter()
            .map(|abstraction| (
                abstraction.metadata.collection_goal_id,
                abstraction.metadata.collection_variant_id,
                abstraction.metadata.progressive_refinement_root,
                abstraction.metadata.stop_reason,
                abstraction.metadata.concrete_plan_operator_ids.as_deref(),
            ))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        abstractions
            .iter()
            .map(|abstraction| abstraction.metadata.collection_goal_id)
            .collect::<Vec<_>>(),
        vec![Some(0), Some(1)]
    );
    assert_eq!(
        abstractions[0]
            .metadata
            .concrete_plan_operator_ids
            .as_deref(),
        Some([0].as_slice())
    );
    assert_eq!(
        abstractions[1]
            .metadata
            .concrete_plan_operator_ids
            .as_deref(),
        Some([].as_slice()),
        "the dead checkpoint must be rebuilt independently at the satisfying initial state"
    );
    assert!(!abstractions[1].metadata.progressive_refinement_root);
}

#[test]
fn progressive_goal_roots_retry_an_earlier_unsatisfied_goal_after_advancing() {
    let task = NumericRootTask::new(
        2,
        Metric::new(true, None),
        vec![
            comparison_variable("x-at-least-four"),
            comparison_variable("x-at-least-two"),
        ],
        vec![
            NumericVariable::new("x".into(), NumericType::Regular, None),
            NumericVariable::new("one".into(), NumericType::Constant, None),
            NumericVariable::new("four".into(), NumericType::Constant, None),
            NumericVariable::new("two".into(), NumericType::Constant, None),
        ],
        vec![ExplicitFact::new(0, 0), ExplicitFact::new(1, 0)],
        vec![],
        vec![2, 2],
        vec![0.0, 1.0, 4.0, 2.0],
        vec![Operator::new(
            "increment".into(),
            vec![],
            vec![],
            vec![AssignmentEffect::new(
                0,
                AssignmentOperation::Plus,
                1,
                false,
                vec![],
            )],
            1,
        )],
        vec![],
        vec![
            ComparisonAxiom::new(0, 0, 2, ComparisonOperator::GreaterThanOrEqual),
            ComparisonAxiom::new(1, 0, 3, ComparisonOperator::GreaterThanOrEqual),
        ],
        vec![],
        ExplicitFact::new(0, 0),
    );

    let abstractions =
        CartesianAbstractionCollectionGenerator::new(CartesianAbstractionCollectionConfig {
            abstraction: CartesianAbstractionConfig {
                max_states: 3,
                ..Default::default()
            },
            collection_strategy: CollectionStrategy::Standard,
            variants_per_goal: 1,
            max_collection_states: 9,
            total_max_time: None,
            progressive_goal_roots: true,
        })
        .unwrap()
        .generate(&task)
        .unwrap();

    assert_eq!(abstractions.len(), 3);
    assert_eq!(
        abstractions
            .iter()
            .map(|abstraction| abstraction.metadata.collection_goal_id)
            .collect::<Vec<_>>(),
        vec![Some(0), Some(1), Some(0)]
    );
    assert!(
        abstractions[0]
            .metadata
            .concrete_plan_operator_ids
            .is_none()
    );
    assert_eq!(
        abstractions[1]
            .metadata
            .concrete_plan_operator_ids
            .as_ref()
            .unwrap()
            .len(),
        2
    );
    assert_eq!(
        abstractions[2]
            .metadata
            .concrete_plan_operator_ids
            .as_ref()
            .unwrap()
            .len(),
        2
    );

    for abstraction in &abstractions {
        for x in 0..=4 {
            let propositions = vec![usize::from(x < 4), usize::from(x < 2)];
            let numeric = vec![x as f64, 1.0, 4.0, 2.0];
            let state_id = abstraction
                .abstract_state_id(&propositions, &numeric)
                .unwrap();
            let h = abstraction.distance_table.distances[state_id];
            let goal = abstraction.metadata.collection_goal_id.unwrap();
            let target: usize = if goal == 0 { 4 } else { 2 };
            assert!(
                h <= target.saturating_sub(x) as f64,
                "goal {goal} abstraction overestimates at x={x}: h={h}"
            );
        }
    }
}

#[test]
fn regression_splits_at_comparison_target_while_progression_splits_at_witness() {
    let task = numeric_goal_task(
        vec![comparison_variable("x-at-least-three")],
        vec![
            NumericVariable::new("x".into(), NumericType::Regular, None),
            NumericVariable::new("one".into(), NumericType::Constant, None),
            NumericVariable::new("three".into(), NumericType::Constant, None),
        ],
        vec![0.0, 1.0, 3.0],
        vec![Operator::new(
            "increment".into(),
            vec![],
            vec![],
            vec![AssignmentEffect::new(
                0,
                AssignmentOperation::Plus,
                1,
                false,
                vec![],
            )],
            1,
        )],
        vec![ComparisonAxiom::new(
            0,
            0,
            2,
            ComparisonOperator::GreaterThanOrEqual,
        )],
        vec![],
    );

    let first_numeric_boundary = |direction| {
        let abstraction = CartesianAbstractionGenerator::new(CartesianAbstractionConfig {
            max_states: 2,
            refinement_direction: direction,
            ..Default::default()
        })
        .unwrap()
        .generate(&task)
        .unwrap();
        abstraction
            .hierarchy
            .nodes
            .iter()
            .find_map(|node| match node {
                RefinementNode::Numeric { boundary, .. } => Some(*boundary),
                _ => None,
            })
            .expect("one Cartesian numeric split")
    };

    assert_eq!(
        first_numeric_boundary(CartesianRefinementDirection::Progression),
        0.0
    );
    assert_eq!(
        first_numeric_boundary(CartesianRefinementDirection::Regression),
        3.0
    );
}

#[test]
fn collection_schedule_covers_each_goal_before_focusing_the_strongest() {
    let mut counts = vec![0, 0, 0];
    let strengths = vec![2.0, 5.0, 3.0];
    let mut selected = Vec::new();
    for _ in 0..9 {
        let goal = select_next_cartesian_collection_goal(&counts, &strengths, 3).unwrap();
        selected.push(goal);
        counts[goal] += 1;
    }

    assert_eq!(selected, vec![0, 1, 2, 0, 1, 2, 1, 2, 0]);
    assert_eq!(counts, vec![3, 3, 3]);
}

#[test]
fn goal_collection_preserves_empty_goal_tasks() {
    let task = NumericRootTask::new(
        0,
        Metric::new(true, None),
        vec![ExplicitVariable::new(
            2,
            "value".into(),
            vec!["zero".into(), "one".into()],
            None,
            0,
        )],
        vec![],
        vec![],
        vec![],
        vec![0],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    );
    let abstractions = CartesianAbstractionCollectionGenerator::new(
        CartesianAbstractionCollectionConfig::default(),
    )
    .unwrap()
    .generate(&task)
    .unwrap();

    assert_eq!(abstractions.len(), 1);
    assert_eq!(
        abstractions[0].distance_table.distances[abstractions[0].distance_table.initial_state_hash],
        0.0
    );
    assert!(abstractions[0].metadata.solved_by_self);
    assert_eq!(abstractions[0].metadata.collection_goal_id, None);
    assert_eq!(abstractions[0].metadata.collection_variant_id, None);
}

#[test]
fn collection_time_limit_keeps_mandatory_first_abstraction() {
    let task = numeric_goal_task(
        vec![comparison_variable("x-at-three")],
        vec![
            NumericVariable::new("x".into(), NumericType::Regular, None),
            NumericVariable::new("one".into(), NumericType::Constant, None),
            NumericVariable::new("three".into(), NumericType::Constant, None),
        ],
        vec![0.0, 1.0, 3.0],
        vec![Operator::new(
            "increment".into(),
            vec![],
            vec![],
            vec![AssignmentEffect::new(
                0,
                AssignmentOperation::Plus,
                1,
                false,
                vec![],
            )],
            1,
        )],
        vec![ComparisonAxiom::new(
            0,
            0,
            2,
            ComparisonOperator::GreaterThanOrEqual,
        )],
        vec![],
    );

    let abstractions =
        CartesianAbstractionCollectionGenerator::new(CartesianAbstractionCollectionConfig {
            abstraction: CartesianAbstractionConfig {
                max_states: 64,
                ..Default::default()
            },
            collection_strategy: CollectionStrategy::Complementary,
            variants_per_goal: 3,
            max_collection_states: 192,
            total_max_time: Some(Duration::ZERO),
            progressive_goal_roots: false,
        })
        .unwrap()
        .generate(&task)
        .unwrap();

    assert_eq!(abstractions.len(), 1);
    assert_eq!(
        abstractions[0].metadata.stop_reason,
        CartesianStopReason::TimeLimit
    );
    assert_eq!(abstractions[0].metadata.collection_goal_id, Some(0));
    assert_eq!(abstractions[0].metadata.collection_variant_id, Some(0));
}

fn comparison_variable(name: &str) -> ExplicitVariable {
    ExplicitVariable::new(
        3,
        name.into(),
        vec!["true".into(), "false".into(), "unknown".into()],
        Some(0),
        2,
    )
}

fn numeric_goal_task(
    variables: Vec<ExplicitVariable>,
    numeric_variables: Vec<NumericVariable>,
    initial_numeric: Vec<f64>,
    operators: Vec<Operator>,
    comparison_axioms: Vec<ComparisonAxiom>,
    assignment_axioms: Vec<AssignmentAxiom>,
) -> NumericRootTask {
    NumericRootTask::new(
        1,
        Metric::new(true, None),
        variables,
        numeric_variables,
        vec![ExplicitFact::new(0, 0)],
        vec![],
        vec![2],
        initial_numeric,
        operators,
        vec![],
        comparison_axioms,
        assignment_axioms,
        ExplicitFact::new(0, 0),
    )
}

fn assert_solved_with_h(task: &NumericRootTask, expected_h: f64) {
    for flaw_kind in [FlawKind::Progression, FlawKind::ExecuteEntirePlan] {
        let abstraction = CartesianAbstractionGenerator::new(CartesianAbstractionConfig {
            max_states: 64,
            flaw_kind,
            ..Default::default()
        })
        .unwrap()
        .generate(task)
        .unwrap();
        assert_eq!(
            abstraction.distance_table.distances[abstraction.distance_table.initial_state_hash],
            expected_h,
            "unexpected initial h for {flaw_kind}"
        );
        assert!(
            abstraction.metadata.solved_by_self,
            "{flaw_kind} failed to produce a concrete plan"
        );
        assert_eq!(
            abstraction.metadata.stop_reason,
            CartesianStopReason::ConcretePlan
        );
        assert_eq!(
            abstraction.abstract_operator_footprints.len(),
            abstraction.transition_system.transitions.len()
        );
    }
}
