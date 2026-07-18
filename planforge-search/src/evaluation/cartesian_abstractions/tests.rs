use std::time::Duration;

use planforge_sas::axioms::{
    AssignmentAxiom, CalOperator, ComparisonAxiom, ComparisonOperator, PropositionalAxiom,
};
use planforge_sas::numeric_task::{
    AssignmentEffect, AssignmentOperation, Effect, ExplicitFact, ExplicitVariable, Metric,
    NumericRootTask, NumericType, NumericVariable, Operator,
};

use super::{
    CartesianAbstractionCollectionConfig, CartesianAbstractionCollectionGenerator,
    CartesianAbstractionConfig, CartesianAbstractionGenerator, CartesianSemantics,
    CartesianStopReason, OperatorBitSet, ShortestPaths, Split, StateRegion, TransitionKey,
    WorkingAbstraction, numeric_split_choice_key, retain_min_growth_splits,
};
use crate::evaluation::domain_abstractions::comparison_expression::Interval;

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
    let semantics = CartesianSemantics::new(&task, None).unwrap();
    let mut working = WorkingAbstraction::new(
        StateRegion {
            propositions: vec![vec![0, 1]],
            numeric: vec![
                Interval::unbounded(),
                Interval::unbounded(),
                Interval::singleton(1.0),
            ],
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
        description: String::new(),
    };

    let mut candidates = vec![split(0), split(1)];
    retain_min_growth_splits(&working, &semantics, &mut candidates, |candidate| candidate).unwrap();
    assert_eq!(candidates.len(), 1);
    assert!(matches!(candidates[0], Split::Numeric { var_id: 1, .. }));
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
            propositions: Vec::new(),
            numeric: Vec::new(),
        },
        8,
    );
    working.states.push(StateRegion {
        propositions: Vec::new(),
        numeric: Vec::new(),
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
    let plus_semantics = CartesianSemantics::new(&plus, None).unwrap();
    assert_eq!(
        plus_semantics
            .numeric_effect_preimage(Interval::new(5.0, 10.0, false, true), 0, 0)
            .unwrap(),
        Interval::new(4.0, 9.0, false, true)
    );

    let times = affine_effect_task(AssignmentOperation::Times, -2.0);
    let times_semantics = CartesianSemantics::new(&times, None).unwrap();
    assert_eq!(
        times_semantics
            .numeric_effect_preimage(Interval::new(-10.0, -4.0, false, true), 0, 0)
            .unwrap(),
        Interval::new(2.0, 5.0, true, false)
    );

    let divide = affine_effect_task(AssignmentOperation::Divide, -2.0);
    let divide_semantics = CartesianSemantics::new(&divide, None).unwrap();
    assert_eq!(
        divide_semantics
            .numeric_effect_preimage(Interval::new(-5.0, -2.0, false, true), 0, 0)
            .unwrap(),
        Interval::new(4.0, 10.0, true, false)
    );
}

#[test]
fn assignment_preimage_is_universal_exactly_when_target_contains_rhs() {
    let task = affine_effect_task(AssignmentOperation::Assign, 3.0);
    let semantics = CartesianSemantics::new(&task, None).unwrap();
    assert_eq!(
        semantics
            .numeric_effect_preimage(Interval::new(2.0, 3.0, true, true), 0, 0)
            .unwrap(),
        Interval::unbounded()
    );
    assert!(
        semantics
            .numeric_effect_preimage(Interval::new(2.0, 3.0, true, false), 0, 0)
            .is_err()
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
            variants_per_goal: 3,
            max_collection_states: 384,
            total_max_time: None,
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
    assert_eq!(initial_h, vec![2.0, 2.0, 2.0, 3.0, 3.0, 3.0]);
    assert_eq!(abstractions[0].metadata.collection_goal_id, Some(0));
    assert_eq!(abstractions[0].metadata.collection_variant_id, Some(0));
    assert_eq!(abstractions[2].metadata.collection_goal_id, Some(0));
    assert_eq!(abstractions[2].metadata.collection_variant_id, Some(2));
    assert_eq!(abstractions[3].metadata.collection_goal_id, Some(1));
    assert_eq!(abstractions[3].metadata.collection_variant_id, Some(0));
    assert_eq!(abstractions[5].metadata.collection_goal_id, Some(1));
    assert_eq!(abstractions[5].metadata.collection_variant_id, Some(2));
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
            variants_per_goal: 3,
            max_collection_states: 192,
            total_max_time: Some(Duration::ZERO),
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
    let abstraction = CartesianAbstractionGenerator::new(CartesianAbstractionConfig {
        max_states: 64,
        ..Default::default()
    })
    .unwrap()
    .generate(task)
    .unwrap();
    assert_eq!(
        abstraction.distance_table.distances[abstraction.distance_table.initial_state_hash],
        expected_h
    );
    assert!(abstraction.metadata.solved_by_self);
    assert_eq!(
        abstraction.metadata.stop_reason,
        CartesianStopReason::ConcretePlan
    );
    assert_eq!(
        abstraction.abstract_operator_footprints.len(),
        abstraction.transition_system.transitions.len()
    );
}
