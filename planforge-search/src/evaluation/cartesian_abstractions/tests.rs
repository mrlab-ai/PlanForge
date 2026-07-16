use planforge_sas::axioms::{
    AssignmentAxiom, CalOperator, ComparisonAxiom, ComparisonOperator, PropositionalAxiom,
};
use planforge_sas::numeric_task::{
    AssignmentEffect, AssignmentOperation, Effect, ExplicitFact, ExplicitVariable, Metric,
    NumericRootTask, NumericType, NumericVariable, Operator,
};

use super::{
    CartesianAbstractionCollectionConfig, CartesianAbstractionCollectionGenerator,
    CartesianAbstractionConfig, CartesianAbstractionGenerator, CartesianStopReason, ShortestPaths,
    Split, StateRegion, TransitionKey, WorkingAbstraction, retain_min_growth_splits,
};
use crate::evaluation::domain_abstractions::comparison_expression::Interval;

#[test]
fn min_growth_is_sticky_after_the_initial_tie() {
    let mut working = WorkingAbstraction::new(StateRegion {
        propositions: Vec::new(),
        numeric: vec![Interval::unbounded(), Interval::unbounded()],
    });
    let split = |var_id| Split::Numeric {
        state_id: 0,
        var_id,
        boundary: 0.0,
        lower_includes_boundary: true,
        witness_value: 0.0,
        description: String::new(),
    };

    let mut tied = vec![split(0), split(1)];
    retain_min_growth_splits(&working, &mut tied, |candidate| candidate);
    assert_eq!(tied.len(), 2);

    working.numeric_refinement_counts[0] = 1;
    let mut candidates = vec![split(0), split(1)];
    retain_min_growth_splits(&working, &mut candidates, |candidate| candidate);
    assert_eq!(candidates.len(), 1);
    assert!(matches!(candidates[0], Split::Numeric { var_id: 0, .. }));
}

#[test]
fn removed_transitions_are_unlinked_and_their_slots_are_reused() {
    let mut working = WorkingAbstraction::new(StateRegion {
        propositions: Vec::new(),
        numeric: Vec::new(),
    });
    working.add_transition(0, 7, 0);
    assert_eq!(working.transitions.len(), 1);

    let removed = working.remove_incident_transitions(0);
    assert_eq!(removed.len(), 1);
    assert!(working.outgoing[0].is_empty());
    assert!(working.incoming[0].is_empty());
    assert!(working.transitions[0].is_none());

    working.add_transition(0, 7, 0);
    assert_eq!(working.transitions.len(), 1);
    assert_eq!(working.outgoing[0], vec![0]);
    assert_eq!(working.incoming[0], vec![0]);
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
