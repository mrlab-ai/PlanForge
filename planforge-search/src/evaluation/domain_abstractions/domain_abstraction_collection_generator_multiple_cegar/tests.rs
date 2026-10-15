use super::*;

#[test]
fn standard_collection_defaults_match_numeric_fd_canonical_configuration() {
    let config = DomainAbstractionCollectionGeneratorMultipleCegarConfig::default();
    assert_eq!(config.collection_strategy, CollectionStrategy::Standard);
    assert_eq!(config.max_abstraction_size, 1_000_000);
    assert_eq!(config.max_collection_size, 10_000_000);
    assert!(!config.use_wildcard_plans);
    assert_eq!(config.random_seed, Some(2011));
    assert_eq!(config.flaw_kind, FlawKind::ExecuteEntirePlan);
    assert!(!config.interleave_split_directions);
}
use planforge_sas::axioms::{AssignmentAxiom, CalOperator};
use planforge_sas::numeric_task::{
    AssignmentEffect, AssignmentOperation, Effect, ExplicitFact, ExplicitValueIndex,
    ExplicitVariable, Metric, NumericRootTask, NumericRootTaskParts, NumericVariable, OperatorCost,
    VariableIndex,
};

#[test]
fn collection_builds_one_abstraction_before_enforcing_its_time_limit() {
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 1,
        metric: Metric::new(true, None),
        variables: vec![ExplicitVariable::new(
            2,
            "goal".into(),
            vec!["false".into(), "true".into()],
            None,
            ExplicitValueIndex::new(1),
        )],
        numeric_variables: vec![],
        goals: vec![ExplicitFact::propositional(0, 1)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![],
        operators: vec![Operator::new(
            "set-goal".into(),
            vec![],
            vec![Effect::new(
                vec![],
                VariableIndex::new(0),
                Some(ExplicitValueIndex::new(0)),
                ExplicitValueIndex::new(1),
            )],
            vec![],
            OperatorCost::new(1),
        )],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });
    let config = DomainAbstractionCollectionGeneratorMultipleCegarConfig {
        max_abstraction_size: 10,
        max_collection_size: 100,
        abstraction_generation_max_time: 0.0,
        total_max_time: 0.0,
        compute_operator_regions: false,
        ..Default::default()
    };
    let abstractions = DomainAbstractionCollectionGeneratorMultipleCegar::new(config)
        .generate_collection(&task)
        .unwrap();

    assert_eq!(abstractions.len(), 1);
    assert_eq!(
        abstractions[0].metadata.abstraction_use,
        AbstractionUse::CollectionMember
    );
}

#[test]
fn single_init_split_selection_uses_round_robin_iteration_order() {
    let candidates = [
        VariableIndex::from_usize(0usize),
        VariableIndex::from_usize(1),
        VariableIndex::from_usize(2),
        VariableIndex::from_usize(3),
        VariableIndex::from_usize(4),
    ];
    let selected = (1..=8)
        .map(|iteration| select_single_init_split_var(&candidates, iteration).unwrap())
        .collect::<Vec<_>>();

    assert_eq!(
        selected,
        vec![
            VariableIndex::from_usize(1),
            VariableIndex::from_usize(2),
            VariableIndex::from_usize(3),
            VariableIndex::from_usize(4),
            VariableIndex::from_usize(0),
            VariableIndex::from_usize(1),
            VariableIndex::from_usize(2),
            VariableIndex::from_usize(3)
        ]
    );
}

#[test]
fn single_init_split_selection_handles_empty_candidates() {
    assert_eq!(select_single_init_split_var(&[], 1), None);
}

#[test]
fn standard_uses_configured_full_goal_flaw_kind() {
    let config = DomainAbstractionCollectionGeneratorMultipleCegarConfig {
        collection_strategy: CollectionStrategy::Standard,
        flaw_kind: FlawKind::SequenceBidirectional,
        ..Default::default()
    };
    let generator = DomainAbstractionCollectionGeneratorMultipleCegar::new(config);

    assert!(generator.uses_full_goal_task(11, 1));
    assert!(generator.uses_full_goal_task(11, 2));
    assert_eq!(
        generator.flaw_kind_for_goal_count(11, 1),
        FlawKind::SequenceBidirectional
    );
}

#[test]
fn standard_can_interleave_split_directions_on_the_full_task() {
    let config = DomainAbstractionCollectionGeneratorMultipleCegarConfig {
        collection_strategy: CollectionStrategy::Standard,
        flaw_kind: FlawKind::ExecuteEntirePlan,
        interleave_split_directions: true,
        ..Default::default()
    };
    let generator = DomainAbstractionCollectionGeneratorMultipleCegar::new(config);

    assert!(generator.uses_full_goal_task(18, 1));
    assert!(generator.uses_full_goal_task(18, 2));
    assert_eq!(
        generator.flaw_kind_for_goal_count(18, 1),
        FlawKind::ExecuteEntirePlan
    );
    assert_eq!(
        generator.split_direction_for_iteration(1),
        Some(SplitDirection::Forward)
    );
    assert_eq!(
        generator.split_direction_for_iteration(2),
        Some(SplitDirection::Backward)
    );
    assert_eq!(
        generator.split_direction_for_iteration(3),
        Some(SplitDirection::Forward)
    );
}

#[test]
fn interleaving_rejects_ambiguous_direction_configuration() {
    let config = DomainAbstractionCollectionGeneratorMultipleCegarConfig {
        interleave_split_directions: true,
        split_direction: Some(SplitDirection::Forward),
        ..Default::default()
    };
    let error = DomainAbstractionCollectionGeneratorMultipleCegar::new(config)
        .validate_supported_options()
        .unwrap_err();

    assert!(error.to_string().contains("cannot be combined"));
}

#[test]
fn interleaving_rejects_single_goal_complementary_collections() {
    let config = DomainAbstractionCollectionGeneratorMultipleCegarConfig {
        collection_strategy: CollectionStrategy::Complementary,
        interleave_split_directions: true,
        ..Default::default()
    };
    let error = DomainAbstractionCollectionGeneratorMultipleCegar::new(config)
        .validate_supported_options()
        .unwrap_err();

    assert!(error.to_string().contains("collection_strategy=standard"));
}

#[test]
fn complementary_schedule_cycles_every_goal_group_and_direction() {
    let mut goal_index = 0;
    let mut group_index = 0;
    let mut direction = ComplementaryDirection::Regression;
    let mut visited = Vec::new();
    let group_counts = [2, 1];

    loop {
        visited.push((goal_index, group_index, direction));
        let wrapped = advance_complementary_schedule(
            group_counts.len(),
            group_counts[goal_index],
            &mut goal_index,
            &mut group_index,
            &mut direction,
        );
        if wrapped {
            break;
        }
    }

    assert_eq!(
        visited,
        vec![
            (0, 0, ComplementaryDirection::Regression),
            (0, 0, ComplementaryDirection::Progression),
            (0, 1, ComplementaryDirection::Regression),
            (0, 1, ComplementaryDirection::Progression),
            (1, 0, ComplementaryDirection::Regression),
            (1, 0, ComplementaryDirection::Progression),
        ]
    );
    assert_eq!(goal_index, 0);
    assert_eq!(group_index, 0);
    assert_eq!(direction, ComplementaryDirection::Regression);
}

#[test]
fn numeric_seed_shells_are_interleaved_across_dimensions() {
    let numeric = |numeric_var_id, value| InitialSeedSplit::Numeric {
        numeric_var_id,
        value,
        include_in_lower: true,
    };
    let mut seeds = vec![InitialSeedSplit::Propositional {
        var_id: VariableIndex::from_usize(3),
        value: ExplicitValueIndex::new(1),
    }];

    append_interleaved_numeric_seeds(
        &mut seeds,
        vec![
            vec![
                numeric(VariableIndex::from_usize(9), NumericValue::new(0.0)),
                numeric(VariableIndex::from_usize(9), NumericValue::new(1.0)),
            ],
            vec![
                numeric(VariableIndex::from_usize(2), NumericValue::new(0.0)),
                numeric(VariableIndex::from_usize(2), NumericValue::new(1.0)),
                numeric(VariableIndex::from_usize(2), NumericValue::new(2.0)),
            ],
        ],
    );

    assert_eq!(
        seeds,
        vec![
            InitialSeedSplit::Propositional {
                var_id: VariableIndex::from_usize(3),
                value: ExplicitValueIndex::new(1),
            },
            numeric(VariableIndex::from_usize(2), NumericValue::new(0.0)),
            numeric(VariableIndex::from_usize(9), NumericValue::new(0.0)),
            numeric(VariableIndex::from_usize(2), NumericValue::new(1.0)),
            numeric(VariableIndex::from_usize(9), NumericValue::new(1.0)),
            numeric(VariableIndex::from_usize(2), NumericValue::new(2.0)),
        ]
    );
}

#[test]
fn affine_root_groups_share_immutable_anchors_without_merging_independent_ones() {
    let numeric_variables = vec![
        NumericVariable::new("mutable-a".into(), NumericType::Regular, None),
        NumericVariable::new("mutable-b".into(), NumericType::Regular, None),
        NumericVariable::new("anchor-a".into(), NumericType::Regular, None),
        NumericVariable::new("anchor-b".into(), NumericType::Regular, None),
        NumericVariable::new("first-coordinate".into(), NumericType::Derived, None),
        NumericVariable::new("second-coordinate".into(), NumericType::Derived, None),
        NumericVariable::new("independent-coordinate".into(), NumericType::Derived, None),
        NumericVariable::new("one".into(), NumericType::Constant, None),
    ];
    let operators = vec![
        Operator::new(
            "change-a".into(),
            vec![],
            vec![],
            vec![AssignmentEffect::new(
                VariableIndex::from_usize(0),
                AssignmentOperation::Plus,
                VariableIndex::from_usize(7),
                false,
                vec![],
            )],
            OperatorCost::new(1),
        ),
        Operator::new(
            "change-b".into(),
            vec![],
            vec![],
            vec![AssignmentEffect::new(
                VariableIndex::from_usize(1),
                AssignmentOperation::Plus,
                VariableIndex::from_usize(7),
                false,
                vec![],
            )],
            OperatorCost::new(1),
        ),
    ];
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, None),
        variables: vec![ExplicitVariable::new(
            1,
            "global-constraint".into(),
            vec!["true".into()],
            None,
            ExplicitValueIndex::new(0),
        )],
        numeric_variables,
        goals: vec![],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![
            NumericValue::new(0.0),
            NumericValue::new(0.0),
            NumericValue::new(10.0),
            NumericValue::new(20.0),
            NumericValue::new(0.0),
            NumericValue::new(0.0),
            NumericValue::new(0.0),
            NumericValue::new(1.0),
        ],
        operators,
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![
            AssignmentAxiom::new(
                VariableIndex::from_usize(4),
                CalOperator::Difference,
                VariableIndex::from_usize(0),
                VariableIndex::from_usize(2),
            ),
            AssignmentAxiom::new(
                VariableIndex::from_usize(5),
                CalOperator::Difference,
                VariableIndex::from_usize(1),
                VariableIndex::from_usize(2),
            ),
            AssignmentAxiom::new(
                VariableIndex::from_usize(6),
                CalOperator::Difference,
                VariableIndex::from_usize(0),
                VariableIndex::from_usize(3),
            ),
        ],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    let first = numeric_root_group_key(&task, &task, VariableIndex::from_usize(4)).unwrap();
    let second = numeric_root_group_key(&task, &task, VariableIndex::from_usize(5)).unwrap();
    let independent = numeric_root_group_key(&task, &task, VariableIndex::from_usize(6)).unwrap();

    assert_eq!(first, second);
    assert_ne!(first, independent);
}
