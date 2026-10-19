use super::*;

#[test]
fn goal_collection_builds_every_goal_with_source_sets() {
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
                VariableIndex::from_usize(0),
                AssignmentOperation::Plus,
                VariableIndex::from_usize(2),
                false,
                vec![],
            )],
            OperatorCost::new(1),
        ),
        Operator::new(
            "increment-y".into(),
            vec![],
            vec![],
            vec![AssignmentEffect::new(
                VariableIndex::from_usize(1),
                AssignmentOperation::Plus,
                VariableIndex::from_usize(2),
                false,
                vec![],
            )],
            OperatorCost::new(1),
        ),
    ];
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 2,
        metric: Metric::new(true, None),
        variables,
        numeric_variables,
        goals: vec![
            ExplicitFact::propositional(0, 0),
            ExplicitFact::propositional(1, 0),
        ],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(2), ExplicitValueIndex::new(2)],
        numeric_state: vec![
            NumericValue::new(0.0),
            NumericValue::new(0.0),
            NumericValue::new(1.0),
            NumericValue::new(2.0),
            NumericValue::new(3.0),
        ],
        operators,
        axioms: vec![],
        comparison_axioms: vec![
            ComparisonAxiom::new(
                VariableIndex::from_usize(0),
                VariableIndex::from_usize(0),
                VariableIndex::from_usize(3),
                ComparisonOperator::GreaterThanOrEqual,
            ),
            ComparisonAxiom::new(
                VariableIndex::from_usize(1),
                VariableIndex::from_usize(1),
                VariableIndex::from_usize(4),
                ComparisonOperator::GreaterThanOrEqual,
            ),
        ],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    let abstractions =
        CartesianAbstractionCollectionGenerator::new(CartesianAbstractionCollectionConfig {
            abstraction: CartesianAbstractionConfig {
                max_states: 64,
                compute_source_sets: true,
                ..Default::default()
            },
            collection_strategy: CollectionStrategy::Complementary,
            variants_per_goal: 3,
            max_collection_states: 384,
            total_max_time: None,
            progressive_goal_roots: false,
            alternate_split_kinds: true,
            single_goal_tasks: true,
        })
        .unwrap()
        .generate(&task)
        .unwrap();

    assert_eq!(abstractions.len(), 6);
    let initial_h = abstractions
        .iter()
        .map(|abstraction| {
            assert_eq!(
                abstraction.abstract_source_sets.len(),
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
    assert_eq!(
        abstractions[0]
            .metadata
            .collection_member
            .map(|member| member.goal_id),
        Some(0)
    );
    assert_eq!(
        abstractions[0]
            .metadata
            .collection_member
            .map(|member| member.variant_id),
        Some(0)
    );
    assert_eq!(
        abstractions[2]
            .metadata
            .collection_member
            .map(|member| member.goal_id),
        Some(0)
    );
    assert_eq!(
        abstractions[2]
            .metadata
            .collection_member
            .map(|member| member.variant_id),
        Some(1)
    );
    assert_eq!(
        abstractions[3]
            .metadata
            .collection_member
            .map(|member| member.goal_id),
        Some(1)
    );
    assert_eq!(
        abstractions[3]
            .metadata
            .collection_member
            .map(|member| member.variant_id),
        Some(1)
    );
    assert_eq!(
        abstractions[4]
            .metadata
            .collection_member
            .map(|member| member.goal_id),
        Some(1)
    );
    assert_eq!(
        abstractions[4]
            .metadata
            .collection_member
            .map(|member| member.variant_id),
        Some(2)
    );
    assert_eq!(
        abstractions[5]
            .metadata
            .collection_member
            .map(|member| member.goal_id),
        Some(0)
    );
    assert_eq!(
        abstractions[5]
            .metadata
            .collection_member
            .map(|member| member.variant_id),
        Some(2)
    );
    assert_eq!(abstractions[0].metadata.split_kind, SplitKind::Concrete);
    assert_eq!(abstractions[0].metadata.split_selection_rank, Some(0));
    assert_eq!(abstractions[2].metadata.split_kind, SplitKind::Wanted);
    assert_eq!(abstractions[2].metadata.split_selection_rank, Some(0));
    assert_eq!(abstractions[4].metadata.split_kind, SplitKind::Concrete);
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
            alternate_split_kinds: true,
            single_goal_tasks: true,
        })
        .unwrap()
        .generate(&task)
        .unwrap();
    assert!(standard.iter().all(|abstraction| {
        abstraction.metadata.split_kind == SplitKind::Concrete
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
            alternate_split_kinds: true,
            single_goal_tasks: true,
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
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 2,
        metric: Metric::new(true, None),
        variables: vec![
            comparison_variable("x-at-least-two"),
            comparison_variable("x-at-least-four"),
        ],
        numeric_variables: vec![
            NumericVariable::new("x".into(), NumericType::Regular, None),
            NumericVariable::new("one".into(), NumericType::Constant, None),
            NumericVariable::new("two".into(), NumericType::Constant, None),
            NumericVariable::new("four".into(), NumericType::Constant, None),
        ],
        goals: vec![
            ExplicitFact::propositional(0, 0),
            ExplicitFact::propositional(1, 0),
        ],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(2), ExplicitValueIndex::new(2)],
        numeric_state: vec![
            NumericValue::new(0.0),
            NumericValue::new(1.0),
            NumericValue::new(2.0),
            NumericValue::new(4.0),
        ],
        operators: vec![Operator::new(
            "increment".into(),
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
        comparison_axioms: vec![
            ComparisonAxiom::new(
                VariableIndex::new(0),
                VariableIndex::new(0),
                VariableIndex::new(2),
                ComparisonOperator::GreaterThanOrEqual,
            ),
            ComparisonAxiom::new(
                VariableIndex::new(1),
                VariableIndex::new(0),
                VariableIndex::new(3),
                ComparisonOperator::GreaterThanOrEqual,
            ),
        ],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

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
            alternate_split_kinds: true,
            single_goal_tasks: true,
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
    assert_eq!(
        abstractions[2]
            .metadata
            .collection_member
            .map(|member| member.goal_id),
        Some(1)
    );
    assert_eq!(
        abstractions[2]
            .metadata
            .collection_member
            .map(|member| member.variant_id),
        Some(1)
    );
    assert!(
        !abstractions[2]
            .metadata
            .collection_member
            .is_some_and(|member| member.progressive_root)
    );
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
        let propositions = vec![
            ExplicitValueIndex::from_usize(usize::from(x < 2)),
            ExplicitValueIndex::from_usize(usize::from(x < 4)),
        ];
        let numeric = vec![
            NumericValue::new(x as f64),
            NumericValue::new(1.0),
            NumericValue::new(2.0),
            NumericValue::new(4.0),
        ];
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
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 2,
        metric: Metric::new(true, None),
        variables: vec![
            comparison_variable("x-at-least-four"),
            comparison_variable("x-at-least-two"),
        ],
        numeric_variables: vec![
            NumericVariable::new("x".into(), NumericType::Regular, None),
            NumericVariable::new("one".into(), NumericType::Constant, None),
            NumericVariable::new("four".into(), NumericType::Constant, None),
            NumericVariable::new("two".into(), NumericType::Constant, None),
        ],
        goals: vec![
            ExplicitFact::propositional(0, 0),
            ExplicitFact::propositional(1, 0),
        ],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(2), ExplicitValueIndex::new(2)],
        numeric_state: vec![
            NumericValue::new(0.0),
            NumericValue::new(1.0),
            NumericValue::new(4.0),
            NumericValue::new(2.0),
        ],
        operators: vec![Operator::new(
            "increment".into(),
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
        comparison_axioms: vec![
            ComparisonAxiom::new(
                VariableIndex::new(0),
                VariableIndex::new(0),
                VariableIndex::new(2),
                ComparisonOperator::GreaterThanOrEqual,
            ),
            ComparisonAxiom::new(
                VariableIndex::new(1),
                VariableIndex::new(0),
                VariableIndex::new(3),
                ComparisonOperator::GreaterThanOrEqual,
            ),
        ],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

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
            alternate_split_kinds: true,
            single_goal_tasks: true,
        })
        .unwrap()
        .generate(&task)
        .unwrap();

    assert_eq!(
        abstractions
            .iter()
            .map(|abstraction| abstraction
                .metadata
                .collection_member
                .map(|member| member.goal_id))
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
            .map(|abstraction| abstraction
                .metadata
                .collection_member
                .is_some_and(|member| member.progressive_root))
            .collect::<Vec<_>>(),
        vec![false, false],
        "a completed lane must not be progressed or retried again"
    );
}

#[test]
fn progressive_goal_roots_make_a_lane_terminal_after_a_dead_root() {
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 2,
        metric: Metric::new(true, None),
        variables: vec![
            comparison_variable("x-at-least-one"),
            comparison_variable("x-at-most-zero"),
        ],
        numeric_variables: vec![
            NumericVariable::new("x".into(), NumericType::Regular, None),
            NumericVariable::new("one".into(), NumericType::Constant, None),
            NumericVariable::new("zero".into(), NumericType::Constant, None),
        ],
        goals: vec![
            ExplicitFact::propositional(0, 0),
            ExplicitFact::propositional(1, 0),
        ],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(2), ExplicitValueIndex::new(2)],
        numeric_state: vec![
            NumericValue::new(0.0),
            NumericValue::new(1.0),
            NumericValue::new(0.0),
        ],
        operators: vec![Operator::new(
            "increment".into(),
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
        comparison_axioms: vec![
            ComparisonAxiom::new(
                VariableIndex::new(0),
                VariableIndex::new(0),
                VariableIndex::new(1),
                ComparisonOperator::GreaterThanOrEqual,
            ),
            ComparisonAxiom::new(
                VariableIndex::new(1),
                VariableIndex::new(0),
                VariableIndex::new(2),
                ComparisonOperator::LessThanOrEqual,
            ),
        ],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

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
            alternate_split_kinds: true,
            single_goal_tasks: true,
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
                abstraction
                    .metadata
                    .collection_member
                    .map(|member| member.goal_id),
                abstraction
                    .metadata
                    .collection_member
                    .map(|member| member.variant_id),
                abstraction
                    .metadata
                    .collection_member
                    .is_some_and(|member| member.progressive_root),
                abstraction.metadata.stop_reason,
                abstraction.metadata.concrete_plan_operator_ids.as_deref(),
            ))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        abstractions
            .iter()
            .map(|abstraction| abstraction
                .metadata
                .collection_member
                .map(|member| member.goal_id))
            .collect::<Vec<_>>(),
        vec![Some(0), Some(1)]
    );
    assert_eq!(
        abstractions[0]
            .metadata
            .concrete_plan_operator_ids
            .as_deref(),
        Some([OperatorIndex::new(0)].as_slice())
    );
    assert_eq!(
        abstractions[1]
            .metadata
            .concrete_plan_operator_ids
            .as_deref(),
        Some([].as_slice()),
        "the dead checkpoint must be rebuilt independently at the satisfying initial state"
    );
    assert!(
        !abstractions[1]
            .metadata
            .collection_member
            .is_some_and(|member| member.progressive_root)
    );
}

#[test]
fn progressive_goal_roots_retry_an_earlier_unsatisfied_goal_after_advancing() {
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 2,
        metric: Metric::new(true, None),
        variables: vec![
            comparison_variable("x-at-least-four"),
            comparison_variable("x-at-least-two"),
        ],
        numeric_variables: vec![
            NumericVariable::new("x".into(), NumericType::Regular, None),
            NumericVariable::new("one".into(), NumericType::Constant, None),
            NumericVariable::new("four".into(), NumericType::Constant, None),
            NumericVariable::new("two".into(), NumericType::Constant, None),
        ],
        goals: vec![
            ExplicitFact::propositional(0, 0),
            ExplicitFact::propositional(1, 0),
        ],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(2), ExplicitValueIndex::new(2)],
        numeric_state: vec![
            NumericValue::new(0.0),
            NumericValue::new(1.0),
            NumericValue::new(4.0),
            NumericValue::new(2.0),
        ],
        operators: vec![Operator::new(
            "increment".into(),
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
        comparison_axioms: vec![
            ComparisonAxiom::new(
                VariableIndex::new(0),
                VariableIndex::new(0),
                VariableIndex::new(2),
                ComparisonOperator::GreaterThanOrEqual,
            ),
            ComparisonAxiom::new(
                VariableIndex::new(1),
                VariableIndex::new(0),
                VariableIndex::new(3),
                ComparisonOperator::GreaterThanOrEqual,
            ),
        ],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

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
            alternate_split_kinds: true,
            single_goal_tasks: true,
        })
        .unwrap()
        .generate(&task)
        .unwrap();

    assert_eq!(abstractions.len(), 3);
    assert_eq!(
        abstractions
            .iter()
            .map(|abstraction| abstraction
                .metadata
                .collection_member
                .map(|member| member.goal_id))
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
            let propositions = vec![
                ExplicitValueIndex::from_usize(usize::from(x < 4)),
                ExplicitValueIndex::from_usize(usize::from(x < 2)),
            ];
            let numeric = vec![
                NumericValue::new(x as f64),
                NumericValue::new(1.0),
                NumericValue::new(4.0),
                NumericValue::new(2.0),
            ];
            let state_id = abstraction
                .abstract_state_id(&propositions, &numeric)
                .unwrap();
            let h = abstraction.distance_table.distances[state_id];
            let goal = abstraction
                .metadata
                .collection_member
                .map(|member| member.goal_id)
                .unwrap();
            let target: usize = if goal == 0 { 4 } else { 2 };
            assert!(
                h <= target.saturating_sub(x) as f64,
                "goal {goal} abstraction overestimates at x={x}: h={h}"
            );
        }
    }
}

#[test]
fn goal_collection_preserves_empty_goal_tasks() {
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 0,
        metric: Metric::new(true, None),
        variables: vec![ExplicitVariable::new(
            2,
            "value".into(),
            vec!["zero".into(), "one".into()],
            None,
            ExplicitValueIndex::new(0),
        )],
        numeric_variables: vec![],
        goals: vec![],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![],
        operators: vec![],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });
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
    assert_eq!(
        abstractions[0]
            .metadata
            .collection_member
            .map(|member| member.goal_id),
        None
    );
    assert_eq!(
        abstractions[0]
            .metadata
            .collection_member
            .map(|member| member.variant_id),
        None
    );
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
        vec![
            NumericValue::new(0.0),
            NumericValue::new(1.0),
            NumericValue::new(3.0),
        ],
        vec![Operator::new(
            "increment".into(),
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
        vec![ComparisonAxiom::new(
            VariableIndex::from_usize(0),
            VariableIndex::from_usize(0),
            VariableIndex::from_usize(2),
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
            alternate_split_kinds: true,
            single_goal_tasks: true,
        })
        .unwrap()
        .generate(&task)
        .unwrap();

    assert_eq!(abstractions.len(), 1);
    assert_eq!(
        abstractions[0].metadata.stop_reason,
        CartesianStopReason::TimeLimit
    );
    assert_eq!(
        abstractions[0]
            .metadata
            .collection_member
            .map(|member| member.goal_id),
        Some(0)
    );
    assert_eq!(
        abstractions[0]
            .metadata
            .collection_member
            .map(|member| member.variant_id),
        Some(0)
    );
}
