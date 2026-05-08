use super::*;
use crate::numeric::evaluation::domain_abstractions::max_domain_abstraction_heuristic::MaxDomainAbstractionHeuristic;
use crate::numeric::evaluation::evaluator::EvaluationState;
use crate::numeric::evaluation::heuristic::Heuristic;
use planners_sas::numeric::axioms::AxiomEvaluator;
use planners_sas::numeric::axioms::{ComparisonAxiom, ComparisonOperator};
use planners_sas::numeric::numeric_task::{
    AssignmentEffect, AssignmentOperation, Effect, ExplicitFact, ExplicitVariable, Metric,
    NumericRootTask, NumericType, NumericVariable, Operator,
};
use planners_sas::numeric::state_registry::StateRegistry;
use planners_sas::numeric::utils::int_packer::IntDoublePacker;

#[test]
fn single_init_split_selection_uses_round_robin_iteration_order() {
    let candidates = [0usize, 1, 2, 3, 4];
    let selected = (1..=8)
        .map(|iteration| select_single_init_split_var(&candidates, iteration).unwrap())
        .collect::<Vec<_>>();

    assert_eq!(selected, vec![1, 2, 3, 4, 0, 1, 2, 3]);
}

#[test]
fn single_init_split_selection_handles_empty_candidates() {
    assert_eq!(select_single_init_split_var(&[], 1), None);
}

#[test]
fn view_diverse_seed_splits_rotate_false_comparison_views_by_deficit() {
    let variables = vec![
        ExplicitVariable::new(
            3,
            "cmp_x".into(),
            vec!["true".into(), "false".into(), "unknown".into()],
            Some(0),
            2,
        ),
        ExplicitVariable::new(
            3,
            "cmp_y".into(),
            vec!["true".into(), "false".into(), "unknown".into()],
            Some(1),
            2,
        ),
    ];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("c10".into(), NumericType::Constant, None),
        NumericVariable::new("y".into(), NumericType::Regular, None),
        NumericVariable::new("c20".into(), NumericType::Constant, None),
    ];
    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        numeric_variables,
        vec![ExplicitFact::new(1, COMPARISON_TRUE_VALUE)],
        vec![],
        vec![0, 0],
        vec![0.0, 10.0, 5.0, 20.0],
        vec![Operator::new("noop".into(), vec![], vec![], vec![], 1)],
        vec![],
        vec![
            ComparisonAxiom::new(0, 0, 1, ComparisonOperator::GreaterThanOrEqual),
            ComparisonAxiom::new(1, 2, 3, ComparisonOperator::GreaterThanOrEqual),
        ],
        vec![],
        ExplicitFact::new(0, 0),
    );
    let config = DomainAbstractionCollectionGeneratorMultipleCegarConfig {
        portfolio_strategy: PortfolioStrategy::ViewDiverse,
        ..Default::default()
    };
    let generator = DomainAbstractionCollectionGeneratorMultipleCegar::new(config);

    let first = generator.initial_seed_splits(&task, 1);
    assert!(first.contains(&InitialSeedSplit::Numeric {
        numeric_var_id: 2,
        value: 5.0,
        include_in_lower: true,
    }));
    assert!(first.contains(&InitialSeedSplit::Propositional {
        var_id: 1,
        value: COMPARISON_TRUE_VALUE,
    }));

    let second = generator.initial_seed_splits(&task, 2);
    assert!(second.contains(&InitialSeedSplit::Numeric {
        numeric_var_id: 2,
        value: 5.0,
        include_in_lower: true,
    }));
    assert!(second.contains(&InitialSeedSplit::Propositional {
        var_id: 1,
        value: COMPARISON_TRUE_VALUE,
    }));

    let third = generator.initial_seed_splits(&task, 3);
    assert!(third.contains(&InitialSeedSplit::Numeric {
        numeric_var_id: 0,
        value: 0.0,
        include_in_lower: true,
    }));
    assert!(third.contains(&InitialSeedSplit::Propositional {
        var_id: 0,
        value: COMPARISON_TRUE_VALUE,
    }));
    assert_eq!(
        generator.flaw_kind_for_iteration(1),
        FlawKind::SequenceProgression
    );
    assert_eq!(
        generator.flaw_kind_for_iteration(2),
        FlawKind::SequenceRegression
    );
}

#[test]
fn complementary_seed_splits_rotate_false_views_with_route_shells() {
    let variables = vec![
        ExplicitVariable::new(
            3,
            "cmp_x".into(),
            vec!["true".into(), "false".into(), "unknown".into()],
            Some(0),
            2,
        ),
        ExplicitVariable::new(
            3,
            "cmp_y".into(),
            vec!["true".into(), "false".into(), "unknown".into()],
            Some(1),
            2,
        ),
    ];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("c10".into(), NumericType::Constant, None),
        NumericVariable::new("y".into(), NumericType::Regular, None),
        NumericVariable::new("c20".into(), NumericType::Constant, None),
        NumericVariable::new("delta3".into(), NumericType::Constant, None),
    ];
    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        numeric_variables,
        vec![ExplicitFact::new(1, COMPARISON_TRUE_VALUE)],
        vec![],
        vec![0, 0],
        vec![0.0, 10.0, 5.0, 20.0, 3.0],
        vec![Operator::new(
            "inc_y".into(),
            vec![],
            vec![],
            vec![AssignmentEffect::new(
                2,
                AssignmentOperation::Plus,
                4,
                false,
                vec![],
            )],
            1,
        )],
        vec![],
        vec![
            ComparisonAxiom::new(0, 0, 1, ComparisonOperator::GreaterThanOrEqual),
            ComparisonAxiom::new(1, 2, 3, ComparisonOperator::GreaterThanOrEqual),
        ],
        vec![],
        ExplicitFact::new(0, 0),
    );
    let config = DomainAbstractionCollectionGeneratorMultipleCegarConfig {
        portfolio_strategy: PortfolioStrategy::Complementary,
        ..Default::default()
    };
    let generator = DomainAbstractionCollectionGeneratorMultipleCegar::new(config);

    let first = generator.initial_seed_splits(&task, 1);
    assert!(first.contains(&InitialSeedSplit::Numeric {
        numeric_var_id: 2,
        value: 5.0,
        include_in_lower: true,
    }));
    assert!(first.contains(&InitialSeedSplit::Propositional {
        var_id: 1,
        value: COMPARISON_TRUE_VALUE,
    }));
    for value in [20.0, 17.0, 14.0, 11.0, 8.0] {
        assert!(
            first.contains(&InitialSeedSplit::Numeric {
                numeric_var_id: 2,
                value,
                include_in_lower: false,
            }),
            "missing complementary route shell at {value}: {first:?}"
        );
    }

    let second = generator.initial_seed_splits(&task, 2);
    assert!(second.contains(&InitialSeedSplit::Numeric {
        numeric_var_id: 0,
        value: 0.0,
        include_in_lower: true,
    }));
    assert!(second.contains(&InitialSeedSplit::Propositional {
        var_id: 0,
        value: COMPARISON_TRUE_VALUE,
    }));
    assert_eq!(
        generator.flaw_kind_for_iteration(1),
        DomainAbstractionCollectionGeneratorMultipleCegarConfig::default().flaw_kind
    );
    assert_eq!(
        generator.flaw_kind_for_iteration(2),
        FlawKind::SequenceRegression
    );
}

#[test]
fn complementary_seed_splits_group_alternative_goal_achiever_views() {
    let variables = vec![
        ExplicitVariable::new(
            3,
            "cmp_a".into(),
            vec!["true".into(), "false".into(), "unknown".into()],
            Some(0),
            2,
        ),
        ExplicitVariable::new(
            3,
            "cmp_b".into(),
            vec!["true".into(), "false".into(), "unknown".into()],
            Some(1),
            2,
        ),
        ExplicitVariable::new(2, "goal".into(), vec!["no".into(), "yes".into()], None, 0),
    ];
    let numeric_variables = vec![
        NumericVariable::new("a".into(), NumericType::Regular, None),
        NumericVariable::new("b".into(), NumericType::Regular, None),
        NumericVariable::new("c10".into(), NumericType::Constant, None),
        NumericVariable::new("delta".into(), NumericType::Constant, None),
    ];
    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        numeric_variables,
        vec![ExplicitFact::new(2, 1)],
        vec![],
        vec![0, 0, 0],
        vec![0.0, 1.0, 10.0, 1.0],
        vec![
            Operator::new(
                "achieve_a".into(),
                vec![ExplicitFact::new(0, COMPARISON_TRUE_VALUE)],
                vec![Effect::new(vec![], 2, Some(0), 1)],
                vec![AssignmentEffect::new(
                    0,
                    AssignmentOperation::Plus,
                    3,
                    false,
                    vec![],
                )],
                1,
            ),
            Operator::new(
                "achieve_b".into(),
                vec![ExplicitFact::new(1, COMPARISON_TRUE_VALUE)],
                vec![Effect::new(vec![], 2, Some(0), 1)],
                vec![AssignmentEffect::new(
                    1,
                    AssignmentOperation::Plus,
                    3,
                    false,
                    vec![],
                )],
                1,
            ),
        ],
        vec![],
        vec![
            ComparisonAxiom::new(0, 0, 2, ComparisonOperator::GreaterThanOrEqual),
            ComparisonAxiom::new(1, 1, 2, ComparisonOperator::GreaterThanOrEqual),
        ],
        vec![],
        ExplicitFact::new(0, 0),
    );
    let config = DomainAbstractionCollectionGeneratorMultipleCegarConfig {
        portfolio_strategy: PortfolioStrategy::Complementary,
        ..Default::default()
    };
    let generator = DomainAbstractionCollectionGeneratorMultipleCegar::new(config);

    let seeds = generator.initial_seed_splits(&task, 1);
    assert!(seeds.contains(&InitialSeedSplit::Propositional {
        var_id: 0,
        value: COMPARISON_TRUE_VALUE,
    }));
    assert!(seeds.contains(&InitialSeedSplit::Propositional {
        var_id: 1,
        value: COMPARISON_TRUE_VALUE,
    }));
    assert!(seeds.contains(&InitialSeedSplit::Numeric {
        numeric_var_id: 0,
        value: 10.0,
        include_in_lower: false,
    }));
    assert!(seeds.contains(&InitialSeedSplit::Numeric {
        numeric_var_id: 1,
        value: 10.0,
        include_in_lower: false,
    }));
}

#[test]
fn route_shell_seed_splits_create_separate_backward_windows() {
    let variables = vec![
        ExplicitVariable::new(
            3,
            "cmp_x".into(),
            vec!["true".into(), "false".into(), "unknown".into()],
            Some(0),
            2,
        ),
        ExplicitVariable::new(2, "saved".into(), vec!["no".into(), "yes".into()], None, 0),
    ];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("target".into(), NumericType::Constant, None),
        NumericVariable::new("delta".into(), NumericType::Constant, None),
    ];
    let task = NumericRootTask::new(
        3,
        Metric::new(true, None),
        variables,
        numeric_variables,
        vec![ExplicitFact::new(1, 1)],
        vec![],
        vec![0, 0],
        vec![0.0, 10.0, 1.0],
        vec![
            Operator::new(
                "inc".into(),
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
                "save".into(),
                vec![ExplicitFact::new(0, COMPARISON_TRUE_VALUE)],
                vec![Effect::new(vec![], 1, Some(0), 1)],
                vec![],
                1,
            ),
        ],
        vec![],
        vec![ComparisonAxiom::new(
            0,
            0,
            1,
            ComparisonOperator::GreaterThanOrEqual,
        )],
        vec![],
        ExplicitFact::new(0, 0),
    );
    let config = DomainAbstractionCollectionGeneratorMultipleCegarConfig {
        portfolio_strategy: PortfolioStrategy::RouteShells,
        ..Default::default()
    };
    let generator = DomainAbstractionCollectionGeneratorMultipleCegar::new(config);

    let shell_0 = generator.initial_seed_splits(&task, 1);
    let shell_1 = generator.initial_seed_splits(&task, 2);

    for value in [6.0, 7.0, 8.0, 9.0, 10.0] {
        assert!(
            shell_0.contains(&InitialSeedSplit::Numeric {
                numeric_var_id: 0,
                value,
                include_in_lower: false,
            }),
            "missing shell 0 split {value}: {shell_0:?}"
        );
    }
    for value in [2.0, 3.0, 4.0, 5.0, 6.0, 10.0] {
        assert!(
            shell_1.contains(&InitialSeedSplit::Numeric {
                numeric_var_id: 0,
                value,
                include_in_lower: false,
            }),
            "missing shell 1 split {value}: {shell_1:?}"
        );
    }
    assert!(shell_0.contains(&InitialSeedSplit::Propositional {
        var_id: 1,
        value: 1,
    }));
    assert!(shell_1.contains(&InitialSeedSplit::Propositional {
        var_id: 1,
        value: 1,
    }));
}

#[test]
fn complementary_seed_splits_include_bounded_propositional_achiever_chain() {
    let variables = vec![
        ExplicitVariable::new(
            2,
            "at_item_target".into(),
            vec!["false".into(), "true".into()],
            None,
            0,
        ),
        ExplicitVariable::new(
            2,
            "in_arm".into(),
            vec!["false".into(), "true".into()],
            None,
            0,
        ),
        ExplicitVariable::new(
            2,
            "at_bot_target".into(),
            vec!["false".into(), "true".into()],
            None,
            0,
        ),
        ExplicitVariable::new(
            2,
            "at_item_source".into(),
            vec!["false".into(), "true".into()],
            None,
            1,
        ),
        ExplicitVariable::new(
            2,
            "at_bot_source".into(),
            vec!["false".into(), "true".into()],
            None,
            1,
        ),
        ExplicitVariable::new(
            2,
            "free_arm".into(),
            vec!["false".into(), "true".into()],
            None,
            1,
        ),
    ];
    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        vec![],
        vec![ExplicitFact::new(0, 1)],
        vec![],
        vec![0, 0, 0, 1, 1, 1],
        vec![],
        vec![
            Operator::new(
                "drop".into(),
                vec![ExplicitFact::new(1, 1), ExplicitFact::new(2, 1)],
                vec![Effect::new(vec![], 0, Some(0), 1)],
                vec![],
                1,
            ),
            Operator::new(
                "pick".into(),
                vec![
                    ExplicitFact::new(3, 1),
                    ExplicitFact::new(4, 1),
                    ExplicitFact::new(5, 1),
                ],
                vec![Effect::new(vec![], 1, Some(0), 1)],
                vec![],
                1,
            ),
            Operator::new(
                "move".into(),
                vec![ExplicitFact::new(4, 1)],
                vec![Effect::new(vec![], 2, Some(0), 1)],
                vec![],
                1,
            ),
        ],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    );
    let config = DomainAbstractionCollectionGeneratorMultipleCegarConfig {
        portfolio_strategy: PortfolioStrategy::Complementary,
        ..Default::default()
    };
    let generator = DomainAbstractionCollectionGeneratorMultipleCegar::new(config);

    let seeds = generator.initial_seed_splits(&task, 1);
    for (var_id, value) in [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)] {
        assert!(
            seeds.contains(&InitialSeedSplit::Propositional { var_id, value }),
            "missing propositional achiever seed p{var_id}={value}: {seeds:?}"
        );
    }
}

#[test]
fn region_landmarks_seed_splits_build_goal_achiever_shells() {
    let variables = vec![
        ExplicitVariable::new(
            3,
            "cmp".into(),
            vec!["true".into(), "false".into(), "unknown".into()],
            Some(0),
            2,
        ),
        ExplicitVariable::new(2, "goal".into(), vec!["no".into(), "yes".into()], None, 0),
    ];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("delta".into(), NumericType::Constant, None),
        NumericVariable::new("threshold".into(), NumericType::Constant, None),
    ];
    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        numeric_variables,
        vec![ExplicitFact::new(1, 1)],
        vec![],
        vec![1, 0],
        vec![0.0, 3.0, 10.0],
        vec![
            Operator::new(
                "inc".into(),
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
            ),
            Operator::new(
                "achieve".into(),
                vec![ExplicitFact::new(0, COMPARISON_TRUE_VALUE)],
                vec![Effect::new(vec![], 1, Some(0), 1)],
                vec![],
                1,
            ),
        ],
        vec![],
        vec![ComparisonAxiom::new(
            0,
            0,
            2,
            ComparisonOperator::GreaterThanOrEqual,
        )],
        vec![],
        ExplicitFact::new(0, 0),
    );
    let config = DomainAbstractionCollectionGeneratorMultipleCegarConfig {
        portfolio_strategy: PortfolioStrategy::RegionLandmarks,
        ..Default::default()
    };
    let generator = DomainAbstractionCollectionGeneratorMultipleCegar::new(config);

    let seeds = generator.initial_seed_splits(&task, 1);
    assert!(seeds.contains(&InitialSeedSplit::Propositional {
        var_id: 1,
        value: 1,
    }));
    assert!(seeds.contains(&InitialSeedSplit::Propositional {
        var_id: 0,
        value: COMPARISON_TRUE_VALUE,
    }));
    for value in [10.0, 7.0, 4.0, 1.0] {
        assert!(
            seeds.contains(&InitialSeedSplit::Numeric {
                numeric_var_id: 0,
                value,
                include_in_lower: false,
            }),
            "missing shell split at {value}: {seeds:?}"
        );
    }
    assert_eq!(
        generator.flaw_kind_for_iteration(1),
        FlawKind::SequenceBidirectional
    );
}

#[test]
fn region_landmarks_caps_single_abstraction_budget() {
    let variables = vec![
        ExplicitVariable::new(
            3,
            "cmp".into(),
            vec!["true".into(), "false".into(), "unknown".into()],
            Some(0),
            2,
        ),
        ExplicitVariable::new(2, "goal".into(), vec!["no".into(), "yes".into()], None, 0),
    ];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("delta".into(), NumericType::Constant, None),
        NumericVariable::new("threshold".into(), NumericType::Constant, None),
    ];
    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        numeric_variables,
        vec![ExplicitFact::new(1, 1)],
        vec![],
        vec![1, 0],
        vec![0.0, 1.0, 10.0],
        vec![
            Operator::new(
                "inc".into(),
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
            ),
            Operator::new(
                "achieve".into(),
                vec![ExplicitFact::new(0, COMPARISON_TRUE_VALUE)],
                vec![Effect::new(vec![], 1, Some(0), 1)],
                vec![],
                1,
            ),
        ],
        vec![],
        vec![ComparisonAxiom::new(
            0,
            0,
            2,
            ComparisonOperator::GreaterThanOrEqual,
        )],
        vec![],
        ExplicitFact::new(0, 0),
    );
    let config = DomainAbstractionCollectionGeneratorMultipleCegarConfig {
        portfolio_strategy: PortfolioStrategy::RegionLandmarks,
        max_abstraction_size: 10_000,
        max_collection_size: 90,
        random_seed: Some(1),
        ..Default::default()
    };
    let generator = DomainAbstractionCollectionGeneratorMultipleCegar::new(config);
    let abstractions = generator.generate_collection(&task).unwrap();

    assert!(!abstractions.is_empty());
    assert!(
        abstractions.iter().all(|abstraction| {
            abstraction
                .metadata
                .max_abstraction_size
                .is_some_and(|budget| budget <= 30)
        }),
        "region landmarks should cap each abstraction at one third of collection budget"
    );
}

#[test]
#[ignore = "requires the local pfile2.sas regression input"]
fn pfile2_multi_domain_abstractions_initial_heuristic_is_finite() {
    let task_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../pfile2.sas");
    assert!(
        task_path.exists(),
        "expected local regression input at {}",
        task_path.display()
    );
    let task = NumericRootTask::from_file(&task_path);

    let config = DomainAbstractionCollectionGeneratorMultipleCegarConfig {
        max_abstraction_size: 100_000,
        max_collection_size: 1_000_000,
        total_max_time: 150.0,
        debug: true,
        random_seed: Some(0),
        ..Default::default()
    };
    let generator = DomainAbstractionCollectionGeneratorMultipleCegar::new(config);
    let abstractions = generator.generate_collection(&task).unwrap();

    let heuristic = MaxDomainAbstractionHeuristic::new(None, abstractions);
    let packer = IntDoublePacker::from_task(&task);
    let axiom_evaluator = AxiomEvaluator::new(&task, &packer);
    let mut registry = StateRegistry::new(&task, &packer, &axiom_evaluator);
    let initial_state = registry.get_initial_state();
    let eval_state =
        EvaluationState::new_with_registry(&initial_state, 0.0, false, &task, &registry);
    let initial_h = heuristic.compute_heuristic(&eval_state).unwrap();

    assert!(
        initial_h.is_finite(),
        "multi_domain_abstractions initial heuristic should be finite for pfile2"
    );
}

#[test]
#[ignore = "requires the local pfile2.sas regression input"]
fn pfile2_collection_inf_abstraction_reduced_case() {
    let task_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../pfile2.sas");
    assert!(
        task_path.exists(),
        "expected local regression input at {}",
        task_path.display()
    );
    let task = NumericRootTask::from_file(&task_path);
    let sas = std::fs::read_to_string(&task_path).unwrap();
    assert!(sas.contains(
        "begin_variable\nvar9\n-1\n2\nAtom clear(pallet2)\nNegatedAtom clear(pallet2)\nend_variable"
    ));
    assert_eq!(task.get_variable_name(13).unwrap(), "var9");
    assert_eq!(task.get_variable_name(0).unwrap(), "var20");

    let goal = ExplicitFact::new(25, 10);
    let single_goal_task = SingleGoalTask::new(&task, goal.clone());
    let config = CegarConfig {
        max_abstraction_size: 100_000,
        debug: true,
        random_seed: Some(11_890_779_981_456_599_205),
        init_split_method: InitSplitMethod::InitValue,
        init_split_var_ids: Some(HashSet::from([13])),
        ..Default::default()
    };

    let outcome =
        crate::numeric::evaluation::domain_abstractions::cegar::Cegar::new(config.clone())
            .unwrap()
            .build_abstraction(&single_goal_task)
            .unwrap();
    assert!(
        outcome.last_step.wildcard_plan.is_some(),
        "the reduced pfile2 abstraction should have an abstract plan"
    );
    let distance_table = outcome
        .final_state
        .factory
        .build_abstract_distance_table(&single_goal_task, config.combine_labels, false)
        .unwrap();
    let initial_h = distance_table.distances[distance_table.initial_state_hash];
    assert!(
        initial_h.is_finite(),
        "reduced pfile2 collection abstraction should not make the initial abstract state a dead end"
    );
}
