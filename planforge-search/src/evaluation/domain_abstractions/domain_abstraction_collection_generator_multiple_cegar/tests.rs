use super::*;
use planforge_sas::axioms::CalOperator;
use planforge_sas::numeric_task::{AssignmentEffect, AssignmentOperation, Metric, NumericRootTask};

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
fn standard_uses_configured_full_goal_flaw_kind() {
    let config = DomainAbstractionCollectionGeneratorMultipleCegarConfig {
        portfolio_strategy: PortfolioStrategy::Standard,
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
fn numeric_seed_shells_are_interleaved_across_dimensions() {
    let numeric = |numeric_var_id, value| InitialSeedSplit::Numeric {
        numeric_var_id,
        value,
        include_in_lower: true,
    };
    let mut seeds = vec![InitialSeedSplit::Propositional {
        var_id: 3,
        value: 1,
    }];

    append_interleaved_numeric_seeds(
        &mut seeds,
        vec![
            vec![numeric(9, 0.0), numeric(9, 1.0)],
            vec![numeric(2, 0.0), numeric(2, 1.0), numeric(2, 2.0)],
        ],
    );

    assert_eq!(
        seeds,
        vec![
            InitialSeedSplit::Propositional {
                var_id: 3,
                value: 1,
            },
            numeric(2, 0.0),
            numeric(9, 0.0),
            numeric(2, 1.0),
            numeric(9, 1.0),
            numeric(2, 2.0),
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
                0,
                AssignmentOperation::Plus,
                7,
                false,
                vec![],
            )],
            1,
        ),
        Operator::new(
            "change-b".into(),
            vec![],
            vec![],
            vec![AssignmentEffect::new(
                1,
                AssignmentOperation::Plus,
                7,
                false,
                vec![],
            )],
            1,
        ),
    ];
    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        vec![],
        numeric_variables,
        vec![],
        vec![],
        vec![],
        vec![0.0, 0.0, 10.0, 20.0, 0.0, 0.0, 0.0, 1.0],
        operators,
        vec![],
        vec![],
        vec![
            AssignmentAxiom::new(4, CalOperator::Difference, 0, 2),
            AssignmentAxiom::new(5, CalOperator::Difference, 1, 2),
            AssignmentAxiom::new(6, CalOperator::Difference, 0, 3),
        ],
        ExplicitFact::new(0, 0),
    );

    let first = numeric_root_group_key(&task, &task, 4).unwrap();
    let second = numeric_root_group_key(&task, &task, 5).unwrap();
    let independent = numeric_root_group_key(&task, &task, 6).unwrap();

    assert_eq!(first, second);
    assert_ne!(first, independent);
}
