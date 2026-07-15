use crate::numeric::evaluation::domain_abstractions::cegar::flaw_search::{
    Flaw, NumericFlaw, PropFlaw, get_flaws,
};

use super::*;
use rand::{SeedableRng, rngs::SmallRng};

use planforge_sas::numeric::axioms::PropositionalAxiom;
use planforge_sas::numeric::axioms::{ComparisonAxiom, ComparisonOperator};

use planforge_sas::numeric::numeric_task::{
    AssignmentEffect, AssignmentOperation, Effect, ExplicitFact, ExplicitVariable, Metric,
    NumericRootTask, NumericType, NumericVariable, Operator,
};

fn one_dimensional_sailing_like_task() -> NumericRootTask {
    let variables = vec![
        ExplicitVariable::new(
            3,
            "x_gt_9".into(),
            vec!["true".into(), "false".into(), "unknown".into()],
            Some(0),
            2,
        ),
        ExplicitVariable::new(
            2,
            "saved".into(),
            vec!["false".into(), "true".into()],
            None,
            0,
        ),
    ];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("one".into(), NumericType::Constant, None),
        NumericVariable::new("nine".into(), NumericType::Constant, None),
    ];
    let go_east = Operator::new(
        "go_east".into(),
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
    let save = Operator::new(
        "save".into(),
        vec![ExplicitFact::new(0, 0)],
        vec![Effect::new(vec![], 1, Some(0), 1)],
        vec![],
        1,
    );

    NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        numeric_variables,
        vec![ExplicitFact::new(1, 1)],
        vec![],
        vec![2, 0],
        vec![0.0, 1.0, 9.0],
        vec![go_east, save],
        vec![],
        vec![ComparisonAxiom::new(
            0,
            0,
            2,
            ComparisonOperator::GreaterThan,
        )],
        vec![],
        ExplicitFact::new(0, 0),
    )
}

#[test]
fn build_abstraction_produces_singleton_plan_without_wildcards() {
    let variables = vec![ExplicitVariable::new(
        2,
        "v".into(),
        vec!["v0".into(), "v1".into()],
        None,
        0,
    )];
    let numeric_variables: Vec<NumericVariable> = vec![];
    let goals = vec![ExplicitFact::new(0, 1)];
    let op0 = Operator::new(
        "set0".into(),
        vec![ExplicitFact::new(0, 0)],
        vec![planforge_sas::numeric::numeric_task::Effect::new(
            vec![],
            0,
            Some(0),
            1,
        )],
        vec![],
        1,
    );
    let op1 = Operator::new(
        "set1".into(),
        vec![ExplicitFact::new(0, 0)],
        vec![planforge_sas::numeric::numeric_task::Effect::new(
            vec![],
            0,
            Some(0),
            1,
        )],
        vec![],
        1,
    );
    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        numeric_variables,
        goals,
        vec![],
        vec![0],
        vec![],
        vec![op0, op1],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    );

    let config = CegarConfig {
        use_wildcard_plans: false,
        max_iterations: 2,
        random_seed: Some(1),
        ..Default::default()
    };

    let cegar = Cegar::new(config).unwrap();
    let outcome = cegar.build_abstraction(&task).unwrap();
    let plan = outcome.last_step.wildcard_plan.expect("plan exists");
    assert_eq!(plan.wildcard_plan.len(), 1);
    assert!(matches!(plan.wildcard_plan[0].as_slice(), [0] | [1]));
}

#[test]
fn empty_wildcard_plan_is_real_exactly_when_initial_state_is_goal() {
    let variables = vec![ExplicitVariable::new(
        2,
        "v".into(),
        vec!["v0".into(), "v1".into()],
        None,
        0,
    )];
    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables.clone(),
        vec![],
        vec![ExplicitFact::new(0, 0)],
        vec![],
        vec![0],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    );
    let empty_plan = WildcardPlanResult {
        wildcard_plan: vec![],
        abstract_state_hashes: vec![0],
        abstract_prop_states: vec![vec![0]],
        abstract_numeric_states: vec![vec![]],
    };
    assert!(wildcard_plan_is_real(&task, &empty_plan).unwrap());

    let non_goal_task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        vec![],
        vec![ExplicitFact::new(0, 1)],
        vec![],
        vec![0],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 1),
    );
    assert!(!wildcard_plan_is_real(&non_goal_task, &empty_plan).unwrap());
}

#[test]
fn get_flaws_reports_numeric_deviation_flaw() {
    use crate::numeric::evaluation::domain_abstractions::comparison_expression::Interval;
    use planforge_sas::numeric::axioms::{ComparisonAxiom, ComparisonOperator};
    use planforge_sas::numeric::numeric_task::{
        AssignmentEffect, AssignmentOperation, NumericType,
    };

    let variables = vec![
        ExplicitVariable::new(
            3,
            "gt".into(),
            vec!["true".into(), "false".into(), "unknown".into()],
            Some(0),
            2,
        ),
        ExplicitVariable::new(2, "g".into(), vec!["g0".into(), "g1".into()], None, 0),
    ];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("c".into(), NumericType::Constant, None),
        NumericVariable::new("thresh".into(), NumericType::Constant, None),
    ];
    let comparison_axioms = vec![ComparisonAxiom::new(
        0,
        0,
        2,
        ComparisonOperator::GreaterThan,
    )];
    let op0 = Operator::new(
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
    );
    let op1 = Operator::new(
        "set_g".into(),
        vec![ExplicitFact::new(0, 0)],
        vec![Effect::new(vec![], 1, Some(0), 1)],
        vec![],
        1,
    );
    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        numeric_variables,
        vec![ExplicitFact::new(1, 1)],
        vec![],
        vec![2, 0],
        vec![-10.0, 3.0, -5.0],
        vec![op0, op1],
        vec![],
        comparison_axioms,
        vec![],
        ExplicitFact::new(0, 0),
    );

    let partitions = NumericPartitions::with_partitions(vec![
        vec![
            Interval::new(f64::NEG_INFINITY, -5.0, false, true),
            Interval::new(-5.0, f64::INFINITY, false, false),
        ],
        vec![Interval::singleton(3.0)],
        vec![Interval::singleton(-5.0)],
    ]);
    let plan = WildcardPlanResult {
        wildcard_plan: vec![vec![0]],
        abstract_state_hashes: vec![],
        abstract_prop_states: vec![],
        abstract_numeric_states: vec![vec![0, 0, 0], vec![1, 0, 0]],
    };
    let domain_mapping = vec![vec![0, 1], vec![0, 1]];

    let flaws = get_flaws(
        &task,
        &partitions,
        &domain_mapping,
        &plan,
        FlawKind::Progression,
    )
    .unwrap();

    assert!(
        flaws.iter().any(|flaw| matches!(flaw, Flaw::Numeric(_))),
        "expected a numeric deviation flaw"
    );
}

#[test]
fn cegar_default_config_matches_current_port_defaults() {
    let config = CegarConfig::default();

    assert_eq!(config.max_abstraction_size, usize::MAX);
    assert_eq!(config.max_iterations, 10_000);
    assert!(config.use_wildcard_plans);
    assert_eq!(config.random_seed, None);
    assert_eq!(
        config.flaw_treatment,
        FlawTreatmentVariants::RandomSingleAtom
    );
    assert_eq!(config.init_split_method, InitSplitMethod::InitValue);
}

#[test]
fn seeded_shuffle_indices_is_not_identity() {
    let mut indices = vec![0, 1, 2, 3, 4, 5];
    let original = indices.clone();
    let mut rng = SmallRng::seed_from_u64(7);

    shuffle_indices_with_rng(&mut indices, &mut rng);

    assert_eq!(indices.len(), original.len());
    let mut sorted = indices.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, original);
    assert_ne!(indices, original);
}

#[test]
fn fix_flaws_respects_max_abstraction_size_limit() {
    let variables = vec![ExplicitVariable::new(
        2,
        "v".into(),
        vec!["v0".into(), "v1".into()],
        None,
        0,
    )];
    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        vec![],
        vec![ExplicitFact::new(0, 1)],
        vec![],
        vec![0],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    );

    let config = CegarConfig {
        max_abstraction_size: 1,
        ..Default::default()
    };

    let mut domain_mapping = vec![vec![0, 0]];
    let mut domain_sizes = vec![1];
    let mut partitions = NumericPartitions::trivial(&task);
    let mut numeric_domain_sizes = vec![];
    let mut rng = SmallRng::seed_from_u64(7);
    let mut blacklisted_prop_var_ids = HashSet::new();
    let mut blacklisted_numeric_var_ids = HashSet::new();
    let flaws = vec![Flaw::Propositional(PropFlaw {
        fact: ExplicitFact::new(0, 1),
        dependent_numeric_flaws: vec![],
        step: 0,
    })];

    let refined = fix_flaws(
        &config,
        &task,
        &flaws,
        &mut domain_mapping,
        &mut domain_sizes,
        &mut partitions,
        &mut numeric_domain_sizes,
        &mut rng,
        &mut blacklisted_prop_var_ids,
        &mut blacklisted_numeric_var_ids,
        0,
    )
    .unwrap();

    assert!(refined.is_empty());
    assert_eq!(domain_sizes, vec![1]);
    assert_eq!(domain_mapping, vec![vec![0, 0]]);
    assert!(blacklisted_prop_var_ids.contains(&0));
}
#[test]
fn blacklisted_propositional_vars_are_not_refined() {
    let variables = vec![ExplicitVariable::new(
        2,
        "v".into(),
        vec!["v0".into(), "v1".into()],
        None,
        0,
    )];
    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        vec![],
        vec![ExplicitFact::new(0, 1)],
        vec![],
        vec![0],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    );

    let mut config = CegarConfig::default();
    config.blacklisted_prop_var_ids.insert(0);
    let cegar = Cegar::new(config).unwrap();

    let mut domain_mapping = vec![vec![0, 0]];
    let mut domain_sizes = vec![1];
    let mut partitions = NumericPartitions::trivial(&task);
    let mut numeric_domain_sizes = vec![];
    let mut rng = SmallRng::seed_from_u64(7);
    let mut blacklisted_prop_var_ids = HashSet::from([0usize]);
    let mut blacklisted_numeric_var_ids = HashSet::new();
    let flaws = vec![Flaw::Propositional(PropFlaw {
        fact: ExplicitFact::new(0, 1),
        dependent_numeric_flaws: vec![],
        step: 0,
    })];

    let refined = fix_flaws(
        &cegar.config,
        &task,
        &flaws,
        &mut domain_mapping,
        &mut domain_sizes,
        &mut partitions,
        &mut numeric_domain_sizes,
        &mut rng,
        &mut blacklisted_prop_var_ids,
        &mut blacklisted_numeric_var_ids,
        0,
    )
    .unwrap();

    assert!(refined.is_empty());
    assert_eq!(domain_sizes, vec![1]);
}

#[test]
fn init_value_split_uses_true_branch_for_comparison_variables() {
    let variables = vec![ExplicitVariable::new(
        3,
        "cmp".into(),
        vec!["true".into(), "false".into(), "unknown".into()],
        Some(0),
        2,
    )];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("y".into(), NumericType::Regular, None),
    ];
    let comparison_axioms = vec![ComparisonAxiom::new(
        0,
        0,
        1,
        ComparisonOperator::GreaterThan,
    )];
    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        numeric_variables,
        vec![ExplicitFact::new(0, 0)],
        vec![],
        vec![2],
        vec![1.0, 0.0],
        vec![],
        vec![],
        comparison_axioms,
        vec![],
        ExplicitFact::new(0, 0),
    );

    let mut config = CegarConfig {
        init_split_method: InitSplitMethod::InitValue,
        ..Default::default()
    };
    config.init_split_method = InitSplitMethod::InitValue;
    let mut rng = SmallRng::seed_from_u64(7);

    let (new_domain_size, mapping) =
        compute_initial_split_mapping(&task, &config, 0, Some(0), &mut rng).unwrap();

    assert_eq!(new_domain_size, 2);
    assert_eq!(mapping, vec![1, 0, 0]);
}

#[test]
fn goal_variable_values_expand_goal_axiom_preconditions() {
    let variables = vec![
        ExplicitVariable::new(2, "need_a".into(), vec!["f".into(), "t".into()], None, 0),
        ExplicitVariable::new(2, "need_b".into(), vec!["f".into(), "t".into()], None, 0),
        ExplicitVariable::new(
            2,
            "goal_flag".into(),
            vec!["off".into(), "on".into()],
            None,
            0,
        ),
    ];
    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        vec![],
        vec![ExplicitFact::new(2, 1)],
        vec![],
        vec![0, 0, 0],
        vec![],
        vec![],
        vec![PropositionalAxiom::new(
            vec![ExplicitFact::new(0, 1), ExplicitFact::new(1, 1)],
            2,
            0,
            1,
        )],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    );

    assert_eq!(
        goal_variable_values(&task),
        vec![ExplicitFact::new(0, 1), ExplicitFact::new(1, 1)]
    );
}

#[test]
fn numeric_init_split_is_applied_for_encoded_init_split_var() {
    let variables = vec![ExplicitVariable::new(
        2,
        "g".into(),
        vec!["g0".into(), "g1".into()],
        None,
        0,
    )];
    let numeric_variables = vec![NumericVariable::new("x".into(), NumericType::Regular, None)];
    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        numeric_variables,
        vec![ExplicitFact::new(0, 1)],
        vec![],
        vec![0],
        vec![3.5],
        vec![],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    );

    let config = CegarConfig {
        init_split_method: InitSplitMethod::Identity,
        init_split_var_ids: Some(HashSet::from([1usize])),
        ..Default::default()
    };

    let mut rng = SmallRng::seed_from_u64(7);
    let mut domain_mapping = vec![vec![0, 0]];
    let mut domain_sizes = vec![1];
    let mut partitions = NumericPartitions::trivial(&task);
    let mut numeric_domain_sizes = vec![1];

    apply_initial_goal_splits(
        &task,
        &config,
        &mut rng,
        &HashSet::new(),
        &HashSet::new(),
        &mut domain_mapping,
        &mut domain_sizes,
        &mut partitions,
        &mut numeric_domain_sizes,
    );

    assert_eq!(numeric_domain_sizes, vec![2]);
    let parts = partitions.partitions(0).unwrap();
    assert_eq!(parts.len(), 2);
    assert!(parts[0].contains(3.5) || parts[1].contains(3.5));
}

#[test]
fn cegar_progress_fail_fast_does_not_fire_on_normal_runs() {
    let task = one_dimensional_sailing_like_task();
    let config = CegarConfig {
        flaw_kind: FlawKind::TargetCentered,
        split_direction: Some(SplitDirection::Backward),
        flaw_treatment: FlawTreatmentVariants::MaxRefinedSingleAtom,
        max_iterations: 64,
        random_seed: Some(7),
        ..Default::default()
    };

    let outcome = Cegar::new(config)
        .unwrap()
        .build_abstraction(&task)
        .expect("normal one-dimensional CEGAR run should make progress");

    let x_partitions = outcome
        .final_state
        .factory
        .partitions()
        .partitions(0)
        .expect("x partitions should exist");
    assert!(
        x_partitions.len() > 1,
        "CEGAR should refine x on the one-dimensional task"
    );
}

#[test]
fn target_centered_builds_goal_side_partitions() {
    fn split_points(outcome: &CegarOutcome) -> Vec<f64> {
        let mut points = outcome
            .final_state
            .factory
            .partitions()
            .partitions(0)
            .expect("x partitions should exist")
            .iter()
            .flat_map(|interval| [interval.lower, interval.upper])
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        points.sort_by(|a, b| a.partial_cmp(b).unwrap());
        points.dedup_by(|a, b| a.to_bits() == b.to_bits());
        points
    }

    let task = one_dimensional_sailing_like_task();
    let target_centered = Cegar::new(CegarConfig {
        flaw_kind: FlawKind::TargetCentered,
        split_direction: Some(SplitDirection::Backward),
        flaw_treatment: FlawTreatmentVariants::MaxRefinedSingleAtom,
        max_iterations: 5,
        random_seed: Some(7),
        ..Default::default()
    })
    .unwrap()
    .build_abstraction(&task)
    .expect("target-centered CEGAR should build goal-side partitions");
    let progression = Cegar::new(CegarConfig {
        flaw_kind: FlawKind::Progression,
        flaw_treatment: FlawTreatmentVariants::MaxRefinedSingleAtom,
        max_iterations: 5,
        random_seed: Some(7),
        ..Default::default()
    })
    .unwrap()
    .build_abstraction(&task)
    .expect("progression CEGAR should build start-side partitions");

    let target_points = split_points(&target_centered);
    let progression_points = split_points(&progression);
    let target_goal_side = target_points.iter().filter(|&&point| point >= 5.0).count();
    let progression_start_side = progression_points
        .iter()
        .filter(|&&point| point <= 5.0)
        .count();

    assert!(
        target_goal_side * 2 >= target_points.len(),
        "target-centered splits should cluster near the goal side: {target_points:?}"
    );
    assert!(
        progression_start_side * 2 >= progression_points.len(),
        "progression splits should cluster near the start side: {progression_points:?}"
    );
}

#[test]
fn max_refined_single_atom_is_sticky() {
    let variables = vec![ExplicitVariable::new(
        1,
        "p".into(),
        vec!["p0".into()],
        None,
        0,
    )];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("y".into(), NumericType::Regular, None),
    ];
    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        numeric_variables,
        vec![],
        vec![],
        vec![0],
        vec![0.0, 0.0],
        vec![],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    );
    let config = CegarConfig {
        flaw_treatment: FlawTreatmentVariants::MaxRefinedSingleAtom,
        random_seed: Some(7),
        ..Default::default()
    };
    let mut rng = SmallRng::seed_from_u64(7);
    let mut domain_mapping = vec![vec![0]];
    let mut domain_sizes = vec![1];
    let mut partitions = NumericPartitions::trivial(&task);
    let mut numeric_domain_sizes = vec![1, 1];
    let mut blacklisted_prop_var_ids = HashSet::new();
    let mut blacklisted_numeric_var_ids = HashSet::new();

    for step in 0..4 {
        let value = (step + 1) as f64;
        let flaws = vec![
            Flaw::Numeric(NumericFlaw {
                numeric_var_id: 0,
                value,
                include_in_lower: true,
                step,
            }),
            Flaw::Numeric(NumericFlaw {
                numeric_var_id: 1,
                value,
                include_in_lower: true,
                step,
            }),
        ];
        let refined = fix_flaws(
            &config,
            &task,
            &flaws,
            &mut domain_mapping,
            &mut domain_sizes,
            &mut partitions,
            &mut numeric_domain_sizes,
            &mut rng,
            &mut blacklisted_prop_var_ids,
            &mut blacklisted_numeric_var_ids,
            1,
        )
        .unwrap();
        assert_eq!(refined.refined_numeric_vars, HashSet::from([0]));
    }

    assert_eq!(numeric_domain_sizes, vec![5, 1]);
}

#[test]
fn stale_numeric_flaw_at_existing_boundary_contributes_empty_summary() {
    let task = one_dimensional_sailing_like_task();
    let config = CegarConfig {
        flaw_treatment: FlawTreatmentVariants::OneSplitPerAtom,
        random_seed: Some(1),
        ..Default::default()
    };
    let mut rng = SmallRng::seed_from_u64(1);
    let (mut domain_mapping, mut domain_sizes) = trivial_domain_mapping_and_sizes(&task).unwrap();
    let mut partitions = NumericPartitions::trivial(&task);
    assert!(partitions.split_at(0, 0.0, true));
    let mut numeric_domain_sizes = vec![2, 1, 1];
    let mut blacklisted_prop_var_ids = HashSet::new();
    let mut blacklisted_numeric_var_ids = HashSet::new();
    let flaws = vec![Flaw::Numeric(NumericFlaw {
        numeric_var_id: 0,
        value: -0.0,
        include_in_lower: true,
        step: 0,
    })];

    let refined = fix_flaws(
        &config,
        &task,
        &flaws,
        &mut domain_mapping,
        &mut domain_sizes,
        &mut partitions,
        &mut numeric_domain_sizes,
        &mut rng,
        &mut blacklisted_prop_var_ids,
        &mut blacklisted_numeric_var_ids,
        1,
    )
    .unwrap();

    assert!(refined.is_empty());
    assert_eq!(numeric_domain_sizes[0], 2);
}
