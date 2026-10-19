use crate::evaluation::abstractions::domain::flaws::{Flaw, NumericFlaw, PropFlaw, get_flaws};

use std::collections::HashSet;

use planforge_sas::numeric_conditions::ConditionValue;

use super::*;
use crate::evaluation::abstractions::domain::flaws::SplitDirection;
use crate::evaluation::abstractions::domain::initial_splits::compute_initial_split_mapping;
use crate::evaluation::abstractions::domain::split_selection::{
    FlawTreatmentVariants, InitSplitMethod,
};
use rand::{SeedableRng, rngs::SmallRng};

use planforge_sas::axioms::{ComparisonAxiom, ComparisonOperator};

use planforge_sas::numeric_task::{
    AssignmentEffect, AssignmentOperation, Effect, ExplicitFact, ExplicitValueIndex,
    ExplicitVariable, INF_VALUE, Metric, NEG_INF_VALUE, NumericRootTask, NumericRootTaskParts,
    NumericType, NumericValue, NumericVariable, Operator, OperatorCost, OperatorIndex,
};

fn one_dimensional_sailing_like_task() -> NumericRootTask {
    let variables = vec![
        ExplicitVariable::new(
            ConditionValue::DOMAIN_SIZE,
            "x_gt_9".into(),
            vec!["true".into(), "false".into()],
            Some(0),
            ExplicitValueIndex::from_usize(ConditionValue::False.as_usize()),
        ),
        ExplicitVariable::new(
            2,
            "saved".into(),
            vec!["false".into(), "true".into()],
            None,
            ExplicitValueIndex::new(0),
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
            VariableIndex::from_usize(0),
            AssignmentOperation::Plus,
            VariableIndex::from_usize(1),
            false,
            vec![],
        )],
        OperatorCost::new(1),
    );
    let save = Operator::new(
        "save".into(),
        vec![ExplicitFact::propositional(0, 0)],
        vec![Effect::new(
            vec![],
            VariableIndex::from_usize(1),
            Some(ExplicitValueIndex::new(0)),
            ExplicitValueIndex::new(1),
        )],
        vec![],
        OperatorCost::new(1),
    );

    NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, None),
        variables,
        numeric_variables,
        goals: vec![ExplicitFact::propositional(1, 1)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(2), ExplicitValueIndex::new(0)],
        numeric_state: vec![
            NumericValue::new(0.0),
            NumericValue::new(1.0),
            NumericValue::new(9.0),
        ],
        operators: vec![go_east, save],
        axioms: vec![],
        comparison_axioms: vec![ComparisonAxiom::new(
            VariableIndex::new(0),
            VariableIndex::new(0),
            VariableIndex::new(2),
            ComparisonOperator::GreaterThan,
        )],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    })
}

#[test]
fn build_abstraction_produces_singleton_plan_without_wildcards() {
    let variables = vec![ExplicitVariable::new(
        2,
        "v".into(),
        vec!["v0".into(), "v1".into()],
        None,
        ExplicitValueIndex::new(0),
    )];
    let numeric_variables: Vec<NumericVariable> = vec![];
    let goals = vec![ExplicitFact::propositional(0, 1)];
    let op0 = Operator::new(
        "set0".into(),
        vec![ExplicitFact::propositional(0, 0)],
        vec![planforge_sas::numeric_task::Effect::new(
            vec![],
            VariableIndex::new(0),
            Some(ExplicitValueIndex::new(0)),
            ExplicitValueIndex::new(1),
        )],
        vec![],
        OperatorCost::new(1),
    );
    let op1 = Operator::new(
        "set1".into(),
        vec![ExplicitFact::propositional(0, 0)],
        vec![planforge_sas::numeric_task::Effect::new(
            vec![],
            VariableIndex::new(0),
            Some(ExplicitValueIndex::new(0)),
            ExplicitValueIndex::new(1),
        )],
        vec![],
        OperatorCost::new(1),
    );
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, None),
        variables,
        numeric_variables,
        goals,
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![],
        operators: vec![op0, op1],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

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
    let op0 = OperatorIndex::new(0);
    let op1 = OperatorIndex::new(1);
    assert!(plan.wildcard_plan[0].as_slice() == [op0] || plan.wildcard_plan[0].as_slice() == [op1]);
}

#[test]
fn empty_wildcard_plan_is_real_exactly_when_initial_state_is_goal() {
    let variables = vec![ExplicitVariable::new(
        2,
        "v".into(),
        vec!["v0".into(), "v1".into()],
        None,
        ExplicitValueIndex::new(0),
    )];
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, None),
        variables: variables.clone(),
        numeric_variables: vec![],
        goals: vec![ExplicitFact::propositional(0, 0)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![],
        operators: vec![],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });
    let empty_plan = WildcardPlanResult {
        wildcard_plan: vec![],
        abstract_state_hashes: vec![0],
        abstract_prop_states: vec![vec![0]],
        abstract_numeric_states: vec![vec![]],
    };
    assert!(wildcard_plan_is_real(&task, &empty_plan).unwrap());

    let non_goal_task = NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, None),
        variables,
        numeric_variables: vec![],
        goals: vec![ExplicitFact::propositional(0, 1)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![],
        operators: vec![],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 1),
    });
    assert!(!wildcard_plan_is_real(&non_goal_task, &empty_plan).unwrap());
}

#[test]
fn get_flaws_reports_numeric_deviation_flaw() {
    use planforge_sas::axioms::{ComparisonAxiom, ComparisonOperator};
    use planforge_sas::numeric_task::{AssignmentEffect, AssignmentOperation, NumericType};
    use planforge_sas::utils::interval::Interval;

    let variables = vec![
        ExplicitVariable::new(
            ConditionValue::DOMAIN_SIZE,
            "gt".into(),
            vec!["true".into(), "false".into()],
            Some(0),
            ExplicitValueIndex::from_usize(ConditionValue::False.as_usize()),
        ),
        ExplicitVariable::new(
            2,
            "g".into(),
            vec!["g0".into(), "g1".into()],
            None,
            ExplicitValueIndex::new(0),
        ),
    ];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("c".into(), NumericType::Constant, None),
        NumericVariable::new("thresh".into(), NumericType::Constant, None),
    ];
    let comparison_axioms = vec![ComparisonAxiom::new(
        VariableIndex::new(0),
        VariableIndex::new(0),
        VariableIndex::new(2),
        ComparisonOperator::GreaterThan,
    )];
    let op0 = Operator::new(
        "inc".into(),
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
    );
    let op1 = Operator::new(
        "set_g".into(),
        vec![ExplicitFact::propositional(0, 0)],
        vec![Effect::new(
            vec![],
            VariableIndex::from_usize(1),
            Some(ExplicitValueIndex::new(0)),
            ExplicitValueIndex::new(1),
        )],
        vec![],
        OperatorCost::new(1),
    );
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, None),
        variables,
        numeric_variables,
        goals: vec![ExplicitFact::propositional(1, 1)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(2), ExplicitValueIndex::new(0)],
        numeric_state: vec![
            NumericValue::new(-10.0),
            NumericValue::new(3.0),
            NumericValue::new(-5.0),
        ],
        operators: vec![op0, op1],
        axioms: vec![],
        comparison_axioms,
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    let partitions = NumericPartitions::with_partitions(vec![
        vec![
            Interval::new(NEG_INF_VALUE, NumericValue::new(-5.0), false, true),
            Interval::new(NumericValue::new(-5.0), INF_VALUE, false, false),
        ],
        vec![Interval::singleton(NumericValue::new(3.0))],
        vec![Interval::singleton(NumericValue::new(-5.0))],
    ]);
    let plan = WildcardPlanResult {
        wildcard_plan: vec![vec![OperatorIndex::new(0)]],
        abstract_state_hashes: vec![],
        abstract_prop_states: vec![],
        abstract_numeric_states: vec![vec![0, 0, 0], vec![1, 0, 0]],
    };
    let domain_mapping = vec![
        vec![ExplicitValueIndex::new(0), ExplicitValueIndex::new(1)],
        vec![ExplicitValueIndex::new(0), ExplicitValueIndex::new(1)],
    ];

    let flaws = get_flaws(
        &task,
        &partitions,
        &domain_mapping,
        &plan,
        FlawKind::Progression,
    )
    .unwrap()
    .flaws;

    assert!(
        flaws.iter().any(|flaw| matches!(flaw, Flaw::Numeric(_))),
        "expected a numeric deviation flaw"
    );
}

#[test]
fn cegar_default_config_matches_numeric_fd_defaults() {
    let config = CegarConfig::default();

    assert_eq!(config.max_abstraction_size, 100_000);
    assert_eq!(config.max_iterations, usize::MAX);
    assert!(!config.use_wildcard_plans);
    assert_eq!(config.random_seed, Some(2011));
    assert_eq!(
        config.flaw_treatment,
        FlawTreatmentVariants::RandomSingleAtom
    );
    assert_eq!(config.init_split_method, InitSplitMethod::InitValue);
}

#[test]
fn fix_flaws_respects_max_abstraction_size_limit() {
    let variables = vec![ExplicitVariable::new(
        2,
        "v".into(),
        vec!["v0".into(), "v1".into()],
        None,
        ExplicitValueIndex::new(0),
    )];
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, None),
        variables,
        numeric_variables: vec![],
        goals: vec![ExplicitFact::propositional(0, 1)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![],
        operators: vec![],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    let config = CegarConfig {
        max_abstraction_size: 1,
        ..Default::default()
    };

    let mut domain_mapping = vec![vec![ExplicitValueIndex::new(0), ExplicitValueIndex::new(0)]];
    let mut domain_sizes = vec![1];
    let mut partitions = NumericPartitions::trivial(&task);
    let mut numeric_domain_sizes = vec![];
    let mut rng = SmallRng::seed_from_u64(7);
    let mut blacklisted_prop_var_ids = HashSet::new();
    let mut blacklisted_numeric_var_ids = HashSet::new();
    let flaws = vec![Flaw::Propositional(PropFlaw {
        fact: ExplicitFact::propositional(0, 1),
        dependent_numeric_flaws: vec![],
        step: 0,
    })];

    let refined = fix_flaws(
        &config,
        &task,
        &flaws,
        RefinementState {
            domain_mapping: &mut domain_mapping,
            domain_sizes: &mut domain_sizes,
            partitions: &mut partitions,
            numeric_domain_sizes: &mut numeric_domain_sizes,
            blacklisted_prop_var_ids: &mut blacklisted_prop_var_ids,
            blacklisted_numeric_var_ids: &mut blacklisted_numeric_var_ids,
        },
        &mut rng,
        0,
    )
    .unwrap();

    assert!(refined.is_empty());
    assert_eq!(domain_sizes, vec![1]);
    assert_eq!(
        domain_mapping,
        vec![vec![
            ExplicitValueIndex::from_usize(0),
            ExplicitValueIndex::from_usize(0)
        ]]
    );
    assert!(blacklisted_prop_var_ids.contains(&VariableIndex::from_usize(0)));
}
#[test]
fn blacklisted_propositional_vars_are_not_refined() {
    let variables = vec![ExplicitVariable::new(
        2,
        "v".into(),
        vec!["v0".into(), "v1".into()],
        None,
        ExplicitValueIndex::new(0),
    )];
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, None),
        variables,
        numeric_variables: vec![],
        goals: vec![ExplicitFact::propositional(0, 1)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![],
        operators: vec![],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    let mut config = CegarConfig::default();
    config
        .blacklisted_prop_var_ids
        .insert(VariableIndex::from_usize(0));
    let cegar = Cegar::new(config).unwrap();

    let mut domain_mapping = vec![vec![ExplicitValueIndex::new(0), ExplicitValueIndex::new(0)]];
    let mut domain_sizes = vec![1];
    let mut partitions = NumericPartitions::trivial(&task);
    let mut numeric_domain_sizes = vec![];
    let mut rng = SmallRng::seed_from_u64(7);
    let mut blacklisted_prop_var_ids = HashSet::from([VariableIndex::from_usize(0usize)]);
    let mut blacklisted_numeric_var_ids = HashSet::new();
    let flaws = vec![Flaw::Propositional(PropFlaw {
        fact: ExplicitFact::propositional(0, 1),
        dependent_numeric_flaws: vec![],
        step: 0,
    })];

    let refined = fix_flaws(
        &cegar.config,
        &task,
        &flaws,
        RefinementState {
            domain_mapping: &mut domain_mapping,
            domain_sizes: &mut domain_sizes,
            partitions: &mut partitions,
            numeric_domain_sizes: &mut numeric_domain_sizes,
            blacklisted_prop_var_ids: &mut blacklisted_prop_var_ids,
            blacklisted_numeric_var_ids: &mut blacklisted_numeric_var_ids,
        },
        &mut rng,
        0,
    )
    .unwrap();

    assert!(refined.is_empty());
    assert_eq!(domain_sizes, vec![1]);
}

#[test]
fn init_value_split_uses_true_branch_for_comparison_variables() {
    let variables = vec![ExplicitVariable::new(
        ConditionValue::DOMAIN_SIZE,
        "cmp".into(),
        vec!["true".into(), "false".into()],
        Some(0),
        ExplicitValueIndex::from_usize(ConditionValue::False.as_usize()),
    )];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("y".into(), NumericType::Regular, None),
    ];
    let comparison_axioms = vec![ComparisonAxiom::new(
        VariableIndex::new(0),
        VariableIndex::new(0),
        VariableIndex::new(1),
        ComparisonOperator::GreaterThan,
    )];
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, None),
        variables,
        numeric_variables,
        goals: vec![ExplicitFact::propositional(0, 0)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(2)],
        numeric_state: vec![NumericValue::new(1.0), NumericValue::new(0.0)],
        operators: vec![],
        axioms: vec![],
        comparison_axioms,
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    let mut config = CegarConfig {
        init_split_method: InitSplitMethod::InitValue,
        ..Default::default()
    };
    config.init_split_method = InitSplitMethod::InitValue;
    let mut rng = SmallRng::seed_from_u64(7);

    let (new_domain_size, mapping) = compute_initial_split_mapping(
        &task,
        &config,
        VariableIndex::from_usize(0),
        Some(ExplicitValueIndex::new(0)),
        &mut rng,
    )
    .unwrap();

    assert_eq!(new_domain_size, 2);
    assert_eq!(
        mapping,
        vec![ExplicitValueIndex::new(1), ExplicitValueIndex::new(0)]
    );
}

#[test]
fn numeric_init_split_is_applied_for_encoded_init_split_var() {
    let variables = vec![ExplicitVariable::new(
        2,
        "g".into(),
        vec!["g0".into(), "g1".into()],
        None,
        ExplicitValueIndex::new(0),
    )];
    let numeric_variables = vec![NumericVariable::new("x".into(), NumericType::Regular, None)];
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, None),
        variables,
        numeric_variables,
        goals: vec![ExplicitFact::propositional(0, 1)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![NumericValue::new(3.5)],
        operators: vec![],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    let config = CegarConfig {
        init_split_method: InitSplitMethod::Identity,
        init_split_var_ids: Some(HashSet::from([VariableIndex::from_usize(1usize)])),
        ..Default::default()
    };

    let mut rng = SmallRng::seed_from_u64(7);
    let mut domain_mapping = vec![vec![ExplicitValueIndex::new(0), ExplicitValueIndex::new(0)]];
    let mut domain_sizes = vec![1];
    let mut partitions = NumericPartitions::trivial(&task);
    let mut numeric_domain_sizes = vec![1];

    apply_initial_goal_splits(
        &task,
        &config,
        &mut rng,
        RefinementState {
            domain_mapping: &mut domain_mapping,
            domain_sizes: &mut domain_sizes,
            partitions: &mut partitions,
            numeric_domain_sizes: &mut numeric_domain_sizes,
            blacklisted_prop_var_ids: &mut HashSet::new(),
            blacklisted_numeric_var_ids: &mut HashSet::new(),
        },
    )
    .unwrap();

    assert_eq!(numeric_domain_sizes, vec![2]);
    let parts = partitions.partitions(VariableIndex::from_usize(0)).unwrap();
    assert_eq!(parts.len(), 2);
    assert!(parts[0].contains(NumericValue::new(3.5)) || parts[1].contains(NumericValue::new(3.5)));
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
        .partitions(VariableIndex::from_usize(0))
        .expect("x partitions should exist");
    assert!(
        x_partitions.len() > 1,
        "CEGAR should refine x on the one-dimensional task"
    );
}

#[test]
fn target_centered_builds_goal_side_partitions() {
    fn split_points(outcome: &CegarOutcome) -> Vec<NumericValue> {
        let mut points = outcome
            .final_state
            .factory
            .partitions()
            .partitions(VariableIndex::from_usize(0))
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
    let target_goal_side = target_points
        .iter()
        .filter(|&&point| point >= NumericValue::new(5.0))
        .count();
    let progression_start_side = progression_points
        .iter()
        .filter(|&&point| point <= NumericValue::new(5.0))
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
        ExplicitValueIndex::new(0),
    )];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("y".into(), NumericType::Regular, None),
    ];
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, None),
        variables,
        numeric_variables,
        goals: vec![],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![NumericValue::new(0.0), NumericValue::new(0.0)],
        operators: vec![],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });
    let config = CegarConfig {
        flaw_treatment: FlawTreatmentVariants::MaxRefinedSingleAtom,
        random_seed: Some(7),
        ..Default::default()
    };
    let mut rng = SmallRng::seed_from_u64(7);
    let mut domain_mapping = vec![vec![ExplicitValueIndex::new(0)]];
    let mut domain_sizes = vec![1];
    let mut partitions = NumericPartitions::trivial(&task);
    let mut numeric_domain_sizes = vec![1, 1];
    let mut blacklisted_prop_var_ids = HashSet::new();
    let mut blacklisted_numeric_var_ids = HashSet::new();

    for step in 0..4 {
        let value = (step + 1) as f64;
        let flaws = vec![
            Flaw::Numeric(NumericFlaw {
                numeric_var_id: VariableIndex::new(0),
                value: NumericValue::new(value),
                include_in_lower: true,
                step,
            }),
            Flaw::Numeric(NumericFlaw {
                numeric_var_id: VariableIndex::new(1),
                value: NumericValue::new(value),
                include_in_lower: true,
                step,
            }),
        ];
        let refined = fix_flaws(
            &config,
            &task,
            &flaws,
            RefinementState {
                domain_mapping: &mut domain_mapping,
                domain_sizes: &mut domain_sizes,
                partitions: &mut partitions,
                numeric_domain_sizes: &mut numeric_domain_sizes,
                blacklisted_prop_var_ids: &mut blacklisted_prop_var_ids,
                blacklisted_numeric_var_ids: &mut blacklisted_numeric_var_ids,
            },
            &mut rng,
            1,
        )
        .unwrap();
        assert_eq!(
            refined.refined_numeric_vars,
            HashSet::from([VariableIndex::from_usize(0)])
        );
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
    assert!(partitions.split_at(VariableIndex::from_usize(0), NumericValue::new(0.0), true));
    let mut numeric_domain_sizes = vec![2, 1, 1];
    let mut blacklisted_prop_var_ids = HashSet::new();
    let mut blacklisted_numeric_var_ids = HashSet::new();
    let flaws = vec![Flaw::Numeric(NumericFlaw {
        numeric_var_id: VariableIndex::from_usize(0),
        value: NumericValue::new(-0.0),
        include_in_lower: true,
        step: 0,
    })];

    let refined = fix_flaws(
        &config,
        &task,
        &flaws,
        RefinementState {
            domain_mapping: &mut domain_mapping,
            domain_sizes: &mut domain_sizes,
            partitions: &mut partitions,
            numeric_domain_sizes: &mut numeric_domain_sizes,
            blacklisted_prop_var_ids: &mut blacklisted_prop_var_ids,
            blacklisted_numeric_var_ids: &mut blacklisted_numeric_var_ids,
        },
        &mut rng,
        1,
    )
    .unwrap();

    assert!(refined.is_empty());
    assert_eq!(numeric_domain_sizes[0], 2);
}

/// A counter `x` from `initial` with the goal `x >= 20` and a unit increment.
fn counter_to_twenty(initial: f64) -> NumericRootTask {
    NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, None),
        variables: vec![ExplicitVariable::new(
            ConditionValue::DOMAIN_SIZE,
            "x_ge_20".into(),
            vec!["true".into(), "false".into()],
            Some(0),
            ExplicitValueIndex::from_usize(ConditionValue::False.as_usize()),
        )],
        numeric_variables: vec![
            NumericVariable::new("x".into(), NumericType::Regular, None),
            NumericVariable::new("one".into(), NumericType::Constant, None),
            NumericVariable::new("twenty".into(), NumericType::Constant, None),
        ],
        goals: vec![ExplicitFact::propositional(0, 0)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::from_usize(usize::from(initial < 20.0))],
        numeric_state: [initial, 1.0, 20.0].map(NumericValue::new).to_vec(),
        operators: vec![Operator::new(
            "increment".into(),
            vec![],
            vec![],
            vec![AssignmentEffect::new(
                VariableIndex::new(0),
                AssignmentOperation::Plus,
                VariableIndex::new(1),
                false,
                vec![],
            )],
            OperatorCost::new(1),
        )],
        axioms: vec![],
        comparison_axioms: vec![ComparisonAxiom::new(
            VariableIndex::new(0),
            VariableIndex::new(0),
            VariableIndex::new(2),
            ComparisonOperator::GreaterThanOrEqual,
        )],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    })
}

/// CEGAR splits at concrete state values, so every state of the plan lies on
/// a partition boundary. The flaw search, the distance table's initial-state
/// hash, and the heuristic's state lookup must map those boundary values to
/// the same partitions, whether the variable has few or many partitions: the
/// converged abstraction is then exact on the plan, and its h(s0) is the one
/// its distance table stores.
#[test]
fn cegar_distance_tables_and_lookups_agree_on_boundary_values() {
    use crate::evaluation::abstractions::component::AbstractionComponent;
    use crate::evaluation::abstractions::domain::generator::DomainAbstractionGenerator;

    for initial in [0.0, 7.0, 15.0] {
        let task = counter_to_twenty(initial);
        let abstraction = DomainAbstractionGenerator::new(CegarConfig {
            max_abstraction_size: 10_000,
            flaw_kind: FlawKind::Progression,
            init_split_var_ids: Some(HashSet::new()),
            ..Default::default()
        })
        .unwrap()
        .generate(&task)
        .unwrap();
        assert!(abstraction.metadata.solved_by_self, "x0={initial}");
        let partition_count = abstraction.factory.numeric_domain_sizes()[0];
        // Both sides of the old small/large lookup split are covered.
        assert_eq!(
            partition_count > 8,
            initial < 15.0,
            "x0={initial}: {partition_count} partitions"
        );
        let stored_h =
            abstraction.distance_table.distances[abstraction.distance_table.initial_state_hash];
        assert_eq!(stored_h, 20.0 - initial, "x0={initial}");

        let component = AbstractionComponent::domain(None, abstraction);
        for step in 0..=(20.0 - initial) as usize {
            let x = initial + step as f64;
            let goal_value = if x >= 20.0 {
                ConditionValue::True
            } else {
                ConditionValue::False
            };
            let h = component
                .standalone_value_from_state_values(
                    &[ExplicitValueIndex::from_usize(goal_value.as_usize())],
                    &[x, 1.0, 20.0].map(NumericValue::new),
                )
                .unwrap();
            assert_eq!(
                h,
                20.0 - x,
                "x0={initial}, {partition_count} partitions: lookup of plan state x={x}"
            );
        }
    }
}
