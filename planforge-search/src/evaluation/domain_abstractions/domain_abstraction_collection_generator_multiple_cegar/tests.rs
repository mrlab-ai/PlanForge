use super::*;
use planforge_sas::axioms::{ComparisonAxiom, ComparisonOperator};
use planforge_sas::numeric_conditions::ConditionValue;
use planforge_sas::numeric_task::{
    AssignmentEffect, AssignmentOperation, Effect, ExplicitFact, ExplicitValueIndex,
    ExplicitVariable, Metric, NumericRootTask, NumericRootTaskParts, NumericValue, NumericVariable,
    Operator, OperatorCost, OperatorIndex,
};

use crate::evaluation::abstraction_collections::component::{
    AbstractionComponent, GoalFocusedMember,
};
use crate::evaluation::abstraction_collections::goal_focused::MemberKind;
use crate::evaluation::cartesian_abstractions::{
    CartesianAbstractionCollectionConfig, CartesianAbstractionCollectionGenerator,
    CartesianAbstractionConfig,
};

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
    assert_eq!(config.variants_per_goal, 1);
    assert!(!config.progressive_goal_roots);
}

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
    for collection_strategy in [
        CollectionStrategy::Standard,
        CollectionStrategy::Complementary,
    ] {
        let config = DomainAbstractionCollectionGeneratorMultipleCegarConfig {
            max_abstraction_size: 10,
            max_collection_size: 100,
            abstraction_generation_max_time: 0.0,
            total_max_time: 0.0,
            compute_operator_regions: false,
            collection_strategy,
            ..Default::default()
        };
        let abstractions = DomainAbstractionCollectionGeneratorMultipleCegar::new(config)
            .generate_collection(&task)
            .unwrap();

        assert_eq!(abstractions.len(), 1, "{collection_strategy}");
        assert_eq!(
            abstractions[0].metadata.abstraction_use,
            AbstractionUse::CollectionMember
        );
    }
}

#[test]
fn single_init_split_selection_uses_round_robin_iteration_order() {
    let candidates = [0, 1, 2, 3, 4].map(VariableIndex::new);
    let selected = (1..=8)
        .map(|iteration| select_single_init_split_var(&candidates, iteration).unwrap())
        .collect::<Vec<_>>();

    assert_eq!(selected, [1, 2, 3, 4, 0, 1, 2, 3].map(VariableIndex::new));
}

#[test]
fn single_init_split_selection_handles_empty_candidates() {
    assert_eq!(select_single_init_split_var(&[], 1), None);
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

fn validation_error(config: DomainAbstractionCollectionGeneratorMultipleCegarConfig) -> String {
    DomainAbstractionCollectionGeneratorMultipleCegar::new(config)
        .validate_supported_options()
        .unwrap_err()
        .to_string()
}

#[test]
fn interleaving_rejects_ambiguous_direction_configuration() {
    let error = validation_error(DomainAbstractionCollectionGeneratorMultipleCegarConfig {
        interleave_split_directions: true,
        split_direction: Some(SplitDirection::Forward),
        ..Default::default()
    });

    assert!(error.contains("cannot be combined"));
}

#[test]
fn interleaving_rejects_single_goal_complementary_collections() {
    let error = validation_error(DomainAbstractionCollectionGeneratorMultipleCegarConfig {
        collection_strategy: CollectionStrategy::Complementary,
        interleave_split_directions: true,
        ..Default::default()
    });

    assert!(error.contains("collection_strategy=standard"));
}

#[test]
fn complementary_rejects_options_the_schedule_decides_or_cannot_follow() {
    let complementary = DomainAbstractionCollectionGeneratorMultipleCegarConfig {
        collection_strategy: CollectionStrategy::Complementary,
        ..Default::default()
    };
    let error = validation_error(DomainAbstractionCollectionGeneratorMultipleCegarConfig {
        split_direction: Some(SplitDirection::Backward),
        ..complementary.clone()
    });
    assert!(error.contains("split direction per variant"), "{error}");

    for flaw_kind in [
        FlawKind::Regression,
        FlawKind::TargetCentered,
        FlawKind::SequenceProgression,
        FlawKind::SequenceRegression,
        FlawKind::SequenceBidirectional,
    ] {
        let error = validation_error(DomainAbstractionCollectionGeneratorMultipleCegarConfig {
            flaw_kind,
            ..complementary.clone()
        });
        assert!(
            error.contains("flaw_kind=progression"),
            "{flaw_kind}: {error}"
        );
    }

    let error = validation_error(DomainAbstractionCollectionGeneratorMultipleCegarConfig {
        variants_per_goal: 0,
        ..complementary
    });
    assert!(error.contains("variants_per_goal"), "{error}");
}

#[test]
fn standard_rejects_goal_focused_options() {
    let error = validation_error(DomainAbstractionCollectionGeneratorMultipleCegarConfig {
        variants_per_goal: 2,
        ..Default::default()
    });
    assert!(
        error.contains("collection_strategy=complementary"),
        "{error}"
    );

    let error = validation_error(DomainAbstractionCollectionGeneratorMultipleCegarConfig {
        progressive_goal_roots: true,
        ..Default::default()
    });
    assert!(
        error.contains("collection_strategy=complementary"),
        "{error}"
    );
}

#[test]
fn goal_focused_members_keep_the_flaw_kind_and_split_by_variant_parity() {
    let config = DomainAbstractionCollectionGeneratorMultipleCegarConfig {
        collection_strategy: CollectionStrategy::Complementary,
        flaw_kind: FlawKind::Progression,
        flaw_treatment: FlawTreatmentVariants::MinGrowthSingleAtom,
        max_abstraction_size: 50,
        ..Default::default()
    };
    let backend = GoalFocusedDomainBackend {
        config: &config,
        member_max_time: Some(Duration::from_secs(7)),
    };
    let variant = |direction, tie_rank, seed| GoalVariant {
        goal_id: Some(1),
        variant_id: 3,
        kind: MemberKind::Scheduled,
        direction: Some(direction),
        tie_rank: Some(tie_rank),
        seed,
    };
    let budget = MemberBudget {
        remaining_states: 30,
        remaining_time: Some(Duration::from_secs(5)),
    };

    let forward =
        backend.member_cegar_config(&variant(MemberDirection::Progression, 1, None), budget);
    assert_eq!(forward.flaw_kind, FlawKind::Progression);
    assert_eq!(forward.split_direction, Some(SplitDirection::Forward));
    assert_eq!(forward.flaw_selection_rank, Some(1));
    assert_eq!(forward.random_seed, None);
    assert_eq!(forward.max_abstraction_size, 30);
    assert_eq!(forward.max_time, Some(Duration::from_secs(5)));
    assert_eq!(forward.init_split_var_ids, Some(HashSet::new()));
    assert!(forward.blacklisted_prop_var_ids.is_empty());
    assert!(forward.blacklisted_numeric_var_ids.is_empty());

    let backward =
        backend.member_cegar_config(&variant(MemberDirection::Regression, 2, Some(9)), budget);
    assert_eq!(
        backward.flaw_kind,
        FlawKind::Progression,
        "regression variants keep the configured flaw kind"
    );
    assert_eq!(backward.split_direction, Some(SplitDirection::Backward));
    assert_eq!(backward.flaw_selection_rank, Some(2));
    assert_eq!(backward.random_seed, Some(9));
}

fn comparison_variable(name: &str) -> ExplicitVariable {
    ExplicitVariable::new(
        ConditionValue::DOMAIN_SIZE,
        name.into(),
        vec!["true".into(), "false".into()],
        Some(0),
        ExplicitValueIndex::from_usize(ConditionValue::False.as_usize()),
    )
}

fn increment(name: &str, numeric_var_id: u32, one_id: u32) -> Operator {
    Operator::new(
        name.into(),
        vec![],
        vec![],
        vec![AssignmentEffect::new(
            VariableIndex::new(numeric_var_id),
            AssignmentOperation::Plus,
            VariableIndex::new(one_id),
            false,
            vec![],
        )],
        OperatorCost::new(1),
    )
}

fn comparison_axiom(
    affected_var_id: u32,
    left_var_id: u32,
    right_var_id: u32,
    operator: ComparisonOperator,
) -> ComparisonAxiom {
    ComparisonAxiom::new(
        VariableIndex::new(affected_var_id),
        VariableIndex::new(left_var_id),
        VariableIndex::new(right_var_id),
        operator,
    )
}

/// Goals `x >= 2` and `y >= 3` over two independent counters.
fn two_counter_task() -> NumericRootTask {
    NumericRootTask::new(NumericRootTaskParts {
        version: 2,
        metric: Metric::new(true, None),
        variables: vec![
            comparison_variable("x-at-two"),
            comparison_variable("y-at-three"),
        ],
        numeric_variables: vec![
            NumericVariable::new("x".into(), NumericType::Regular, None),
            NumericVariable::new("y".into(), NumericType::Regular, None),
            NumericVariable::new("one".into(), NumericType::Constant, None),
            NumericVariable::new("two".into(), NumericType::Constant, None),
            NumericVariable::new("three".into(), NumericType::Constant, None),
        ],
        goals: vec![
            ExplicitFact::propositional(0, 0),
            ExplicitFact::propositional(1, 0),
        ],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(1); 2],
        numeric_state: [0.0, 0.0, 1.0, 2.0, 3.0].map(NumericValue::new).to_vec(),
        operators: vec![
            increment("increment-x", 0, 2),
            increment("increment-y", 1, 2),
        ],
        axioms: vec![],
        comparison_axioms: vec![
            comparison_axiom(0, 0, 3, ComparisonOperator::GreaterThanOrEqual),
            comparison_axiom(1, 1, 4, ComparisonOperator::GreaterThanOrEqual),
        ],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    })
}

/// One counter `x` with goals `x >= first` and `x >= second`, or, with
/// `upper_second`, `x <= second`.
fn one_counter_task(first: f64, second: f64, upper_second: bool) -> NumericRootTask {
    NumericRootTask::new(NumericRootTaskParts {
        version: 2,
        metric: Metric::new(true, None),
        variables: vec![comparison_variable("first"), comparison_variable("second")],
        numeric_variables: vec![
            NumericVariable::new("x".into(), NumericType::Regular, None),
            NumericVariable::new("one".into(), NumericType::Constant, None),
            NumericVariable::new("first".into(), NumericType::Constant, None),
            NumericVariable::new("second".into(), NumericType::Constant, None),
        ],
        goals: vec![
            ExplicitFact::propositional(0, 0),
            ExplicitFact::propositional(1, 0),
        ],
        mutexes: vec![],
        state: vec![
            ExplicitValueIndex::new(1),
            ExplicitValueIndex::new(u32::from(!upper_second)),
        ],
        numeric_state: [0.0, 1.0, first, second].map(NumericValue::new).to_vec(),
        operators: vec![increment("increment", 0, 1)],
        axioms: vec![],
        comparison_axioms: vec![
            comparison_axiom(0, 0, 2, ComparisonOperator::GreaterThanOrEqual),
            comparison_axiom(
                1,
                0,
                3,
                if upper_second {
                    ComparisonOperator::LessThanOrEqual
                } else {
                    ComparisonOperator::GreaterThanOrEqual
                },
            ),
        ],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    })
}

fn domain_goal_focused_config(
    variants_per_goal: usize,
    progressive_goal_roots: bool,
) -> DomainAbstractionCollectionGeneratorMultipleCegarConfig {
    DomainAbstractionCollectionGeneratorMultipleCegarConfig {
        collection_strategy: CollectionStrategy::Complementary,
        flaw_kind: FlawKind::Progression,
        max_abstraction_size: 64,
        max_collection_size: 10_000,
        total_max_time: f64::INFINITY,
        variants_per_goal,
        progressive_goal_roots,
        random_seed: Some(1),
        compute_operator_regions: false,
        ..Default::default()
    }
}

/// The backend-neutral member facts, the initial h, and the concrete plan of
/// every member, in collection order.
type MemberSummary = (Option<GoalFocusedMember>, f64, Option<Vec<OperatorIndex>>);

fn domain_members(
    task: &NumericRootTask,
    config: DomainAbstractionCollectionGeneratorMultipleCegarConfig,
) -> Vec<MemberSummary> {
    DomainAbstractionCollectionGeneratorMultipleCegar::new(config)
        .generate_collection(task)
        .unwrap()
        .into_iter()
        .map(|abstraction| {
            let initial_h =
                abstraction.distance_table.distances[abstraction.distance_table.initial_state_hash];
            let plan = abstraction.metadata.concrete_plan_operator_ids.clone();
            let member = AbstractionComponent::domain(None, abstraction).goal_focused_member();
            (member, initial_h, plan)
        })
        .collect()
}

fn cartesian_members(
    task: &NumericRootTask,
    variants_per_goal: usize,
    progressive_goal_roots: bool,
) -> Vec<MemberSummary> {
    CartesianAbstractionCollectionGenerator::new(CartesianAbstractionCollectionConfig {
        abstraction: CartesianAbstractionConfig {
            max_states: 64,
            random_seed: Some(1),
            ..Default::default()
        },
        collection_strategy: CollectionStrategy::Complementary,
        variants_per_goal,
        max_collection_states: 10_000,
        total_max_time: None,
        progressive_goal_roots,
    })
    .unwrap()
    .generate(task)
    .unwrap()
    .into_iter()
    .map(|abstraction| {
        let initial_h =
            abstraction.distance_table.distances[abstraction.distance_table.initial_state_hash];
        let plan = abstraction.metadata.concrete_plan_operator_ids.clone();
        let member = AbstractionComponent::cartesian(None, abstraction).goal_focused_member();
        (member, initial_h, plan)
    })
    .collect()
}

/// Everything but the backend-specific transition count.
fn schedule_view(
    members: &[MemberSummary],
) -> Vec<(GoalFocusedMember, f64, Option<Vec<OperatorIndex>>)> {
    members
        .iter()
        .map(|(member, initial_h, plan)| {
            let member = GoalFocusedMember {
                retained_transitions: 0,
                ..member.expect("every member of a goal task is a collection member")
            };
            (member, *initial_h, plan.clone())
        })
        .collect()
}

#[test]
fn complementary_domain_collection_follows_the_cartesian_member_sequence() {
    let task = two_counter_task();
    let domain = domain_members(&task, domain_goal_focused_config(3, false));
    let cartesian = cartesian_members(&task, 3, false);

    assert_eq!(schedule_view(&domain), schedule_view(&cartesian));
    assert_eq!(
        schedule_view(&domain)
            .iter()
            .map(|(member, initial_h, _)| (
                member.membership.goal_id,
                member.membership.variant_id,
                member.direction,
                member.tie_rank,
                *initial_h
            ))
            .collect::<Vec<_>>(),
        vec![
            (0, 0, MemberDirection::Progression, Some(0), 2.0),
            (1, 0, MemberDirection::Progression, Some(0), 3.0),
            (0, 1, MemberDirection::Regression, Some(0), 2.0),
            (1, 1, MemberDirection::Regression, Some(0), 3.0),
            (1, 2, MemberDirection::Progression, Some(1), 3.0),
            (0, 2, MemberDirection::Progression, Some(1), 2.0),
        ]
    );
}

#[test]
fn progressive_domain_collection_follows_the_cartesian_lanes() {
    let task = one_counter_task(2.0, 4.0, false);
    let domain = domain_members(&task, domain_goal_focused_config(2, true));
    let cartesian = cartesian_members(&task, 2, true);

    assert_eq!(schedule_view(&domain), schedule_view(&cartesian));
    assert!(
        schedule_view(&domain)
            .iter()
            .any(|(member, _, _)| member.membership.progressive_root),
        "a lane must refine from an advanced root"
    );
}

#[test]
fn a_dead_progressive_domain_root_is_rebuilt_from_the_initial_state() {
    let task = one_counter_task(1.0, 0.0, true);
    let members = schedule_view(&domain_members(&task, domain_goal_focused_config(1, true)));

    assert_eq!(
        members
            .iter()
            .map(|(member, _, plan)| (
                member.membership.goal_id,
                member.membership.progressive_root,
                plan.clone()
            ))
            .collect::<Vec<_>>(),
        vec![
            (0, false, Some(vec![OperatorIndex::new(0)])),
            (1, false, Some(vec![]))
        ],
        "the dead checkpoint x=1 must be rebuilt at the satisfying initial state"
    );
}
