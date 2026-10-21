use std::collections::HashSet;
use std::time::{Duration, Instant};

use super::*;
use crate::evaluation::abstractions::collections::config::MemberSpecialization;
use crate::evaluation::abstractions::collections::generator::AbstractionGenerator;
use crate::evaluation::abstractions::collections::goal_focused::{GoalVariant, MemberBudget};
use crate::evaluation::abstractions::collections::strategy::CollectionStrategy;
use crate::evaluation::abstractions::common::abstraction::{AbstractionUse, CegarAbstraction};
use crate::evaluation::abstractions::common::cegar::options::CegarOptions;
use crate::evaluation::abstractions::common::cegar::{
    CegarStopReason, FlawKind, FlawScope, SplitKind, SplitSelection,
};
use crate::evaluation::abstractions::domain::config::{
    DomainAbstractionCollectionConfig, DomainAbstractionConfig,
};
use planforge_sas::axioms::{ComparisonAxiom, ComparisonOperator};
use planforge_sas::numeric_conditions::ConditionValue;
use planforge_sas::numeric_task::{
    AssignmentEffect, AssignmentOperation, Effect, ExplicitFact, ExplicitValueIndex,
    ExplicitVariable, Metric, NumericRootTask, NumericRootTaskParts, NumericType, NumericValue,
    NumericVariable, Operator, OperatorCost, OperatorIndex,
};

use crate::evaluation::abstractions::cartesian::{
    CartesianAbstractionCollectionConfig, CartesianAbstractionCollectionGenerator,
    CartesianAbstractionConfig, CartesianAbstractionGenerator, RefinementNode,
};
use crate::evaluation::abstractions::collections::goal_focused::MemberKind;
use crate::evaluation::abstractions::collections::standard::StandardBackend;
use crate::evaluation::abstractions::component::{AbstractionComponent, GoalFocusedMember};

#[test]
fn standard_collection_defaults() {
    let config = DomainAbstractionCollectionConfig::default();
    assert_eq!(config.collection_strategy, CollectionStrategy::Standard);
    assert_eq!(config.member.cegar.max_states, 1_000_000);
    assert_eq!(config.max_collection_size, 10_000_000);
    assert!(!config.member.use_wildcard_plans);
    assert_eq!(config.random_seed, Some(2011));
    assert_eq!(config.member.cegar.flaw_kind, FlawKind::Progression);
    assert_eq!(config.member.cegar.flaw_scope, FlawScope::First);
    assert_eq!(config.alternate_split_kinds, None);
    assert_eq!(config.variants_per_goal, 1);
    assert!(!config.progress_states);
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
        CollectionStrategy::GoalFocused,
    ] {
        let config = DomainAbstractionCollectionConfig {
            max_collection_size: 100,
            total_max_time: Some(Duration::ZERO),
            collection_strategy,
            member: member_config(10, Some(Duration::ZERO)),
            ..Default::default()
        };
        let abstractions = DomainAbstractionCollectionGenerator::new(config)
            .unwrap()
            .generate(&task)
            .unwrap();

        assert_eq!(abstractions.len(), 1, "{collection_strategy}");
        assert_eq!(
            abstractions[0].metadata.abstraction_use,
            AbstractionUse::CollectionMember
        );
    }
}

#[test]
fn standard_collection_stops_at_a_member_whose_abstract_plan_is_concrete() {
    // Without the stop, every restart rebuilds the same solved abstraction
    // until the collection time limit.
    let config = DomainAbstractionCollectionConfig {
        max_collection_size: 100_000,
        total_max_time: Some(Duration::from_secs(600)),
        stagnation_limit: 600.0,
        member: member_config(1_000, Some(Duration::from_secs(600))),
        ..Default::default()
    };
    let start = Instant::now();
    let abstractions = DomainAbstractionCollectionGenerator::new(config)
        .unwrap()
        .generate(&two_counter_task())
        .unwrap();

    assert!(start.elapsed() < Duration::from_secs(60));
    assert_eq!(abstractions.len(), 1);
    assert!(abstractions[0].metadata.solved_by_self());
}

/// The default member with `max_states` and `max_time`, without operator
/// regions.
fn member_config(max_states: usize, max_time: Option<Duration>) -> DomainAbstractionConfig {
    let member = DomainAbstractionCollectionConfig::default().member;
    DomainAbstractionConfig {
        cegar: CegarOptions {
            max_states,
            max_time,
            ..member.cegar
        },
        compute_source_sets: false,
        ..member
    }
}

/// The default collection with the member's `flaw_kind` and `flaw_scope`.
fn with_flaws(
    config: DomainAbstractionCollectionConfig,
    flaw_kind: FlawKind,
    flaw_scope: FlawScope,
) -> DomainAbstractionCollectionConfig {
    let mut config = config;
    config.member.cegar.flaw_kind = flaw_kind;
    config.member.cegar.flaw_scope = flaw_scope;
    config
}

#[test]
fn standard_can_alternate_split_kinds_on_the_full_task() {
    let config = with_flaws(
        DomainAbstractionCollectionConfig {
            collection_strategy: CollectionStrategy::Standard,
            alternate_split_kinds: Some(true),
            ..Default::default()
        },
        FlawKind::Progression,
        FlawScope::WholePlan,
    );
    let schedule = config.standard_schedule();

    assert_eq!(schedule.split_kind(1), Some(SplitKind::Concrete));
    assert_eq!(schedule.split_kind(2), Some(SplitKind::Wanted));
    assert_eq!(schedule.split_kind(3), Some(SplitKind::Concrete));
}

fn validation_error(config: DomainAbstractionCollectionConfig) -> String {
    config.validate().unwrap_err().to_string()
}

#[test]
fn alternation_rejects_a_fixed_split_kind() {
    let error = validation_error(DomainAbstractionCollectionConfig {
        alternate_split_kinds: Some(true),
        split_kind: Some(SplitKind::Concrete),
        ..Default::default()
    });

    assert!(error.contains("cannot be combined"));
}

#[test]
fn per_goal_collections_reject_alternation() {
    let error = validation_error(DomainAbstractionCollectionConfig {
        collection_strategy: CollectionStrategy::PerGoal,
        alternate_split_kinds: Some(true),
        ..Default::default()
    });

    assert!(error.contains("collection_strategy=standard"), "{error}");
}

#[test]
fn goal_focused_rejects_options_the_schedule_decides_or_cannot_follow() {
    let goal_focused = DomainAbstractionCollectionConfig {
        collection_strategy: CollectionStrategy::GoalFocused,
        ..Default::default()
    };
    let error = validation_error(DomainAbstractionCollectionConfig {
        split_kind: Some(SplitKind::Wanted),
        ..goal_focused.clone()
    });
    assert!(error.contains("split kind per member"), "{error}");
    let without_alternation = DomainAbstractionCollectionConfig {
        split_kind: Some(SplitKind::Wanted),
        alternate_split_kinds: Some(false),
        single_goal_tasks: true,
        ..goal_focused.clone()
    };
    assert!(
        without_alternation.validate().is_ok(),
        "without alternation the configured split kind applies to every variant"
    );

    for flaw_scope in [FlawScope::First, FlawScope::WholePlan] {
        let error = validation_error(with_flaws(
            DomainAbstractionCollectionConfig {
                progress_states: true,
                ..goal_focused.clone()
            },
            FlawKind::Regression,
            flaw_scope,
        ));
        assert!(
            error.contains("flaw_kind=progression"),
            "{flaw_scope}: {error}"
        );
    }

    let error = validation_error(DomainAbstractionCollectionConfig {
        variants_per_goal: 0,
        ..goal_focused
    });
    assert!(error.contains("variants_per_goal"), "{error}");
}

#[test]
fn standard_rejects_goal_focused_options() {
    let error = validation_error(DomainAbstractionCollectionConfig {
        variants_per_goal: 2,
        ..Default::default()
    });
    assert!(error.contains("`variants_per_goal` requires"), "{error}");

    let error = validation_error(DomainAbstractionCollectionConfig {
        progress_states: true,
        ..Default::default()
    });
    assert!(error.contains("`progress_states=true` requires"), "{error}");
}

#[test]
fn goal_focused_members_keep_the_flaw_kind_and_split_by_variant_parity() {
    let mut config = DomainAbstractionCollectionConfig {
        collection_strategy: CollectionStrategy::GoalFocused,
        ..Default::default()
    };
    config.member.cegar.flaw_kind = FlawKind::Progression;
    config.member.cegar.split_selection = SplitSelection::MinGrowth;
    config.member.cegar.max_states = 50;
    config.member.cegar.max_time = Some(Duration::from_secs(7));
    let variant = |direction, tie_rank, seed| GoalVariant {
        goal_id: Some(1),
        full_task: false,
        variant_id: 3,
        kind: MemberKind::Scheduled,
        split_kind: Some(direction),
        tie_rank: Some(tie_rank),
        seed,
    };
    let budget = MemberBudget {
        remaining_states: 30,
        remaining_time: Some(Duration::from_secs(5)),
    };

    let member = |variant: GoalVariant| {
        config
            .member
            .for_variant(&variant, config.split_kind, budget)
    };
    let forward = member(variant(SplitKind::Concrete, 1, None));
    assert_eq!(forward.cegar.flaw_kind, FlawKind::Progression);
    assert_eq!(forward.cegar.split_kind, SplitKind::Concrete);
    assert_eq!(forward.cegar.split_selection_rank, Some(1));
    assert_eq!(forward.cegar.random_seed, None);
    assert_eq!(forward.cegar.max_states, 30);
    assert_eq!(forward.cegar.max_time, Some(Duration::from_secs(5)));
    assert_eq!(forward.init_split_var_ids, Some(HashSet::new()));
    assert!(forward.blacklisted_prop_var_ids.is_empty());
    assert!(forward.blacklisted_numeric_var_ids.is_empty());

    let backward = member(variant(SplitKind::Wanted, 2, Some(9)));
    assert_eq!(
        backward.cegar.flaw_kind,
        FlawKind::Progression,
        "regression variants keep the configured flaw kind"
    );
    assert_eq!(backward.cegar.split_kind, SplitKind::Wanted);
    assert_eq!(backward.cegar.split_selection_rank, Some(2));
    assert_eq!(backward.cegar.random_seed, Some(9));
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
    progress_states: bool,
) -> DomainAbstractionCollectionConfig {
    DomainAbstractionCollectionConfig {
        collection_strategy: CollectionStrategy::GoalFocused,
        max_collection_size: 10_000,
        total_max_time: None,
        variants_per_goal,
        progress_states,
        random_seed: Some(1),
        member: member_config(64, None),
        ..Default::default()
    }
}

/// The backend-neutral member facts, the initial h, and the concrete plan of
/// every member, in collection order.
type MemberSummary = (Option<GoalFocusedMember>, f64, Option<Vec<OperatorIndex>>);

fn domain_members(
    task: &NumericRootTask,
    config: DomainAbstractionCollectionConfig,
) -> Vec<MemberSummary> {
    DomainAbstractionCollectionGenerator::new(config)
        .unwrap()
        .generate(task)
        .unwrap()
        .into_iter()
        .map(|abstraction| {
            let initial_h =
                abstraction.distance_table.distances[abstraction.distance_table.initial_state];
            let plan = abstraction.metadata.concrete_plan_operator_ids.clone();
            let member = AbstractionComponent::from(abstraction).goal_focused_member();
            (member, initial_h, plan)
        })
        .collect()
}

fn cartesian_members(
    task: &NumericRootTask,
    variants_per_goal: usize,
    progress_states: bool,
) -> Vec<MemberSummary> {
    cartesian_members_with(
        task,
        CartesianAbstractionCollectionConfig {
            member: CartesianAbstractionConfig {
                cegar: CegarOptions {
                    max_states: 64,
                    random_seed: Some(1),
                    ..CartesianAbstractionConfig::default().cegar
                },
                ..Default::default()
            },
            collection_strategy: CollectionStrategy::GoalFocused,
            variants_per_goal,
            max_collection_size: 10_000,
            total_max_time: None,
            progress_states,
            alternate_split_kinds: Some(true),
            single_goal_tasks: true,
            split_kind: None,
            random_seed: None,
            ..Default::default()
        },
    )
}

fn cartesian_members_with(
    task: &NumericRootTask,
    config: CartesianAbstractionCollectionConfig,
) -> Vec<MemberSummary> {
    CartesianAbstractionCollectionGenerator::new(config)
        .unwrap()
        .generate(task)
        .unwrap()
        .into_iter()
        .map(|abstraction| {
            let initial_h =
                abstraction.distance_table.distances[abstraction.distance_table.initial_state];
            let plan = abstraction.metadata.concrete_plan_operator_ids.clone();
            let member = AbstractionComponent::from(abstraction).goal_focused_member();
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
                abstract_operators: 0,
                ..member.expect("every member of a goal task is a collection member")
            };
            (member, *initial_h, plan.clone())
        })
        .collect()
}

#[test]
fn goal_focused_domain_collection_follows_the_cartesian_member_sequence() {
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
                member.split_kind,
                member.tie_rank,
                *initial_h
            ))
            .collect::<Vec<_>>(),
        vec![
            (0, 0, SplitKind::Concrete, Some(0), 2.0),
            (1, 0, SplitKind::Concrete, Some(0), 3.0),
            (0, 1, SplitKind::Wanted, Some(0), 2.0),
            (1, 1, SplitKind::Wanted, Some(0), 3.0),
            (1, 2, SplitKind::Concrete, Some(1), 3.0),
            (0, 2, SplitKind::Concrete, Some(1), 2.0),
        ]
    );
}

#[test]
fn domain_progress_states_follow_the_cartesian_variants() {
    let task = one_counter_task(2.0, 4.0, false);
    let domain = domain_members(&task, domain_goal_focused_config(2, true));
    let cartesian = cartesian_members(&task, 2, true);

    assert_eq!(schedule_view(&domain), schedule_view(&cartesian));
    assert!(
        schedule_view(&domain)
            .iter()
            .any(|(member, _, _)| member.membership.from_progress_state),
        "a variant must refine from an advanced progress state"
    );
}

#[test]
fn a_dead_domain_progress_state_is_rebuilt_from_the_initial_state() {
    let task = one_counter_task(1.0, 0.0, true);
    let members = schedule_view(&domain_members(&task, domain_goal_focused_config(1, true)));

    assert_eq!(
        members
            .iter()
            .map(|(member, _, plan)| (
                member.membership.goal_id,
                member.membership.from_progress_state,
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

#[test]
fn without_alternation_domain_and_cartesian_members_keep_the_configured_direction() {
    let task = two_counter_task();
    let domain = domain_members(
        &task,
        DomainAbstractionCollectionConfig {
            alternate_split_kinds: Some(false),
            single_goal_tasks: true,
            split_kind: Some(SplitKind::Wanted),
            ..domain_goal_focused_config(3, true)
        },
    );
    let cartesian = cartesian_members_with(
        &task,
        CartesianAbstractionCollectionConfig {
            member: CartesianAbstractionConfig {
                cegar: CegarOptions {
                    max_states: 64,
                    random_seed: Some(1),
                    split_kind: SplitKind::Wanted,
                    ..CartesianAbstractionConfig::default().cegar
                },
                ..Default::default()
            },
            collection_strategy: CollectionStrategy::GoalFocused,
            variants_per_goal: 3,
            max_collection_size: 10_000,
            total_max_time: None,
            progress_states: true,
            alternate_split_kinds: Some(false),
            single_goal_tasks: true,
            split_kind: None,
            random_seed: None,
            ..Default::default()
        },
    );

    assert_eq!(schedule_view(&domain), schedule_view(&cartesian));
    assert!(
        schedule_view(&domain)
            .iter()
            .all(|(member, _, _)| member.split_kind == SplitKind::Wanted)
    );
    assert_eq!(
        schedule_view(&domain)
            .iter()
            .map(|(member, _, _)| member.tie_rank)
            .collect::<Vec<_>>(),
        vec![Some(0), Some(0), Some(0), Some(0), Some(1), Some(1)],
        "ranks still follow the variant"
    );
}

#[test]
fn per_goal_cartesian_collections_reject_alternation() {
    let error =
        CartesianAbstractionCollectionGenerator::new(CartesianAbstractionCollectionConfig {
            collection_strategy: CollectionStrategy::PerGoal,
            alternate_split_kinds: Some(true),
            single_goal_tasks: true,
            ..Default::default()
        })
        .err()
        .unwrap();
    assert!(
        error.to_string().contains("collection_strategy=standard"),
        "{error:#}"
    );
}

#[test]
fn full_task_domain_and_cartesian_members_follow_the_same_round_robin() {
    let task = two_counter_task();
    let domain = DomainAbstractionCollectionGenerator::new(DomainAbstractionCollectionConfig {
        single_goal_tasks: false,
        ..domain_goal_focused_config(2, false)
    })
    .unwrap()
    .generate(&task)
    .unwrap();
    let cartesian =
        CartesianAbstractionCollectionGenerator::new(CartesianAbstractionCollectionConfig {
            member: CartesianAbstractionConfig {
                cegar: CegarOptions {
                    max_states: 64,
                    random_seed: Some(1),
                    ..CartesianAbstractionConfig::default().cegar
                },
                ..Default::default()
            },
            collection_strategy: CollectionStrategy::GoalFocused,
            variants_per_goal: 2,
            max_collection_size: 10_000,
            total_max_time: None,
            progress_states: false,
            alternate_split_kinds: Some(true),
            split_kind: None,
            random_seed: None,
            single_goal_tasks: false,
            stagnation_limit: 20.0,
            blacklist_trigger_percentage: 0.75,
            enable_blacklist_on_stagnation: true,
        })
        .unwrap()
        .generate(&task)
        .unwrap();

    assert_eq!(domain.len(), 4);
    assert_eq!(cartesian.len(), 4);
    for (index, (domain, cartesian)) in domain.iter().zip(&cartesian).enumerate() {
        assert_eq!(
            domain.metadata.collection.map(|record| record.full_task),
            Some(true)
        );
        assert_eq!(domain.metadata.collection_member, None);
        assert_eq!(cartesian.metadata.collection_member, None);
        let expected_backward = index >= 2;
        assert_eq!(
            domain.metadata.split_kind,
            if expected_backward {
                SplitKind::Wanted
            } else {
                SplitKind::Concrete
            }
        );
        assert_eq!(
            cartesian.metadata.split_kind,
            if expected_backward {
                SplitKind::Wanted
            } else {
                SplitKind::Concrete
            }
        );
        assert_eq!(domain.metadata.split_selection_rank, Some(0));
        assert_eq!(cartesian.metadata.split_selection_rank, Some(0));
        // Both abstract the full task: the plan reaches x >= 2 and y >= 3.
        for (backend, initial_h, plan) in [
            (
                "domain",
                domain.distance_table.distances[domain.distance_table.initial_state],
                domain.metadata.concrete_plan_operator_ids.as_ref(),
            ),
            (
                "Cartesian",
                cartesian.distance_table.distances[cartesian.distance_table.initial_state],
                cartesian.metadata.concrete_plan_operator_ids.as_ref(),
            ),
        ] {
            assert_eq!(initial_h, 5.0, "{backend} member {index}");
            assert_eq!(plan.map(Vec::len), Some(5), "{backend} member {index}");
        }
        assert!(
            AbstractionComponent::from(domain.clone())
                .goal_focused_member()
                .is_none()
        );
    }
}

#[test]
fn full_task_members_are_rejected_where_they_are_ill_defined() {
    let error = validation_error(DomainAbstractionCollectionConfig {
        single_goal_tasks: false,
        ..Default::default()
    });
    assert!(
        error.contains("collection_strategy=goal_focused"),
        "{error}"
    );
    let error = validation_error(DomainAbstractionCollectionConfig {
        single_goal_tasks: false,
        ..domain_goal_focused_config(2, true)
    });
    assert!(error.contains("progress_states=true"), "{error}");

    for (collection_strategy, progress_states, expected) in [
        (
            CollectionStrategy::Standard,
            false,
            "collection_strategy=goal_focused",
        ),
        (
            CollectionStrategy::GoalFocused,
            true,
            "progress_states=true",
        ),
    ] {
        let error =
            CartesianAbstractionCollectionGenerator::new(CartesianAbstractionCollectionConfig {
                collection_strategy,
                progress_states,
                single_goal_tasks: false,
                ..Default::default()
            })
            .err()
            .unwrap();
        assert!(error.to_string().contains(expected), "{error:#}");
    }
}

fn cartesian_collection(
    strategy: CollectionStrategy,
    max_states: usize,
    max_collection_size: usize,
) -> CartesianAbstractionCollectionConfig {
    CartesianAbstractionCollectionConfig {
        member: CartesianAbstractionConfig {
            cegar: CegarOptions {
                max_states,
                random_seed: Some(1),
                ..CartesianAbstractionConfig::default().cegar
            },
            ..Default::default()
        },
        collection_strategy: strategy,
        max_collection_size,
        random_seed: Some(1),
        ..Default::default()
    }
}

#[test]
fn both_representations_build_one_per_goal_member_per_goal() {
    let task = two_counter_task();
    let domain = DomainAbstractionCollectionGenerator::new(DomainAbstractionCollectionConfig {
        collection_strategy: CollectionStrategy::PerGoal,
        max_collection_size: 10_000,
        total_max_time: None,
        random_seed: Some(1),
        member: member_config(64, None),
        ..Default::default()
    })
    .unwrap()
    .generate(&task)
    .unwrap();
    let cartesian = CartesianAbstractionCollectionGenerator::new(cartesian_collection(
        CollectionStrategy::PerGoal,
        64,
        10_000,
    ))
    .unwrap()
    .generate(&task)
    .unwrap();
    let domain_goals: Vec<_> = domain
        .iter()
        .map(|member| member.metadata.collection_member.unwrap().goal_id)
        .collect();
    let cartesian_goals: Vec<_> = cartesian
        .iter()
        .map(|member| member.metadata.collection_member.unwrap().goal_id)
        .collect();
    assert_eq!(domain_goals, [0, 1]);
    assert_eq!(cartesian_goals, [0, 1]);
}

#[test]
fn both_representations_stop_a_standard_collection_at_a_member_that_settles_the_task() {
    let task = two_counter_task();
    let domain = DomainAbstractionCollectionGenerator::new(DomainAbstractionCollectionConfig {
        max_collection_size: 100_000,
        total_max_time: None,
        random_seed: Some(1),
        member: member_config(1_000, None),
        ..Default::default()
    })
    .unwrap()
    .generate(&task)
    .unwrap();
    let cartesian = CartesianAbstractionCollectionGenerator::new(cartesian_collection(
        CollectionStrategy::Standard,
        1_000,
        100_000,
    ))
    .unwrap()
    .generate(&task)
    .unwrap();
    assert_eq!(domain.len(), 1);
    assert!(domain[0].metadata.solved_by_self());
    assert_eq!(cartesian.len(), 1);
    assert!(cartesian[0].metadata.solved_by_self());
}

#[test]
fn standard_cartesian_restarts_are_distinct_full_task_abstractions() {
    let task = two_counter_task();
    let members = CartesianAbstractionCollectionGenerator::new(cartesian_collection(
        CollectionStrategy::Standard,
        3,
        12,
    ))
    .unwrap()
    .generate(&task)
    .unwrap();
    assert!(members.len() > 1, "only {} restart", members.len());
    let generator = CartesianAbstractionCollectionGenerator::new(cartesian_collection(
        CollectionStrategy::Standard,
        3,
        12,
    ))
    .unwrap();
    let keys: HashSet<_> = members
        .iter()
        .map(|member| StandardBackend::key(&generator, member))
        .collect();
    assert_eq!(keys.len(), members.len());
    assert!(
        members
            .iter()
            .all(|member| member.metadata.collection_member.is_none())
    );
}

#[test]
fn cartesian_cegar_leaves_blacklisted_variables_alone() {
    let task = two_counter_task();
    let abstraction = CartesianAbstractionGenerator::new(CartesianAbstractionConfig {
        cegar: CegarOptions {
            max_states: 100,
            ..CartesianAbstractionConfig::default().cegar
        },
        blacklisted_numeric_var_ids: [0, 1].map(VariableIndex::from_usize).into(),
        ..Default::default()
    })
    .unwrap()
    .generate(&task)
    .unwrap();
    assert_eq!(
        abstraction.metadata.stop_reason,
        CegarStopReason::NoRefinableFlaw
    );
    assert_eq!(abstraction.num_states(), 1);
}

#[test]
fn a_cartesian_initial_split_on_a_goal_condition_splits_its_operand_at_the_initial_value() {
    let task = two_counter_task();
    let abstraction = CartesianAbstractionGenerator::new(CartesianAbstractionConfig {
        cegar: CegarOptions {
            max_states: 2,
            ..CartesianAbstractionConfig::default().cegar
        },
        initial_split_var: Some(VariableIndex::from_usize(0)),
        ..Default::default()
    })
    .unwrap()
    .generate(&task)
    .unwrap();
    assert!(
        matches!(
            abstraction.hierarchy.nodes[0],
            RefinementNode::Numeric { var_id, boundary, .. }
                if var_id == VariableIndex::from_usize(0) && boundary == NumericValue::new(0.0)
        ),
        "{:?}",
        abstraction.hierarchy.nodes[0]
    );
}
