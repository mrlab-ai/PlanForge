use std::collections::HashSet;

use planforge_sas::numeric_conditions::ConditionValue;

use super::*;
use crate::evaluation::abstractions::domain::config::FlawTreatmentVariants;
use crate::evaluation::abstractions::domain::config::SplitDirection;
use crate::evaluation::abstractions::domain::initial_splits::initial_split_mapping;

use planforge_sas::axioms::{ComparisonAxiom, ComparisonOperator};

use planforge_sas::numeric_task::{
    AssignmentEffect, AssignmentOperation, Effect, ExplicitFact, ExplicitValueIndex,
    ExplicitVariable, Metric, NumericRootTask, NumericRootTaskParts, NumericType, NumericValue,
    NumericVariable, Operator, OperatorCost, OperatorIndex,
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
fn build_abstraction_finds_a_one_step_concrete_plan() {
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

    let outcome = Cegar::new(config)
        .unwrap()
        .build_abstraction(&task)
        .unwrap();
    assert!(outcome.solved_by_self);
    let plan = outcome
        .concrete_plan_operator_ids
        .expect("a solved abstraction has a plan");
    assert!(plan == [OperatorIndex::new(0)] || plan == [OperatorIndex::new(1)]);
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
}

#[test]
fn cegar_stops_when_the_flaw_split_exceeds_the_size_limit() {
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

    let outcome = Cegar::new(CegarConfig {
        max_abstraction_size: 1,
        ..Default::default()
    })
    .unwrap()
    .build_abstraction(&task)
    .unwrap();

    assert_eq!(outcome.stop_reason, CegarStopReason::NoRefinableFlaw);
    assert_eq!(outcome.final_state.factory.domain_sizes(), [1]);
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
    let outcome = Cegar::new(config)
        .unwrap()
        .build_abstraction(&task)
        .unwrap();

    assert_eq!(outcome.stop_reason, CegarStopReason::NoRefinableFlaw);
    assert_eq!(outcome.final_state.factory.domain_sizes(), [1]);
}

#[test]
fn init_value_split_separates_the_initial_value() {
    let variables = vec![ExplicitVariable::new(
        3,
        "v".into(),
        vec!["v0".into(), "v1".into(), "v2".into()],
        None,
        ExplicitValueIndex::new(0),
    )];
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, None),
        variables,
        numeric_variables: vec![],
        goals: vec![ExplicitFact::propositional(0, 2)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(1)],
        numeric_state: vec![],
        operators: vec![],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    let mapping = initial_split_mapping(&task, VariableIndex::from_usize(0)).unwrap();

    assert_eq!(
        mapping,
        vec![
            ExplicitValueIndex::new(0),
            ExplicitValueIndex::new(1),
            ExplicitValueIndex::new(0)
        ]
    );
}

#[test]
fn cegar_progress_fail_fast_does_not_fire_on_normal_runs() {
    let task = one_dimensional_sailing_like_task();
    let config = CegarConfig {
        flaw_kind: FlawKind::Progression,
        split_direction: Some(SplitDirection::Backward),
        flaw_treatment: FlawTreatmentVariants::MinGrowthSingleAtom,
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
fn backward_splits_build_goal_side_partitions() {
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
    let backward = Cegar::new(CegarConfig {
        flaw_kind: FlawKind::Progression,
        split_direction: Some(SplitDirection::Backward),
        flaw_treatment: FlawTreatmentVariants::MinGrowthSingleAtom,
        max_iterations: 5,
        random_seed: Some(7),
        ..Default::default()
    })
    .unwrap()
    .build_abstraction(&task)
    .expect("backward-split CEGAR should build goal-side partitions");
    let progression = Cegar::new(CegarConfig {
        flaw_kind: FlawKind::Progression,
        flaw_treatment: FlawTreatmentVariants::MinGrowthSingleAtom,
        max_iterations: 5,
        random_seed: Some(7),
        ..Default::default()
    })
    .unwrap()
    .build_abstraction(&task)
    .expect("progression CEGAR should build start-side partitions");

    let target_points = split_points(&backward);
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
        "backward splits should cluster near the goal side: {target_points:?}"
    );
    assert!(
        progression_start_side * 2 >= progression_points.len(),
        "progression splits should cluster near the start side: {progression_points:?}"
    );
}

/// CEGAR splits at concrete state values, so every state of the plan lies on
/// a partition boundary. The flaw search, the distance table's initial-state
/// hash, and the heuristic's state lookup must map those boundary values to
/// the same partitions, whether the variable has few or many partitions: the
/// converged abstraction is then exact on the plan, and its h(s0) is the one
/// its distance table stores.
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

/// Domain-abstraction CEGAR with regression flaws alone, first or whole-plan,
/// reaches a concrete optimal plan on the tasks where regression and
/// progression differ, with both split directions.
#[test]
fn regression_cegar_solves_the_regression_tasks_optimally() {
    use crate::evaluation::abstractions::common::cegar::regression::test_tasks::{
        diagonal_task, jump_task, linear_goal_task,
    };
    use crate::evaluation::abstractions::domain::generator::DomainAbstractionGenerator;

    for (name, task, optimal_cost) in [
        ("jump", jump_task(), 4.0),
        ("linear goal", linear_goal_task(), 2.0),
        ("diagonal", diagonal_task(), 2.0),
    ] {
        for flaw_kind in [FlawKind::Regression, FlawKind::RegressEntirePlan] {
            for split_direction in [SplitDirection::Forward, SplitDirection::Backward] {
                let abstraction = DomainAbstractionGenerator::new(CegarConfig {
                    max_abstraction_size: 1_000,
                    flaw_kind,
                    split_direction: Some(split_direction),
                    random_seed: Some(1),
                    ..Default::default()
                })
                .unwrap()
                .generate(&task)
                .unwrap();
                assert!(
                    abstraction.metadata.solved_by_self,
                    "{name}, {flaw_kind}, {split_direction}"
                );
                assert_eq!(
                    abstraction.distance_table.distances
                        [abstraction.distance_table.initial_state_hash],
                    optimal_cost,
                    "{name}, {flaw_kind}, {split_direction}"
                );
            }
        }
    }
}
