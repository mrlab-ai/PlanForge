// `numeric_potentials` is behind the `cplex` feature, which needs a proprietary
// CPLEX installation. Neither CI nor a checkout without one can compile this
// file, so no clippy run can tell whether these two fire; `-D warnings` under
// `--features cplex` would break for whoever has CPLEX. Both stay until someone
// with an installation can look. `field_reassign_with_default` is live by
// inspection (`let mut config = ...::default();` followed by field writes).
#![allow(clippy::arc_with_non_send_sync, clippy::field_reassign_with_default)]

use std::sync::Arc;

use planforge_sas::axioms::{
    AssignmentAxiom, CalOperator, ComparisonAxiom, ComparisonOperator, PropositionalAxiom,
};
use planforge_sas::numeric_task::{
    AssignmentEffect, AssignmentOperation, Effect, ExplicitFact, ExplicitVariable, Metric,
    NumericRootTask, NumericRootTaskParts, NumericType, NumericVariable, Operator, TaskRef,
};
use planforge_sas::state_registry::StateRegistry;

use crate::config::{ApplyOptions, ConfigArg, ConfigValue};
use crate::evaluation::domain_abstractions::cegar::CegarConfig;
use crate::evaluation::domain_abstractions::domain_abstraction_generator::DomainAbstractionGenerator;
use crate::evaluation::{EvaluationState, Heuristic};

use super::rays::RayGenerator;
use super::{
    BoundsProvider, NumericPotentialConfig, NumericPotentialHeuristic, NumericPotentialOptimizer,
    OptimizationOutcome, OptimizeFor, PotentialAbstractionOcpHeuristic, PotentialTask,
};

#[test]
fn cpp_defaults_validate() {
    NumericPotentialConfig::default().validate().unwrap();
}

fn option(name: &str, value: &str) -> ConfigArg {
    ConfigArg::new(Some(name.to_string()), ConfigValue::Atom(value.to_string()))
}

#[test]
fn every_cpp_numeric_potential_option_parses() {
    let args = [
        ("opt", "diverse_samples"),
        ("num_samples", "228"),
        ("num_heuristics", "10"),
        ("max_diverse_generation_time", "4.5"),
        ("include_initial_state_potential", "false"),
        ("include_all_states_potential", "true"),
        ("diverse_fallback", "random"),
        ("rays", "3"),
        ("max_ray_generation_time", "5"),
        ("ray_epsilon", "1e-7"),
        ("ray_certificate_file", "/tmp/ray.json"),
        ("max_potential", "infinity"),
        ("ignore_numeric_variables", "true"),
        ("bounds", "all"),
        ("simple_action_bounds", "true"),
        ("goal_conditioned", "false"),
        ("goal_cost_partitioning", "false"),
        ("num_goal_cost_partitions", "2"),
        ("num_goal_conditioned_heuristics", "3"),
        ("num_goal_conditioned_samples", "17"),
        ("max_conditioned_generation_time", "6"),
        ("max_online_heuristics", "7"),
        ("online_reoptimization_interval", "8"),
        ("max_consecutive_online_misses", "9"),
        ("max_online_misses", "10"),
        ("max_online_lp_solves", "11"),
        ("invalidate_online_cache_on_growth", "true"),
        ("online_reoptimization_on_new_states_only", "true"),
        ("cache_estimates", "true"),
        ("precision", "1e-8"),
        ("epsilon", "0.25"),
        ("dump_lp", "true"),
        ("validate_duality", "true"),
    ]
    .map(|(name, value)| option(name, value));
    let mut config = NumericPotentialConfig::default();
    config.apply_options(&args).unwrap();
    config.validate().unwrap();
    assert_eq!(config.opt, OptimizeFor::DiverseSamples);
    assert_eq!(config.num_samples, 228);
    assert_eq!(config.num_heuristics, 10);
    assert_eq!(config.bounds, BoundsProvider::All);
    assert_eq!(config.max_online_functions, 7);
    assert!(config.max_potential.is_infinite());
    assert!(config.validate_duality);

    assert!(
        config
            .apply_options(&[option("not_a_cpp_option", "1")])
            .unwrap_err()
            .contains("unknown option")
    );
    let mut invalid = NumericPotentialConfig::default();
    invalid.num_samples = 0;
    assert!(invalid.validate().unwrap_err().contains("num_samples"));
    invalid = NumericPotentialConfig::default();
    invalid.online_reoptimization_interval = 0;
    assert!(
        invalid
            .validate()
            .unwrap_err()
            .contains("online_reoptimization_interval")
    );
}

fn optimize_initial(task: NumericRootTask) -> (f64, f64) {
    let task: TaskRef<'static> = Arc::new(task);
    let mut registry = StateRegistry::for_task(task.clone());
    let initial = registry.get_initial_state();
    let mut optimizer =
        NumericPotentialOptimizer::new(&*task, &NumericPotentialConfig::default()).unwrap();
    let columns = optimizer.num_columns() as f64;
    let OptimizationOutcome::Optimal { value, function } =
        optimizer.optimize_for_state(&initial, &registry).unwrap()
    else {
        panic!("expected bounded optimal potential")
    };
    let h = function
        .value(
            &initial,
            &registry,
            optimizer.task(),
            &mut Vec::new(),
            &mut Vec::new(),
        )
        .unwrap();
    assert!((h - value).abs() < 1e-7);
    let (dual_value, _, _) = optimizer
        .validate_duality(&initial, &registry, value, 1e-7)
        .unwrap();
    assert!((dual_value - value).abs() < 1e-7);
    (value, columns)
}

#[test]
fn one_step_classical_potential_equals_optimal_cost() {
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(false, None),
        variables: vec![ExplicitVariable::new(
            2,
            "location".into(),
            vec!["start".into(), "goal".into()],
            None,
            0,
        )],
        numeric_variables: vec![],
        goals: vec![ExplicitFact::propositional(0, 1)],
        mutexes: vec![],
        state: vec![0],
        numeric_state: vec![],
        operators: vec![Operator::new(
            "finish".into(),
            vec![ExplicitFact::propositional(0, 0)],
            vec![Effect::new(vec![], 0, Some(0), 1)],
            vec![],
            1,
        )],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });
    let (value, _) = optimize_initial(task);
    assert!((value - 1.0).abs() < 1e-7, "got {value}");
}

#[test]
fn additive_numeric_potential_equals_two_required_increments() {
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(false, None),
        variables: vec![ExplicitVariable::new(
            2,
            "x-at-least-two".into(),
            vec!["true".into(), "false".into()],
            Some(0),
            1,
        )],
        numeric_variables: vec![
            NumericVariable::new("x".into(), NumericType::Regular, None),
            NumericVariable::new("two".into(), NumericType::Constant, None),
            NumericVariable::new("one".into(), NumericType::Constant, None),
        ],
        goals: vec![ExplicitFact::propositional(0, 0)],
        mutexes: vec![],
        state: vec![1],
        numeric_state: vec![0.0, 2.0, 1.0],
        operators: vec![Operator::new(
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
        )],
        axioms: vec![],
        comparison_axioms: vec![ComparisonAxiom::new(
            0,
            0,
            1,
            ComparisonOperator::GreaterThanOrEqual,
        )],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });
    let (value, _) = optimize_initial(task);
    assert!((value - 2.0).abs() < 1e-7, "got {value}");
}

#[test]
fn thirty_deterministic_random_duality_instances() {
    let mut random_state = 0x8f3d_29a7_c15b_04e1_u64;
    let mut next = || {
        random_state = random_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        random_state
    };

    for instance in 0..30 {
        let target = (next() % 9 + 1) as f64;
        let operator_count = (next() % 3 + 2) as usize;
        let mut numeric_variables = vec![
            NumericVariable::new("x".into(), NumericType::Regular, None),
            NumericVariable::new("target".into(), NumericType::Constant, None),
        ];
        let mut initial_numeric = vec![0.0, target];
        let mut operators = Vec::with_capacity(operator_count);
        let mut expected_ratio = f64::INFINITY;
        for operator_id in 0..operator_count {
            let delta = (next() % 4 + 1) as f64;
            let cost = next() % 7 + 1;
            let delta_var = numeric_variables.len();
            numeric_variables.push(NumericVariable::new(
                format!("delta-{operator_id}"),
                NumericType::Constant,
                None,
            ));
            initial_numeric.push(delta);
            operators.push(Operator::new(
                format!("increment-{operator_id}"),
                vec![],
                vec![],
                vec![AssignmentEffect::new(
                    0,
                    AssignmentOperation::Plus,
                    delta_var,
                    false,
                    vec![],
                )],
                cost,
            ));
            expected_ratio = expected_ratio.min(cost as f64 / delta);
        }
        let task = NumericRootTask::new(NumericRootTaskParts {
            version: 4,
            metric: Metric::new(false, None),
            variables: vec![ExplicitVariable::new(
                2,
                "x-at-target".into(),
                vec!["true".into(), "false".into()],
                Some(0),
                1,
            )],
            numeric_variables,
            goals: vec![ExplicitFact::propositional(0, 0)],
            mutexes: vec![],
            state: vec![1],
            numeric_state: initial_numeric,
            operators,
            axioms: vec![],
            comparison_axioms: vec![ComparisonAxiom::new(
                0,
                0,
                1,
                ComparisonOperator::GreaterThanOrEqual,
            )],
            assignment_axioms: vec![],
            global_constraint: ExplicitFact::propositional(0, 0),
        });
        let (value, _) = optimize_initial(task);
        let expected = target * expected_ratio;
        assert!(
            (value - expected).abs() < 1e-7,
            "random duality instance {instance}: expected {expected}, got {value}"
        );
    }
}

#[test]
fn conjunctive_numeric_goal_helper_keeps_original_conditions_separate() {
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(false, None),
        variables: vec![
            ExplicitVariable::new(
                2,
                "x-at-least-two".into(),
                vec!["true".into(), "false".into()],
                Some(0),
                1,
            ),
            ExplicitVariable::new(
                2,
                "y-at-least-three".into(),
                vec!["true".into(), "false".into()],
                Some(0),
                1,
            ),
            ExplicitVariable::new(
                2,
                "numeric-goal".into(),
                vec!["true".into(), "false".into()],
                Some(1),
                1,
            ),
        ],
        numeric_variables: vec![
            NumericVariable::new("x".into(), NumericType::Regular, None),
            NumericVariable::new("y".into(), NumericType::Regular, None),
            NumericVariable::new("two".into(), NumericType::Constant, None),
            NumericVariable::new("three".into(), NumericType::Constant, None),
            NumericVariable::new("one".into(), NumericType::Constant, None),
        ],
        goals: vec![ExplicitFact::propositional(2, 0)],
        mutexes: vec![],
        state: vec![1, 1, 1],
        numeric_state: vec![0.0, 0.0, 2.0, 3.0, 1.0],
        operators: vec![
            Operator::new(
                "increment-x".into(),
                vec![],
                vec![],
                vec![AssignmentEffect::new(
                    0,
                    AssignmentOperation::Plus,
                    4,
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
                    4,
                    false,
                    vec![],
                )],
                1,
            ),
        ],
        axioms: vec![PropositionalAxiom::new(
            vec![
                ExplicitFact::propositional(0, 0),
                ExplicitFact::propositional(1, 0),
            ],
            2,
            1,
            0,
        )],
        comparison_axioms: vec![
            ComparisonAxiom::new(0, 0, 2, ComparisonOperator::GreaterThanOrEqual),
            ComparisonAxiom::new(1, 1, 3, ComparisonOperator::GreaterThanOrEqual),
        ],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(2, 0),
    });
    let (value, _) = optimize_initial(task);
    assert!((value - 5.0).abs() < 1e-7, "got {value}");
}

#[test]
fn classical_only_mode_ignores_numeric_action_conditions() {
    let task: TaskRef<'static> = Arc::new(NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(false, None),
        variables: vec![
            ExplicitVariable::new(
                2,
                "x-at-least-zero".into(),
                vec!["true".into(), "false".into()],
                Some(0),
                1,
            ),
            ExplicitVariable::new(
                2,
                "done".into(),
                vec!["true".into(), "false".into()],
                None,
                0,
            ),
        ],
        numeric_variables: vec![
            NumericVariable::new("x".into(), NumericType::Regular, None),
            NumericVariable::new("zero".into(), NumericType::Constant, None),
        ],
        goals: vec![ExplicitFact::propositional(1, 0)],
        mutexes: vec![],
        state: vec![1, 1],
        numeric_state: vec![0.0, 0.0],
        operators: vec![Operator::new(
            "finish".into(),
            vec![
                ExplicitFact::propositional(0, 0),
                ExplicitFact::propositional(1, 1),
            ],
            vec![Effect::new(vec![], 1, Some(1), 0)],
            vec![],
            1,
        )],
        axioms: vec![],
        comparison_axioms: vec![ComparisonAxiom::new(
            0,
            0,
            1,
            ComparisonOperator::GreaterThanOrEqual,
        )],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(1, 0),
    }));
    let mut config = NumericPotentialConfig::default();
    config.ignore_numeric_variables = true;
    let mut registry = StateRegistry::for_task(task.clone());
    let initial = registry.get_initial_state();
    let mut optimizer = NumericPotentialOptimizer::new(&*task, &config).unwrap();
    assert!(!optimizer.task().features.is_empty());
    assert!(optimizer.conditionable_goals().is_empty());
    let OptimizationOutcome::Optimal { value, .. } =
        optimizer.optimize_for_state(&initial, &registry).unwrap()
    else {
        panic!("expected a bounded classical potential")
    };
    assert!((value - 1.0).abs() < 1e-7, "got {value}");
}

#[test]
fn ocp_retains_stuttering_action_constraints() {
    let task: TaskRef<'static> = Arc::new(NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(false, None),
        variables: vec![ExplicitVariable::new(
            2,
            "x-at-least-two".into(),
            vec!["true".into(), "false".into()],
            Some(0),
            1,
        )],
        numeric_variables: vec![
            NumericVariable::new("x".into(), NumericType::Regular, None),
            NumericVariable::new("two".into(), NumericType::Constant, None),
            NumericVariable::new("one".into(), NumericType::Constant, None),
        ],
        goals: vec![ExplicitFact::propositional(0, 0)],
        mutexes: vec![],
        state: vec![1],
        numeric_state: vec![0.0, 2.0, 1.0],
        operators: vec![Operator::new(
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
        )],
        axioms: vec![],
        comparison_axioms: vec![ComparisonAxiom::new(
            0,
            0,
            1,
            ComparisonOperator::GreaterThanOrEqual,
        )],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    }));
    let cegar_config = CegarConfig {
        max_abstraction_size: 1,
        max_iterations: 1,
        compute_operator_footprints: false,
        ..CegarConfig::default()
    };
    let generator = DomainAbstractionGenerator::new(cegar_config).unwrap();
    let abstraction = generator.generate(&*task).unwrap();
    let heuristic = PotentialAbstractionOcpHeuristic::new(
        &*task,
        task.clone(),
        abstraction,
        NumericPotentialConfig::default(),
        false,
        100,
    )
    .unwrap();
    let mut registry = StateRegistry::for_task(task.clone());
    let initial = registry.get_initial_state();
    let eval_state = EvaluationState::new_with_registry(&initial, 0.0, false, &*task, &registry);
    let value = heuristic.compute_heuristic(&eval_state).unwrap();
    assert!(
        (value - 2.0).abs() < 1e-7,
        "the abstract self-loop must enforce a nonnegative abstraction cost share; got {value}"
    );

    let capped = PotentialAbstractionOcpHeuristic::new(
        &*task,
        task.clone(),
        generator.generate(&*task).unwrap(),
        NumericPotentialConfig::default(),
        false,
        0,
    )
    .unwrap();
    let capped_value = capped.compute_heuristic(&eval_state).unwrap();
    assert!(
        (capped_value - 2.0).abs() < 1e-7,
        "a capped transition system must fall back to the independently admissible potential; got {capped_value}"
    );
}

#[test]
fn conditioned_achiever_couples_numeric_precondition_to_goal_cost() {
    let task: TaskRef<'static> = Arc::new(NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(false, None),
        variables: vec![
            ExplicitVariable::new(
                2,
                "x-at-least-two".into(),
                vec!["true".into(), "false".into()],
                Some(0),
                1,
            ),
            ExplicitVariable::new(
                2,
                "goal".into(),
                vec!["true".into(), "false".into()],
                None,
                0,
            ),
        ],
        numeric_variables: vec![
            NumericVariable::new("x".into(), NumericType::Regular, None),
            NumericVariable::new("two".into(), NumericType::Constant, None),
            NumericVariable::new("one".into(), NumericType::Constant, None),
        ],
        goals: vec![ExplicitFact::propositional(1, 0)],
        mutexes: vec![],
        state: vec![1, 1],
        numeric_state: vec![0.0, 2.0, 1.0],
        operators: vec![
            Operator::new(
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
            ),
            Operator::new(
                "finish".into(),
                vec![
                    ExplicitFact::propositional(0, 0),
                    ExplicitFact::propositional(1, 1),
                ],
                vec![Effect::new(vec![], 1, Some(1), 0)],
                vec![],
                1,
            ),
        ],
        axioms: vec![],
        comparison_axioms: vec![ComparisonAxiom::new(
            0,
            0,
            1,
            ComparisonOperator::GreaterThanOrEqual,
        )],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(1, 0),
    }));
    let mut registry = StateRegistry::for_task(task.clone());
    let initial = registry.get_initial_state();
    let mut optimizer =
        NumericPotentialOptimizer::new(&*task, &NumericPotentialConfig::default()).unwrap();
    assert_eq!(optimizer.conditionable_goals(), [(1, 0)]);
    assert_eq!(optimizer.goal_achievers(1, 0), [1]);
    let OptimizationOutcome::Optimal { value, function } = optimizer
        .optimize_for_conditioned_goal(1, 0, 1, &initial, &registry)
        .unwrap()
    else {
        panic!("expected conditioned optimum")
    };
    assert!((value - 3.0).abs() < 1e-7, "got {value}");
    assert!(
        (function
            .value(
                &initial,
                &registry,
                optimizer.task(),
                &mut Vec::new(),
                &mut Vec::new(),
            )
            .unwrap()
            - 3.0)
            .abs()
            < 1e-7
    );

    let mut online_config = NumericPotentialConfig::default();
    online_config.opt = OptimizeFor::DiverseSamples;
    online_config.num_samples = 1;
    online_config.num_heuristics = 1;
    online_config.online_reoptimization_interval = 1;
    online_config.max_online_functions = 1;
    let heuristic =
        NumericPotentialHeuristic::from_config(&*task, task.clone(), online_config).unwrap();
    let mut online_registry = StateRegistry::for_task(task.clone());
    let online_initial = online_registry.get_initial_state();
    let eval_state =
        EvaluationState::new_with_registry(&online_initial, 0.0, false, &*task, &online_registry);
    assert!(
        (heuristic.compute_heuristic(&eval_state).unwrap() - 3.0).abs() < 1e-7,
        "online reoptimization must not conflict with evaluation scratch borrows"
    );
}

#[test]
fn monotone_and_aibr_bounds_preserve_numeric_optimum() {
    for bounds in [
        BoundsProvider::Monotone,
        BoundsProvider::Aibr,
        BoundsProvider::All,
    ] {
        let task: TaskRef<'static> = Arc::new(NumericRootTask::new(NumericRootTaskParts {
            version: 4,
            metric: Metric::new(false, None),
            variables: vec![ExplicitVariable::new(
                2,
                "x-at-least-two".into(),
                vec!["true".into(), "false".into()],
                Some(0),
                1,
            )],
            numeric_variables: vec![
                NumericVariable::new("x".into(), NumericType::Regular, None),
                NumericVariable::new("two".into(), NumericType::Constant, None),
                NumericVariable::new("one".into(), NumericType::Constant, None),
            ],
            goals: vec![ExplicitFact::propositional(0, 0)],
            mutexes: vec![],
            state: vec![1],
            numeric_state: vec![0.0, 2.0, 1.0],
            operators: vec![Operator::new(
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
            )],
            axioms: vec![],
            comparison_axioms: vec![ComparisonAxiom::new(
                0,
                0,
                1,
                ComparisonOperator::GreaterThanOrEqual,
            )],
            assignment_axioms: vec![],
            global_constraint: ExplicitFact::propositional(0, 0),
        }));
        let mut registry = StateRegistry::for_task(task.clone());
        let initial = registry.get_initial_state();
        let mut config = NumericPotentialConfig::default();
        config.bounds = bounds;
        let mut optimizer = NumericPotentialOptimizer::new(&*task, &config).unwrap();
        let OptimizationOutcome::Optimal { value, .. } =
            optimizer.optimize_for_state(&initial, &registry).unwrap()
        else {
            panic!("expected bounded optimum for {bounds:?}")
        };
        assert!((value - 2.0).abs() < 1e-7, "{bounds:?}: got {value}");
    }
}

#[test]
fn exact_ray_certifies_numeric_dead_end() {
    let task: TaskRef<'static> = Arc::new(NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(false, None),
        variables: vec![ExplicitVariable::new(
            2,
            "x-at-least-two".into(),
            vec!["true".into(), "false".into()],
            Some(0),
            1,
        )],
        numeric_variables: vec![
            NumericVariable::new("x".into(), NumericType::Regular, None),
            NumericVariable::new("two".into(), NumericType::Constant, None),
            NumericVariable::new("minus-one".into(), NumericType::Constant, None),
        ],
        goals: vec![ExplicitFact::propositional(0, 0)],
        mutexes: vec![],
        state: vec![1],
        numeric_state: vec![0.0, 2.0, -1.0],
        operators: vec![Operator::new(
            "decrement".into(),
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
        )],
        axioms: vec![],
        comparison_axioms: vec![ComparisonAxiom::new(
            0,
            0,
            1,
            ComparisonOperator::GreaterThanOrEqual,
        )],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    }));
    let mut registry = StateRegistry::for_task(task.clone());
    let initial = registry.get_initial_state();
    let config = NumericPotentialConfig::default();
    let mut optimizer = NumericPotentialOptimizer::new(&*task, &config).unwrap();
    let mut generator = RayGenerator::new(&optimizer, config.ray_epsilon).unwrap();
    let ray = generator
        .try_certify(&mut optimizer, &initial, &registry)
        .unwrap()
        .expect("the decreasing-only task must have an exact dead-end ray");
    assert!(ray.coefficients().iter().any(|value| value.abs() > 0.0));
    assert!(
        ray.value(
            &initial,
            &registry,
            optimizer.task(),
            &mut Vec::new(),
            &mut Vec::new(),
        )
        .unwrap()
            > config.ray_epsilon
    );
}

#[test]
fn ray_goal_intervals_use_provider_bounds_for_goal_free_resources() {
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(false, None),
        variables: vec![ExplicitVariable::new(
            2,
            "done".into(),
            vec!["true".into(), "false".into()],
            None,
            0,
        )],
        numeric_variables: vec![
            NumericVariable::new("resource".into(), NumericType::Regular, None),
            NumericVariable::new("one".into(), NumericType::Constant, None),
        ],
        goals: vec![ExplicitFact::propositional(0, 0)],
        mutexes: vec![],
        state: vec![1],
        numeric_state: vec![3.0, 1.0],
        operators: vec![Operator::new(
            "increase-and-finish".into(),
            vec![ExplicitFact::propositional(0, 1)],
            vec![Effect::new(vec![], 0, Some(1), 0)],
            vec![AssignmentEffect::new(
                0,
                AssignmentOperation::Plus,
                1,
                false,
                vec![],
            )],
            1,
        )],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });
    let potential_task =
        PotentialTask::build(&task, 1e-6, 0.0, false, BoundsProvider::Monotone, false).unwrap();
    assert_eq!(potential_task.feature_goal_bounds[0].lower, 3.0);
    assert_eq!(potential_task.ray_feature_goal_bounds[0].lower, 3.0);
    assert!(
        potential_task.ray_feature_goal_bounds[0]
            .upper
            .is_infinite()
    );
}

#[test]
fn impossible_reachable_bounds_skip_the_ordinary_lp() {
    let task: TaskRef<'static> = Arc::new(NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(false, None),
        variables: vec![ExplicitVariable::new(
            2,
            "x-at-least-two".into(),
            vec!["true".into(), "false".into()],
            Some(0),
            1,
        )],
        numeric_variables: vec![
            NumericVariable::new("x".into(), NumericType::Regular, None),
            NumericVariable::new("two".into(), NumericType::Constant, None),
            NumericVariable::new("minus-one".into(), NumericType::Constant, None),
        ],
        goals: vec![ExplicitFact::propositional(0, 0)],
        mutexes: vec![],
        state: vec![1],
        numeric_state: vec![0.0, 2.0, -1.0],
        operators: vec![Operator::new(
            "decrement".into(),
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
        )],
        axioms: vec![],
        comparison_axioms: vec![ComparisonAxiom::new(
            0,
            0,
            1,
            ComparisonOperator::GreaterThanOrEqual,
        )],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    }));
    let config = NumericPotentialConfig {
        bounds: BoundsProvider::Monotone,
        ..Default::default()
    };
    let mut registry = StateRegistry::for_task(task.clone());
    let initial = registry.get_initial_state();
    let mut optimizer = NumericPotentialOptimizer::new(&*task, &config).unwrap();
    assert_eq!(optimizer.num_columns(), 0);
    assert_eq!(optimizer.num_rows(), 0);
    assert!(matches!(
        optimizer.optimize_for_state(&initial, &registry).unwrap(),
        OptimizationOutcome::Unbounded { .. }
    ));
}

#[test]
fn affine_auxiliary_features_match_cpp_numeric_proxy() {
    let task: TaskRef<'static> = Arc::new(NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(false, None),
        variables: vec![
            ExplicitVariable::new(
                2,
                "difference-at-least-zero".into(),
                vec!["true".into(), "false".into()],
                Some(2),
                1,
            ),
            ExplicitVariable::new(
                2,
                "done".into(),
                vec!["true".into(), "false".into()],
                None,
                0,
            ),
        ],
        numeric_variables: vec![
            NumericVariable::new("z".into(), NumericType::Regular, None),
            NumericVariable::new("y".into(), NumericType::Regular, None),
            NumericVariable::new("x".into(), NumericType::Regular, None),
            NumericVariable::new("x+y".into(), NumericType::Derived, Some(0)),
            NumericVariable::new("x+y-z".into(), NumericType::Derived, Some(1)),
            NumericVariable::new("zero".into(), NumericType::Constant, None),
            NumericVariable::new("one".into(), NumericType::Constant, None),
        ],
        goals: vec![ExplicitFact::propositional(1, 0)],
        mutexes: vec![],
        state: vec![1, 1],
        numeric_state: vec![5.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        operators: vec![
            Operator::new(
                "decrease-z".into(),
                vec![],
                vec![],
                vec![AssignmentEffect::new(
                    0,
                    AssignmentOperation::Minus,
                    6,
                    false,
                    vec![],
                )],
                1,
            ),
            Operator::new(
                "finish".into(),
                vec![
                    ExplicitFact::propositional(0, 0),
                    ExplicitFact::propositional(1, 1),
                ],
                vec![Effect::new(vec![], 1, Some(1), 0)],
                vec![],
                1,
            ),
            Operator::new(
                "increase-x".into(),
                vec![],
                vec![],
                vec![AssignmentEffect::new(
                    2,
                    AssignmentOperation::Plus,
                    6,
                    false,
                    vec![],
                )],
                1,
            ),
            Operator::new(
                "increase-y".into(),
                vec![],
                vec![],
                vec![AssignmentEffect::new(
                    1,
                    AssignmentOperation::Plus,
                    6,
                    false,
                    vec![],
                )],
                1,
            ),
        ],
        axioms: vec![],
        comparison_axioms: vec![ComparisonAxiom::new(
            0,
            4,
            5,
            ComparisonOperator::GreaterThanOrEqual,
        )],
        assignment_axioms: vec![
            AssignmentAxiom::new(3, CalOperator::Sum, 2, 1),
            AssignmentAxiom::new(4, CalOperator::Difference, 3, 0),
        ],
        global_constraint: ExplicitFact::propositional(1, 0),
    }));
    let potential_task =
        PotentialTask::build(&*task, 1e-6, 0.0, false, BoundsProvider::All, false).unwrap();
    assert_eq!(
        potential_task
            .features
            .iter()
            .map(|feature| feature.name.as_str())
            .collect::<Vec<_>>(),
        ["z", "y", "x", "x+y", "x+y-z"]
    );
    assert_eq!(
        potential_task
            .global_feature_bounds
            .iter()
            .filter(|bounds| bounds.lower.is_finite())
            .count(),
        4
    );
    assert_eq!(
        potential_task
            .global_feature_bounds
            .iter()
            .filter(|bounds| bounds.upper.is_finite())
            .count(),
        1
    );

    let mut config = NumericPotentialConfig::default();
    config.bounds = BoundsProvider::All;
    let mut optimizer = NumericPotentialOptimizer::new(&*task, &config).unwrap();
    assert_eq!(optimizer.num_columns(), 17);
    let mut registry = StateRegistry::for_task(task.clone());
    let initial = registry.get_initial_state();
    let OptimizationOutcome::Optimal { value, .. } = optimizer
        .optimize_for_conditioned_goal(1, 0, 1, &initial, &registry)
        .unwrap()
    else {
        panic!("expected an optimal conditioned affine potential")
    };
    assert!((value - 3.0).abs() < 1e-7, "got {value}");
}
