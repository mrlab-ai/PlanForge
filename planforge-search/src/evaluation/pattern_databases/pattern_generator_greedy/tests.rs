use planforge_sas::axioms::{AssignmentAxiom, CalOperator};
use planforge_sas::axioms::{ComparisonAxiom, ComparisonOperator, PropositionalAxiom};
use planforge_sas::numeric_task::{
    AssignmentEffect, AssignmentOperation, ExplicitFact, ExplicitVariable, Metric, NumericRootTask,
    NumericType, NumericVariable, Operator,
};

use super::*;
use crate::evaluation::pattern_databases::variable_order_finder::GreedyVariableOrderType;
use crate::task_restriction::build_restricted_task;

fn simple_var(name: &str, axiom_layer: Option<usize>) -> ExplicitVariable {
    ExplicitVariable::new(
        2,
        name.to_string(),
        vec![format!("{name}=0"), format!("{name}=1")],
        axiom_layer,
        1,
    )
}

fn sample_task() -> NumericRootTask {
    NumericRootTask::new(
        1,
        Metric::new(true, None),
        vec![
            simple_var("p", None),
            ExplicitVariable::new(
                3,
                "cmp".to_string(),
                vec!["t".to_string(), "f".to_string(), "u".to_string()],
                Some(0),
                2,
            ),
        ],
        vec![
            NumericVariable::new("c".to_string(), NumericType::Constant, None),
            NumericVariable::new("x".to_string(), NumericType::Regular, None),
        ],
        vec![ExplicitFact::new(1, 0)],
        vec![],
        vec![0, 2],
        vec![1.0, 0.0],
        vec![Operator::new(
            "inc".to_string(),
            vec![],
            vec![],
            vec![AssignmentEffect::new(
                1,
                AssignmentOperation::Plus,
                0,
                false,
                vec![],
            )],
            1,
        )],
        vec![PropositionalAxiom::new(
            vec![ExplicitFact::new(1, 0)],
            0,
            1,
            0,
        )],
        vec![ComparisonAxiom::new(
            1,
            1,
            0,
            ComparisonOperator::GreaterThanOrEqual,
        )],
        vec![],
        ExplicitFact::new(0, 0),
    )
}

fn operator_predecessor_task() -> NumericRootTask {
    NumericRootTask::new(
        1,
        Metric::new(true, None),
        vec![
            simple_var("pre", None),
            simple_var("goal", None),
            simple_var("other", None),
        ],
        vec![],
        vec![ExplicitFact::new(1, 1)],
        vec![],
        vec![0, 0, 0],
        vec![],
        vec![Operator::new(
            "achieve-goal".to_string(),
            vec![ExplicitFact::new(0, 1)],
            vec![planforge_sas::numeric_task::Effect::new(
                vec![],
                1,
                Some(0),
                1,
            )],
            vec![],
            1,
        )],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    )
}

fn operator_comparison_predecessor_task() -> NumericRootTask {
    NumericRootTask::new(
        1,
        Metric::new(true, None),
        vec![
            simple_var("goal", None),
            ExplicitVariable::new(
                3,
                "cmp".to_string(),
                vec!["t".to_string(), "f".to_string(), "u".to_string()],
                Some(0),
                2,
            ),
        ],
        vec![
            NumericVariable::new("c5".to_string(), NumericType::Constant, None),
            NumericVariable::new("x".to_string(), NumericType::Regular, None),
            NumericVariable::new("y".to_string(), NumericType::Regular, None),
            NumericVariable::new("sum".to_string(), NumericType::Derived, Some(0)),
        ],
        vec![ExplicitFact::new(0, 0)],
        vec![],
        vec![0, 2],
        vec![5.0, 0.0, 0.0, 0.0],
        vec![Operator::new(
            "achieve-goal".to_string(),
            vec![ExplicitFact::new(1, 0)],
            vec![planforge_sas::numeric_task::Effect::new(
                vec![],
                0,
                Some(1),
                0,
            )],
            vec![],
            1,
        )],
        vec![],
        vec![ComparisonAxiom::new(
            1,
            3,
            0,
            ComparisonOperator::GreaterThanOrEqual,
        )],
        vec![AssignmentAxiom::new(3, CalOperator::Sum, 1, 2)],
        ExplicitFact::new(0, 0),
    )
}
fn numeric_goal_task() -> NumericRootTask {
    NumericRootTask::new(
        1,
        Metric::new(true, None),
        vec![simple_var("cmp", Some(0))],
        vec![
            NumericVariable::new("threshold".to_string(), NumericType::Constant, None),
            NumericVariable::new("x".to_string(), NumericType::Regular, None),
        ],
        vec![ExplicitFact::new(0, 1)],
        vec![],
        vec![0],
        vec![1.0, 0.0],
        vec![],
        vec![],
        vec![ComparisonAxiom::new(
            0,
            1,
            0,
            ComparisonOperator::GreaterThanOrEqual,
        )],
        vec![],
        ExplicitFact::new(0, 0),
    )
}

#[test]
fn greedy_pattern_config_defaults_match_fd_defaults() {
    let config = GreedyPatternGeneratorConfig::default();
    assert_eq!(config.max_pdb_states, 100_000);
    assert!(config.numeric_first);
    assert_eq!(config.random_seed, 0);
    assert_eq!(
        config.variable_order_type,
        GreedyVariableOrderType::GoalCgLevel
    );
}

#[test]
fn greedy_pattern_uses_fd_goal_ordering() {
    let task = numeric_goal_task();
    let pattern = generate_greedy_pattern(&task, GreedyPatternGeneratorConfig::default());
    assert_eq!(pattern.numeric.first().copied(), Some(1));
}

#[test]
fn greedy_pattern_respects_budget_like_fd() {
    let task = numeric_goal_task();
    let pattern = generate_greedy_pattern(
        &task,
        GreedyPatternGeneratorConfig {
            max_pdb_states: 0,
            ..GreedyPatternGeneratorConfig::default()
        },
    );
    assert!(pattern.regular.is_empty());
    assert!(pattern.numeric.is_empty());
}

#[test]
fn greedy_pattern_prefers_goal_variables() {
    let task = sample_task();
    let pattern = generate_greedy_pattern(&task, GreedyPatternGeneratorConfig::default());

    assert!(pattern.numeric.contains(&1));
}

#[test]
fn greedy_pattern_config_defaults_match_expected_port_defaults() {
    let config = GreedyPatternGeneratorConfig::default();

    assert_eq!(config.max_pdb_states, 100_000);
    assert!(config.numeric_first);
    assert_eq!(config.random_seed, 0);
    assert_eq!(
        config.variable_order_type,
        GreedyVariableOrderType::GoalCgLevel
    );
}

#[test]
fn greedy_pattern_expands_via_causal_predecessors_not_all_variables() {
    let task = operator_predecessor_task();
    let pattern = generate_greedy_pattern(
        &task,
        GreedyPatternGeneratorConfig {
            max_pdb_states: 32,
            ..GreedyPatternGeneratorConfig::default()
        },
    );

    assert!(pattern.regular.contains(&1));
    assert!(pattern.regular.contains(&0));
    assert!(!pattern.regular.contains(&2));
}

#[test]
fn greedy_pattern_respects_estimated_numeric_domain_size_budget() {
    let task = sample_task();
    let pattern = generate_greedy_pattern(
        &task,
        GreedyPatternGeneratorConfig {
            max_pdb_states: 2,
            ..GreedyPatternGeneratorConfig::default()
        },
    );

    assert!(!pattern.numeric.contains(&1));
}

#[test]
fn greedy_pattern_collects_regular_numeric_dependencies_from_comparison_trees() {
    let task = NumericRootTask::new(
        1,
        Metric::new(true, None),
        vec![ExplicitVariable::new(
            2,
            "goal".to_string(),
            vec!["off".to_string(), "on".to_string()],
            Some(0),
            0,
        )],
        vec![
            NumericVariable::new("c5".to_string(), NumericType::Constant, None),
            NumericVariable::new("x".to_string(), NumericType::Regular, None),
            NumericVariable::new("y".to_string(), NumericType::Regular, None),
            NumericVariable::new("sum".to_string(), NumericType::Derived, Some(0)),
        ],
        vec![ExplicitFact::new(0, 0)],
        vec![],
        vec![0],
        vec![5.0, 0.0, 0.0, 0.0],
        vec![],
        vec![],
        vec![ComparisonAxiom::new(
            0,
            3,
            0,
            ComparisonOperator::GreaterThanOrEqual,
        )],
        vec![AssignmentAxiom::new(3, CalOperator::Sum, 1, 2)],
        ExplicitFact::new(0, 0),
    );

    let restricted = build_restricted_task(&task).unwrap().unwrap();
    let pattern =
        generate_greedy_pattern(restricted.task(), GreedyPatternGeneratorConfig::default());

    assert!(pattern.numeric.contains(&1));
}

#[test]
fn greedy_pattern_collects_operator_comparison_support_after_restriction() {
    let task = operator_comparison_predecessor_task();
    let restricted = build_restricted_task(&task).unwrap().unwrap();
    let pattern =
        generate_greedy_pattern(restricted.task(), GreedyPatternGeneratorConfig::default());

    assert!(pattern.regular.contains(&0));
    assert!(pattern.numeric.contains(&1));
}
