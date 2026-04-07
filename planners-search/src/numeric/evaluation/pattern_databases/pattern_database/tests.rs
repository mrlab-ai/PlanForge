use planners_sas::numeric::axioms::{
    AssignmentAxiom, CalOperator, ComparisonAxiom, ComparisonOperator,
};
use planners_sas::numeric::numeric_task::{
    AssignmentEffect, AssignmentOperation, ExplicitVariable, Metric, NumericRootTask, NumericType,
    NumericVariable,
};

use super::*;
use crate::numeric::evaluation::pattern_databases::projected_task::{Pattern, ProjectedTask};

fn build_pdb_from_abstract_task<T: AbstractNumericTask>(
    task: T,
    max_states: usize,
) -> PatternDatabase<T> {
    PatternDatabase::new(task, max_states).unwrap()
}

fn simple_var(name: &str, axiom_layer: Option<usize>) -> ExplicitVariable {
    ExplicitVariable::new(
        2,
        name.to_string(),
        vec![format!("{name}=0"), format!("{name}=1")],
        axiom_layer,
        1,
    )
}

fn propositional_task() -> NumericRootTask {
    NumericRootTask::new(
        1,
        Metric::new(true, None),
        vec![simple_var("p", None)],
        vec![NumericVariable::new(
            "x".to_string(),
            NumericType::Regular,
            None,
        )],
        vec![ExplicitFact::new(0, 1)],
        vec![],
        vec![0],
        vec![0.0],
        vec![Operator::new(
            "set-goal".to_string(),
            vec![ExplicitFact::new(0, 0)],
            vec![planners_sas::numeric::numeric_task::Effect::new(
                vec![],
                0,
                Some(0),
                1,
            )],
            vec![],
            3,
        )],
        vec![],
        vec![],
        vec![AssignmentAxiom::new(0, CalOperator::Sum, 0, 0)],
        ExplicitFact::new(0, 0),
    )
}

fn comparison_guarded_task() -> NumericRootTask {
    NumericRootTask::new(
        1,
        Metric::new(true, None),
        vec![simple_var("cmp", Some(0)), simple_var("goal", None)],
        vec![
            NumericVariable::new("threshold".to_string(), NumericType::Constant, None),
            NumericVariable::new("x".to_string(), NumericType::Regular, None),
        ],
        vec![ExplicitFact::new(1, 1)],
        vec![],
        vec![2, 0],
        vec![0.0, 0.0],
        vec![Operator::new(
            "advance".to_string(),
            vec![ExplicitFact::new(0, 0)],
            vec![planners_sas::numeric::numeric_task::Effect::new(
                vec![],
                1,
                Some(0),
                1,
            )],
            vec![],
            1,
        )],
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

fn truncated_chain_task() -> NumericRootTask {
    NumericRootTask::new(
        1,
        Metric::new(true, None),
        vec![ExplicitVariable::new(
            3,
            "p".to_string(),
            vec!["p=0".to_string(), "p=1".to_string(), "p=2".to_string()],
            None,
            2,
        )],
        vec![],
        vec![ExplicitFact::new(0, 2)],
        vec![],
        vec![0],
        vec![],
        vec![
            Operator::new(
                "step-1".to_string(),
                vec![ExplicitFact::new(0, 0)],
                vec![planners_sas::numeric::numeric_task::Effect::new(
                    vec![],
                    0,
                    Some(0),
                    1,
                )],
                vec![],
                5,
            ),
            Operator::new(
                "step-2".to_string(),
                vec![ExplicitFact::new(0, 1)],
                vec![planners_sas::numeric::numeric_task::Effect::new(
                    vec![],
                    0,
                    Some(1),
                    2,
                )],
                vec![],
                5,
            ),
        ],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    )
}

fn cost_only_hidden_numeric_task() -> NumericRootTask {
    NumericRootTask::new(
        1,
        Metric::new(true, Some(0)),
        vec![simple_var("p", None)],
        vec![
            NumericVariable::new("total-cost".to_string(), NumericType::Cost, None),
            NumericVariable::new("c1".to_string(), NumericType::Constant, None),
        ],
        vec![ExplicitFact::new(0, 1)],
        vec![],
        vec![0],
        vec![0.0, 1.0],
        vec![
            Operator::new(
                "wait".to_string(),
                vec![ExplicitFact::new(0, 0)],
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
                "finish".to_string(),
                vec![ExplicitFact::new(0, 0)],
                vec![planners_sas::numeric::numeric_task::Effect::new(
                    vec![],
                    0,
                    Some(0),
                    1,
                )],
                vec![AssignmentEffect::new(
                    0,
                    AssignmentOperation::Plus,
                    1,
                    false,
                    vec![],
                )],
                1,
            ),
        ],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    )
}

fn zero_metric_cost_hidden_numeric_task() -> NumericRootTask {
    NumericRootTask::new(
        1,
        Metric::new(true, Some(0)),
        vec![simple_var("p", None)],
        vec![
            NumericVariable::new("total-cost".to_string(), NumericType::Cost, None),
            NumericVariable::new("zero".to_string(), NumericType::Constant, None),
        ],
        vec![ExplicitFact::new(0, 1)],
        vec![],
        vec![0],
        vec![0.0, 0.0],
        vec![Operator::new(
            "finish".to_string(),
            vec![ExplicitFact::new(0, 0)],
            vec![planners_sas::numeric::numeric_task::Effect::new(
                vec![],
                0,
                Some(0),
                1,
            )],
            vec![AssignmentEffect::new(
                0,
                AssignmentOperation::Plus,
                1,
                false,
                vec![],
            )],
            1,
        )],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    )
}

#[test]
fn lookup_returns_distance_for_reached_state() {
    let task = propositional_task();
    let projected_task = ProjectedTask::new(
        &task,
        &Pattern {
            regular: vec![0],
            numeric: vec![0],
        },
    )
    .unwrap();
    let pdb = PatternDatabase::new(projected_task, 32).unwrap();

    assert_eq!(pdb.lookup(&[0], &[0.0]), Some(3.0));
    assert_eq!(pdb.lookup(&[1], &[0.0]), Some(0.0));
}

#[test]
fn pattern_database_accepts_numeric_abstract_task_boundary() {
    let task = propositional_task();
    let projected_task = ProjectedTask::new(
        &task,
        &Pattern {
            regular: vec![0],
            numeric: vec![0],
        },
    )
    .unwrap();

    let pdb = build_pdb_from_abstract_task(projected_task, 32);

    assert_eq!(pdb.lookup(&[0], &[0.0]), Some(3.0));
}

#[test]
fn lookup_miss_returns_zero_for_goal_state() {
    let task = propositional_task();
    let projected_task = ProjectedTask::new(
        &task,
        &Pattern {
            regular: vec![0],
            numeric: vec![0],
        },
    )
    .unwrap();
    let pdb = PatternDatabase::new(projected_task, 1).unwrap();

    assert_eq!(pdb.lookup(&[1], &[0.0]), None);
    assert_eq!(pdb.lookup_or_fallback(&[1], &[0.0]), 0.0);
}

#[test]
fn lookup_miss_returns_min_operator_cost_for_non_goal_state() {
    let task = propositional_task();
    let projected_task = ProjectedTask::new(
        &task,
        &Pattern {
            regular: vec![0],
            numeric: vec![0],
        },
    )
    .unwrap();
    let pdb = PatternDatabase::new(projected_task, 1).unwrap();

    assert_eq!(pdb.lookup(&[0], &[42.0]), None);
    assert_eq!(pdb.lookup_or_fallback(&[0], &[42.0]), 3.0);
}

#[test]
fn pdb_build_expands_from_axiom_closed_initial_state() {
    let task = comparison_guarded_task();
    let projected_task = ProjectedTask::new(
        &task,
        &Pattern {
            regular: vec![1],
            numeric: vec![1],
        },
    )
    .unwrap();

    let (initial_prop, initial_num) = projected_task.evaluated_initial_state_values().unwrap();
    assert_eq!(initial_prop, vec![0, 0]);

    let pdb = PatternDatabase::new(projected_task, 16).unwrap();

    assert!(pdb.states.len() > 1);
    assert_eq!(pdb.lookup(&initial_prop, &initial_num), Some(1.0));
    assert!(pdb.distances.contains(&0.0));
}

#[test]
fn truncated_pdb_propagates_frontier_seed_costs() {
    let task = truncated_chain_task();
    let projected_task = ProjectedTask::new(
        &task,
        &Pattern {
            regular: vec![0],
            numeric: vec![],
        },
    )
    .unwrap();

    let pdb = PatternDatabase::new(projected_task, 2).unwrap();

    assert!(pdb.truncated);
    assert_eq!(pdb.reached_goal_states, 0);
    assert_eq!(pdb.frontier_states, vec![1]);
    assert_eq!(pdb.lookup(&[1], &[]), Some(5.0));
    assert_eq!(pdb.lookup(&[0], &[]), Some(10.0));
    assert_eq!(pdb.lookup_or_fallback(&[0], &[]), 10.0);
}

#[test]
fn pdb_collapses_hidden_cost_dimensions_outside_pattern() {
    let task = cost_only_hidden_numeric_task();
    let projected_task = ProjectedTask::new(
        &task,
        &Pattern {
            regular: vec![0],
            numeric: vec![],
        },
    )
    .unwrap();

    let pdb = PatternDatabase::new(projected_task, 64).unwrap();

    assert_eq!(pdb.states.len(), 2);
    assert_eq!(pdb.lookup(&[0], &[]), Some(1.0));
    assert_eq!(pdb.lookup(&[1], &[]), Some(0.0));
}

#[test]
fn pdb_uses_metric_delta_costs_even_when_metric_var_is_hidden() {
    let task = zero_metric_cost_hidden_numeric_task();
    let projected_task = ProjectedTask::new(
        &task,
        &Pattern {
            regular: vec![0],
            numeric: vec![],
        },
    )
    .unwrap();

    let pdb = PatternDatabase::new(projected_task, 64).unwrap();

    assert_eq!(pdb.min_operator_cost(), 0.0);
    assert_eq!(pdb.lookup(&[0], &[]), Some(0.0));
    assert_eq!(pdb.lookup(&[1], &[]), Some(0.0));
}
