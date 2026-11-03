use planforge_sas::axioms::{AssignmentAxiom, CalOperator};
use planforge_sas::axioms::{ComparisonAxiom, ComparisonOperator, PropositionalAxiom};
use planforge_sas::numeric_task::{
    AssignmentEffect, AssignmentOperation, ExplicitFact, ExplicitValueIndex, ExplicitVariable,
    Metric, NumericType, NumericValue, NumericVariable, Operator, OperatorCost, Task, TaskParts,
    VariableIndex,
};

use super::*;
use planforge_search::task_transformations::restriction::build_restricted_task;

fn simple_var(name: &str, axiom_layer: Option<usize>) -> ExplicitVariable {
    ExplicitVariable::new(
        2,
        name.to_string(),
        vec![format!("{name}=0"), format!("{name}=1")],
        axiom_layer,
        ExplicitValueIndex::new(1),
    )
}

fn sample_task() -> Task {
    Task::new(TaskParts {
        metric: Metric::new(true, None),
        variables: vec![
            simple_var("p", None),
            ExplicitVariable::new(
                3,
                "cmp".to_string(),
                vec!["t".to_string(), "f".to_string(), "u".to_string()],
                Some(0),
                ExplicitValueIndex::new(2),
            ),
        ],
        numeric_variables: vec![
            NumericVariable::new("c".to_string(), NumericType::Constant, None),
            NumericVariable::new("x".to_string(), NumericType::Regular, None),
        ],
        goals: vec![ExplicitFact::propositional(1, 0)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0), ExplicitValueIndex::new(2)],
        numeric_state: vec![NumericValue::new(1.0), NumericValue::new(0.0)],
        operators: vec![Operator::new(
            "inc".to_string(),
            vec![],
            vec![],
            vec![AssignmentEffect::new(
                VariableIndex::from_usize(1),
                AssignmentOperation::Plus,
                VariableIndex::from_usize(0),
                false,
                vec![],
            )],
            OperatorCost::new(1),
        )],
        axioms: vec![PropositionalAxiom::new(
            vec![ExplicitFact::propositional(1, 0)],
            VariableIndex::from_usize(0),
            ExplicitValueIndex::new(1),
            ExplicitValueIndex::new(0),
        )],
        comparison_axioms: vec![ComparisonAxiom::new(
            VariableIndex::new(1),
            VariableIndex::new(1),
            VariableIndex::new(0),
            ComparisonOperator::GreaterThanOrEqual,
        )],
        assignment_axioms: vec![],
        global_constraint: Some(ExplicitFact::propositional(0, 0)),
    })
}

fn operator_predecessor_task() -> Task {
    Task::new(TaskParts {
        metric: Metric::new(true, None),
        variables: vec![
            simple_var("pre", None),
            simple_var("goal", None),
            simple_var("other", None),
        ],
        numeric_variables: vec![],
        goals: vec![ExplicitFact::propositional(1, 1)],
        mutexes: vec![],
        state: vec![
            ExplicitValueIndex::new(0),
            ExplicitValueIndex::new(0),
            ExplicitValueIndex::new(0),
        ],
        numeric_state: vec![],
        operators: vec![Operator::new(
            "achieve-goal".to_string(),
            vec![ExplicitFact::propositional(0, 1)],
            vec![planforge_sas::numeric_task::Effect::new(
                vec![],
                VariableIndex::new(1),
                Some(ExplicitValueIndex::from_usize(0)),
                ExplicitValueIndex::new(1),
            )],
            vec![],
            OperatorCost::new(1),
        )],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: Some(ExplicitFact::propositional(0, 0)),
    })
}

fn operator_comparison_predecessor_task() -> Task {
    Task::new(TaskParts {
        metric: Metric::new(true, None),
        variables: vec![
            simple_var("goal", None),
            ExplicitVariable::new(
                3,
                "cmp".to_string(),
                vec!["t".to_string(), "f".to_string(), "u".to_string()],
                Some(1),
                ExplicitValueIndex::new(2),
            ),
        ],
        numeric_variables: vec![
            NumericVariable::new("c5".to_string(), NumericType::Constant, None),
            NumericVariable::new("x".to_string(), NumericType::Regular, None),
            NumericVariable::new("y".to_string(), NumericType::Regular, None),
            NumericVariable::new("sum".to_string(), NumericType::Derived, Some(0)),
        ],
        goals: vec![ExplicitFact::propositional(0, 0)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0), ExplicitValueIndex::new(2)],
        numeric_state: vec![
            NumericValue::new(5.0),
            NumericValue::new(0.0),
            NumericValue::new(0.0),
            NumericValue::new(0.0),
        ],
        operators: vec![Operator::new(
            "achieve-goal".to_string(),
            vec![ExplicitFact::propositional(1, 0)],
            vec![planforge_sas::numeric_task::Effect::new(
                vec![],
                VariableIndex::new(0),
                Some(ExplicitValueIndex::new(1)),
                ExplicitValueIndex::new(0),
            )],
            vec![],
            OperatorCost::new(1),
        )],
        axioms: vec![],
        comparison_axioms: vec![ComparisonAxiom::new(
            VariableIndex::new(1),
            VariableIndex::new(3),
            VariableIndex::new(0),
            ComparisonOperator::GreaterThanOrEqual,
        )],
        assignment_axioms: vec![AssignmentAxiom::new(
            VariableIndex::from_usize(3),
            CalOperator::Sum,
            VariableIndex::from_usize(1),
            VariableIndex::from_usize(2),
        )],
        global_constraint: Some(ExplicitFact::propositional(0, 0)),
    })
}
fn numeric_goal_task() -> Task {
    Task::new(TaskParts {
        metric: Metric::new(true, None),
        variables: vec![simple_var("cmp", Some(0))],
        numeric_variables: vec![
            NumericVariable::new("threshold".to_string(), NumericType::Constant, None),
            NumericVariable::new("x".to_string(), NumericType::Regular, None),
        ],
        goals: vec![ExplicitFact::propositional(0, 1)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![NumericValue::new(1.0), NumericValue::new(0.0)],
        operators: vec![],
        axioms: vec![],
        comparison_axioms: vec![ComparisonAxiom::new(
            VariableIndex::new(0),
            VariableIndex::new(1),
            VariableIndex::new(0),
            ComparisonOperator::GreaterThanOrEqual,
        )],
        assignment_axioms: vec![],
        global_constraint: Some(ExplicitFact::propositional(0, 0)),
    })
}

#[test]
fn greedy_pattern_uses_fd_goal_ordering() {
    let task = numeric_goal_task();
    let pattern = generate_greedy_pattern(&task, GreedyPatternGeneratorConfig::default());
    assert_eq!(
        pattern.numeric.first().copied(),
        Some(VariableIndex::from_usize(1))
    );
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

    assert!(pattern.numeric.contains(&VariableIndex::from_usize(1)));
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

    assert!(pattern.regular.contains(&VariableIndex::from_usize(1)));
    assert!(pattern.regular.contains(&VariableIndex::from_usize(0)));
    assert!(!pattern.regular.contains(&VariableIndex::from_usize(2)));
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

    assert!(!pattern.numeric.contains(&VariableIndex::from_usize(1)));
}

#[test]
fn greedy_pattern_collects_regular_numeric_dependencies_from_comparison_trees() {
    let task = Task::new(TaskParts {
        metric: Metric::new(true, None),
        variables: vec![ExplicitVariable::new(
            2,
            "goal".to_string(),
            vec!["off".to_string(), "on".to_string()],
            Some(1),
            ExplicitValueIndex::new(0),
        )],
        numeric_variables: vec![
            NumericVariable::new("c5".to_string(), NumericType::Constant, None),
            NumericVariable::new("x".to_string(), NumericType::Regular, None),
            NumericVariable::new("y".to_string(), NumericType::Regular, None),
            NumericVariable::new("sum".to_string(), NumericType::Derived, Some(0)),
        ],
        goals: vec![ExplicitFact::propositional(0, 0)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![
            NumericValue::new(5.0),
            NumericValue::new(0.0),
            NumericValue::new(0.0),
            NumericValue::new(0.0),
        ],
        operators: vec![],
        axioms: vec![],
        comparison_axioms: vec![ComparisonAxiom::new(
            VariableIndex::new(0),
            VariableIndex::new(3),
            VariableIndex::new(0),
            ComparisonOperator::GreaterThanOrEqual,
        )],
        assignment_axioms: vec![AssignmentAxiom::new(
            VariableIndex::from_usize(3),
            CalOperator::Sum,
            VariableIndex::from_usize(1),
            VariableIndex::from_usize(2),
        )],
        global_constraint: Some(ExplicitFact::propositional(0, 0)),
    });

    let restricted = build_restricted_task(&task).unwrap().unwrap();
    let pattern = generate_greedy_pattern(&restricted, GreedyPatternGeneratorConfig::default());

    assert!(pattern.numeric.contains(&VariableIndex::from_usize(1)));
}

#[test]
fn greedy_pattern_collects_operator_comparison_support_after_restriction() {
    let task = operator_comparison_predecessor_task();
    let restricted = build_restricted_task(&task).unwrap().unwrap();
    let pattern = generate_greedy_pattern(&restricted, GreedyPatternGeneratorConfig::default());

    assert!(pattern.regular.contains(&VariableIndex::from_usize(0)));
    assert!(pattern.numeric.contains(&VariableIndex::from_usize(1)));
}
