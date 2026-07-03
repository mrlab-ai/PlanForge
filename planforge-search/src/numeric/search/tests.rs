use super::*;

use planforge_sas::numeric::numeric_task::{
    ExplicitFact, ExplicitVariable, Metric, NumericRootTask, NumericType, NumericVariable,
    Operator, TaskRef,
};
use planforge_sas::numeric::state_registry::StateRegistry;
use std::sync::Arc;
use std::time::Duration;

#[test]
fn test_compute_effective_operator_costs_plus_constants() {
    // Metric var 0 (cost), incremented by constants 1 and 2.
    let version = 4;
    let metric = Metric::new(true, Some(0));

    let variables = vec![ExplicitVariable::new(
        2,
        "v".to_string(),
        vec!["a".to_string(), "b".to_string()],
        None,
        0,
    )];

    let numeric_variables = vec![
        NumericVariable::new("total_cost()".to_string(), NumericType::Cost, None),
        NumericVariable::new("c1".to_string(), NumericType::Constant, None),
        NumericVariable::new("c2".to_string(), NumericType::Constant, None),
    ];

    let op1 = Operator::new(
        "op1".to_string(),
        vec![],
        vec![],
        vec![planforge_sas::numeric::numeric_task::AssignmentEffect::new(
            0,
            planforge_sas::numeric::numeric_task::AssignmentOperation::Plus,
            1,
            false,
            vec![],
        )],
        1,
    );
    let op2 = Operator::new(
        "op2".to_string(),
        vec![],
        vec![],
        vec![planforge_sas::numeric::numeric_task::AssignmentEffect::new(
            0,
            planforge_sas::numeric::numeric_task::AssignmentOperation::Plus,
            2,
            false,
            vec![],
        )],
        1,
    );

    let task = NumericRootTask::new(
        version,
        metric,
        variables,
        numeric_variables,
        vec![],
        vec![],
        vec![0],
        vec![0.0, 0.5, 0.002],
        vec![op1, op2],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    );

    let task: TaskRef = Arc::new(task);
    let mut state_registry = StateRegistry::for_task(task.clone());
    let initial_state = state_registry.get_initial_state();

    let d0 = state_registry
        .metric_delta_applying_operator(&initial_state, &task.get_operators()[0])
        .unwrap();
    let d1 = state_registry
        .metric_delta_applying_operator(&initial_state, &task.get_operators()[1])
        .unwrap();
    assert!((d0 - 0.5).abs() < 1e-12);
    assert!((d1 - 0.002).abs() < 1e-12);

    let operator_costs = compute_effective_operator_costs(&*task, &state_registry, &initial_state);
    assert_eq!(operator_costs.len(), 2);
    assert!((operator_costs[0] - 0.5).abs() < 1e-12);
    assert!((operator_costs[1] - 0.002).abs() < 1e-12);
    let min_cost = operator_costs
        .iter()
        .copied()
        .fold(f64::INFINITY, |left, right| left.min(right));
    assert!((min_cost - 0.002).abs() < 1e-12);
}

#[test]
fn test_search_status_enum() {
    // Test basic enum functionality
    assert_eq!(SearchStatus::InProgress, SearchStatus::InProgress);
    assert_ne!(SearchStatus::Solved(0), SearchStatus::Failed);
    assert_ne!(SearchStatus::MemoryLimitReached, SearchStatus::Timeout);
}

#[test]
fn test_search_result_creation() {
    let result = SearchResult {
        status: SearchStatus::Failed,
        plan: None,
        solution_cost: None,
        nodes_expanded: 0,
        nodes_reopened: 0,
        nodes_evaluated: 0,
        evaluations: 0,
        nodes_generated: 0,
        dead_ends: 0,
        nodes_expanded_until_last_jump: 0,
        nodes_reopened_until_last_jump: 0,
        nodes_evaluated_until_last_jump: 0,
        nodes_generated_until_last_jump: 0,
        registered_states: 0,
        search_time: Duration::from_millis(100),
    };

    assert_eq!(result.status, SearchStatus::Failed);
    assert!(result.plan.is_none());
    assert_eq!(result.nodes_expanded, 0);
}

#[test]
fn test_progress_format_dedupes_rounding_equal_f_layers() {
    assert_eq!(format_progress_value(95.4940004), "95.494000");
    assert_eq!(format_progress_value(95.49400049), "95.494000");
}
