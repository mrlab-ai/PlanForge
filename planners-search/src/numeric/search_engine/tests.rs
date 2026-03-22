use super::*;

use planners_sas::numeric::axioms::AxiomEvaluator;
use planners_sas::numeric::numeric_task::{
    ExplicitVariable, Metric, NumericRootTask, NumericType, NumericVariable, Operator,
};
use planners_sas::numeric::state_registry::StateRegistry;
use planners_sas::numeric::utils::int_packer::IntDoublePacker;

#[test]
fn test_min_action_cost_from_initial_metric_deltas_plus_constants() {
    // Metric var 0 (cost), incremented by constants 1 and 2.
    let version = 4;
    let metric = Metric::new(true, 0);

    let variables = vec![ExplicitVariable::new(
        2,
        "v".to_string(),
        vec!["a".to_string(), "b".to_string()],
        -1,
        0,
    )];

    let numeric_variables = vec![
        NumericVariable::new("total_cost()".to_string(), NumericType::Cost, -1),
        NumericVariable::new("c1".to_string(), NumericType::Constant, -1),
        NumericVariable::new("c2".to_string(), NumericType::Constant, -1),
    ];

    let op1 = Operator::new(
        "op1".to_string(),
        vec![],
        vec![],
        vec![planners_sas::numeric::numeric_task::AssignmentEffect::new(
            0,
            planners_sas::numeric::numeric_task::AssignmentOperation::Plus,
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
        vec![planners_sas::numeric::numeric_task::AssignmentEffect::new(
            0,
            planners_sas::numeric::numeric_task::AssignmentOperation::Plus,
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
        (0, 0),
    );

    let state_packer = IntDoublePacker::from_task(&task);
    let axiom_evaluator = AxiomEvaluator::new(&task, &state_packer);
    let mut state_registry = StateRegistry::new(&task, &state_packer, &axiom_evaluator);
    let initial_state = state_registry.get_initial_state();

    let d0 = state_registry
        .metric_delta_applying_operator(&initial_state, &task.get_operators()[0])
        .unwrap();
    let d1 = state_registry
        .metric_delta_applying_operator(&initial_state, &task.get_operators()[1])
        .unwrap();
    assert!((d0 - 0.5).abs() < 1e-12);
    assert!((d1 - 0.002).abs() < 1e-12);

    let min_cost = min_action_cost_from_initial_metric_deltas(
        &state_registry,
        &initial_state,
        task.get_operators(),
    );
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
        nodes_generated: 0,
        search_time: Duration::from_millis(100),
    };

    assert_eq!(result.status, SearchStatus::Failed);
    assert!(result.plan.is_none());
    assert_eq!(result.nodes_expanded, 0);
}
