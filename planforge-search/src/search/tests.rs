use super::*;
use crate::evaluation::{EvaluationError, EvaluationState, Heuristic};

use planforge_sas::numeric_task::{
    Effect, ExplicitFact, ExplicitVariable, Metric, NumericRootTask, NumericType, NumericVariable,
    Operator, TaskRef,
};
use planforge_sas::state_registry::StateRegistry;
use std::cell::Cell;
use std::rc::Rc;
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
        vec![planforge_sas::numeric_task::AssignmentEffect::new(
            0,
            planforge_sas::numeric_task::AssignmentOperation::Plus,
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
        vec![planforge_sas::numeric_task::AssignmentEffect::new(
            0,
            planforge_sas::numeric_task::AssignmentOperation::Plus,
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

struct FailingHeuristic;

impl Heuristic for FailingHeuristic {
    fn compute_heuristic(
        &self,
        _eval_state: &EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError> {
        Err(EvaluationError::ComputationFailed(
            "construction deadline".to_string(),
        ))
    }
}

#[test]
fn initial_evaluation_error_is_not_reported_as_no_solution() {
    let task: TaskRef = Arc::new(NumericRootTask::new(
        4,
        Metric::new(false, None),
        vec![ExplicitVariable::new(
            1,
            "v".to_string(),
            vec!["value".to_string()],
            None,
            0,
        )],
        vec![],
        vec![],
        vec![],
        vec![0],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    ));
    let registry = StateRegistry::for_task(task.clone());
    let mut search = AStarSearch::new(task, registry, Some(Box::new(FailingHeuristic)), None, None);

    let error = search.search().unwrap_err();
    assert!(
        error
            .to_string()
            .contains("initial state evaluation failed")
    );
    assert!(format!("{error:#}").contains("construction deadline"));
}

struct RevisionControlledHeuristic {
    revision: Rc<Cell<u64>>,
    calls: Rc<Cell<usize>>,
    reevaluate_on_every_pop: bool,
}

impl Heuristic for RevisionControlledHeuristic {
    fn compute_heuristic(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError> {
        self.calls.set(self.calls.get() + 1);
        Ok(if eval_state.is_goal() {
            0.0
        } else {
            self.revision.get() as f64
        })
    }

    fn revision(&self) -> u64 {
        self.revision.get()
    }

    fn reevaluate_on_every_pop(&self) -> bool {
        self.reevaluate_on_every_pop
    }
}

fn one_step_task() -> TaskRef<'static> {
    Arc::new(NumericRootTask::new(
        4,
        Metric::new(false, None),
        vec![ExplicitVariable::new(
            2,
            "location".to_string(),
            vec!["start".to_string(), "goal".to_string()],
            None,
            0,
        )],
        vec![],
        vec![ExplicitFact::new(0, 1)],
        vec![],
        vec![0],
        vec![],
        vec![Operator::new(
            "finish".to_string(),
            vec![ExplicitFact::new(0, 0)],
            vec![Effect::new(vec![], 0, Some(0), 1)],
            vec![],
            1,
        )],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    ))
}

#[test]
fn mpd_reevaluates_and_reinserts_a_stale_open_entry() {
    let revision = Rc::new(Cell::new(0));
    let calls = Rc::new(Cell::new(0));
    let heuristic = RevisionControlledHeuristic {
        revision: Rc::clone(&revision),
        calls: Rc::clone(&calls),
        reevaluate_on_every_pop: false,
    };
    let task = one_step_task();
    let registry = StateRegistry::for_task(task.clone());
    let mut search =
        AStarSearch::new_with_mpd(task, registry, Some(Box::new(heuristic)), None, None, true);

    search.initialize().unwrap();
    assert_eq!(calls.get(), 1);

    revision.set(1);
    assert_eq!(search.step().unwrap(), SearchStatus::InProgress);
    assert_eq!(calls.get(), 2, "stale initial entry must be re-evaluated");

    let result = loop {
        match search.step().unwrap() {
            SearchStatus::InProgress => {}
            terminal => break search.finish(terminal),
        }
    };
    assert!(matches!(result.status, SearchStatus::Solved(_)));
    assert_eq!(result.solution_cost, Some(1.0));
    assert_eq!(result.nodes_evaluated, 2);
    assert_eq!(result.evaluations, 3);
}

#[test]
fn static_astar_does_not_pay_for_revision_checks() {
    let revision = Rc::new(Cell::new(0));
    let calls = Rc::new(Cell::new(0));
    let heuristic = RevisionControlledHeuristic {
        revision: Rc::clone(&revision),
        calls: Rc::clone(&calls),
        reevaluate_on_every_pop: false,
    };
    let task = one_step_task();
    let registry = StateRegistry::for_task(task.clone());
    let mut search = AStarSearch::new(task, registry, Some(Box::new(heuristic)), None, None);

    search.initialize().unwrap();
    revision.set(1);
    let result = loop {
        match search.step().unwrap() {
            SearchStatus::InProgress => {}
            terminal => break search.finish(terminal),
        }
    };

    assert!(matches!(result.status, SearchStatus::Solved(_)));
    assert_eq!(calls.get(), 2);
    assert_eq!(result.nodes_evaluated, result.evaluations);
}

#[test]
fn uncached_mpd_reevaluates_every_popped_entry() {
    let revision = Rc::new(Cell::new(0));
    let calls = Rc::new(Cell::new(0));
    let heuristic = RevisionControlledHeuristic {
        revision,
        calls: Rc::clone(&calls),
        reevaluate_on_every_pop: true,
    };
    let task = one_step_task();
    let registry = StateRegistry::for_task(task.clone());
    let mut search =
        AStarSearch::new_with_mpd(task, registry, Some(Box::new(heuristic)), None, None, true);

    search.initialize().unwrap();
    let result = loop {
        match search.step().unwrap() {
            SearchStatus::InProgress => {}
            terminal => break search.finish(terminal),
        }
    };

    assert!(matches!(result.status, SearchStatus::Solved(_)));
    assert_eq!(calls.get(), 4);
    assert_eq!(result.nodes_evaluated, 2);
    assert_eq!(result.evaluations, 4);
}
