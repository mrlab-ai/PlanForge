use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;

use planforge_sas::numeric_task::{
    ExplicitFact, ExplicitValueIndex, ExplicitVariable, Metric, NumericRootTask,
    NumericRootTaskParts, NumericType, NumericValue, NumericVariable, Operator, OperatorCost,
    TaskRef, VariableIndex,
};
use planforge_sas::state_registry::StateRegistry;

use super::*;
use crate::evaluation::heuristic::BlindHeuristic;
use crate::evaluation::{EvaluationError, EvaluationState, Heuristic};
use crate::search::SearchOutcome;
use crate::test_tasks::one_step_task;

/// Step `search` until it terminates.
fn finish(mut search: AStarSearch<'_>) -> SearchResult {
    loop {
        if let Step::Terminate(termination) = search.step().unwrap() {
            return search.core.finish(termination);
        }
    }
}

fn solved_cost(result: &SearchResult) -> f64 {
    match result.outcome {
        SearchOutcome::Solved { cost, .. } => cost,
        ref outcome => panic!("expected a plan, got {outcome:?}"),
    }
}

#[test]
fn metric_operator_costs_are_the_constant_increments() {
    // Metric var 0 (cost), incremented by constants 1 and 2.
    let version = 4;
    let metric = Metric::new(true, Some(VariableIndex::from_usize(0)));

    let variables = vec![ExplicitVariable::new(
        2,
        "v".to_string(),
        vec!["a".to_string(), "b".to_string()],
        None,
        ExplicitValueIndex::new(0),
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
            VariableIndex::from_usize(0),
            planforge_sas::numeric_task::AssignmentOperation::Plus,
            VariableIndex::from_usize(1),
            false,
            vec![],
        )],
        OperatorCost::new(1),
    );
    let op2 = Operator::new(
        "op2".to_string(),
        vec![],
        vec![],
        vec![planforge_sas::numeric_task::AssignmentEffect::new(
            VariableIndex::from_usize(0),
            planforge_sas::numeric_task::AssignmentOperation::Plus,
            VariableIndex::from_usize(2),
            false,
            vec![],
        )],
        OperatorCost::new(1),
    );

    let task = NumericRootTask::new(NumericRootTaskParts {
        version,
        metric,
        variables,
        numeric_variables,
        goals: vec![],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![
            NumericValue::new(0.0),
            NumericValue::new(0.5),
            NumericValue::new(0.002),
        ],
        operators: vec![op1, op2],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    let task: TaskRef = Arc::new(task);
    let mut state_registry = StateRegistry::for_task(task.clone());
    let initial_state = state_registry.get_initial_state();

    let mut step_cost = |operator| {
        state_registry
            .get_successor_state_with_buffers_and_cost(
                &initial_state,
                operator,
                &mut Vec::new(),
                &mut Vec::new(),
            )
            .unwrap()
            .1
    };
    let d0 = step_cost(&task.operators()[0]);
    let d1 = step_cost(&task.operators()[1]);
    assert!((d0 - 0.5).abs() < 1e-12);
    assert!((d1 - 0.002).abs() < 1e-12);

    let operator_costs = task.effective_operator_costs();
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
fn progress_values_print_six_decimals() {
    assert_eq!(format_progress_value(95.4940004), "95.494000");
    assert_eq!(format_progress_value(95.49400049), "95.494000");
}

struct FailingHeuristic;

impl Heuristic for FailingHeuristic {
    fn evaluate(&mut self, _eval_state: &EvaluationState<'_, '_>) -> Result<f64, EvaluationError> {
        Err(EvaluationError::ComputationFailed(
            "construction deadline".to_string(),
        ))
    }

    fn name(&self) -> &str {
        "failing"
    }
}

#[test]
fn initial_evaluation_error_is_not_reported_as_no_solution() {
    let task: TaskRef = Arc::new(NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(false, None),
        variables: vec![ExplicitVariable::new(
            1,
            "v".to_string(),
            vec!["value".to_string()],
            None,
            ExplicitValueIndex::new(0),
        )],
        numeric_variables: vec![],
        goals: vec![],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![],
        operators: vec![],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    }));
    let search = AStarSearch::new(
        task.clone(),
        SearchParams {
            heuristic: Box::new(FailingHeuristic),
            lazy: None,
            mpd: false,
            time_limit: None,
            max_memory_bytes: None,
        },
    );

    let error = search.run().unwrap_err();
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
    fn evaluate(&mut self, eval_state: &EvaluationState<'_, '_>) -> Result<f64, EvaluationError> {
        self.calls.set(self.calls.get() + 1);
        // Above the cheapest action cost, which search floors non-goal
        // values at, so a new revision raises the value.
        Ok(if eval_state.is_goal() {
            0.0
        } else {
            1.0 + self.revision.get() as f64
        })
    }

    fn revision(&self) -> u64 {
        self.revision.get()
    }

    fn reevaluate_on_every_pop(&self) -> bool {
        self.reevaluate_on_every_pop
    }

    fn name(&self) -> &str {
        "revision_controlled"
    }
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
    let mut search = AStarSearch::new(
        task.clone(),
        SearchParams {
            heuristic: Box::new(heuristic),
            lazy: None,
            mpd: true,
            time_limit: None,
            max_memory_bytes: None,
        },
    );

    search.core.initialize().unwrap();
    assert_eq!(calls.get(), 1);

    revision.set(1);
    assert_eq!(search.step().unwrap(), Step::Continue);
    assert_eq!(calls.get(), 2, "stale initial entry must be re-evaluated");

    let result = finish(search);
    assert_eq!(solved_cost(&result), 1.0);
    assert_eq!(result.statistics.nodes.evaluated, 2);
    assert_eq!(result.statistics.evaluations, 3);
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
    let mut search = AStarSearch::new(
        task.clone(),
        SearchParams {
            heuristic: Box::new(heuristic),
            lazy: None,
            mpd: false,
            time_limit: None,
            max_memory_bytes: None,
        },
    );

    search.core.initialize().unwrap();
    revision.set(1);
    let result = finish(search);

    assert_eq!(solved_cost(&result), 1.0);
    assert_eq!(calls.get(), 2);
    assert_eq!(
        result.statistics.nodes.evaluated,
        result.statistics.evaluations
    );
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
    let mut search = AStarSearch::new(
        task.clone(),
        SearchParams {
            heuristic: Box::new(heuristic),
            lazy: None,
            mpd: true,
            time_limit: None,
            max_memory_bytes: None,
        },
    );

    search.core.initialize().unwrap();
    let result = finish(search);

    assert_eq!(solved_cost(&result), 1.0);
    assert_eq!(calls.get(), 4);
    assert_eq!(result.statistics.nodes.evaluated, 2);
    assert_eq!(result.statistics.evaluations, 4);
}

struct LazyHeuristic {
    estimate: f64,
    calls: Rc<Cell<usize>>,
}

impl Heuristic for LazyHeuristic {
    fn evaluate(&mut self, state: &EvaluationState<'_, '_>) -> Result<f64, EvaluationError> {
        self.calls.set(self.calls.get() + 1);
        Ok(if state.is_goal() { 0.0 } else { self.estimate })
    }

    fn name(&self) -> &str {
        "lazy_test"
    }
}

#[test]
fn lazy_astar_defers_each_expansion_even_when_the_estimate_does_not_increase() {
    let task = one_step_task();
    let calls = Rc::new(Cell::new(0));
    let mut params = SearchParams::new(Box::new(BlindHeuristic::for_task(&*task)));
    // Below the non-goal floor: the single evaluation path must preserve it.
    params.lazy = Some(Box::new(LazyHeuristic {
        estimate: 0.0,
        calls: calls.clone(),
    }));
    let mut search = AStarSearch::new(task.clone(), params);
    search.core.initialize().unwrap();
    assert_eq!(
        calls.get(),
        0,
        "initialization evaluates only the primary heuristic"
    );
    assert_eq!(search.step().unwrap(), Step::Continue);
    assert_eq!(
        calls.get(),
        1,
        "first pop evaluates and requeues the initial entry"
    );
    assert_eq!(search.step().unwrap(), Step::Continue);
    assert_eq!(
        calls.get(),
        1,
        "second pop expands without another lazy evaluation"
    );
    assert_eq!(search.step().unwrap(), Step::Continue);
    assert_eq!(calls.get(), 2, "goal entries also get a lazy evaluation");
    let result = finish(search);
    assert_eq!(solved_cost(&result), 1.0);
    assert_eq!(result.statistics.nodes.expanded, 2);
    assert_eq!(result.statistics.nodes.generated, 1);
    assert_eq!(result.statistics.nodes.evaluated, 2);
    assert_eq!(
        result.statistics.evaluations, 4,
        "count both primary and lazy evaluations"
    );
    assert_eq!(calls.get(), 2);
}

#[test]
fn lazy_astar_prunes_a_lazy_dead_end() {
    let task = one_step_task();
    let calls = Rc::new(Cell::new(0));
    let mut params = SearchParams::new(Box::new(BlindHeuristic::for_task(&*task)));
    params.lazy = Some(Box::new(LazyHeuristic {
        estimate: f64::INFINITY,
        calls: calls.clone(),
    }));
    let result = AStarSearch::new(task.clone(), params).run().unwrap();
    assert_eq!(result.outcome, SearchOutcome::Unsolvable);
    assert_eq!(result.statistics.nodes.expanded, 0);
    assert_eq!(result.statistics.dead_ends, 1);
    assert_eq!(
        result.statistics.evaluations, 2,
        "count a lazy dead-end evaluation too"
    );
    assert_eq!(calls.get(), 1);
}

#[test]
fn lazy_astar_propagates_evaluation_failures() {
    let task = one_step_task();
    let mut params = SearchParams::new(Box::new(BlindHeuristic::for_task(&*task)));
    params.lazy = Some(Box::new(FailingHeuristic));
    let error = AStarSearch::new(task.clone(), params).run().unwrap_err();
    assert!(format!("{error:#}").contains("slow heuristic evaluation failed"));
    assert!(format!("{error:#}").contains("construction deadline"));
}

#[test]
fn mpd_and_lazy_compare_dynamic_estimates_with_the_primary_estimate() {
    let task = one_step_task();
    let revision = Rc::new(Cell::new(0));
    let primary_calls = Rc::new(Cell::new(0));
    let lazy_calls = Rc::new(Cell::new(0));
    let mut params = SearchParams::new(Box::new(RevisionControlledHeuristic {
        revision: revision.clone(),
        calls: primary_calls.clone(),
        reevaluate_on_every_pop: false,
    }));
    params.mpd = true;
    params.lazy = Some(Box::new(LazyHeuristic {
        estimate: 5.0,
        calls: lazy_calls.clone(),
    }));
    let mut search = AStarSearch::new(task.clone(), params);
    search.core.initialize().unwrap();
    assert_eq!(search.step().unwrap(), Step::Continue);
    assert_eq!(lazy_calls.get(), 1);
    revision.set(1);
    // Primary h grows from 1 to 2 while the lazy entry already has h=5.
    assert_eq!(search.step().unwrap(), Step::Continue);
    assert_eq!(
        primary_calls.get(),
        3,
        "re-evaluate primary, then generate the goal"
    );
    assert_eq!(lazy_calls.get(), 1);
    assert_eq!(search.step().unwrap(), Step::Continue);
    assert_eq!(solved_cost(&finish(search)), 1.0);
}
