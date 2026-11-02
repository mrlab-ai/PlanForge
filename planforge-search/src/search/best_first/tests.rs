use std::sync::Arc;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicU64, AtomicUsize};

use planforge_sas::numeric_task::{
    ExplicitFact, ExplicitValueIndex, ExplicitVariable, Metric, NumericType, NumericValue,
    NumericVariable, Operator, OperatorCost, Task, TaskParts, VariableIndex,
};
use planforge_sas::state_registry::StateRegistry;

use super::*;
use crate::evaluation::BlindHeuristic;
use crate::evaluation::{Estimate, EvaluationError, EvaluationState, Heuristic};
use crate::search::SearchOutcome;
use crate::test_tasks::{one_step_task, start_to_goal_task};

/// Step `search` until it terminates.
fn finish(mut search: AStarSearch) -> SearchResult {
    loop {
        if let Some(outcome) = search.step().unwrap() {
            return search.progress.finish(outcome, &search.space);
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

    let task = Task::new(TaskParts {
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
        global_constraint: Some(ExplicitFact::propositional(0, 0)),
    });

    let task: Arc<Task> = Arc::new(task);
    let mut state_registry = StateRegistry::for_task(task.clone());
    let initial_state = state_registry.register_initial_state();

    let mut step_cost = |operator| {
        state_registry
            .successor(&initial_state, operator)
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

struct FailingHeuristic;

impl Heuristic for FailingHeuristic {
    fn evaluate(
        &mut self,
        _eval_state: &EvaluationState<'_, '_>,
    ) -> Result<Estimate, EvaluationError> {
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
    let task: Arc<Task> = Arc::new(Task::new(TaskParts {
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
        global_constraint: Some(ExplicitFact::propositional(0, 0)),
    }));
    let search = AStarSearch::new(task.clone(), SearchParams::new(Box::new(FailingHeuristic)));

    let error = search.run().unwrap_err();
    assert!(
        error
            .to_string()
            .contains("initial state evaluation failed")
    );
    assert!(format!("{error:#}").contains("construction deadline"));
}

struct RevisionControlledHeuristic {
    revision: Arc<AtomicU64>,
    calls: Arc<AtomicUsize>,
    reevaluate_on_every_pop: bool,
}

impl Heuristic for RevisionControlledHeuristic {
    fn evaluate(
        &mut self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<Estimate, EvaluationError> {
        self.calls.fetch_add(1, Relaxed);
        // Above the cheapest action cost, which search floors non-goal
        // values at, so a new revision raises the value.
        Ok(Estimate::new(if eval_state.is_goal() {
            0.0
        } else {
            1.0 + self.revision.load(Relaxed) as f64
        }))
    }

    fn revision(&self) -> u64 {
        self.revision.load(Relaxed)
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
    let revision = Arc::new(AtomicU64::new(0));
    let calls = Arc::new(AtomicUsize::new(0));
    let heuristic = RevisionControlledHeuristic {
        revision: Arc::clone(&revision),
        calls: Arc::clone(&calls),
        reevaluate_on_every_pop: false,
    };
    let task = one_step_task();
    let mut search = AStarSearch::new(
        task.clone(),
        SearchParams {
            mpd: true,
            ..SearchParams::new(Box::new(heuristic))
        },
    );

    search.initialize().unwrap();
    assert_eq!(calls.load(Relaxed), 1);

    revision.store(1, Relaxed);
    assert_eq!(search.step().unwrap(), None);
    assert_eq!(
        calls.load(Relaxed),
        2,
        "stale initial entry must be re-evaluated"
    );

    let result = finish(search);
    assert_eq!(solved_cost(&result), 1.0);
    assert_eq!(result.statistics.nodes.evaluated, 2);
    assert_eq!(result.statistics.evaluations, 3);
}

#[test]
fn static_astar_does_not_pay_for_revision_checks() {
    let revision = Arc::new(AtomicU64::new(0));
    let calls = Arc::new(AtomicUsize::new(0));
    let heuristic = RevisionControlledHeuristic {
        revision: Arc::clone(&revision),
        calls: Arc::clone(&calls),
        reevaluate_on_every_pop: false,
    };
    let task = one_step_task();
    let mut search = AStarSearch::new(task.clone(), SearchParams::new(Box::new(heuristic)));

    search.initialize().unwrap();
    revision.store(1, Relaxed);
    let result = finish(search);

    assert_eq!(solved_cost(&result), 1.0);
    assert_eq!(calls.load(Relaxed), 2);
    assert_eq!(
        result.statistics.nodes.evaluated,
        result.statistics.evaluations
    );
}

#[test]
fn uncached_mpd_reevaluates_every_popped_entry() {
    let revision = Arc::new(AtomicU64::new(0));
    let calls = Arc::new(AtomicUsize::new(0));
    let heuristic = RevisionControlledHeuristic {
        revision,
        calls: Arc::clone(&calls),
        reevaluate_on_every_pop: true,
    };
    let task = one_step_task();
    let mut search = AStarSearch::new(
        task.clone(),
        SearchParams {
            mpd: true,
            ..SearchParams::new(Box::new(heuristic))
        },
    );

    search.initialize().unwrap();
    let result = finish(search);

    assert_eq!(solved_cost(&result), 1.0);
    assert_eq!(calls.load(Relaxed), 4);
    assert_eq!(result.statistics.nodes.evaluated, 2);
    assert_eq!(result.statistics.evaluations, 4);
}

struct LazyHeuristic {
    estimate: Estimate,
    calls: Arc<AtomicUsize>,
}

impl Heuristic for LazyHeuristic {
    fn evaluate(&mut self, state: &EvaluationState<'_, '_>) -> Result<Estimate, EvaluationError> {
        self.calls.fetch_add(1, Relaxed);
        Ok(if state.is_goal() {
            Estimate::new(0.0)
        } else {
            self.estimate
        })
    }

    fn name(&self) -> &str {
        "lazy_test"
    }
}

#[test]
fn lazy_astar_defers_each_expansion_even_when_the_estimate_does_not_increase() {
    let task = one_step_task();
    let calls = Arc::new(AtomicUsize::new(0));
    let mut params = SearchParams::new(Box::new(BlindHeuristic::for_task(&task)));
    // Below the non-goal floor: the single evaluation path must preserve it.
    params.lazy = Some(Box::new(LazyHeuristic {
        estimate: Estimate::new(0.0),
        calls: calls.clone(),
    }));
    let mut search = AStarSearch::new(task.clone(), params);
    search.initialize().unwrap();
    assert_eq!(
        calls.load(Relaxed),
        0,
        "initialization evaluates only the primary heuristic"
    );
    assert_eq!(search.step().unwrap(), None);
    assert_eq!(
        calls.load(Relaxed),
        1,
        "first pop evaluates and requeues the initial entry"
    );
    assert_eq!(search.step().unwrap(), None);
    assert_eq!(
        calls.load(Relaxed),
        1,
        "second pop expands without another lazy evaluation"
    );
    assert_eq!(search.step().unwrap(), None);
    assert_eq!(
        calls.load(Relaxed),
        2,
        "goal entries also get a lazy evaluation"
    );
    let result = finish(search);
    assert_eq!(solved_cost(&result), 1.0);
    assert_eq!(result.statistics.nodes.expanded, 2);
    assert_eq!(result.statistics.nodes.generated, 1);
    assert_eq!(result.statistics.nodes.evaluated, 2);
    assert_eq!(
        result.statistics.evaluations, 4,
        "count both primary and lazy evaluations"
    );
    assert_eq!(calls.load(Relaxed), 2);
}

#[test]
fn lazy_astar_prunes_a_lazy_dead_end() {
    let task = one_step_task();
    let calls = Arc::new(AtomicUsize::new(0));
    let mut params = SearchParams::new(Box::new(BlindHeuristic::for_task(&task)));
    params.lazy = Some(Box::new(LazyHeuristic {
        estimate: Estimate::DEAD_END,
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
    assert_eq!(calls.load(Relaxed), 1);
}

#[test]
fn lazy_astar_propagates_evaluation_failures() {
    let task = one_step_task();
    let mut params = SearchParams::new(Box::new(BlindHeuristic::for_task(&task)));
    params.lazy = Some(Box::new(FailingHeuristic));
    let error = AStarSearch::new(task.clone(), params).run().unwrap_err();
    assert!(format!("{error:#}").contains("slow heuristic evaluation failed"));
    assert!(format!("{error:#}").contains("construction deadline"));
}

#[test]
fn mpd_and_lazy_compare_dynamic_estimates_with_the_primary_estimate() {
    let task = one_step_task();
    let revision = Arc::new(AtomicU64::new(0));
    let primary_calls = Arc::new(AtomicUsize::new(0));
    let lazy_calls = Arc::new(AtomicUsize::new(0));
    let mut params = SearchParams::new(Box::new(RevisionControlledHeuristic {
        revision: revision.clone(),
        calls: primary_calls.clone(),
        reevaluate_on_every_pop: false,
    }));
    params.mpd = true;
    params.lazy = Some(Box::new(LazyHeuristic {
        estimate: Estimate::new(5.0),
        calls: lazy_calls.clone(),
    }));
    let mut search = AStarSearch::new(task.clone(), params);
    search.initialize().unwrap();
    assert_eq!(search.step().unwrap(), None);
    assert_eq!(lazy_calls.load(Relaxed), 1);
    revision.store(1, Relaxed);
    // Primary h grows from 1 to 2 while the lazy entry already has h=5.
    assert_eq!(search.step().unwrap(), None);
    assert_eq!(
        primary_calls.load(Relaxed),
        3,
        "re-evaluate primary, then generate the goal"
    );
    assert_eq!(lazy_calls.load(Relaxed), 1);
    assert_eq!(search.step().unwrap(), None);
    assert_eq!(solved_cost(&finish(search)), 1.0);
}

#[test]
fn a_cheaper_duplicate_within_one_expansion_is_evaluated_and_queued_again() {
    // `expensive` is generated first.
    let task = start_to_goal_task(&[("expensive", 2), ("cheap", 1)]);
    let result = AStarSearch::new(
        task.clone(),
        SearchParams::new(Box::new(BlindHeuristic::for_task(&task))),
    )
    .run()
    .unwrap();
    assert_eq!(solved_cost(&result), 1.0);
    let nodes = result.statistics.nodes;
    assert_eq!(
        (nodes.expanded, nodes.generated, nodes.evaluated),
        (2, 2, 2)
    );
    assert_eq!(result.statistics.evaluations, 3);
}

#[test]
#[should_panic(expected = "non-negative")]
fn an_estimate_rejects_nan() {
    Estimate::new(f64::NAN);
}
