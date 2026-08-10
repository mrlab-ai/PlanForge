use super::*;

use planforge_sas::numeric_task::{
    Effect, ExplicitFact, ExplicitVariable, Metric, NumericRootTask,
};
use std::sync::Arc;

/// Chain task: one variable stepping `0 -> 1 -> 2`, goal `v = 2`, unit costs.
/// The true goal distance is 2 from the initial state and 1 after one step.
fn chain_task() -> TaskRef<'static> {
    let variables = vec![ExplicitVariable::new(
        3,
        "v".to_string(),
        vec!["v=0".to_string(), "v=1".to_string(), "v=2".to_string()],
        None,
        0,
    )];
    let step = |name: &str, from: usize, to: usize| {
        Operator::new(
            name.to_string(),
            vec![ExplicitFact::propositional(0, from)],
            vec![Effect::new(vec![], 0, Some(from), to)],
            vec![],
            1,
        )
    };

    Arc::new(NumericRootTask::new(
        3,
        Metric::new(true, None),
        variables,
        vec![],
        vec![ExplicitFact::propositional(0, 2)],
        vec![],
        vec![0],
        vec![],
        vec![step("step_0_1", 0, 1), step("step_1_2", 1, 2)],
        vec![],
        vec![],
        vec![],
        ExplicitFact::propositional(0, 0),
    ))
}

/// Returns a fixed value for every non-goal state, whatever the task says.
struct ConstantHeuristic {
    value: f64,
    name: String,
}

impl ConstantHeuristic {
    fn boxed(value: f64, name: &str) -> Box<dyn Heuristic + 'static> {
        Box::new(Self {
            value,
            name: name.to_string(),
        })
    }
}

impl Heuristic for ConstantHeuristic {
    fn compute_heuristic(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError> {
        Ok(if eval_state.is_goal() {
            0.0
        } else {
            self.value
        })
    }

    fn heuristic_name(&self) -> String {
        self.name.clone()
    }
}

fn evaluate_initial_state(
    task: TaskRef<'static>,
    inner: Option<Box<dyn Heuristic + 'static>>,
) -> Result<f64, EvaluationError> {
    let heuristic = CheckAdmissibleHeuristic::new(inner, task.clone())
        .expect("the chain task has finite non-negative costs");
    let mut registry = StateRegistry::for_task(task.clone());
    let initial_state = registry.get_initial_state();
    let mut eval_state =
        EvaluationState::new_with_registry(&initial_state, 0.0, false, &*task, &registry);
    eval_state.set_is_goal(false);
    heuristic.compute_heuristic(&eval_state)
}

#[test]
fn rejects_an_inadmissible_heuristic() {
    let error =
        evaluate_initial_state(chain_task(), Some(ConstantHeuristic::boxed(1000.0, "huge")))
            .expect_err("h = 1000 must be rejected against a true goal distance of 2");

    let EvaluationError::ComputationFailed(message) = error else {
        panic!("expected a ComputationFailed error, got {error:?}");
    };
    assert!(message.contains("huge"), "{message}");
    assert!(message.contains("inadmissible"), "{message}");
    assert!(message.contains("1000"), "{message}");
    assert!(message.contains("h* = 2"), "{message}");
    assert!(message.contains("h - h* = 998"), "{message}");
}

#[test]
fn accepts_an_admissible_heuristic() {
    let h_value = evaluate_initial_state(chain_task(), Some(ConstantHeuristic::boxed(1.0, "one")))
        .expect("h = 1 is below the true goal distance of 2");
    assert_eq!(h_value, 1.0);
}

#[test]
fn accepts_a_perfect_heuristic() {
    let h_value =
        evaluate_initial_state(chain_task(), Some(ConstantHeuristic::boxed(2.0, "exact")))
            .expect("h = 2 equals the true goal distance and stays admissible");
    assert_eq!(h_value, 2.0);
}

#[test]
fn rejects_a_heuristic_that_invents_a_dead_end() {
    let error = evaluate_initial_state(
        chain_task(),
        Some(ConstantHeuristic::boxed(f64::INFINITY, "fake_dead_end")),
    )
    .expect_err("a solvable state must not be reported as a dead end");
    assert!(matches!(error, EvaluationError::ComputationFailed(_)));
}

#[test]
fn materializes_blind_when_no_inner_heuristic_is_given() {
    let h_value = evaluate_initial_state(chain_task(), None)
        .expect("the blind estimate is the minimum action cost");
    assert_eq!(h_value, 1.0);
}

#[test]
fn wrapper_name_differs_from_the_wrapped_name() {
    let heuristic =
        CheckAdmissibleHeuristic::new(Some(ConstantHeuristic::boxed(1.0, "inner")), chain_task())
            .expect("the chain task has finite non-negative costs");
    assert_eq!(heuristic.heuristic_name(), "check_admissible_inner");
}

#[test]
fn goal_distance_is_computed_per_state() {
    let task = chain_task();
    let heuristic = CheckAdmissibleHeuristic::new(None, task.clone())
        .expect("the chain task has finite non-negative costs");
    let mut registry = StateRegistry::for_task(task.clone());
    let initial_state = registry.get_initial_state();
    let successor = registry
        .get_successor_state(&initial_state, &task.get_operators()[0])
        .expect("`step_0_1` is applicable in the initial state");

    let mut oracle = heuristic.oracle.borrow_mut();
    assert_eq!(
        oracle
            .goal_distance(&initial_state, &registry)
            .expect("the goal is reachable from the initial state"),
        2.0
    );
    assert_eq!(
        oracle
            .goal_distance(&successor, &registry)
            .expect("the goal is reachable after one step"),
        1.0
    );
}

#[test]
fn rejects_a_task_with_a_negative_operator_cost() {
    let error = minimum_action_cost(&[1.0, -1.0]).expect_err("negative costs break Dijkstra");
    assert!(error.contains("operator 1"), "{error}");
}
