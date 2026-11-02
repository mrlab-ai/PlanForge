use super::*;
use planforge_sas::numeric_task::Operator;

use planforge_sas::numeric_task::{
    Effect, ExplicitFact, ExplicitValueIndex, ExplicitVariable, Metric, OperatorCost, Task,
    TaskParts, VariableIndex,
};
use std::sync::Arc;

/// Chain task: one variable stepping `0 -> 1 -> 2`, goal `v = 2`, unit costs.
/// The true goal distance is 2 from the initial state and 1 after one step.
fn chain_task() -> Arc<Task> {
    let variables = vec![ExplicitVariable::new(
        3,
        "v".to_string(),
        vec!["v=0".to_string(), "v=1".to_string(), "v=2".to_string()],
        None,
        ExplicitValueIndex::new(0),
    )];
    let step = |name: &str, from: usize, to: usize| {
        Operator::new(
            name.to_string(),
            vec![ExplicitFact::propositional(0, from)],
            vec![Effect::new(
                vec![],
                VariableIndex::new(0),
                Some(ExplicitValueIndex::from_usize(from)),
                ExplicitValueIndex::from_usize(to),
            )],
            vec![],
            OperatorCost::new(1),
        )
    };

    Arc::new(Task::new(TaskParts {
        metric: Metric::new(true, None),
        variables,
        numeric_variables: vec![],
        goals: vec![ExplicitFact::propositional(0, 2)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![],
        operators: vec![step("step_0_1", 0, 1), step("step_1_2", 1, 2)],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: Some(ExplicitFact::propositional(0, 0)),
    }))
}

/// Returns a fixed value for every non-goal state, whatever the task says.
struct ConstantHeuristic {
    value: f64,
    name: String,
}

impl ConstantHeuristic {
    fn boxed(value: f64, name: &str) -> Box<dyn Heuristic> {
        Box::new(Self {
            value,
            name: name.to_string(),
        })
    }
}

impl Heuristic for ConstantHeuristic {
    fn evaluate(
        &mut self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<Estimate, EvaluationError> {
        Ok(Estimate::new(if eval_state.is_goal() {
            0.0
        } else {
            self.value
        }))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

fn evaluate_initial_state(
    task: Arc<Task>,
    inner: Box<dyn Heuristic>,
) -> Result<f64, EvaluationError> {
    let mut heuristic = CheckAdmissibleHeuristic::new(inner, task.clone())
        .expect("the chain task has finite non-negative costs");
    let mut registry = StateRegistry::for_task(task.clone());
    let initial_state = registry.get_initial_state();
    let eval_state = EvaluationState::new(&initial_state, &task, &registry);
    heuristic.evaluate(&eval_state).map(Estimate::value)
}

#[test]
fn rejects_an_inadmissible_heuristic() {
    let error = evaluate_initial_state(chain_task(), ConstantHeuristic::boxed(1000.0, "huge"))
        .expect_err("h = 1000 must be rejected against a true goal distance of 2");

    assert!(
        matches!(error, EvaluationError::Inadmissible { .. }),
        "{error}"
    );
    let message = error.to_string();
    assert!(message.contains("huge"), "{message}");
    assert!(message.contains("inadmissible"), "{message}");
    assert!(message.contains("1000"), "{message}");
    assert!(message.contains("h* = 2"), "{message}");
    assert!(message.contains("h - h* = 998"), "{message}");
}

#[test]
fn accepts_an_admissible_heuristic() {
    let h_value = evaluate_initial_state(chain_task(), ConstantHeuristic::boxed(1.0, "one"))
        .expect("h = 1 is below the true goal distance of 2");
    assert_eq!(h_value, 1.0);
}

#[test]
fn accepts_a_perfect_heuristic() {
    let h_value = evaluate_initial_state(chain_task(), ConstantHeuristic::boxed(2.0, "exact"))
        .expect("h = 2 equals the true goal distance and stays admissible");
    assert_eq!(h_value, 2.0);
}

#[test]
fn rejects_a_heuristic_that_invents_a_dead_end() {
    let error = evaluate_initial_state(
        chain_task(),
        ConstantHeuristic::boxed(f64::INFINITY, "fake_dead_end"),
    )
    .expect_err("a solvable state must not be reported as a dead end");
    assert!(matches!(error, EvaluationError::Inadmissible { .. }));
}

#[test]
fn checks_a_blind_heuristic() {
    let h_value = evaluate_initial_state(
        chain_task(),
        Box::new(planforge_search::evaluation::BlindHeuristic::for_task(
            &chain_task(),
        )),
    )
    .expect("the blind estimate is the minimum action cost");
    assert_eq!(h_value, 1.0);
}

#[test]
fn wrapper_name_differs_from_the_wrapped_name() {
    let heuristic =
        CheckAdmissibleHeuristic::new(ConstantHeuristic::boxed(1.0, "inner"), chain_task())
            .expect("the chain task has finite non-negative costs");
    assert_eq!(heuristic.name(), "check_admissible_inner");
}

#[test]
fn goal_distance_is_computed_per_state() {
    let task = chain_task();
    let mut heuristic = CheckAdmissibleHeuristic::new(
        Box::new(planforge_search::evaluation::BlindHeuristic::for_task(
            &task,
        )),
        task.clone(),
    )
    .expect("the chain task has finite non-negative costs");
    let mut registry = StateRegistry::for_task(task.clone());
    let initial_state = registry.get_initial_state();
    let successor = registry
        .get_successor_state(&initial_state, &task.operators()[0])
        .expect("`step_0_1` is applicable in the initial state");

    let oracle = &mut heuristic.oracle;
    assert_eq!(
        oracle
            .goal_distance(registry.view(&initial_state), initial_state.get_id())
            .expect("the goal is reachable from the initial state"),
        2.0
    );
    assert_eq!(
        oracle
            .goal_distance(registry.view(&successor), successor.get_id())
            .expect("the goal is reachable after one step"),
        1.0
    );
}

#[test]
fn the_wrapped_heuristic_is_one_positional_heuristic() {
    let spec =
        planforge_search::config::parse_call("check_admissible([domain_collection()])").unwrap();
    let error = super::single_wrapped_heuristic_spec("check_admissible", &spec.args).unwrap_err();
    assert_eq!(error.to_string(), "expected heuristic, got a list");
    let spec = planforge_search::config::parse_call("check_admissible(h=blind())").unwrap();
    let error = super::single_wrapped_heuristic_spec("check_admissible", &spec.args).unwrap_err();
    assert!(error.to_string().contains("positionally"), "{error}");
}
