use std::cell::Cell;
use std::rc::Rc;

use planforge_sas::numeric_task::OperatorIndex;
use planforge_sas::state_registry::StateRegistry;

use super::MaximumHeuristic;
use planforge_search::evaluation::{EvaluationError, EvaluationState, Heuristic};
use planforge_search::heuristic_factory::BuildContext;
use planforge_search::search::{AStarConfig, parse_search_spec};
use planforge_search::test_tasks::one_step_task;

struct Child {
    estimate: Result<f64, EvaluationError>,
    preferred: Vec<OperatorIndex>,
    revision: Rc<Cell<u64>>,
    every_pop: bool,
}

impl Heuristic for Child {
    fn evaluate(&mut self, _: &EvaluationState<'_, '_>) -> Result<f64, EvaluationError> {
        self.estimate.clone()
    }

    fn evaluate_with_preferred(
        &mut self,
        state: &EvaluationState<'_, '_>,
        preferred: &mut Vec<OperatorIndex>,
    ) -> Result<f64, EvaluationError> {
        preferred.extend_from_slice(&self.preferred);
        self.evaluate(state)
    }

    fn name(&self) -> &str {
        "test child"
    }

    fn revision(&self) -> u64 {
        self.revision.get()
    }

    fn reevaluate_on_every_pop(&self) -> bool {
        self.every_pop
    }
}

fn child(estimate: Result<f64, EvaluationError>) -> Box<Child> {
    Box::new(Child {
        estimate,
        preferred: Vec::new(),
        revision: Rc::new(Cell::new(0)),
        every_pop: false,
    })
}

#[test]
fn maximum_combines_estimates_preferences_and_independent_revisions() {
    crate::register_heuristics_for_tests();
    let first = Rc::new(Cell::new(100));
    let second = Rc::new(Cell::new(1));
    let mut maximum = MaximumHeuristic::new(
        "max".into(),
        vec![
            Box::new(Child {
                estimate: Ok(2.0),
                preferred: vec![OperatorIndex::from_usize(2)],
                revision: first.clone(),
                every_pop: false,
            }),
            Box::new(Child {
                estimate: Ok(5.0),
                preferred: vec![OperatorIndex::from_usize(0), OperatorIndex::from_usize(2)],
                revision: second.clone(),
                every_pop: true,
            }),
        ],
    );
    let task = one_step_task();
    let mut registry = StateRegistry::for_task(task.clone());
    let state = registry.get_initial_state();
    let eval = EvaluationState::new(&state, &task, &registry);
    assert_eq!(maximum.evaluate(&eval).unwrap(), 5.0);
    let mut preferred = Vec::new();
    assert_eq!(
        maximum
            .evaluate_with_preferred(&eval, &mut preferred)
            .unwrap(),
        5.0
    );
    assert_eq!(
        preferred,
        vec![OperatorIndex::from_usize(0), OperatorIndex::from_usize(2)]
    );
    assert!(maximum.reevaluate_on_every_pop());
    assert_eq!(maximum.revision(), 101);
    second.set(2);
    assert_eq!(
        maximum.revision(),
        102,
        "a smaller child's revision still invalidates the maximum"
    );
    first.set(101);
    assert_eq!(maximum.revision(), 103);
}

#[test]
fn maximum_propagates_child_errors_and_rejects_nan() {
    crate::register_heuristics_for_tests();
    let task = one_step_task();
    let mut registry = StateRegistry::for_task(task.clone());
    let state = registry.get_initial_state();
    let eval = EvaluationState::new(&state, &task, &registry);
    let error = EvaluationError::ComputationFailed("failure".into());
    let mut maximum = MaximumHeuristic::new("max".into(), vec![child(Err(error.clone()))]);
    assert_eq!(maximum.evaluate(&eval), Err(error));

    let mut maximum = MaximumHeuristic::new("max".into(), vec![child(Ok(f64::NAN))]);
    assert!(matches!(
        maximum.evaluate(&eval),
        Err(EvaluationError::ComputationFailed(_))
    ));
}

#[test]
fn maximum_of_a_dead_end_is_a_dead_end() {
    crate::register_heuristics_for_tests();
    let task = one_step_task();
    let mut registry = StateRegistry::for_task(task.clone());
    let state = registry.get_initial_state();
    let eval = EvaluationState::new(&state, &task, &registry);
    let mut maximum =
        MaximumHeuristic::new("max".into(), vec![child(Ok(3.0)), child(Ok(f64::INFINITY))]);
    assert_eq!(maximum.evaluate(&eval), Ok(f64::INFINITY));
}

#[test]
fn configured_maximum_builds_arbitrary_and_mixed_children() {
    crate::register_heuristics_for_tests();
    let task = one_step_task();
    let context = BuildContext::new(&task);
    let mut registry = StateRegistry::for_task(task.clone());
    let state = registry.get_initial_state();
    let eval = EvaluationState::new(&state, &task, &registry);
    for text in [
        "astar(max(ff()))",
        "astar(max([blind(), ff(), lmcutnumeric()]))",
        "astar(max([domain(), ff(), cartesian()]))",
        "astar(max([max(blind()), ff()]))",
        "astar(max([domain()]))",
        "astar(max([domain(),cartesian()]))",
    ] {
        let parsed = parse_search_spec(text).unwrap();
        let config = parsed
            .config::<AStarConfig>()
            .unwrap_or_else(|| panic!("{text} is an A* configuration"));
        let mut heuristic = config.heuristic.build(&context).unwrap();
        assert_eq!(heuristic.evaluate(&eval).unwrap(), 1.0, "{text}");
    }
}
