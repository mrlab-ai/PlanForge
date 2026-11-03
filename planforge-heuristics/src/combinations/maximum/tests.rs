use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::Relaxed;

use planforge_sas::numeric_task::OperatorIndex;
use planforge_sas::state_registry::StateRegistry;

use super::MaximumHeuristic;
use planforge_search::evaluation::{Estimate, EvaluationError, EvaluationState, Heuristic};
use planforge_search::heuristic_features::BuildContext;
use planforge_search::search::AStarConfig;
use planforge_search::test_tasks::one_step_task;

struct Child {
    estimate: Result<Estimate, EvaluationError>,
    preferred: Vec<OperatorIndex>,
    revision: Arc<AtomicU64>,
    every_pop: bool,
}

impl Heuristic for Child {
    fn evaluate(&mut self, _: &EvaluationState<'_, '_>) -> Result<Estimate, EvaluationError> {
        self.estimate.clone()
    }

    fn evaluate_with_preferred(
        &mut self,
        state: &EvaluationState<'_, '_>,
        preferred: &mut Vec<OperatorIndex>,
    ) -> Result<Estimate, EvaluationError> {
        preferred.extend_from_slice(&self.preferred);
        self.evaluate(state)
    }

    fn name(&self) -> &str {
        "test child"
    }

    fn revision(&self) -> u64 {
        self.revision.load(Relaxed)
    }

    fn reevaluate_on_every_pop(&self) -> bool {
        self.every_pop
    }
}

fn child(estimate: Result<Estimate, EvaluationError>) -> Box<Child> {
    Box::new(Child {
        estimate,
        preferred: Vec::new(),
        revision: Arc::new(AtomicU64::new(0)),
        every_pop: false,
    })
}

#[test]
fn maximum_combines_estimates_preferences_and_independent_revisions() {
    let first = Arc::new(AtomicU64::new(100));
    let second = Arc::new(AtomicU64::new(1));
    let mut maximum = MaximumHeuristic::new(
        "max".into(),
        vec![
            Box::new(Child {
                estimate: Ok(Estimate::new(2.0)),
                preferred: vec![OperatorIndex::from_usize(2)],
                revision: first.clone(),
                every_pop: false,
            }),
            Box::new(Child {
                estimate: Ok(Estimate::new(5.0)),
                preferred: vec![OperatorIndex::from_usize(0), OperatorIndex::from_usize(2)],
                revision: second.clone(),
                every_pop: true,
            }),
        ],
    );
    let task = one_step_task();
    let mut registry = StateRegistry::for_task(task.clone());
    let state = registry.register_initial_state();
    let eval = EvaluationState::new(&state, &task, &registry);
    assert_eq!(maximum.evaluate(&eval).unwrap(), Estimate::new(5.0));
    let mut preferred = Vec::new();
    assert_eq!(
        maximum
            .evaluate_with_preferred(&eval, &mut preferred)
            .unwrap(),
        Estimate::new(5.0)
    );
    assert_eq!(
        preferred,
        vec![OperatorIndex::from_usize(0), OperatorIndex::from_usize(2)]
    );
    assert!(maximum.reevaluate_on_every_pop());
    assert_eq!(maximum.revision(), 101);
    second.store(2, Relaxed);
    assert_eq!(
        maximum.revision(),
        102,
        "a smaller child's revision still invalidates the maximum"
    );
    first.store(101, Relaxed);
    assert_eq!(maximum.revision(), 103);
}

#[test]
fn maximum_propagates_child_errors() {
    let task = one_step_task();
    let mut registry = StateRegistry::for_task(task.clone());
    let state = registry.register_initial_state();
    let eval = EvaluationState::new(&state, &task, &registry);
    let error = EvaluationError::ComputationFailed("failure".into());
    let mut maximum = MaximumHeuristic::new("max".into(), vec![child(Err(error.clone()))]);
    assert_eq!(maximum.evaluate(&eval), Err(error));
}

#[test]
fn maximum_of_a_dead_end_is_a_dead_end() {
    let task = one_step_task();
    let mut registry = StateRegistry::for_task(task.clone());
    let state = registry.register_initial_state();
    let eval = EvaluationState::new(&state, &task, &registry);
    let mut maximum = MaximumHeuristic::new(
        "max".into(),
        vec![child(Ok(Estimate::new(3.0))), child(Ok(Estimate::DEAD_END))],
    );
    assert_eq!(maximum.evaluate(&eval), Ok(Estimate::DEAD_END));
}

#[test]
fn configured_maximum_builds_arbitrary_and_mixed_children() {
    let features = crate::features();
    let task = one_step_task();
    let context = BuildContext::new(task.clone());
    let mut registry = StateRegistry::for_task(task.clone());
    let state = registry.register_initial_state();
    let eval = EvaluationState::new(&state, &task, &registry);
    for text in [
        "astar(max(ff()))",
        "astar(max([blind(), ff(), lmcutnumeric()]))",
        "astar(max([domain(), ff(), cartesian()]))",
        "astar(max([max(blind()), ff()]))",
        "astar(max([domain()]))",
        "astar(max([domain(),cartesian()]))",
    ] {
        let parsed = features.parse_search(text).unwrap();
        let config = parsed
            .config::<AStarConfig>()
            .unwrap_or_else(|| panic!("{text} is an A* configuration"));
        let mut heuristic = config.heuristic.build(&context).unwrap();
        assert_eq!(heuristic.evaluate(&eval).unwrap().value(), 1.0, "{text}");
    }
}
