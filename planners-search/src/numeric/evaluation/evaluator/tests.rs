use super::*;

fn create_test_state(id: usize) -> ConcreteState {
    ConcreteState::new(id)
}

struct TestEvaluator {
    name: String,
    value: f64,
}

impl TestEvaluator {
    fn new(name: &str, value: f64) -> Self {
        Self {
            name: name.to_string(),
            value,
        }
    }
}

impl Evaluator for TestEvaluator {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn evaluate_state(&self, eval_state: &mut EvaluationState<'_>) -> Result<f64, EvaluationError> {
        eval_state
            .result_mut()
            .set_heuristic_value(self.name(), self.value);
        Ok(self.value)
    }
}

#[test]
fn test_evaluation_state_basic() {
    let state = create_test_state(1);
    let state_owned = state.clone();
    let mut eval_state = EvaluationState::new(&state_owned, 5.0, false);

    assert_eq!(eval_state.result().g_value, 5.0);
    assert!(!eval_state.result().is_preferred);
    assert!(!eval_state.is_computed("test"));

    eval_state.mark_computed("test");
    assert!(eval_state.is_computed("test"));
}

#[test]
fn test_evaluator_basic() {
    let state = create_test_state(1);
    let evaluator = TestEvaluator::new("test_h", 42.0);
    let result = evaluator.evaluate(&state, 5.0).unwrap();

    assert_eq!(result.g_value, 5.0);
    assert_eq!(result.get_heuristic_value("test_h"), 42.0);
    assert!(!result.is_dead_end);
}

#[test]
fn test_evaluator_collection() {
    let state = create_test_state(1);
    let mut collection = EvaluatorCollection::new("test_collection".to_string());
    collection.add_evaluator(Rc::new(TestEvaluator::new("h1", 10.0)));
    collection.add_evaluator(Rc::new(TestEvaluator::new("h2", 20.0)));

    let result = collection.evaluate_all(&state, 5.0).unwrap();

    assert_eq!(result.get_heuristic_value("h1"), 10.0);
    assert_eq!(result.get_heuristic_value("h2"), 20.0);
    assert_eq!(result.g_value, 5.0);
}

#[test]
fn test_caching_behavior() {
    let state = create_test_state(1);
    let state_owned = state.clone();
    let mut eval_state = EvaluationState::new(&state_owned, 5.0, false);
    let evaluator = TestEvaluator::new("cached_h", 99.0);

    // First evaluation should compute
    assert!(!eval_state.is_computed("cached_h"));
    let value1 = eval_state.get_or_compute_heuristic(&evaluator).unwrap();
    assert_eq!(value1, 99.0);
    assert!(eval_state.is_computed("cached_h"));

    // Second evaluation should use cache
    let value2 = eval_state.get_or_compute_heuristic(&evaluator).unwrap();
    assert_eq!(value2, 99.0);
}
