use super::*;

fn create_test_state(id: usize) -> ConcreteState {
    ConcreteState::new(id)
}

struct TestHeuristic {
    name: String,
    value: f64,
    is_reliable: bool,
}

impl TestHeuristic {
    fn new(name: &str, value: f64, is_reliable: bool) -> Self {
        Self {
            name: name.to_string(),
            value,
            is_reliable,
        }
    }
}

impl Heuristic for TestHeuristic {
    fn compute_heuristic(
        &self,
        _eval_state: &EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError> {
        if self.value.is_infinite() && self.value.is_sign_positive() {
            Err(EvaluationError::DeadEnd {
                reliable: self.is_reliable,
            })
        } else {
            Ok(self.value)
        }
    }

    fn heuristic_name(&self) -> String {
        self.name.clone()
    }

    fn get_cost_type(&self) -> CostType {
        CostType::Normal
    }
}

#[test]
fn test_blind_heuristic() {
    let state = create_test_state(1);
    let heuristic = BlindHeuristic::new(None);
    let result = heuristic.evaluate(&state, 5.0).unwrap();

    assert_eq!(result.get_heuristic_value("blind_heuristic"), 1.0);
    assert!(!result.is_dead_end);
}

#[test]
fn test_heuristic_dead_end_reliable() {
    let state = create_test_state(1);
    let heuristic = TestHeuristic::new("dead_end_h", f64::INFINITY, true);
    let error = heuristic.evaluate(&state, 5.0).unwrap_err();

    match error {
        EvaluationError::DeadEnd { reliable } => assert!(reliable),
        _ => panic!("Expected dead end error"),
    }
}

#[test]
fn test_heuristic_dead_end_unreliable() {
    let state = create_test_state(1);
    let heuristic = TestHeuristic::new("unreliable_h", f64::INFINITY, false);
    let error = heuristic.evaluate(&state, 5.0).unwrap_err();

    match error {
        EvaluationError::DeadEnd { reliable } => assert!(!reliable),
        _ => panic!("Expected dead end error"),
    }
}

#[test]
fn test_heuristic_normal_value() {
    let state = create_test_state(1);
    let heuristic = TestHeuristic::new("normal_h", 42.0, false);
    let result = heuristic.evaluate(&state, 5.0).unwrap();

    assert_eq!(result.get_heuristic_value("normal_h"), 42.0);
    assert_eq!(result.get_f_value("normal_h"), 47.0); // g + h = 5 + 42
    assert!(!result.is_dead_end);
}

#[test]
fn test_cached_heuristic() {
    let state = create_test_state(1);
    let inner_heuristic = TestHeuristic::new("inner_h", 25.0, false);
    let cached_heuristic = CachedHeuristic::new(inner_heuristic, Some("cached_test".to_string()));

    let result = cached_heuristic.evaluate(&state, 3.0).unwrap();

    assert_eq!(result.get_heuristic_value("cached_test"), 25.0);
    assert_eq!(result.get_f_value("cached_test"), 28.0);
}
