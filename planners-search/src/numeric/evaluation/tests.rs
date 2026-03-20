use super::*;

fn create_test_state(id: usize) -> ConcreteState {
    ConcreteState::new(id)
}

#[test]
fn test_evaluation_result_basic() {
    let state = create_test_state(1);
    let mut result = EvaluationResult::new(state, 5.0, false);

    assert_eq!(result.g_value, 5.0);
    assert!(!result.is_preferred);
    assert!(!result.is_dead_end);
    assert!(!result.has_heuristics());

    // Test heuristic value setting
    result.set_heuristic_value("h1".to_string(), 10.0);
    assert_eq!(result.get_heuristic_value("h1"), 10.0);
    assert_eq!(result.get_f_value("h1"), 15.0);
    assert!(result.has_heuristics());

    // Test infinite heuristic (dead end)
    result.set_heuristic_value("h2".to_string(), f64::INFINITY);
    assert!(result.is_heuristic_infinite("h2"));
    assert!(result.is_dead_end);
    assert!(!result.is_reliable_dead_end);
}

#[test]
fn test_evaluation_result_merge() {
    let state = create_test_state(1);

    let mut result1 = EvaluationResult::new(state.clone(), 5.0, false);
    result1.set_heuristic_value("h1".to_string(), 10.0);

    let mut result2 = EvaluationResult::new(state, 5.0, false);
    result2.set_heuristic_value("h2".to_string(), 15.0);
    result2.set_reliable_dead_end();

    result1.merge(&result2);

    assert_eq!(result1.get_heuristic_value("h1"), 10.0);
    assert_eq!(result1.get_heuristic_value("h2"), 15.0);
    assert!(result1.is_dead_end);
    assert!(result1.is_reliable_dead_end);
}

#[test]
fn test_evaluation_result_heuristic_access() {
    let state = create_test_state(1);
    let mut result = EvaluationResult::new(state, 5.0, false);
    result.set_heuristic_value("existing".to_string(), 42.0);

    // Test existing heuristic
    assert_eq!(result.get_heuristic_value("existing"), 42.0);
    assert_eq!(result.get_heuristic_value_optional("existing"), Some(42.0));

    // Test non-existing heuristic
    assert_eq!(result.get_heuristic_value("missing"), f64::INFINITY);
    assert_eq!(result.get_heuristic_value_optional("missing"), None);
}
