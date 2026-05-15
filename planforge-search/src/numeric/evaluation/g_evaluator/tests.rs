use super::*;
use planforge_sas::numeric::state_registry::ConcreteState;

fn create_test_state(id: usize) -> ConcreteState {
    ConcreteState::new(id)
}

#[test]
fn test_g_evaluator() {
    let state = create_test_state(1);
    let g_evaluator = GEvaluator::new(None);
    let result = g_evaluator.evaluate(&state, 42.5).unwrap();

    assert_eq!(result.g_value, 42.5);
    assert_eq!(result.get_heuristic_value("g"), 42.5);
}

#[test]
fn test_g_evaluator_custom_name() {
    let state = create_test_state(1);
    let g_evaluator = GEvaluator::new(Some("custom_g".to_string()));
    let result = g_evaluator.evaluate(&state, 15.0).unwrap();

    assert_eq!(result.get_heuristic_value("custom_g"), 15.0);
    assert_eq!(result.get_heuristic_value("g"), f64::INFINITY); // Default name not set
}

#[test]
fn test_sum_evaluator() {
    let state = create_test_state(1);

    // Create evaluation state with some values
    let state_owned = state.clone();
    let mut eval_state = EvaluationState::new(&state_owned, 10.0, false);
    eval_state
        .result_mut()
        .set_heuristic_value("g".to_string(), 10.0);
    eval_state
        .result_mut()
        .set_heuristic_value("h".to_string(), 25.0);

    let sum_evaluator = SumEvaluator::f_evaluator("h".to_string());
    let result = sum_evaluator.evaluate_state(&mut eval_state).unwrap();

    assert_eq!(result, 35.0); // 10 + 25
    assert_eq!(eval_state.result().get_heuristic_value("f_h"), 35.0);
}

#[test]
fn test_sum_evaluator_with_infinity() {
    let state = create_test_state(1);

    let state_owned = state.clone();
    let mut eval_state = EvaluationState::new(&state_owned, 10.0, false);
    eval_state
        .result_mut()
        .set_heuristic_value("g".to_string(), 10.0);
    eval_state
        .result_mut()
        .set_heuristic_value("h_inf".to_string(), f64::INFINITY);

    let sum_evaluator =
        SumEvaluator::new("f_inf".to_string(), "g".to_string(), "h_inf".to_string());
    let result = sum_evaluator.evaluate_state(&mut eval_state).unwrap();

    assert!(result.is_infinite());
    assert!(
        eval_state
            .result()
            .get_heuristic_value("f_inf")
            .is_infinite()
    );
}

#[test]
fn test_weighted_evaluator() {
    let state = create_test_state(1);

    let state_owned = state.clone();
    let mut eval_state = EvaluationState::new(&state_owned, 5.0, false);
    eval_state
        .result_mut()
        .set_heuristic_value("h".to_string(), 20.0);

    let weighted_evaluator = WeightedEvaluator::new("weighted_h".to_string(), "h".to_string(), 2.5);
    let result = weighted_evaluator.evaluate_state(&mut eval_state).unwrap();

    assert_eq!(result, 50.0); // 20.0 * 2.5
    assert_eq!(eval_state.result().get_heuristic_value("weighted_h"), 50.0);
}

#[test]
fn test_max_evaluator() {
    let state = create_test_state(1);

    let state_owned = state.clone();
    let mut eval_state = EvaluationState::new(&state_owned, 5.0, false);
    eval_state
        .result_mut()
        .set_heuristic_value("h1".to_string(), 15.0);
    eval_state
        .result_mut()
        .set_heuristic_value("h2".to_string(), 30.0);

    let max_evaluator = MaxEvaluator::new("max_h".to_string(), "h1".to_string(), "h2".to_string());
    let result = max_evaluator.evaluate_state(&mut eval_state).unwrap();

    assert_eq!(result, 30.0); // max(15.0, 30.0)
    assert_eq!(eval_state.result().get_heuristic_value("max_h"), 30.0);
}

#[test]
fn test_evaluator_dependencies() {
    let sum_evaluator = SumEvaluator::f_evaluator("manhattan".to_string());
    let deps = sum_evaluator.get_dependencies();

    assert_eq!(deps.len(), 2);
    assert!(deps.contains(&"g".to_string()));
    assert!(deps.contains(&"manhattan".to_string()));
}
