//! G-evaluator implementation
//!
//! This module provides an evaluator that returns the g-value (cost to reach a state).

use crate::search::numeric::evaluation::evaluator::{EvaluationError, EvaluationState, Evaluator};

/// Evaluator that returns the g-value (path cost) of a state
///
/// This corresponds to the C++ GEvaluator and is useful for
/// implementing uniform-cost search and as a component in f-value calculations.
pub struct GEvaluator {
    name: String,
}

impl GEvaluator {
    /// Creates a new G-evaluator with the given name
    pub fn new(name: Option<String>) -> Self {
        Self {
            name: name.unwrap_or_else(|| "g".to_string()),
        }
    }
}

impl Evaluator for GEvaluator {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn evaluate_state(&self, eval_state: &mut EvaluationState) -> Result<f64, EvaluationError> {
        let g_value = eval_state.result().g_value;
        eval_state
            .result_mut()
            .set_heuristic_value(self.name(), g_value);
        Ok(g_value)
    }

    fn dead_ends_are_reliable(&self) -> bool {
        // G-values don't detect dead ends
        false
    }
}

/// Sum evaluator that combines two evaluators by adding their values
///
/// This is commonly used to create f = g + h evaluators.
pub struct SumEvaluator {
    name: String,
    first_evaluator_name: String,
    second_evaluator_name: String,
}

impl SumEvaluator {
    /// Creates a new sum evaluator
    pub fn new(name: String, first_evaluator_name: String, second_evaluator_name: String) -> Self {
        Self {
            name,
            first_evaluator_name,
            second_evaluator_name,
        }
    }

    /// Convenience constructor for f = g + h
    pub fn f_evaluator(heuristic_name: String) -> Self {
        Self::new(
            format!("f_{}", heuristic_name),
            "g".to_string(),
            heuristic_name,
        )
    }
}

impl Evaluator for SumEvaluator {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn evaluate_state(&self, eval_state: &mut EvaluationState) -> Result<f64, EvaluationError> {
        let first_value = eval_state
            .result()
            .get_heuristic_value(&self.first_evaluator_name);
        let second_value = eval_state
            .result()
            .get_heuristic_value(&self.second_evaluator_name);

        // If either value is infinite, the sum is infinite
        let sum = if first_value.is_infinite() || second_value.is_infinite() {
            f64::INFINITY
        } else {
            first_value + second_value
        };

        eval_state
            .result_mut()
            .set_heuristic_value(self.name(), sum);
        Ok(sum)
    }

    fn dead_ends_are_reliable(&self) -> bool {
        // Sum is only as reliable as its least reliable component
        // For simplicity, we assume it's not reliable unless proven otherwise
        false
    }

    fn get_dependencies(&self) -> Vec<String> {
        vec![
            self.first_evaluator_name.clone(),
            self.second_evaluator_name.clone(),
        ]
    }
}

/// Weighted evaluator that multiplies an evaluator's value by a constant
pub struct WeightedEvaluator {
    name: String,
    base_evaluator_name: String,
    weight: f64,
}

impl WeightedEvaluator {
    /// Creates a new weighted evaluator
    pub fn new(name: String, base_evaluator_name: String, weight: f64) -> Self {
        Self {
            name,
            base_evaluator_name,
            weight,
        }
    }
}

impl Evaluator for WeightedEvaluator {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn evaluate_state(&self, eval_state: &mut EvaluationState) -> Result<f64, EvaluationError> {
        let base_value = eval_state
            .result()
            .get_heuristic_value(&self.base_evaluator_name);

        let weighted_value = if base_value.is_infinite() {
            base_value // Preserve infinity
        } else {
            base_value * self.weight
        };

        eval_state
            .result_mut()
            .set_heuristic_value(self.name(), weighted_value);
        Ok(weighted_value)
    }

    fn dead_ends_are_reliable(&self) -> bool {
        // Weighting doesn't change reliability
        false
    }

    fn get_dependencies(&self) -> Vec<String> {
        vec![self.base_evaluator_name.clone()]
    }
}

/// Maximum evaluator that returns the maximum of two evaluators
pub struct MaxEvaluator {
    name: String,
    first_evaluator_name: String,
    second_evaluator_name: String,
}

impl MaxEvaluator {
    /// Creates a new max evaluator
    pub fn new(name: String, first_evaluator_name: String, second_evaluator_name: String) -> Self {
        Self {
            name,
            first_evaluator_name,
            second_evaluator_name,
        }
    }
}

impl Evaluator for MaxEvaluator {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn evaluate_state(&self, eval_state: &mut EvaluationState) -> Result<f64, EvaluationError> {
        let first_value = eval_state
            .result()
            .get_heuristic_value(&self.first_evaluator_name);
        let second_value = eval_state
            .result()
            .get_heuristic_value(&self.second_evaluator_name);

        let max_value = first_value.max(second_value);

        eval_state
            .result_mut()
            .set_heuristic_value(self.name(), max_value);
        Ok(max_value)
    }

    fn dead_ends_are_reliable(&self) -> bool {
        false
    }

    fn get_dependencies(&self) -> Vec<String> {
        vec![
            self.first_evaluator_name.clone(),
            self.second_evaluator_name.clone(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::numeric::state_registry::ConcreteState;

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
        assert!(eval_state
            .result()
            .get_heuristic_value("f_inf")
            .is_infinite());
    }

    #[test]
    fn test_weighted_evaluator() {
        let state = create_test_state(1);

        let state_owned = state.clone();
        let mut eval_state = EvaluationState::new(&state_owned, 5.0, false);
        eval_state
            .result_mut()
            .set_heuristic_value("h".to_string(), 20.0);

        let weighted_evaluator =
            WeightedEvaluator::new("weighted_h".to_string(), "h".to_string(), 2.5);
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

        let max_evaluator =
            MaxEvaluator::new("max_h".to_string(), "h1".to_string(), "h2".to_string());
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
}
