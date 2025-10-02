//! Modern evaluation system for planning
//! 
//! This module provides a clean, idiomatic Rust implementation for state evaluation
//! that combines the functionality of the C++ EvaluationContext and EvaluationResult
//! into a unified design.

pub mod evaluator;
pub mod heuristic;
pub mod g_evaluator;

pub use evaluator::{Evaluator, EvaluationState, EvaluationError};
pub use heuristic::Heuristic;
pub use g_evaluator::GEvaluator;

use crate::search::numeric::state_registry::ConcreteState;
use std::collections::HashMap;

/// Result of evaluating a state
/// 
/// This combines the C++ EvaluationResult and relevant parts of EvaluationContext
/// into a single, immutable structure that contains all evaluation information.
#[derive(Debug, Clone, PartialEq)]
pub struct EvaluationResult {
    /// The state that was evaluated
    pub state: ConcreteState,
    /// G-value (cost to reach this state)
    pub g_value: f64,
    /// Whether this state was reached by a preferred operator
    pub is_preferred: bool,
    /// Computed heuristic values (evaluator name -> value)
    pub heuristic_values: HashMap<String, f64>,
    /// Whether this evaluation represents a dead end
    pub is_dead_end: bool,
    /// Whether the dead end detection is reliable
    pub is_reliable_dead_end: bool,
}

impl EvaluationResult {
    /// Creates a new evaluation result for the given state
    pub fn new(state: ConcreteState, g_value: f64, is_preferred: bool) -> Self {
        Self {
            state,
            g_value,
            is_preferred,
            heuristic_values: HashMap::new(),
            is_dead_end: false,
            is_reliable_dead_end: false,
        }
    }

    /// Gets a heuristic value by evaluator name
    /// Returns infinity if the heuristic is not available
    pub fn get_heuristic_value(&self, evaluator_name: &str) -> f64 {
        self.heuristic_values.get(evaluator_name).copied().unwrap_or(f64::INFINITY)
    }

    /// Gets a heuristic value by evaluator name, returning None if not computed
    pub fn get_heuristic_value_optional(&self, evaluator_name: &str) -> Option<f64> {
        self.heuristic_values.get(evaluator_name).copied()
    }

    /// Checks if a specific heuristic value is infinite
    pub fn is_heuristic_infinite(&self, evaluator_name: &str) -> bool {
        self.get_heuristic_value(evaluator_name).is_infinite()
    }

    /// Sets a heuristic value
    pub fn set_heuristic_value(&mut self, evaluator_name: String, value: f64) {
        self.heuristic_values.insert(evaluator_name, value);
        // Update dead end status if this heuristic indicates a dead end
        if value.is_infinite() && value.is_sign_positive() {
            self.is_dead_end = true;
        }
    }

    /// Marks this evaluation as a reliable dead end
    pub fn set_reliable_dead_end(&mut self) {
        self.is_dead_end = true;
        self.is_reliable_dead_end = true;
    }

    /// Gets the f-value for a given heuristic (g + h)
    pub fn get_f_value(&self, heuristic_name: &str) -> f64 {
        self.g_value + self.get_heuristic_value(heuristic_name)
    }

    /// Gets all computed heuristic names
    pub fn get_heuristic_names(&self) -> impl Iterator<Item = &String> {
        self.heuristic_values.keys()
    }

    /// Checks if any heuristics have been computed
    pub fn has_heuristics(&self) -> bool {
        !self.heuristic_values.is_empty()
    }

    /// Merges another evaluation result into this one
    /// This is useful for combining results from multiple evaluators
    pub fn merge(&mut self, other: &EvaluationResult) {
        for (name, value) in &other.heuristic_values {
            self.set_heuristic_value(name.clone(), *value);
        }
        self.is_dead_end |= other.is_dead_end;
        self.is_reliable_dead_end |= other.is_reliable_dead_end;
    }
}

#[cfg(test)]
mod tests {
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
}
