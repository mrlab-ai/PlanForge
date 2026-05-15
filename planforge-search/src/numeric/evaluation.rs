//! Modern evaluation system for Planning.
//!
//! This module provides a clean, idiomatic Rust implementation for state evaluation
//! that combines the functionality of the C++ `EvaluationContext` and `EvaluationResult`
//! into a unified design.

pub mod domain_abstractions;
pub mod evaluator;
pub mod ff_heuristic;
pub mod g_evaluator;
pub mod heuristic;
pub mod numeric_landmarks;
pub mod pattern_databases;
#[cfg(test)]
mod tests;

pub use evaluator::{EvaluationError, EvaluationState, Evaluator};
pub use g_evaluator::GEvaluator;
pub use heuristic::Heuristic;

use planforge_sas::numeric::state_registry::ConcreteState;
use planforge_sas::numeric::state_registry::StateID;
use std::collections::HashMap;

/// Light-weight reference to a state used inside `EvaluationResult`.
/// Can either own a `ConcreteState` or store a compact `StateID` to avoid cloning.
#[derive(Debug, Clone, PartialEq)]
pub enum EvalStateRef {
    Owned(ConcreteState),
    Id(StateID),
}

impl EvalStateRef {
    pub fn id(&self) -> StateID {
        match self {
            EvalStateRef::Owned(s) => s.get_id(),
            EvalStateRef::Id(id) => *id,
        }
    }
}

/// Result of evaluating a state.
///
/// This combines the C++ `EvaluationResult` and relevant parts of `EvaluationContext`
/// into a single, immutable structure that contains all evaluation information.
#[derive(Debug, Clone, PartialEq)]
pub struct EvaluationResult {
    /// Reference to the state that was evaluated. May be an owned state or a compact id.
    pub state: EvalStateRef,
    /// `g`-value (cost to reach this state).
    pub g_value: f64,
    /// Whether this state was reached by a preferred operator.
    pub is_preferred: bool,
    /// Computed heuristic values (evaluator name -> value).
    pub heuristic_values: HashMap<String, f64>,
    /// Whether this evaluation represents a dead end.
    pub is_dead_end: bool,
    /// Whether the dead end detection is reliable.
    pub is_reliable_dead_end: bool,
}

impl EvaluationResult {
    /// Create a new evaluation result for the given state.
    pub fn new(state: ConcreteState, g_value: f64, is_preferred: bool) -> Self {
        Self {
            state: EvalStateRef::Owned(state),
            g_value,
            is_preferred,
            heuristic_values: HashMap::new(),
            is_dead_end: false,
            is_reliable_dead_end: false,
        }
    }

    /// Create an evaluation result that stores only a compact state id.
    pub fn new_with_id(state_id: StateID, g_value: f64, is_preferred: bool) -> Self {
        Self {
            state: EvalStateRef::Id(state_id),
            g_value,
            is_preferred,
            heuristic_values: HashMap::new(),
            is_dead_end: false,
            is_reliable_dead_end: false,
        }
    }

    /// Get a heuristic value by evaluator name.
    /// Return infinity if the heuristic is not available.
    pub fn get_heuristic_value(&self, evaluator_name: &str) -> f64 {
        self.heuristic_values
            .get(evaluator_name)
            .copied()
            .unwrap_or(f64::INFINITY)
    }

    /// Get a heuristic value by evaluator name, returning `None` if not computed.
    pub fn get_heuristic_value_optional(&self, evaluator_name: &str) -> Option<f64> {
        self.heuristic_values.get(evaluator_name).copied()
    }

    /// Check if a specific heuristic value is infinite.
    pub fn is_heuristic_infinite(&self, evaluator_name: &str) -> bool {
        self.get_heuristic_value(evaluator_name).is_infinite()
    }

    /// Set a heuristic value.
    pub fn set_heuristic_value(&mut self, evaluator_name: String, value: f64) {
        self.heuristic_values.insert(evaluator_name, value);
        // Update dead end status if this heuristic indicates a dead end
        if value.is_infinite() && value.is_sign_positive() {
            self.is_dead_end = true;
        }
    }

    /// Mark this evaluation as a reliable dead end.
    pub fn set_reliable_dead_end(&mut self) {
        self.is_dead_end = true;
        self.is_reliable_dead_end = true;
    }

    /// Get the `f`-value for a given heuristic (`g` + `h`).
    pub fn get_f_value(&self, heuristic_name: &str) -> f64 {
        self.g_value + self.get_heuristic_value(heuristic_name)
    }

    /// Get all computed heuristic names.
    pub fn get_heuristic_names(&self) -> impl Iterator<Item = &String> {
        self.heuristic_values.keys()
    }

    /// Check if any heuristics have been computed.
    pub fn has_heuristics(&self) -> bool {
        !self.heuristic_values.is_empty()
    }

    /// Merge another evaluation result into this one.
    /// This is useful for combining results from multiple evaluators.
    pub fn merge(&mut self, other: &EvaluationResult) {
        for (name, value) in &other.heuristic_values {
            self.set_heuristic_value(name.clone(), *value);
        }
        self.is_dead_end |= other.is_dead_end;
        self.is_reliable_dead_end |= other.is_reliable_dead_end;
    }
}
