//! Base trait for heuristic evaluators.
//!
//! This module provides the heuristic trait that specializes the general
//! `Evaluator` trait for heuristic functions.

#[cfg(test)]
mod tests;

use crate::numeric::evaluation::evaluator::{EvaluationError, EvaluationState, Evaluator};
use planforge_sas::numeric::numeric_task::Operator;
use planforge_sas::numeric::state_registry::ConcreteState;
use std::collections::HashMap;

/// Base trait for heuristic functions.
///
/// This replaces the C++ Heuristic class with a clean trait-based design.
/// Heuristics are specialized evaluators that estimate the cost to reach the goal.
pub trait Heuristic: Evaluator {
    /// Compute the heuristic value for the given state.
    ///
    /// This is the core method that sub-classes must implement.
    /// Return the estimated cost to reach the goal, or infinity for dead ends.
    fn compute_heuristic(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError>;

    /// Return true if dead ends detected by this heuristic are reliable.
    fn dead_ends_are_reliable(&self) -> bool {
        false
    }

    /// Get the name of this heuristic (it allows custom names).
    fn heuristic_name(&self) -> String {
        // Default implementation uses the type name.
        format!(
            "heuristic_{}",
            std::any::type_name::<Self>()
                .split("::")
                .last()
                .unwrap_or("unknown")
        )
    }

    /// Called when a new state is reached during search.
    ///
    /// This allows heuristics to update internal state or caches.
    /// Return true if the heuristic successfully processed the state.
    fn reach_state(
        &mut self,
        _parent_state: &ConcreteState,
        _operator: &Operator,
        _state: &ConcreteState,
    ) -> bool {
        true
    }

    /// Get preferred operators for the given state.
    ///
    /// Some heuristics can suggest operators that are likely to lead
    /// towards the goal. The default implementation returns no preferences.
    fn get_preferred_operators(&self, _state: &ConcreteState) -> Vec<Operator> {
        vec![]
    }

    /// Return the cost type used by this heuristic.
    fn get_cost_type(&self) -> CostType {
        CostType::Normal
    }

    /// Print statistics about this heuristic.
    fn print_statistics(&self) {
        // Default implementation does nothing.
    }
}

/// Different ways to handle operator costs in heuristics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CostType {
    /// Use normal operator costs.
    Normal,
    /// Treat all operators as having cost 1.
    Unit,
    /// Use only the cost of the most expensive operator.
    Max,
}

/// Automatic implementation of `Evaluator` for all Heuristics.
impl<H: Heuristic> Evaluator for H {
    fn name(&self) -> String {
        self.heuristic_name()
    }

    fn evaluate_state(
        &self,
        eval_state: &mut EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError> {
        let heuristic_name = self.name();

        // Check if already computed.
        if let Some(value) = eval_state
            .result()
            .get_heuristic_value_optional(&heuristic_name)
        {
            return Ok(value);
        }

        // Compute the heuristic value (heuristic can inspect goal flag).
        let h_value = self.compute_heuristic(eval_state)?;

        // Update the evaluation state.
        eval_state
            .result_mut()
            .set_heuristic_value(heuristic_name, h_value);

        // Check for dead ends.
        if h_value.is_infinite() && h_value.is_sign_positive() {
            if Heuristic::dead_ends_are_reliable(self) {
                eval_state.result_mut().set_reliable_dead_end();
            }
            return Err(EvaluationError::DeadEnd {
                reliable: Heuristic::dead_ends_are_reliable(self),
            });
        }

        Ok(h_value)
    }

    fn dead_ends_are_reliable(&self) -> bool {
        Heuristic::dead_ends_are_reliable(self)
    }
}

/// A heuristic that returns `0` for goal states and `min_action_cost` for
/// non-goal states.
/// This implements the classical blind search heuristic behavior.
pub struct BlindHeuristic {
    name: String,
    // Cost to return for non-goal states (minimum action cost).
    min_action_cost: f64,
}

impl BlindHeuristic {
    pub fn new(name: Option<String>) -> Self {
        Self {
            name: name.unwrap_or_else(|| "blind_heuristic".to_string()),
            min_action_cost: 1.0,
        }
    }

    /// Create a `BlindHeuristic` that uses the provided `min_action_cost` for
    /// non-goal states.
    pub fn with_min_action_cost(min_action_cost: f64, name: Option<String>) -> Self {
        Self {
            name: name.unwrap_or_else(|| "blind_heuristic".to_string()),
            min_action_cost,
        }
    }
}

impl Heuristic for BlindHeuristic {
    fn compute_heuristic(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError> {
        // `Blind heuristic`: `0` for goal states, `min_action_cost` otherwise.
        Ok(if eval_state.is_goal() {
            0.0
        } else {
            self.min_action_cost
        })
    }

    fn heuristic_name(&self) -> String {
        self.name.clone()
    }
}

/// A heuristic that caches computed values to avoid recomputation.
pub struct CachedHeuristic<H: Heuristic> {
    inner: H,
    // Using state hash as key.
    cache: HashMap<Vec<u8>, f64>,
    name: String,
}

impl<H: Heuristic> CachedHeuristic<H> {
    pub fn new(inner: H, name: Option<String>) -> Self {
        let heuristic_name = name.unwrap_or_else(|| format!("cached_{}", inner.name()));
        Self {
            inner,
            cache: HashMap::new(),
            name: heuristic_name,
        }
    }

    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }
}

impl<H: Heuristic> Heuristic for CachedHeuristic<H> {
    fn compute_heuristic(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError> {
        // For the direct interface, we bypass caching
        self.inner.compute_heuristic(eval_state)
    }

    fn heuristic_name(&self) -> String {
        self.name.clone()
    }

    fn get_cost_type(&self) -> CostType {
        self.inner.get_cost_type()
    }

    fn get_preferred_operators(&self, state: &ConcreteState) -> Vec<Operator> {
        self.inner.get_preferred_operators(state)
    }
}
