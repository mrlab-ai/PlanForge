//! Base evaluator trait and evaluation state management
//!
//! This module provides the core evaluator trait that replaces the C++ ScalarEvaluator
//! hierarchy with a more modern, type-safe Rust design.

use crate::numeric::evaluation::EvaluationResult;
use planners_sas::numeric::state_registry::ConcreteState;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;

/// Errors that can occur during evaluation
#[derive(Debug, Clone, PartialEq)]
pub enum EvaluationError {
    /// State is a dead end (no solution possible)
    DeadEnd { reliable: bool },
    /// Heuristic computation failed
    ComputationFailed(String),
    /// Invalid state for evaluation
    InvalidState(String),
}

impl fmt::Display for EvaluationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvaluationError::DeadEnd { reliable } => {
                write!(f, "Dead end detected (reliable: {})", reliable)
            }
            EvaluationError::ComputationFailed(msg) => {
                write!(f, "Evaluation failed: {}", msg)
            }
            EvaluationError::InvalidState(msg) => {
                write!(f, "Invalid state: {}", msg)
            }
        }
    }
}

impl std::error::Error for EvaluationError {}

/// Evaluation state that manages caching and context for a single evaluation
///
/// This replaces the C++ `EvaluationContext` by providing a clean interface
/// for managing evaluation state and caching results across multiple evaluators.
pub struct EvaluationState<'a> {
    /// The evaluation result being built (heuristic map and flags). The `state`
    /// inside the `EvaluationResult` will be synchronized with `backing_state`
    /// when `into_result` is called.
    result: EvaluationResult,
    /// Borrowed reference to the concrete state used during evaluation
    backing_state: &'a ConcreteState,
    /// Cache of already computed evaluators to avoid recomputation
    computed_evaluators: RefCell<HashMap<String, bool>>,
    /// Whether the currently evaluated state is a goal
    is_goal: bool,
}

impl<'a> EvaluationState<'a> {
    /// Creates a new evaluation state that borrows `state`.
    /// The internal `EvaluationResult` will be synchronized with the borrowed
    /// state when `into_result` is called.
    pub fn new(state: &'a ConcreteState, g_value: f64, is_preferred: bool) -> Self {
        let placeholder = EvaluationResult::new_with_id(state.get_id(), g_value, is_preferred);
        Self {
            result: placeholder,
            backing_state: state,
            computed_evaluators: RefCell::new(HashMap::new()),
            is_goal: false,
        }
    }

    /// Gets the current evaluation result
    pub fn result(&self) -> &EvaluationResult {
        &self.result
    }

    /// Gets the mutable evaluation result for updating
    pub fn result_mut(&mut self) -> &mut EvaluationResult {
        &mut self.result
    }

    /// Consumes the evaluation state and returns the final result
    /// Consume this EvaluationState and return an owned EvaluationResult. This stores
    /// only the state's ID to avoid cloning the whole ConcreteState.
    pub fn into_result(mut self) -> EvaluationResult {
        self.result.state =
            crate::numeric::evaluation::EvalStateRef::Id(self.backing_state.get_id());
        self.result
    }

    /// Mark whether this state is a goal
    pub fn set_is_goal(&mut self, is_goal: bool) {
        self.is_goal = is_goal;
    }

    /// Query whether this state is a goal
    pub fn is_goal(&self) -> bool {
        self.is_goal
    }

    /// Checks if an evaluator has already been computed
    pub fn is_computed(&self, evaluator_name: &str) -> bool {
        self.computed_evaluators
            .borrow()
            .get(evaluator_name)
            .copied()
            .unwrap_or(false)
    }

    /// Marks an evaluator as computed
    pub fn mark_computed(&self, evaluator_name: &str) {
        self.computed_evaluators
            .borrow_mut()
            .insert(evaluator_name.to_string(), true);
    }

    /// Gets a heuristic value, computing it if necessary
    pub fn get_or_compute_heuristic<E: Evaluator + ?Sized>(
        &mut self,
        evaluator: &E,
    ) -> Result<f64, EvaluationError> {
        let evaluator_name = evaluator.name();

        // Check if already computed
        if let Some(value) = self.result.get_heuristic_value_optional(&evaluator_name) {
            return Ok(value);
        }

        // Compute the value
        let value = evaluator.evaluate_state(self)?;
        self.result
            .set_heuristic_value(evaluator_name.clone(), value);
        self.mark_computed(&evaluator_name);

        Ok(value)
    }
}

/// Core trait for all evaluators (replaces C++ ScalarEvaluator)
///
/// This trait provides a clean, type-safe interface for evaluating states.
/// It supports both heuristics and composite evaluators.
pub trait Evaluator {
    /// Returns the name of this evaluator (used for caching and identification)
    fn name(&self) -> String;

    /// Evaluates a state within the given evaluation context
    ///
    /// This method should update the evaluation state with computed values
    /// and return the primary value for this evaluator.
    fn evaluate_state(&self, eval_state: &mut EvaluationState<'_>) -> Result<f64, EvaluationError>;

    /// Returns true if dead ends detected by this evaluator are reliable
    fn dead_ends_are_reliable(&self) -> bool {
        false
    }

    /// Gets the names of evaluators that this evaluator depends on
    ///
    /// This is used for dependency tracking and ensures proper evaluation order.
    fn get_dependencies(&self) -> Vec<String> {
        vec![]
    }

    /// Convenience method to evaluate a state with default settings
    fn evaluate(
        &self,
        state: &ConcreteState,
        g_value: f64,
    ) -> Result<EvaluationResult, EvaluationError> {
        let state_owned = state.clone();
        let mut eval_state = EvaluationState::new(&state_owned, g_value, false);
        self.evaluate_state(&mut eval_state)?;
        Ok(eval_state.into_result())
    }

    /// Convenience method to just get the heuristic value
    fn get_value(&self, state: &ConcreteState, g_value: f64) -> Result<f64, EvaluationError> {
        let state_owned = state.clone();
        let mut eval_state = EvaluationState::new(&state_owned, g_value, false);
        self.evaluate_state(&mut eval_state)
    }
}

/// Trait object type for dynamic dispatch
pub type DynEvaluator = dyn Evaluator + Send + Sync;

/// Reference-counted evaluator for sharing
pub type EvaluatorRef = Rc<DynEvaluator>;

/// A collection of evaluators for multi-criteria evaluation
///
/// This allows combining multiple evaluators into a single evaluation context.
pub struct EvaluatorCollection {
    evaluators: Vec<EvaluatorRef>,
    name: String,
}

impl EvaluatorCollection {
    /// Creates a new evaluator collection
    pub fn new(name: String) -> Self {
        Self {
            evaluators: Vec::new(),
            name,
        }
    }

    /// Adds an evaluator to the collection
    pub fn add_evaluator(&mut self, evaluator: EvaluatorRef) {
        self.evaluators.push(evaluator);
    }

    /// Evaluates a state with all evaluators in the collection
    pub fn evaluate_all(
        &self,
        state: &ConcreteState,
        g_value: f64,
    ) -> Result<EvaluationResult, EvaluationError> {
        let state_owned = state.clone();
        let mut eval_state = EvaluationState::new(&state_owned, g_value, false);

        for evaluator in &self.evaluators {
            evaluator.evaluate_state(&mut eval_state)?;
        }

        Ok(eval_state.into_result())
    }

    /// Gets all evaluators in the collection
    pub fn evaluators(&self) -> &[EvaluatorRef] {
        &self.evaluators
    }
}

impl Evaluator for EvaluatorCollection {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn evaluate_state(&self, eval_state: &mut EvaluationState<'_>) -> Result<f64, EvaluationError> {
        let mut last_value = 0.0;

        for evaluator in &self.evaluators {
            last_value = evaluator.evaluate_state(eval_state)?;
        }

        Ok(last_value)
    }

    fn dead_ends_are_reliable(&self) -> bool {
        self.evaluators.iter().all(|e| e.dead_ends_are_reliable())
    }

    fn get_dependencies(&self) -> Vec<String> {
        self.evaluators
            .iter()
            .flat_map(|e| e.get_dependencies())
            .collect()
    }
}

#[cfg(test)]
mod tests {
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

        fn evaluate_state(
            &self,
            eval_state: &mut EvaluationState<'_>,
        ) -> Result<f64, EvaluationError> {
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
}
