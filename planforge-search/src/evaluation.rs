//! Modern evaluation system for Planning.
//!
//! This module provides a clean, idiomatic Rust implementation for state evaluation
//! that combines the functionality of the C++ `EvaluationContext` and `EvaluationResult`
//! into a unified design.

pub mod abstraction_collections;
pub(crate) mod abstraction_task;
pub mod cartesian_abstractions;
pub mod cegar;
pub mod check_admissible;
pub mod domain_abstractions;
pub mod evaluator;
pub mod ff_heuristic;
pub mod heuristic;
pub mod numeric_landmarks;
#[cfg(feature = "cplex")]
pub mod numeric_potentials;
pub mod pattern_databases;
#[cfg(test)]
mod tests;

pub use evaluator::{EvaluationError, EvaluationState, Evaluator};
pub use heuristic::Heuristic;

use planforge_sas::numeric_task::AbstractNumericTask;
use planforge_sas::state_registry::ConcreteState;
use planforge_sas::state_registry::StateID;
use std::collections::HashMap;

/// Rejects a task whose goal an abstraction cannot express, naming the variable
/// that makes it one.
///
/// Every abstraction family here supports a *conjunctive* goal: a set of facts an
/// abstract operator can bring about. A propositional variable an axiom writes is
/// not one. No abstract operator ever writes it — abstract operators come from the
/// task's operators, and a derived variable has no operator writing it either — so
/// an abstraction over such a goal has nothing to reach, and reports the goal
/// unreachable or drops it. Both answers have been wrong in this tree before, and
/// both are silent, so the family refuses the task by name instead.
///
/// A numeric comparison is *not* affected, which is the point of the distinction:
/// its variable is derived in the SAS sense too, but it is
/// [`FactNamespace::Condition`](planforge_sas::numeric_task::FactNamespace::Condition),
/// and the abstraction machinery reasons about the comparison itself — refining
/// its operands is what the numeric CEGAR loop does all day.
///
/// What is left, then, is a `:derived` predicate the problem puts in its goal, and
/// the `@goal-reachable` predicate the translator invents for a disjunctive,
/// quantified or nested goal. Every non-abstraction configuration — blind, `ff`,
/// `lmcutnumeric` — still solves both, since it tests the goal fact in a state the
/// axiom evaluator has closed.
pub fn validate_abstractable_goal(
    task: &dyn AbstractNumericTask,
) -> std::result::Result<(), String> {
    for index in 0..task.get_num_goals() {
        let fact = task.get_goal_fact(index);
        if fact.is_condition() {
            continue;
        }
        let variable = &task.variables()[fact.var()];
        if let Some(layer) = variable.axiom_layer() {
            return Err(format!(
                "abstractions support conjunctive goals only: goal fact {fact:?} names the \
                 derived variable {:?} ({:?}) in axiom layer {layer}, which no operator writes",
                variable.name(),
                task.get_fact_name(fact)
            ));
        }
    }
    Ok(())
}

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
