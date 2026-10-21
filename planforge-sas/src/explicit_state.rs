//! Explicit states: the values of a state, outside any state registry.
//!
//! A [`StateRegistry`](crate::state_registry::StateRegistry) interns the
//! states a search reaches and hands out handles into its pool, packed with
//! the numeric values it compresses. Code that follows a few states of its
//! own, such as CEGAR replaying an abstract plan or a collection keeping the
//! states its members refine from, needs the values themselves, unpacked and
//! independent of any search. [`ExplicitStateSpace`] gives them the task's
//! semantics, with the same successors as the registry.

#[cfg(test)]
mod tests;

use std::sync::Arc;

use crate::axioms::AxiomEvaluator;
use crate::numeric_task::{
    AbstractNumericTask, ExplicitFact, ExplicitValueIndex, NumericValue, Operator, OperatorIndex,
};
use crate::utils::errors::AxiomEvalError;
use crate::utils::float_tolerance;
use crate::utils::state_packer::StatePacker;

/// The propositional and numeric values of a state, closed under the axioms of
/// the task whose [`ExplicitStateSpace`] made it. The default is empty, a
/// buffer to fill.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ExplicitState {
    pub propositional: Vec<ExplicitValueIndex>,
    pub numeric: Vec<NumericValue>,
}

impl ExplicitState {
    /// Whether the propositional `fact` holds.
    #[inline]
    pub fn holds(&self, fact: &ExplicitFact) -> bool {
        self.propositional[fact.var()] == fact.value_index()
    }

    /// Whether every precondition of `operator` holds.
    pub fn is_applicable(&self, operator: &Operator) -> bool {
        operator.preconditions().iter().all(|fact| self.holds(fact))
    }
}

/// The states of a task as explicit values: its initial state, successors and
/// goal test.
///
/// Owns the axiom evaluator, compiled once, and the propositional layout the
/// evaluator works on.
pub struct ExplicitStateSpace<'task> {
    task: &'task dyn AbstractNumericTask,
    state_packer: Arc<StatePacker>,
    axiom_evaluator: AxiomEvaluator<'task>,
}

impl<'task> ExplicitStateSpace<'task> {
    pub fn new(task: &'task dyn AbstractNumericTask) -> Self {
        let state_packer = Arc::new(StatePacker::propositional(task));
        let axiom_evaluator = AxiomEvaluator::new(Arc::new(task), state_packer.clone());
        Self {
            task,
            state_packer,
            axiom_evaluator,
        }
    }

    pub fn task(&self) -> &'task dyn AbstractNumericTask {
        self.task
    }

    /// The task's initial state, with numeric values canonical as in the
    /// registry.
    pub fn initial_state(&self) -> Result<ExplicitState, AxiomEvalError> {
        let mut state = ExplicitState {
            propositional: self.task.get_initial_propositional_state_values().to_vec(),
            numeric: self
                .task
                .get_initial_numeric_state_values()
                .iter()
                .map(|&value| float_tolerance::canonicalize(value))
                .collect(),
        };
        self.close_under_axioms(&mut state)?;
        Ok(state)
    }

    /// The successor of `state` under `operator`, which must be applicable.
    pub fn successor(
        &self,
        state: &ExplicitState,
        operator: &Operator,
    ) -> Result<ExplicitState, AxiomEvalError> {
        let mut successor = ExplicitState {
            propositional: Vec::with_capacity(state.propositional.len()),
            numeric: Vec::with_capacity(state.numeric.len()),
        };
        self.successor_into(state, operator, &mut successor)?;
        Ok(successor)
    }

    /// Writes the successor of `state` under `operator`, which must be
    /// applicable, into `successor`.
    pub fn successor_into(
        &self,
        state: &ExplicitState,
        operator: &Operator,
        successor: &mut ExplicitState,
    ) -> Result<(), AxiomEvalError> {
        assert!(
            state.is_applicable(operator),
            "operator {} is inapplicable in {state:?}",
            operator.name()
        );
        self.apply_effects_into(state, operator, successor)
    }

    /// Writes the state the effects of `operator` lead to from `state` into
    /// `successor`, whether or not `operator` is applicable: the
    /// counterfactual continuation of a plan past a failed precondition.
    pub fn apply_effects_into(
        &self,
        state: &ExplicitState,
        operator: &Operator,
        successor: &mut ExplicitState,
    ) -> Result<(), AxiomEvalError> {
        self.assert_layout(state);
        operator.apply_to_explicit_state(
            &state.propositional,
            &state.numeric,
            &mut successor.propositional,
            &mut successor.numeric,
        );
        self.close_under_axioms(successor)
    }

    /// The state `plan` reaches from `state`; each operator must be applicable
    /// where the plan applies it.
    pub fn replay(
        &self,
        state: &ExplicitState,
        plan: &[OperatorIndex],
    ) -> Result<ExplicitState, AxiomEvalError> {
        let operators = self.task.get_operators();
        let mut current = state.clone();
        let mut successor = state.clone();
        for &operator_id in plan {
            self.successor_into(&current, &operators[operator_id.index()], &mut successor)?;
            std::mem::swap(&mut current, &mut successor);
        }
        Ok(current)
    }

    /// Whether `goal_id` holds in `state`.
    pub fn satisfies_goal(&self, state: &ExplicitState, goal_id: usize) -> bool {
        state.holds(self.task.get_goal_fact(goal_id))
    }

    /// How many goals hold in `state`.
    pub fn satisfied_goal_count(&self, state: &ExplicitState) -> usize {
        (0..self.task.get_num_goals())
            .filter(|&goal_id| self.satisfies_goal(state, goal_id))
            .count()
    }

    /// Whether every goal holds in `state`.
    pub fn is_goal(&self, state: &ExplicitState) -> bool {
        (0..self.task.get_num_goals()).all(|goal_id| self.satisfies_goal(state, goal_id))
    }

    /// Re-derives the axiom variables of `state` from its other values.
    fn close_under_axioms(&self, state: &mut ExplicitState) -> Result<(), AxiomEvalError> {
        self.assert_layout(state);
        let mut packed = vec![0u64; self.state_packer.num_bins()];
        for (var_id, value) in state.propositional.iter().enumerate() {
            self.state_packer
                .set(&mut packed, var_id, value.index() as u64);
        }
        self.axiom_evaluator
            .evaluate(&mut packed, &mut state.numeric)?;
        for (var_id, value) in state.propositional.iter_mut().enumerate() {
            *value = ExplicitValueIndex::from_usize(
                usize::try_from(self.state_packer.get(&packed, var_id))
                    .expect("a packed value fits its variable's domain"),
            );
        }
        Ok(())
    }

    fn assert_layout(&self, state: &ExplicitState) {
        assert_eq!(
            state.propositional.len(),
            self.task.get_num_variables(),
            "state has the wrong number of propositional values"
        );
        assert_eq!(
            state.numeric.len(),
            self.task.numeric_variables().len(),
            "state has the wrong number of numeric values"
        );
    }
}
