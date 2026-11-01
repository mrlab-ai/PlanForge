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

use crate::numeric_task::Task;
use std::sync::Arc;

use crate::axioms::AxiomEvaluator;
use crate::numeric_task::{
    ExplicitFact, ExplicitValueIndex, NumericValue, Operator, OperatorIndex,
};
use crate::utils::errors::AssignmentAxiomError;
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
        fact.holds_in(&self.propositional)
    }

    /// Whether every one of `facts` holds.
    pub fn holds_all(&self, facts: &[ExplicitFact]) -> bool {
        facts.iter().all(|fact| self.holds(fact))
    }

    /// The `facts` that do not hold, in order.
    pub fn unsatisfied<'f>(
        &self,
        facts: &'f [ExplicitFact],
    ) -> impl Iterator<Item = &'f ExplicitFact> + use<'_, 'f> {
        facts.iter().filter(|fact| !self.holds(fact))
    }

    /// Whether every precondition of `operator` holds.
    pub fn is_applicable(&self, operator: &Operator) -> bool {
        self.holds_all(operator.preconditions())
    }
}

/// The values of a state as lookups read them: the numeric values decoded and
/// closed under the arithmetic axioms, the propositional values either decoded
/// or read on demand from a registered state's packed bins.
///
/// An abstraction tests a few variables of a state; reading those from the
/// packed bins spares decoding all of them for every evaluated state.
#[derive(Clone, Copy)]
pub struct StateValues<'a> {
    propositional: PropositionalValues<'a>,
    numeric: &'a [NumericValue],
}

#[derive(Clone, Copy)]
enum PropositionalValues<'a> {
    Decoded(&'a [ExplicitValueIndex]),
    /// The bins of a concrete-state packer, whose propositional variables are
    /// its first slots.
    Packed {
        packer: &'a StatePacker,
        bins: &'a [u64],
    },
}

impl<'a> StateValues<'a> {
    pub fn new(propositional: &'a [ExplicitValueIndex], numeric: &'a [NumericValue]) -> Self {
        Self {
            propositional: PropositionalValues::Decoded(propositional),
            numeric,
        }
    }

    /// The values of a state packed into `bins` by `packer`.
    pub fn packed(packer: &'a StatePacker, bins: &'a [u64], numeric: &'a [NumericValue]) -> Self {
        Self {
            propositional: PropositionalValues::Packed { packer, bins },
            numeric,
        }
    }

    /// The value of propositional variable `var`.
    #[inline]
    pub fn propositional(&self, var: usize) -> ExplicitValueIndex {
        match self.propositional {
            PropositionalValues::Decoded(values) => values[var],
            PropositionalValues::Packed { packer, bins } => {
                debug_assert!(var < packer.numeric_slot_offset());
                ExplicitValueIndex::from_usize(packer.get(bins, var) as usize)
            }
        }
    }

    /// The number of propositional variables.
    pub fn num_propositional(&self) -> usize {
        match self.propositional {
            PropositionalValues::Decoded(values) => values.len(),
            PropositionalValues::Packed { packer, .. } => packer.numeric_slot_offset(),
        }
    }

    pub fn numeric(&self) -> &'a [NumericValue] {
        self.numeric
    }

    /// Writes the propositional values into `output`.
    pub fn fill_propositional(&self, output: &mut Vec<ExplicitValueIndex>) {
        output.clear();
        output.extend((0..self.num_propositional()).map(|var| self.propositional(var)));
    }
}

impl<'a> From<&'a ExplicitState> for StateValues<'a> {
    fn from(state: &'a ExplicitState) -> Self {
        Self::new(&state.propositional, &state.numeric)
    }
}

/// The states of a task as explicit values: its initial state, successors and
/// goal test.
///
/// Owns the axiom evaluator, compiled once, and the propositional layout the
/// evaluator works on.
pub struct ExplicitStateSpace<'task> {
    task: &'task Task,
    state_packer: Arc<StatePacker>,
    axiom_evaluator: AxiomEvaluator,
}

impl<'task> ExplicitStateSpace<'task> {
    pub fn new(task: &'task Task) -> Self {
        let state_packer = Arc::new(StatePacker::propositional(task));
        let axiom_evaluator = AxiomEvaluator::new(task, state_packer.clone());
        Self {
            task,
            state_packer,
            axiom_evaluator,
        }
    }

    pub fn task(&self) -> &'task Task {
        self.task
    }

    /// The task's initial state, with numeric values canonical as in the
    /// registry.
    pub fn initial_state(&self) -> Result<ExplicitState, AssignmentAxiomError> {
        let mut state = ExplicitState {
            propositional: self.task.initial_propositional_values().to_vec(),
            numeric: self.task.initial_numeric_values().to_vec(),
        };
        self.close_under_axioms(&mut state)?;
        Ok(state)
    }

    /// The successor of `state` under `operator`, which must be applicable.
    pub fn successor(
        &self,
        state: &ExplicitState,
        operator: &Operator,
    ) -> Result<ExplicitState, AssignmentAxiomError> {
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
    ) -> Result<(), AssignmentAxiomError> {
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
    ) -> Result<(), AssignmentAxiomError> {
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
    ) -> Result<ExplicitState, AssignmentAxiomError> {
        let operators = self.task.operators();
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
        state.holds(&self.task.goals()[goal_id])
    }

    /// How many goals hold in `state`.
    pub fn satisfied_goal_count(&self, state: &ExplicitState) -> usize {
        self.task
            .goals()
            .iter()
            .filter(|goal| state.holds(goal))
            .count()
    }

    /// Whether every goal holds in `state`.
    pub fn is_goal(&self, state: &ExplicitState) -> bool {
        state.holds_all(self.task.goals())
    }

    /// Re-derives the axiom variables of `state` from its other values. The
    /// other values are taken as they are, without canonicalization.
    pub fn close_under_axioms(
        &self,
        state: &mut ExplicitState,
    ) -> Result<(), AssignmentAxiomError> {
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
            self.task.variables().len(),
            "state has the wrong number of propositional values"
        );
        assert_eq!(
            state.numeric.len(),
            self.task.numeric_variables().len(),
            "state has the wrong number of numeric values"
        );
    }
}
