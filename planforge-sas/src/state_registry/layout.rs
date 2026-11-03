//! What a registry fixes at construction.

use std::sync::Arc;

use super::{CompactNumericValues, NumericSlot, NumericTransition};
use crate::axioms::AxiomEvaluator;
use crate::numeric_task::{ExplicitFact, NumericType, NumericValue, Operator};
use crate::utils::errors::StateInsertError;
use crate::utils::state_packer::StatePacker;

/// What a registry fixes at construction: the packer, where each numeric
/// variable lives, the metric, and the axioms closing a state.
#[derive(Debug)]
pub(super) struct StateLayout {
    /// Axiom evaluator for handling derived predicates and numeric axioms.
    pub(super) axiom_evaluator: Arc<AxiomEvaluator>,
    /// State packer for efficient bit-level state representation.
    pub(super) global_state_packer: Arc<StatePacker>,
    /// Where each numeric variable's value lives, fixed at construction.
    pub(super) numeric_slots: Vec<NumericSlot>,
    /// Snapshot of the numeric variable layout, populated at construction.
    /// Iterating this avoids per-call vtable dispatch through `task.numeric_variables()`
    /// in hot paths like numeric decoding.
    pub(super) numeric_var_types: Vec<NumericType>,
    /// Initial numeric row with immutable constants in place and every
    /// state-dependent or derived entry zeroed. Numeric-state materialization
    /// copies this row before filling the two state-dependent layouts below.
    pub(super) numeric_template: Vec<NumericValue>,
    /// `(numeric variable id, packed-state slot)` for regular variables.
    pub(super) regular_numeric_slots: Vec<(usize, usize)>,
    /// `(numeric variable id, dense cost-row slot)` for cost variables.
    pub(super) cost_numeric_indices: Vec<(usize, usize)>,
    /// Cached count returned by `countcost_variables`. Constant for the task.
    pub(super) cost_variable_count: usize,
    /// Cached `task.metric().var_id()` and the metric variable's type when set.
    /// Used to short-circuit `metric_value_for_state` for the common Regular case.
    pub(super) metric_var: Option<(usize, NumericType)>,
    /// Whether the task uses a metric (cached `task.metric().use_metric()`).
    pub(super) metric_use_metric: bool,
    /// Whether the task is a minimization (cached `task.metric().is_min()`).
    pub(super) metric_is_min: bool,
    /// Per-bin mask covering only the bits owned by non-derived (input)
    /// variables: regular propositional vars (axiom_layer == None) and regular
    /// numeric vars. `StateRegistry::lookup` uses it to deduplicate
    /// successors before running the comparison/propositional axiom passes.
    /// Two states with identical non-derived bits are guaranteed to produce
    /// identical full buffers because axioms are deterministic functions of
    /// the inputs, so masked equality matches full equality.
    pub(super) non_derived_bits_mask: Vec<u64>,
    /// True iff the task has at least one comparison or propositional axiom
    /// AND `non_derived_bits_mask` actually masks anything off. When true, the
    /// successor flow defers comparison/propositional axiom evaluation until
    /// after dedup so we can skip it entirely on duplicate states.
    pub(super) has_axiom_derived_bits: bool,
}

impl StateLayout {
    /// Packs the regular numeric values into `buffer` and returns the cost
    /// values.
    pub(super) fn process_numeric_variables(
        &self,
        interner: &mut CompactNumericValues,
        buffer: &mut [u64],
        initial_numeric_values: &[NumericValue],
    ) -> Vec<NumericValue> {
        let mut cost_variables = Vec::with_capacity(self.cost_variable_count);
        for (&value, &slot) in initial_numeric_values.iter().zip(&self.numeric_slots) {
            match slot {
                NumericSlot::Packed(packed_slot) => {
                    self.global_state_packer
                        .set(buffer, packed_slot, interner.intern(value));
                }
                NumericSlot::Cost(cost_index) => {
                    debug_assert_eq!(cost_index, cost_variables.len());
                    cost_variables.push(value);
                }
                NumericSlot::Constant(_) | NumericSlot::Derived => {}
            }
        }
        cost_variables
    }

    /// Process numeric variables during state registration.
    ///
    /// Only regular values enter the packed buffer and only cost values are
    /// returned; constant values are the same in every state and must equal
    /// the task's, and derived values are a function of the inputs that
    /// [`Self::evaluate_axioms`] recomputes right after this call.
    pub(super) fn process_register_numeric_variables(
        &self,
        interner: &mut CompactNumericValues,
        buffer: &mut [u64],
        numeric_values: &[NumericValue],
    ) -> Result<Vec<NumericValue>, StateInsertError> {
        assert_eq!(
            numeric_values.len(),
            self.numeric_slots.len(),
            "a state has one value per numeric variable"
        );
        let mut cost_variables = Vec::with_capacity(self.cost_variable_count);
        for (numeric_var_id, (&value, &slot)) in
            numeric_values.iter().zip(&self.numeric_slots).enumerate()
        {
            match slot {
                NumericSlot::Packed(packed_slot) => {
                    self.global_state_packer
                        .set(buffer, packed_slot, interner.intern(value));
                }
                NumericSlot::Cost(cost_index) => {
                    debug_assert_eq!(cost_index, cost_variables.len());
                    cost_variables.push(value);
                }
                NumericSlot::Constant(constant) => {
                    if constant != value {
                        return Err(StateInsertError {
                            message: format!(
                                "constant numeric variable {numeric_var_id} is {constant}, but the state to register has {value}"
                            ),
                        });
                    }
                }
                NumericSlot::Derived => {}
            }
        }
        Ok(cost_variables)
    }

    /// Whether every condition of a conditional assignment effect holds in the
    /// packed state `buffer`.
    ///
    /// Reads `buffer` directly instead of going through `ExplicitFact::is_hold`
    /// so that callers holding only the parent's raw buffer (and no
    /// `ConcreteState` for it) can check conditions on the hot successor path.
    #[inline]
    pub(super) fn assignment_conditions_met(
        &self,
        conditions: &[ExplicitFact],
        buffer: &[u64],
    ) -> bool {
        conditions.iter().all(|condition| {
            self.global_state_packer.get(buffer, condition.var()) == condition.value() as u64
        })
    }

    /// Apply propositional effects of an operator to the state buffer.
    pub(super) fn apply_propositional_effects(
        &self,
        buffer: &mut [u64],
        previous_buffer: &[u64],
        operator: &Operator,
    ) {
        for effect in operator.effects() {
            if self.assignment_conditions_met(effect.conditions(), previous_buffer) {
                let var_id = effect.var_id();
                let value = effect.value().index() as u64;
                self.global_state_packer.set(buffer, var_id.index(), value);
            }
        }
    }

    /// The packer variable of the regular numeric variable `numeric_var_id`.
    pub(super) fn packed_slot(&self, numeric_var_id: usize) -> usize {
        match self.numeric_slots[numeric_var_id] {
            NumericSlot::Packed(packed_slot) => packed_slot,
            slot => panic!("regular numeric variable {numeric_var_id} lives in {slot:?}"),
        }
    }

    pub(super) fn cost_index(&self, numeric_var_id: usize) -> usize {
        match self.numeric_slots[numeric_var_id] {
            NumericSlot::Cost(cost_index) => cost_index,
            slot => panic!("cost numeric variable {numeric_var_id} lives in {slot:?}"),
        }
    }

    /// Applies `operator`'s numeric effects, interning the new regular values
    /// in `interner`, and closes the successor under the axioms. Without
    /// `run_full_axioms` only the arithmetic axioms run, which the metric
    /// needs; the caller then runs the comparison and propositional passes
    /// once the successor turns out to be new.
    pub(super) fn apply_numeric_effects_inner(
        &self,
        interner: &mut CompactNumericValues,
        transition: NumericTransition<'_>,
        operator: &Operator,
        run_full_axioms: bool,
    ) -> Result<(), StateInsertError> {
        let NumericTransition {
            parent_values,
            current_values,
            cost_part,
            next_buffer,
            previous_buffer,
        } = transition;
        // The effects' semantics -- simultaneous, operands read in the
        // parent, additive repeats accumulated -- are the operator's; this
        // only stores each result where its variable lives. Effect
        // conditions read the parent's buffer.
        let packer = &*self.global_state_packer;
        operator.apply_numeric_effects(
            |conditions| self.assignment_conditions_met(conditions, previous_buffer),
            parent_values,
            current_values,
            |variable, result| match self.numeric_slots[variable.index()] {
                NumericSlot::Packed(packed_slot) => {
                    packer.set(next_buffer, packed_slot, interner.intern(result));
                }
                NumericSlot::Cost(cost_index) => cost_part[cost_index] = result,
                slot @ (NumericSlot::Constant(_) | NumericSlot::Derived) => panic!(
                    "an effect of {} writes numeric variable {variable}, which lives in {slot:?}",
                    operator.name()
                ),
            },
        );

        if run_full_axioms {
            self.axiom_evaluator
                .evaluate(next_buffer, current_values)
                .map_err(|e| StateInsertError {
                    message: format!("Failed to evaluate axioms: {:?}", e),
                })?;
        } else {
            self.axiom_evaluator
                .evaluate_arithmetic_axioms(current_values)
                .map_err(|e| StateInsertError {
                    message: format!("Failed to evaluate arithmetic axioms: {:?}", e),
                })?;
        }

        Ok(())
    }
}
