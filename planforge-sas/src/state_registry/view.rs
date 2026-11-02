//! Registered states and read-only views of their values.

use super::{CompactNumericValues, DenseCostInformation, StateID, StateRegistry};
use crate::axioms::AxiomEvaluator;
use crate::explicit_state::ExplicitState;
use crate::numeric_task::{ExplicitFact, ExplicitValueIndex, NumericValue};
use crate::utils::errors::AssignmentAxiomError;
use crate::utils::state_packer::StatePacker;

/// A state a [`StateRegistry`] registered: its handle, the [`StateID`]
/// indexing the registry's pool, without the values. "Concrete" as opposed to
/// the abstract states of an abstraction.
///
/// The three ways to hold a state:
///
/// - `ConcreteState`: the handle a search keeps per reached state; read its
///   values through [`StateRegistry::view`], which yields a
///   [`ConcreteStateView`].
/// - [`ExplicitState`]: owned, decoded values outside any registry, for code
///   that follows a few states of its own, such as CEGAR replaying a plan.
/// - [`StateValues`](crate::explicit_state::StateValues): borrowed values for
///   lookups, decoded numeric values with propositional values either decoded
///   or read from the packed bins on demand.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConcreteState {
    pub(super) pool_offset: usize,
}

/// A short-lived, read-only view of concrete state values.
///
/// Registered states borrow their registry only for the lifetime of the view;
/// decoded CEGAR states borrow their numeric slice directly. The type cannot
/// outlive either backing store and therefore does not create the long-lived
/// aliasing problem of a `(StateID, &StateRegistry)` state handle.
#[derive(Clone, Copy)]
pub struct ConcreteStateView<'a> {
    pub(super) packer: &'a StatePacker,
    pub(super) prop: &'a [u64],
    pub(super) backing: ConcreteStateViewBacking<'a>,
}

#[derive(Clone, Copy)]
pub(super) enum ConcreteStateViewBacking<'a> {
    Decoded(&'a [NumericValue]),
    Registered {
        state_id: StateID,
        numeric_template: &'a [NumericValue],
        regular_numeric_slots: &'a [(usize, usize)],
        cost_numeric_indices: &'a [(usize, usize)],
        compact_numeric_values: &'a CompactNumericValues,
        cost_info: &'a DenseCostInformation,
        axiom_evaluator: &'a AxiomEvaluator,
    },
}

impl<'a> ConcreteStateView<'a> {
    pub fn from_decoded(
        packer: &'a StatePacker,
        prop: &'a [u64],
        numeric: &'a [NumericValue],
    ) -> Self {
        Self {
            packer,
            prop,
            backing: ConcreteStateViewBacking::Decoded(numeric),
        }
    }
}

impl ExplicitFact {
    pub fn is_held(&self, state: ConcreteStateView<'_>) -> bool {
        let value = state.packer().get(state.propositional(), self.var());
        value == self.value() as u64
    }
}

impl<'a> ConcreteStateView<'a> {
    pub fn packer(self) -> &'a StatePacker {
        self.packer
    }

    pub fn propositional(self) -> &'a [u64] {
        self.prop
    }

    /// Whether every one of `facts` holds.
    pub fn holds_all(self, facts: &[ExplicitFact]) -> bool {
        facts.iter().all(|fact| fact.is_held(self))
    }

    /// The `facts` that do not hold, in order.
    pub fn unsatisfied<'f>(
        self,
        facts: &'f [ExplicitFact],
    ) -> impl Iterator<Item = &'f ExplicitFact> + use<'a, 'f> {
        facts.iter().filter(move |fact| !fact.is_held(self))
    }

    pub fn fill_propositional(self, output: &mut Vec<ExplicitValueIndex>) {
        output.clear();
        output.extend(
            (0..self.packer.numeric_slot_offset()).map(|slot| {
                ExplicitValueIndex::from_usize(self.packer.get(self.prop, slot) as usize)
            }),
        );
    }

    /// Writes the numeric values into `output`, the derived ones evaluated by
    /// the arithmetic axioms.
    pub fn fill_numeric(self, output: &mut Vec<NumericValue>) -> Result<(), AssignmentAxiomError> {
        match self.backing {
            ConcreteStateViewBacking::Decoded(values) => {
                output.clear();
                output.extend_from_slice(values);
            }
            ConcreteStateViewBacking::Registered {
                axiom_evaluator, ..
            } => {
                self.fill_numeric_unevaluated(output);
                if axiom_evaluator.has_numeric_axioms() {
                    axiom_evaluator.evaluate_arithmetic_axioms(output)?;
                }
            }
        }
        Ok(())
    }

    /// Writes the numeric values into `output` before the arithmetic axioms:
    /// the regular, constant and cost values, with every derived value zero.
    ///
    /// # Panics
    /// Panics on a decoded view, whose values are already closed under the
    /// axioms.
    pub fn fill_numeric_unevaluated(self, output: &mut Vec<NumericValue>) {
        let ConcreteStateViewBacking::Registered {
            state_id,
            numeric_template,
            regular_numeric_slots,
            cost_numeric_indices,
            compact_numeric_values,
            cost_info,
            ..
        } = self.backing
        else {
            panic!("a decoded state view has no numeric values before the axioms");
        };
        output.clear();
        output.extend_from_slice(numeric_template);
        let interned = compact_numeric_values;
        for &(out_idx, packed_slot) in regular_numeric_slots {
            output[out_idx] = interned.value(self.packer.get(self.prop, packed_slot));
        }
        let cost_values = cost_info.get(state_id);
        for &(out_idx, cost_idx) in cost_numeric_indices {
            output[out_idx] = cost_values[cost_idx];
        }
    }

    /// Writes the propositional and numeric values, derived ones included,
    /// into `state`.
    pub fn fill_explicit(self, state: &mut ExplicitState) -> Result<(), AssignmentAxiomError> {
        self.fill_propositional(&mut state.propositional);
        self.fill_numeric(&mut state.numeric)
    }
}

impl ConcreteState {
    /// Create a new concrete state with the given pool offset.
    pub const fn new(pool_offset: usize) -> Self {
        Self { pool_offset }
    }

    /// Get the state ID (equivalent to C++ `GlobalState::get_id()`).
    /// This is the index into the state registry's data pool.
    pub fn id(&self) -> usize {
        self.pool_offset
    }

    /// Get the propositional state values as a vector.
    pub fn propositional_values(&self, state_registry: &StateRegistry) -> Vec<ExplicitValueIndex> {
        let mut values = Vec::new();
        state_registry.view(self).fill_propositional(&mut values);
        values
    }

    /// Fill `output` with the propositional state values without allocating a new vector.
    pub fn fill_propositional_values(
        &self,
        state_registry: &StateRegistry,
        output: &mut Vec<ExplicitValueIndex>,
    ) {
        state_registry.view(self).fill_propositional(output);
    }

    /// Get the numeric state values for regular variables.
    pub fn regular_numeric_values(&self, state_registry: &StateRegistry) -> Vec<NumericValue> {
        let buffer = self.buffer(state_registry);
        let packer = &state_registry.layout.global_state_packer;
        let interned = &state_registry.compact_numeric_values;
        state_registry
            .layout
            .regular_numeric_slots
            .iter()
            .map(|&(_, packed_slot)| interned.value(packer.get(buffer, packed_slot)))
            .collect()
    }

    /// Get a reference to the underlying buffer for this state.
    pub fn buffer<'a>(&self, state_registry: &'a StateRegistry) -> &'a [u64] {
        state_registry.buffer(self.pool_offset)
    }
}
