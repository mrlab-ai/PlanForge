//! Numeric domain abstractions refined by CEGAR.
//!
//! A domain abstraction maps every variable on its own: propositional values
//! through a domain mapping, numeric values through interval partitions. The
//! abstract state space is the product of these, so splitting one value
//! splits every abstract state that contains it -- the factored counterpart
//! of a Cartesian abstraction, which splits one abstract state at a time.

use planforge_sas::numeric_task::VariableIndex;

pub mod abstract_operators;
pub mod abstraction;
pub mod cegar;
pub mod collection;
pub mod comparison_inputs;
pub mod config;
pub(crate) mod debug_dump;
pub mod factory;
pub mod flaws;
pub mod generator;
pub mod heuristic;
pub mod initial_splits;
pub mod partitions;
pub mod refinement;
pub mod split_selection;

/// The id a domain abstraction gives numeric variable `numeric_var_id`.
///
/// An abstraction addresses both kinds of variable through one dense id space:
/// the task's `num_propositional_vars` propositional variables first, its
/// numeric variables after them. `hash_multipliers` is indexed by that space,
/// and facts on the upper range carry
/// [`FactNamespace::NumericVariable`](planforge_sas::numeric_task::FactNamespace::NumericVariable)
/// so the offset is readable off the fact instead of off the arithmetic that
/// produced it.
#[inline]
pub const fn abstraction_numeric_var(
    num_propositional_vars: usize,
    numeric_var_id: VariableIndex,
) -> VariableIndex {
    VariableIndex::from_usize(num_propositional_vars + numeric_var_id.index())
}
