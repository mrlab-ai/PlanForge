pub mod abstract_operator_generator;
pub(crate) mod additive_numeric_views;
pub mod domain_abstraction;
pub mod domain_abstraction_collection_generator_multiple_cegar;
pub mod numeric_context;
// Only compiled with the `cplex` feature. Requesting the heuristic in another
// build produces an explicit configuration error in planforge-searcher.
#[cfg(feature = "cplex")]
pub mod posthoc_optimization_heuristic;

pub mod cegar;
pub mod domain_abstraction_factory;
pub mod domain_abstraction_generator;
pub mod domain_abstraction_heuristic;
pub mod utils;

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
    numeric_var_id: usize,
) -> usize {
    num_propositional_vars + numeric_var_id
}
