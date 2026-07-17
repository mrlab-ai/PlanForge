pub mod abstract_operator_generator;
pub(crate) mod additive_numeric_views;
pub mod comparison_expression;
pub mod domain_abstraction;
pub mod domain_abstraction_collection_generator_multiple_cegar;
pub mod numeric_context;
// Only compiled with the `highs` feature; the posthoc-optimization heuristic
// depends on the HiGHS LP solver (which requires libclang to build).
#[cfg(feature = "highs")]
pub mod posthoc_optimization_heuristic;

pub mod cegar;
pub mod domain_abstraction_factory;
pub mod domain_abstraction_generator;
pub mod domain_abstraction_heuristic;
pub mod utils;
