//! Admissible numeric potential heuristics.
//!
//! The implementation follows the potential system in the numeric Fast
//! Downward fork. CPLEX is deliberately the only supported LP backend, so
//! everything but the configuration needs the `cplex` feature: the
//! configuration is parsed in every build, so that requesting the heuristic
//! where it is not compiled in is an error before translation.

mod config;
#[cfg(feature = "cplex")]
mod function;
#[cfg(feature = "cplex")]
mod heuristic;
#[cfg(feature = "cplex")]
mod optimizer;
#[cfg(feature = "cplex")]
mod rays;
#[cfg(feature = "cplex")]
mod sampling;
#[cfg(feature = "cplex")]
mod task;

pub use config::{
    BoundsProvider, DiverseFallback, NumericPotentialConfig, NumericPotentialFeature, OptimizeFor,
};
#[cfg(feature = "cplex")]
pub(crate) use function::NumericPotentialFunction;
#[cfg(feature = "cplex")]
pub use heuristic::NumericPotentialHeuristic;
#[cfg(feature = "cplex")]
pub(crate) use optimizer::{NumericPotentialOptimizer, OptimizationOutcome};
#[cfg(feature = "cplex")]
pub(crate) use task::PotentialTask;

#[cfg(all(test, feature = "cplex"))]
mod tests;
