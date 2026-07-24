//! Admissible numeric potential heuristics.
//!
//! The implementation follows the potential system in the numeric Fast
//! Downward fork. CPLEX is deliberately the only supported LP backend.

mod config;
mod function;
mod heuristic;
mod ocp;
mod optimizer;
mod rays;
mod sampling;
mod task;

pub use config::{BoundsProvider, DiverseFallback, NumericPotentialConfig, OptimizeFor};
pub use function::NumericPotentialFunction;
pub use heuristic::NumericPotentialHeuristic;
pub use ocp::PotentialAbstractionOcpHeuristic;
pub use optimizer::{NumericPotentialOptimizer, OptimizationOutcome};
pub use task::{FeatureBounds, PotentialTask};

pub fn assert_cplex_ready() -> Result<(), String> {
    planforge_cplex::assert_unrestricted_license()
        .map_err(|error| format!("an unrestricted CPLEX installation is required: {error}"))
}

#[cfg(test)]
mod tests;
