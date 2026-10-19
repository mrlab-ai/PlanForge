//! Admissible combinations of abstraction heuristics: maximization,
//! canonical heuristics, saturated cost partitioning and post-hoc
//! optimization.

pub mod canonical;
pub mod costs;
pub mod max;
// Only compiled with the `cplex` feature. Requesting the heuristic in another
// build produces an explicit configuration error in planforge-searcher.
#[cfg(feature = "cplex")]
pub mod posthoc_optimization;
pub mod saturated;
