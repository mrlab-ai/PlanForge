//! State-space search for numeric planning.
//!
//! - [`SearchConfig::solve`] runs the search a [`SearchConfig`] describes.
//! - `config`: the `--search` grammar and its registry,
//!   [`SEARCH_ALGORITHMS`]; `algorithms`: the built-in algorithms.
//!
//! A search algorithm is composed of two building blocks:
//!
//! - [`SearchSpace`]: the registered states, successor generation, duplicate
//!   detection on the search node of each reached state, and plan extraction;
//! - [`SearchProgress`]: the time and memory limits, the
//!   [`SearchStatistics`] and the progress reports.
//!
//! [`best_first`] builds best-first search (A*, greedy best-first search) on
//! them; `tutorials/rust-custom-search` builds enforced hill-climbing.

mod algorithms;
pub mod best_first;
mod config;
mod progress;
mod search_space;
mod statistics;

pub use algorithms::{AStarConfig, AStarFeature, GreedyBestFirstConfig, GreedyBestFirstFeature};
pub use best_first::{AStarSearch, BestFirstSearch, GreedyBestFirstSearch, SearchParams};
pub use config::{
    SEARCH_ALGORITHMS, SearchConfig, SearchEntry, SearchFeature, SearchLimits, entry,
    parse_search_spec, search_algorithm_help,
};
pub use progress::SearchProgress;
pub use search_space::{Reached, SearchNode, SearchSpace, Successor};
pub use statistics::{NodeCounters, SearchStatistics};

use planforge_sas::numeric_task::Operator;
use std::time::Duration;

/// A plan: the operators to apply in order.
pub type Plan = Vec<Operator>;

/// How a search ended.
#[derive(Debug, Clone, PartialEq)]
pub enum SearchOutcome {
    /// A plan and its cost under the cost function the search minimised: the
    /// metric when the task has one, otherwise the operator costs.
    Solved {
        plan: Plan,
        cost: f64,
    },
    /// The open list ran empty: no plan exists, unless an unsafe heuristic
    /// pruned one as a dead end.
    Unsolvable,
    Timeout,
    OutOfMemory,
}

/// A finished search: its outcome and what it counted on the way.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub outcome: SearchOutcome,
    pub statistics: SearchStatistics,
    pub search_time: Duration,
}
