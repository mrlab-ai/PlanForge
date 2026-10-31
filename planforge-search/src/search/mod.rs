//! State-space search for numeric planning.
//!
//! - [`solve_task`] runs the search a [`SearchConfig`] describes.
//! - `config`: the `--search` grammar, the algorithms and their options.
//! - [`best_first`]: best-first search (A*, greedy best-first search,
//!   uniform-cost search) and its open lists and node bookkeeping.
//! - `statistics`: what a search counts.

pub mod best_first;
mod config;
mod solve;
mod statistics;

pub use best_first::{AStarSearch, BestFirstSearch, GreedyBestFirstSearch, SearchParams};
pub use config::{SearchConfig, parse_search_spec, search_algorithm_help};
pub use solve::solve_task;
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
