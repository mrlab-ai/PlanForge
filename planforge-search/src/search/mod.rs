//! State-space search for numeric planning.
//!
//! - [`ConfiguredSearch::solve`] runs the search a [`SearchConfig`] describes;
//!   [`registry::Features::parse_search`](crate::registry::Features::parse_search)
//!   parses one from a `--search` string.
//! - `config`: the `--search` language and [`SearchFeature`];
//!   `algorithms`: the built-in algorithms.
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
pub use config::{ConfiguredSearch, SearchConfig, SearchEntry, SearchFeature, SearchLimits, entry};
pub use progress::SearchProgress;
pub use search_space::{Reached, SearchNode, SearchSpace, Successor};
pub use statistics::{NodeCounters, SearchStatistics};

use std::fmt;
use std::time::Duration;

use planforge_sas::numeric_task::Operator;
use planforge_sas::utils::errors::StateInsertError;

use crate::evaluation::EvaluationError;
use crate::heuristic_features::HeuristicBuildError;

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

/// Why a search failed, as opposed to ending without a plan.
#[derive(Debug)]
pub enum SearchError {
    /// A heuristic the search uses could not be built.
    Build(HeuristicBuildError),
    /// A heuristic failed to evaluate a state; `context` names the state.
    Evaluation {
        context: String,
        source: EvaluationError,
    },
    /// The successors of a state could not be generated; `context` names the
    /// state and operator.
    Expansion {
        context: String,
        source: StateInsertError,
    },
    /// A failure particular to the search algorithm.
    Failed(String),
}

impl fmt::Display for SearchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Build(error) => fmt::Display::fmt(error, f),
            Self::Evaluation { context, source } => write!(f, "{context}: {source}"),
            Self::Expansion { context, source } => write!(f, "{context}: {source}"),
            Self::Failed(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for SearchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Build(error) => Some(error),
            Self::Evaluation { source, .. } => Some(source),
            Self::Expansion { source, .. } => Some(source),
            Self::Failed(_) => None,
        }
    }
}

impl From<HeuristicBuildError> for SearchError {
    fn from(error: HeuristicBuildError) -> Self {
        Self::Build(error)
    }
}

/// A finished search: its outcome and what it counted on the way.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub outcome: SearchOutcome,
    pub statistics: SearchStatistics,
    pub search_time: Duration,
}
