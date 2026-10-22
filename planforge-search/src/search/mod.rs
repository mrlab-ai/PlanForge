//! Best-first search for numeric planning.
//!
//! Split into focused submodules:
//! - `open_list`: the open lists and its entry ordering.
//! - `space`: per-state search bookkeeping (parents, g-values, closed
//!   flags, preferred-operator snapshots) and plan extraction.
//! - `stats`: counters, progress snapshots, and trace flags.
//! - `engine`: the search loop itself.

#[cfg(test)]
mod tests;

mod config;
mod engine;
mod open_list;
mod policy;
mod space;
mod stats;

pub use engine::{AStarSearch, BestFirstSearch, GreedyBestFirstSearch, SearchParams};
pub use open_list::{OpenEntry, OpenList, PreferredFirstOpenList, TieBreakingOpenList};
pub use policy::{G, GPlusH, H, Priority};

use anyhow::Result;
use planforge_sas::numeric_task::Operator;
use planforge_sas::state_registry::StateID;
use std::time::Duration;

/// Search status indicating the outcome of the search.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SearchStatus {
    InProgress,
    Solved(StateID), // Include the goal state ID.
    Failed,
    Timeout,
    MemoryLimitReached,
}

/// A plan is a sequence of operators.
pub type Plan = Vec<Operator>;

/// Search result containing the outcome and optional plan.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub status: SearchStatus,
    pub plan: Option<Plan>,
    /// Solution cost as used by the search engine (`g`-value of the goal state).
    ///
    /// When a metric is defined, this corresponds to the accumulated metric
    /// deltas.
    /// Otherwise, it corresponds to the sum of declared operator costs.
    pub solution_cost: Option<f64>,
    pub nodes_expanded: usize,
    pub nodes_reopened: usize,
    pub nodes_evaluated: usize,
    pub evaluations: usize,
    pub nodes_generated: usize,
    pub dead_ends: usize,
    pub nodes_expanded_until_last_jump: usize,
    pub nodes_reopened_until_last_jump: usize,
    pub nodes_evaluated_until_last_jump: usize,
    pub nodes_generated_until_last_jump: usize,
    pub registered_states: usize,
    pub search_time: Duration,
}

pub trait SearchEngine {
    fn initialize(&mut self) -> Result<()>;
    fn step(&mut self) -> Result<SearchStatus>;
    fn finish(&mut self, status: SearchStatus) -> SearchResult;
    fn search(&mut self) -> Result<SearchResult> {
        self.initialize()?;
        loop {
            match self.step()? {
                SearchStatus::InProgress => continue,
                terminal => return Ok(self.finish(terminal)),
            }
        }
    }
}

pub(crate) fn format_progress_value(value: f64) -> String {
    if value.is_infinite() && value.is_sign_positive() {
        return "infinity".to_string();
    }

    if value.fract().abs() < 1e-9 {
        format!("{:.0}", value)
    } else {
        format!("{:.6}", value)
    }
}

/// Resident memory in KB from `/proc/self/status`; `None` on platforms
/// without that measurement.
#[cfg(target_os = "linux")]
pub(crate) fn current_memory_kb() -> Option<u64> {
    let status =
        std::fs::read_to_string("/proc/self/status").expect("/proc/self/status is readable");
    let kb = status
        .lines()
        .find_map(|line| line.strip_prefix("VmRSS:"))
        .and_then(|value| value.split_whitespace().next())
        .and_then(|kb| kb.parse().ok())
        .expect("/proc/self/status reports VmRSS in KB");
    Some(kb)
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn current_memory_kb() -> Option<u64> {
    None
}

/// The progress-line suffix for resident memory, empty where it is not measured.
pub(crate) fn memory_report() -> String {
    current_memory_kb().map_or_else(String::new, |kb| format!(", {kb} KB"))
}
