//! Best-first search for numeric planning.
//!
//! Split into focused submodules:
//! - [`open_list`]: the dual-queue open list and its entry ordering.
//! - [`space`]: per-state search bookkeeping (parents, g-values, closed
//!   flags, preferred-operator snapshots) and plan extraction.
//! - [`stats`]: counters, progress snapshots, and trace flags.
//! - [`engine`]: the search loop itself.

#[cfg(test)]
mod tests;

mod config;
mod engine;
mod open_list;
mod policy;
mod space;
mod stats;

pub use engine::AStarSearch;
pub use policy::SearchPolicy;

use planforge_sas::numeric::numeric_task::{
    AbstractNumericTask, Operator, metric_operator_cost_from_initial_values,
};
use planforge_sas::numeric::state_registry::{ConcreteState, StateID, StateRegistry};
use std::time::Duration;

pub fn compute_effective_operator_costs(
    task: &dyn AbstractNumericTask,
    _state_registry: &StateRegistry<'_>,
    _initial_state: &ConcreteState,
) -> Vec<f64> {
    task.get_operators()
        .iter()
        .map(|op| metric_operator_cost_from_initial_values(task, op))
        .collect()
}

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
    fn initialize(&mut self);
    fn step(&mut self) -> SearchStatus;
    fn finish(&mut self, status: SearchStatus) -> SearchResult;
    fn search(&mut self) -> SearchResult {
        self.initialize();
        loop {
            match self.step() {
                SearchStatus::InProgress => continue,
                terminal => return self.finish(terminal),
            }
        }
    }
    fn print_initial_h_values(&mut self);
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

#[cfg(target_os = "linux")]
pub(crate) fn current_memory_kb() -> u64 {
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if let Some(value) = line.strip_prefix("VmRSS:")
                && let Some(kb) = value
                    .split_whitespace()
                    .next()
                    .and_then(|part| part.parse::<u64>().ok())
            {
                return kb;
            }
        }
    }

    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    let result = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if result != 0 {
        return 0;
    }
    let usage = unsafe { usage.assume_init() };
    usage.ru_maxrss.max(0) as u64
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn current_memory_kb() -> u64 {
    0
}
