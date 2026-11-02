//! What a search counts, the limits it checks and the progress it reports.

use std::time::{Duration, Instant};

use tracing::info;

use super::search_space::SearchSpace;
use super::{SearchLimits, SearchOutcome, SearchResult, SearchStatistics};
use crate::resource_limits::resident_memory_bytes;

/// The resident memory is measured once per this many expansions.
const MEMORY_CHECK_EXPANSION_INTERVAL: usize = 1024;

/// A running search's clock, limits, statistics and progress reports.
pub struct SearchProgress {
    /// Updated by the search as it expands, generates and evaluates.
    pub statistics: SearchStatistics,
    start: Instant,
    limits: SearchLimits,
    next_memory_check: usize,
    heuristic_name: String,
    best_h: f64,
    /// The highest priority layer reported so far.
    last_layer: Option<i64>,
}

impl SearchProgress {
    /// Starts the clock of a search guided by the heuristic `heuristic_name`.
    pub fn start(limits: SearchLimits, heuristic_name: &str) -> Self {
        assert!(
            limits.memory_bytes.is_none() || resident_memory_bytes().is_some(),
            "a search memory limit needs resident-memory measurement, which only Linux provides"
        );
        Self {
            statistics: SearchStatistics::default(),
            start: Instant::now(),
            limits,
            next_memory_check: 0,
            heuristic_name: heuristic_name.to_owned(),
            best_h: f64::INFINITY,
            last_layer: None,
        }
    }

    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// The outcome of a search that ran out of time or memory, if it did.
    /// Memory is measured once per 1024 expansions.
    pub fn limit_reached(&mut self) -> Option<SearchOutcome> {
        if self
            .limits
            .time
            .is_some_and(|limit| self.start.elapsed() > limit)
        {
            return Some(SearchOutcome::Timeout);
        }
        let max_memory_bytes = self.limits.memory_bytes?;
        let expanded = self.statistics.nodes.expanded;
        if expanded < self.next_memory_check {
            return None;
        }
        self.next_memory_check = expanded + MEMORY_CHECK_EXPANSION_INTERVAL;
        let memory = resident_memory_bytes()
            .expect("a memory limit is only accepted where memory is measured");
        (memory >= max_memory_bytes).then_some(SearchOutcome::OutOfMemory)
    }

    pub fn report_initial_h(&self, h: f64) {
        info!(
            "Initial heuristic value for {}: {}",
            self.heuristic_name,
            format_progress_value(h)
        );
    }

    /// Reports `h`, reached on a path of cost `g`, if it is below every
    /// estimate reported so far; returns whether it was.
    #[inline]
    pub fn report_best_h(&mut self, h: f64, g: f64) -> bool {
        if h >= self.best_h {
            return false;
        }
        self.report_new_best_h(h, g);
        true
    }

    #[cold]
    fn report_new_best_h(&mut self, h: f64, g: f64) {
        self.best_h = h;
        info!(
            "New best heuristic value for {}: {}",
            self.heuristic_name,
            format_progress_value(h),
        );
        info!(
            "[g={}, {} evaluated, {} expanded, t={:.6}s{}]",
            format_progress_value(g),
            self.statistics.nodes.evaluated,
            self.statistics.nodes.expanded,
            self.elapsed().as_secs_f64(),
            memory_report(),
        );
    }

    /// Reports the first expansion in a higher layer of a monotone priority
    /// named `label`, and counts the nodes up to it (Fast Downward's "until
    /// last jump").
    pub fn report_layer(&mut self, label: &str, priority: f64, space: &SearchSpace, open: usize) {
        let layer = priority as i64;
        if self.last_layer.is_some_and(|last| layer <= last) {
            return;
        }
        self.last_layer = Some(layer);
        self.statistics.nodes_until_last_jump = self.statistics.nodes;
        info!(
            "{label} = {layer} [{} evaluated, {} expanded, {} states, {open} open, t={:.6}s{}]",
            self.statistics.nodes.evaluated,
            self.statistics.nodes.expanded,
            space.num_registered_states(),
            self.elapsed().as_secs_f64(),
            memory_report(),
        );
    }

    /// The result of the search of `space` that ended with `outcome`.
    pub fn finish(self, outcome: SearchOutcome, space: &SearchSpace) -> SearchResult {
        SearchResult {
            outcome,
            statistics: SearchStatistics {
                registered_states: space.num_registered_states(),
                ..self.statistics
            },
            search_time: self.start.elapsed(),
        }
    }
}

/// `value` as a progress report prints it: integral values without decimals.
fn format_progress_value(value: f64) -> String {
    if value == f64::INFINITY {
        "infinity".to_owned()
    } else if value.fract().abs() < 1e-9 {
        format!("{value:.0}")
    } else {
        format!("{value:.6}")
    }
}

/// The progress-line suffix for resident memory, empty where it is not
/// measured.
fn memory_report() -> String {
    resident_memory_bytes().map_or_else(String::new, |bytes| format!(", {} KB", bytes / 1024))
}

#[cfg(test)]
mod tests {
    use super::format_progress_value;

    #[test]
    fn progress_values_print_six_decimals() {
        assert_eq!(format_progress_value(95.4940004), "95.494000");
        assert_eq!(format_progress_value(95.49400049), "95.494000");
    }
}
