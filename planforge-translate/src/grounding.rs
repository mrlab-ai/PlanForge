//! Grounding budgets and the counters used to enforce them.

use std::fmt;
use std::time::{Duration, Instant};

use tracing::info;

/// Default maximum number of reachable ground actions.
pub const DEFAULT_MAX_GROUND_ACTIONS: u64 = 10_000_000;

/// Default maximum number of atoms derived by the grounding model.
pub const DEFAULT_MAX_GROUND_ATOMS: u64 = 10_000_000;

/// Default approximate memory budget for materialized grounding structures.
pub const DEFAULT_MAX_GROUNDING_MEMORY: u64 = 4 * 1024 * 1024 * 1024;

const PROGRESS_INTERVAL: Duration = Duration::from_secs(5);
const COUNT_CHECK_INTERVAL: u64 = 4096;
const MEMORY_CHECK_INTERVAL: u64 = 1024 * 1024;
const PROGRESS_CHECK_INTERVAL: u64 = 4096;

/// Limits for the finite structures materialized while grounding a task.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GroundingLimits {
    pub max_ground_actions: u64,
    pub max_ground_atoms: u64,
    pub max_grounding_memory: u64,
}

impl Default for GroundingLimits {
    fn default() -> Self {
        Self {
            max_ground_actions: DEFAULT_MAX_GROUND_ACTIONS,
            max_ground_atoms: DEFAULT_MAX_GROUND_ATOMS,
            max_grounding_memory: DEFAULT_MAX_GROUNDING_MEMORY,
        }
    }
}

/// The grounding resource whose configured limit was exceeded.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GroundingLimitKind {
    Actions,
    Atoms,
    Memory,
}

/// A grounding run stopped before it could produce a complete task.
#[derive(Debug, Eq, PartialEq)]
pub struct GroundingLimitError {
    pub kind: GroundingLimitKind,
    pub value: u64,
    pub limit: u64,
    pub phase: String,
}

impl std::error::Error for GroundingLimitError {}

fn decimal(value: u64) -> String {
    let digits = value.to_string();
    let first = digits.len() % 3;
    let mut result = String::with_capacity(digits.len() + digits.len() / 3);
    if first != 0 {
        result.push_str(&digits[..first]);
    }
    for start in (first..digits.len()).step_by(3) {
        if !result.is_empty() {
            result.push(',');
        }
        result.push_str(&digits[start..start + 3]);
    }
    result
}

fn binary_bytes(value: u64) -> String {
    const GIB: f64 = (1024_u64 * 1024 * 1024) as f64;
    const MIB: f64 = (1024_u64 * 1024) as f64;
    if value >= 1024 * 1024 * 1024 {
        format!("{:.2} GiB", value as f64 / GIB)
    } else {
        format!("{:.2} MiB", value as f64 / MIB)
    }
}

impl fmt::Display for GroundingLimitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (name, value, limit, flag) = match self.kind {
            GroundingLimitKind::Actions => (
                "action",
                format!(
                    "{} ground action{}",
                    decimal(self.value),
                    if self.value == 1 { "" } else { "s" }
                ),
                decimal(self.limit),
                "--max-ground-actions",
            ),
            GroundingLimitKind::Atoms => (
                "atom",
                format!(
                    "{} ground atom{}",
                    decimal(self.value),
                    if self.value == 1 { "" } else { "s" }
                ),
                decimal(self.limit),
                "--max-ground-atoms",
            ),
            GroundingLimitKind::Memory => (
                "memory",
                format!("approximately {}", binary_bytes(self.value)),
                binary_bytes(self.limit),
                "--max-grounding-memory",
            ),
        };
        write!(
            f,
            "grounding exceeded the {name} limit: {value} (limit {limit}) while {}; \
             the task is likely too large to ground. Raise {flag} to continue, or use a smaller \
             instance.",
            self.phase
        )
    }
}

/// Counts only allocations owned by the ground model and instantiated task.
/// It is deliberately an estimate rather than process RSS: parser state,
/// allocator bookkeeping, and temporary formatting buffers are not grounding
/// structures, while shared argument arrays and join tables are.
pub(crate) struct GroundingMonitor {
    limits: GroundingLimits,
    phase: String,
    actions: u64,
    atoms: u64,
    model_bytes: u64,
    transient_bytes: u64,
    materialized_bytes: u64,
    peak_bytes: u64,
    next_action_check: u64,
    next_atom_check: u64,
    next_memory_check: u64,
    work_since_progress_check: u64,
    last_progress: Instant,
}

impl GroundingMonitor {
    pub(crate) fn new(limits: GroundingLimits) -> Self {
        Self {
            limits,
            phase: "building the grounding model".to_owned(),
            actions: 0,
            atoms: 0,
            model_bytes: 0,
            transient_bytes: 0,
            materialized_bytes: 0,
            peak_bytes: 0,
            next_action_check: limits
                .max_ground_actions
                .saturating_add(1)
                .min(COUNT_CHECK_INTERVAL),
            next_atom_check: limits
                .max_ground_atoms
                .saturating_add(1)
                .min(COUNT_CHECK_INTERVAL),
            next_memory_check: limits
                .max_grounding_memory
                .saturating_add(1)
                .min(MEMORY_CHECK_INTERVAL),
            work_since_progress_check: 0,
            last_progress: Instant::now(),
        }
    }

    pub(crate) fn enter_phase(&mut self, phase: impl Into<String>) {
        self.phase = phase.into();
        self.progress(true);
    }

    pub(crate) fn note_model_atom(
        &mut self,
        argument_count: usize,
    ) -> Result<(), GroundingLimitError> {
        self.atoms = self.atoms.saturating_add(1);
        // GroundAtom + the Rc allocation/header. Object ids themselves are
        // four bytes. The per-predicate deduplication entry is transient.
        self.model_bytes = self
            .model_bytes
            .saturating_add(48 + 4 * argument_count as u64);
        self.transient_bytes = self.transient_bytes.saturating_add(16);
        self.check(false)?;
        self.note_progress_work();
        Ok(())
    }

    pub(crate) fn note_model_work(
        &mut self,
        transient_bytes: u64,
    ) -> Result<(), GroundingLimitError> {
        self.transient_bytes = self.transient_bytes.saturating_add(transient_bytes);
        self.check(false)?;
        self.note_progress_work();
        Ok(())
    }

    pub(crate) fn finish_model(&mut self) -> Result<(), GroundingLimitError> {
        // Rule match tables are dropped with the compiled rules here. The
        // returned model retains the unique atom queue and its deduplication
        // table no longer exists.
        self.transient_bytes = 0;
        self.check(true)
    }

    pub(crate) fn note_materialized_bytes(
        &mut self,
        bytes: u64,
    ) -> Result<(), GroundingLimitError> {
        self.materialized_bytes = self.materialized_bytes.saturating_add(bytes);
        self.check(false)
    }

    pub(crate) fn note_action(
        &mut self,
        action_name: &str,
        estimated_bytes: u64,
    ) -> Result<(), GroundingLimitError> {
        self.phase = format!("instantiating action `{action_name}`");
        self.actions = self.actions.saturating_add(1);
        self.materialized_bytes = self.materialized_bytes.saturating_add(estimated_bytes);
        self.check(false)?;
        self.note_progress_work();
        Ok(())
    }

    pub(crate) fn complete(&mut self) -> Result<(), GroundingLimitError> {
        self.check(true)?;
        info!(
            "Grounding complete: {} actions, {} atoms, approximately {} held ({} peak).",
            self.actions,
            self.atoms,
            binary_bytes(self.current_bytes()),
            binary_bytes(self.peak_bytes)
        );
        Ok(())
    }

    fn current_bytes(&self) -> u64 {
        self.model_bytes
            .saturating_add(self.transient_bytes)
            .saturating_add(self.materialized_bytes)
    }

    fn update_peak(&mut self) {
        self.peak_bytes = self.peak_bytes.max(self.current_bytes());
    }

    fn check(&mut self, force: bool) -> Result<(), GroundingLimitError> {
        self.update_peak();
        let bytes = self.current_bytes();
        let due = force
            || self.actions >= self.next_action_check
            || self.atoms >= self.next_atom_check
            || bytes >= self.next_memory_check;
        if !due {
            return Ok(());
        }

        if self.actions > self.limits.max_ground_actions {
            return Err(self.error(
                GroundingLimitKind::Actions,
                self.actions,
                self.limits.max_ground_actions,
            ));
        }
        if self.atoms > self.limits.max_ground_atoms {
            return Err(self.error(
                GroundingLimitKind::Atoms,
                self.atoms,
                self.limits.max_ground_atoms,
            ));
        }
        if bytes > self.limits.max_grounding_memory {
            return Err(self.error(
                GroundingLimitKind::Memory,
                bytes,
                self.limits.max_grounding_memory,
            ));
        }

        self.next_action_check = self.actions.saturating_add(COUNT_CHECK_INTERVAL);
        self.next_atom_check = self.atoms.saturating_add(COUNT_CHECK_INTERVAL);
        self.next_memory_check = bytes.saturating_add(MEMORY_CHECK_INTERVAL);
        Ok(())
    }

    fn error(&self, kind: GroundingLimitKind, value: u64, limit: u64) -> GroundingLimitError {
        GroundingLimitError {
            kind,
            value,
            limit,
            phase: self.phase.clone(),
        }
    }

    fn progress(&mut self, force: bool) {
        if !force && self.last_progress.elapsed() < PROGRESS_INTERVAL {
            return;
        }
        info!(
            "Grounding progress: {} actions, {} atoms, approximately {} held; {}.",
            self.actions,
            self.atoms,
            binary_bytes(self.current_bytes()),
            self.phase
        );
        self.last_progress = Instant::now();
    }

    fn note_progress_work(&mut self) {
        self.work_since_progress_check = self.work_since_progress_check.saturating_add(1);
        if self.work_since_progress_check >= PROGRESS_CHECK_INTERVAL {
            self.work_since_progress_check = 0;
            self.progress(false);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_error_names_the_limit_value_and_phase() {
        let error = GroundingLimitError {
            kind: GroundingLimitKind::Actions,
            value: 12_000_000,
            limit: 10_000_000,
            phase: "instantiating action `pick-up`".to_owned(),
        };
        assert_eq!(
            error.to_string(),
            "grounding exceeded the action limit: 12,000,000 ground actions (limit 10,000,000) \
             while instantiating action `pick-up`; the task is likely too large to ground. Raise \
             --max-ground-actions to continue, or use a smaller instance."
        );
    }
}
