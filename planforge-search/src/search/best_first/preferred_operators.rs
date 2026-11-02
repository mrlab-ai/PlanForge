//! The preferred operators of evaluated states, for open lists that favour
//! the successors they generate.

use planforge_sas::numeric_task::OperatorIndex;
use planforge_sas::state_registry::StateID;

/// The preferred operators of one state: a slice of the arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) struct PreferredRange {
    start: u32,
    len: u32,
}

/// The preferred operators the heuristic reported for each evaluated state,
/// read back when the state is expanded to mark the successors they generate
/// as preferred. The ranges index the arena, which only grows: a state's
/// operators are stored once per evaluation.
#[derive(Debug, Default)]
pub(super) struct PreferredOperators {
    arena: Vec<OperatorIndex>,
    ranges: Vec<PreferredRange>,
}

impl PreferredOperators {
    /// Records `operators` as the preferred operators of `state_id`,
    /// replacing those of an earlier evaluation.
    pub(super) fn store(&mut self, state_id: StateID, operators: &[OperatorIndex]) {
        if operators.is_empty() {
            if let Some(range) = self.ranges.get_mut(state_id) {
                *range = PreferredRange::default();
            }
            return;
        }
        if state_id >= self.ranges.len() {
            self.ranges.resize(state_id + 1, PreferredRange::default());
        }
        self.ranges[state_id] = PreferredRange {
            start: u32::try_from(self.arena.len())
                .expect("preferred operator arena offset must fit in u32"),
            len: u32::try_from(operators.len())
                .expect("preferred operator list length must fit in u32"),
        };
        self.arena.extend_from_slice(operators);
    }

    /// The preferred operators of `state_id`, which the search expands now
    /// and only evaluates again (storing new ones) if it reopens it.
    pub(super) fn take(&mut self, state_id: StateID) -> PreferredRange {
        self.ranges
            .get_mut(state_id)
            .map(std::mem::take)
            .unwrap_or_default()
    }

    pub(super) fn contains(&self, range: PreferredRange, operator_id: OperatorIndex) -> bool {
        let start = range.start as usize;
        self.arena[start..start + range.len as usize].contains(&operator_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_preferred_snapshots_do_not_allocate_per_state_storage() {
        let mut preferred = PreferredOperators::default();
        preferred.store(1_000_000, &[]);
        assert!(preferred.ranges.is_empty());

        preferred.store(3, &[OperatorIndex::new(2), OperatorIndex::new(5)]);
        assert_eq!(preferred.ranges.len(), 4);
        let range = preferred.take(3);
        assert!(preferred.contains(range, OperatorIndex::new(2)));
        assert!(preferred.contains(range, OperatorIndex::new(5)));
        assert!(!preferred.contains(range, OperatorIndex::new(7)));
        let taken = preferred.take(3);
        assert!(!preferred.contains(taken, OperatorIndex::new(2)));
    }
}
