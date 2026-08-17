use super::Plan;
use planforge_sas::numeric_task::AbstractNumericTask;
use planforge_sas::state_registry::StateID;

/// Simple search node information for tracking parent relationships.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SearchNodeInfo {
    pub(crate) parent_state: Option<StateID>,
    pub(crate) parent_operator_id: Option<usize>,
    pub(crate) g_value: f64,
    pub(crate) is_dead_end: bool,
    pub(crate) is_closed: bool,
    pub(crate) is_goal: bool,
}

const NO_COMPACT_ID: u32 = u32::MAX;
const NODE_PRESENT: u8 = 1 << 0;
const NODE_DEAD_END: u8 = 1 << 1;
const NODE_CLOSED: u8 = 1 << 2;
const NODE_GOAL: u8 = 1 << 3;
const NO_PREFERRED_RANGE: (u32, u32) = (u32::MAX, 0);

/// Per-state search bookkeeping: the node table (parents, g-values,
/// dead-end/closed flags) and the per-state preferred-operator snapshots.
/// Both tables are indexed by `StateID`.
#[derive(Debug, Default)]
pub(crate) struct SearchSpace {
    // Structure-of-arrays storage keeps ordinary A* bookkeeping at 17 bytes
    // per registered node instead of padding Option<SearchNodeInfo> to 48.
    parent_states: Vec<u32>,
    parent_operator_ids: Vec<u32>,
    g_values: Vec<f64>,
    node_status: Vec<u8>,
    /// Per-state cache of preferred operator IDs reported by the
    /// heuristic for that state, indexed by `state_id`. Populated right
    /// after `evaluate_state` returns `Ok` (so the snapshot is captured
    /// before the heuristic's internal cache is overwritten by the next
    /// state's evaluation). Read back when the state is *expanded* — we
    /// then mark each successor's open-list entry as preferred iff the
    /// operator that generated it is in this set.
    preferred_pool: Vec<u32>,
    preferred_ranges: Vec<(u32, u32)>,
}

impl SearchSpace {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn node(&self, state_id: StateID) -> Option<SearchNodeInfo> {
        let &status = self.node_status.get(state_id)?;
        if status & NODE_PRESENT == 0 {
            return None;
        }
        Some(SearchNodeInfo {
            parent_state: decode_compact_id(self.parent_states[state_id]),
            parent_operator_id: decode_compact_id(self.parent_operator_ids[state_id]),
            g_value: self.g_values[state_id],
            is_dead_end: status & NODE_DEAD_END != 0,
            is_closed: status & NODE_CLOSED != 0,
            is_goal: status & NODE_GOAL != 0,
        })
    }

    pub(crate) fn contains_node(&self, state_id: StateID) -> bool {
        self.node_status
            .get(state_id)
            .is_some_and(|status| status & NODE_PRESENT != 0)
    }

    pub(crate) fn mark_dead_end(&mut self, state_id: StateID) {
        let status = self
            .node_status
            .get_mut(state_id)
            .expect("cannot mark an unallocated search node as a dead end");
        assert!(
            *status & NODE_PRESENT != 0,
            "cannot mark an absent search node as a dead end"
        );
        *status |= NODE_DEAD_END;
    }

    pub(crate) fn mark_closed(&mut self, state_id: StateID) {
        let status = self
            .node_status
            .get_mut(state_id)
            .expect("cannot close an unallocated search node");
        assert!(
            *status & NODE_PRESENT != 0,
            "cannot close an absent search node"
        );
        *status |= NODE_CLOSED;
    }

    pub(crate) fn is_goal(&self, state_id: StateID) -> bool {
        let status = self
            .node_status
            .get(state_id)
            .unwrap_or_else(|| panic!("missing search node for state {state_id}"));
        assert!(
            status & NODE_PRESENT != 0,
            "missing search node for state {state_id}"
        );
        status & NODE_GOAL != 0
    }

    pub(crate) fn set_node(&mut self, state_id: StateID, info: SearchNodeInfo) {
        if state_id >= self.node_status.len() {
            let new_len = state_id
                .checked_add(1)
                .expect("search node table length overflow");
            self.parent_states.resize(new_len, NO_COMPACT_ID);
            self.parent_operator_ids.resize(new_len, NO_COMPACT_ID);
            self.g_values.resize(new_len, 0.0);
            self.node_status.resize(new_len, 0);
        }
        self.parent_states[state_id] = encode_compact_id(info.parent_state, "parent state");
        self.parent_operator_ids[state_id] =
            encode_compact_id(info.parent_operator_id, "parent operator");
        self.g_values[state_id] = info.g_value;
        self.node_status[state_id] = NODE_PRESENT
            | if info.is_dead_end { NODE_DEAD_END } else { 0 }
            | if info.is_closed { NODE_CLOSED } else { 0 }
            | if info.is_goal { NODE_GOAL } else { 0 };
    }

    pub(crate) fn store_preferred(&mut self, state_id: StateID, ids: &[u32]) {
        if ids.is_empty() {
            if let Some(snapshot) = self.preferred_ranges.get_mut(state_id) {
                *snapshot = NO_PREFERRED_RANGE;
            }
            return;
        }
        if state_id >= self.preferred_ranges.len() {
            self.preferred_ranges
                .resize(state_id + 1, NO_PREFERRED_RANGE);
        }
        let start = u32::try_from(self.preferred_pool.len())
            .expect("preferred operator arena offset must fit in u32");
        let len = u32::try_from(ids.len()).expect("preferred operator list length must fit in u32");
        self.preferred_pool.extend_from_slice(ids);
        self.preferred_ranges[state_id] = (start, len);
    }

    /// Remove and return the cached preferred-op range for `state_id`, if any.
    pub(crate) fn take_preferred(&mut self, state_id: StateID) -> Option<(u32, u32)> {
        let range = self.preferred_ranges.get_mut(state_id)?;
        let result = *range;
        *range = NO_PREFERRED_RANGE;
        (result != NO_PREFERRED_RANGE).then_some(result)
    }

    pub(crate) fn preferred_contains(&self, range: (u32, u32), operator_id: u32) -> bool {
        let (start, len) = range;
        let start = start as usize;
        let end = start
            .checked_add(len as usize)
            .expect("preferred operator range overflow");
        self.preferred_pool
            .get(start..end)
            .expect("preferred operator range must lie in the arena")
            .contains(&operator_id)
    }

    /// Trace back the path from goal state to initial state.
    pub(crate) fn extract_plan(&self, goal_state: StateID, task: &dyn AbstractNumericTask) -> Plan {
        let mut plan = Vec::new();
        let mut current_state = goal_state;

        while let Some(node_info) = self.node(current_state) {
            if let (Some(parent_state), Some(operator_id)) =
                (node_info.parent_state, node_info.parent_operator_id)
            {
                plan.push(task.get_operators()[operator_id].clone());
                current_state = parent_state;
            } else {
                break; // Reached initial state
            }
        }

        plan.reverse();
        plan
    }
}

fn encode_compact_id(id: Option<usize>, kind: &str) -> u32 {
    let Some(id) = id else {
        return NO_COMPACT_ID;
    };
    let compact = u32::try_from(id).unwrap_or_else(|_| panic!("{kind} id {id} exceeds u32"));
    assert_ne!(
        compact, NO_COMPACT_ID,
        "{kind} id {id} collides with the missing-id sentinel"
    );
    compact
}

fn decode_compact_id(id: u32) -> Option<usize> {
    (id != NO_COMPACT_ID).then_some(id as usize)
}

#[cfg(test)]
mod tests {
    use super::{SearchNodeInfo, SearchSpace};

    #[test]
    fn compact_node_storage_round_trips_and_updates_status() {
        let mut space = SearchSpace::new();
        space.set_node(
            7,
            SearchNodeInfo {
                parent_state: Some(3),
                parent_operator_id: Some(11),
                g_value: 4.5,
                is_dead_end: false,
                is_closed: false,
                is_goal: true,
            },
        );

        let node = space.node(7).expect("stored node must exist");
        assert_eq!(node.parent_state, Some(3));
        assert_eq!(node.parent_operator_id, Some(11));
        assert_eq!(node.g_value, 4.5);
        assert!(!node.is_dead_end);
        assert!(!node.is_closed);
        assert!(node.is_goal);
        assert!(space.is_goal(7));

        space.mark_dead_end(7);
        space.mark_closed(7);
        let node = space.node(7).expect("updated node must exist");
        assert!(node.is_dead_end);
        assert!(node.is_closed);
        assert!(!space.contains_node(6));
    }

    #[test]
    fn empty_preferred_snapshots_do_not_allocate_per_state_storage() {
        let mut space = SearchSpace::new();
        space.store_preferred(1_000_000, &[]);
        assert!(space.preferred_ranges.is_empty());

        space.store_preferred(3, &[2, 5]);
        assert_eq!(space.preferred_ranges.len(), 4);
        let range = space.take_preferred(3).expect("preferred range must exist");
        assert!(space.preferred_contains(range, 2));
        assert!(space.preferred_contains(range, 5));
        assert!(!space.preferred_contains(range, 7));
    }
}
