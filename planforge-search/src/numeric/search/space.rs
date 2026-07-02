use super::Plan;
use planforge_sas::numeric::numeric_task::AbstractNumericTask;
use planforge_sas::numeric::state_registry::StateID;

/// Simple search node information for tracking parent relationships.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SearchNodeInfo {
    pub(crate) parent_state: Option<StateID>,
    pub(crate) parent_operator_id: Option<usize>,
    pub(crate) g_value: f64,
    pub(crate) is_dead_end: bool,
    pub(crate) is_closed: bool,
}

/// Per-state search bookkeeping: the node table (parents, g-values,
/// dead-end/closed flags) and the per-state preferred-operator snapshots.
/// Both tables are indexed by `StateID`.
#[derive(Debug, Default)]
pub(crate) struct SearchSpace {
    nodes: Vec<Option<SearchNodeInfo>>,
    /// Per-state cache of preferred operator IDs reported by the
    /// heuristic for that state, indexed by `state_id`. Populated right
    /// after `evaluate_state` returns `Ok` (so the snapshot is captured
    /// before the heuristic's internal cache is overwritten by the next
    /// state's evaluation). Read back when the state is *expanded* — we
    /// then mark each successor's open-list entry as preferred iff the
    /// operator that generated it is in this set.
    preferred_op_ids: Vec<Option<Box<[u32]>>>,
}

impl SearchSpace {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn node(&self, state_id: StateID) -> Option<&SearchNodeInfo> {
        self.nodes.get(state_id).and_then(Option::as_ref)
    }

    pub(crate) fn node_mut(&mut self, state_id: StateID) -> Option<&mut SearchNodeInfo> {
        self.nodes.get_mut(state_id).and_then(Option::as_mut)
    }

    pub(crate) fn set_node(&mut self, state_id: StateID, info: SearchNodeInfo) {
        if state_id >= self.nodes.len() {
            self.nodes.resize(state_id + 1, None);
        }
        self.nodes[state_id] = Some(info);
    }

    pub(crate) fn store_preferred(&mut self, state_id: StateID, ids: Vec<usize>) {
        if state_id >= self.preferred_op_ids.len() {
            self.preferred_op_ids.resize_with(state_id + 1, || None);
        }
        if ids.is_empty() {
            self.preferred_op_ids[state_id] = None;
        } else {
            let packed: Box<[u32]> = ids.into_iter().map(|x| x as u32).collect();
            self.preferred_op_ids[state_id] = Some(packed);
        }
    }

    /// Remove and return the cached preferred-op IDs for `state_id`, if any.
    /// We `take` rather than borrow because the only consumer is the
    /// expansion step, after which the IDs aren't needed again unless the
    /// state is reopened — in which case `evaluate_state` will resnapshot.
    pub(crate) fn take_preferred(&mut self, state_id: StateID) -> Option<Box<[u32]>> {
        self.preferred_op_ids.get_mut(state_id).and_then(Option::take)
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
