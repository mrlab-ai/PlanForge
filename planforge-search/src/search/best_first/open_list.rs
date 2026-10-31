//! The open lists of best-first search.

use ordered_float::OrderedFloat;
use planforge_sas::state_registry::StateID;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// A queued search node. Entries pop by smallest priority, then smallest
/// heuristic estimate, then first in.
#[derive(Debug, Clone, Copy)]
pub struct OpenEntry {
    pub(super) priority: OrderedFloat<f64>,
    pub(super) h_value: OrderedFloat<f64>,
    pub(super) g_value: f64,
    pub(super) insertion_order: u32,
    /// Compact state ID plus the preferred and lazy-evaluated flags in
    /// its two high bits. Keeping all priority values as `f64` while packing
    /// these booleans makes an entry 32 rather than 40 bytes.
    tagged_state_id: u32,
}

const PREFERRED_FLAG: u32 = 1 << 31;
const LAZY_EVALUATED_FLAG: u32 = 1 << 30;
const STATE_ID_MASK: u32 = LAZY_EVALUATED_FLAG - 1;

impl OpenEntry {
    pub(super) fn new(state_id: StateID, g: f64, h: f64, priority: f64, preferred: bool) -> Self {
        let compact = u32::try_from(state_id).expect("open-list state id exceeds u32");
        assert!(
            compact <= STATE_ID_MASK,
            "open-list state id exceeds the 30-bit compact representation"
        );
        Self {
            priority: OrderedFloat(priority),
            h_value: OrderedFloat(h),
            g_value: g,
            insertion_order: 0,
            tagged_state_id: compact | if preferred { PREFERRED_FLAG } else { 0 },
        }
    }

    pub(super) fn state_id(self) -> StateID {
        (self.tagged_state_id & STATE_ID_MASK) as StateID
    }

    /// Whether a preferred operator of its parent generated the node.
    pub(super) fn is_preferred(self) -> bool {
        self.tagged_state_id & PREFERRED_FLAG != 0
    }

    pub(super) fn is_lazy_evaluated(self) -> bool {
        self.tagged_state_id & LAZY_EVALUATED_FLAG != 0
    }

    pub(super) fn with_estimate(mut self, h: f64, priority: f64) -> Self {
        self.h_value = OrderedFloat(h);
        self.priority = OrderedFloat(priority);
        self
    }

    pub(super) fn mark_lazy_evaluated(&mut self) {
        self.tagged_state_id |= LAZY_EVALUATED_FLAG;
    }
}

impl PartialEq for OpenEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for OpenEntry {}

impl PartialOrd for OpenEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OpenEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // `BinaryHeap` is a max-heap, so the comparison is inverted: the
        // smaller entry compares greater and pops first.
        other
            .priority
            .cmp(&self.priority)
            .then_with(|| other.h_value.cmp(&self.h_value))
            .then_with(|| other.insertion_order.cmp(&self.insertion_order))
    }
}

/// The queue discipline of best-first search over [`OpenEntry`]'s order.
pub trait OpenList: Default {
    /// Whether the list treats preferred nodes differently, and the search
    /// must therefore ask its heuristic for preferred operators.
    const USES_PREFERRED_OPERATORS: bool;
    fn push(&mut self, entry: OpenEntry);
    fn pop(&mut self) -> Option<OpenEntry>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// One heap; preferred nodes are not distinguished.
#[derive(Debug, Default)]
pub struct TieBreakingOpenList(BinaryHeap<OpenEntry>);
impl OpenList for TieBreakingOpenList {
    const USES_PREFERRED_OPERATORS: bool = false;
    fn push(&mut self, entry: OpenEntry) {
        self.0.push(entry);
    }
    fn pop(&mut self) -> Option<OpenEntry> {
        self.0.pop()
    }
    fn len(&self) -> usize {
        self.0.len()
    }
}

/// Two heaps; the preferred one is drained strictly before the regular one.
#[derive(Debug, Default)]
pub struct PreferredFirstOpenList {
    preferred: BinaryHeap<OpenEntry>,
    regular: BinaryHeap<OpenEntry>,
}
impl OpenList for PreferredFirstOpenList {
    const USES_PREFERRED_OPERATORS: bool = true;
    fn push(&mut self, entry: OpenEntry) {
        if entry.is_preferred() {
            self.preferred.push(entry);
        } else {
            self.regular.push(entry);
        }
    }
    fn pop(&mut self) -> Option<OpenEntry> {
        self.preferred.pop().or_else(|| self.regular.pop())
    }
    fn len(&self) -> usize {
        self.preferred.len() + self.regular.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_entry_is_32_bytes_and_preserves_flags_and_id() {
        assert_eq!(std::mem::size_of::<OpenEntry>(), 32);
        let mut open = TieBreakingOpenList::default();
        let mut entry = OpenEntry::new(123, 2.0, 3.0, 5.0, true);
        entry.mark_lazy_evaluated();
        open.push(entry);
        let entry = open.pop().unwrap();
        assert_eq!(entry.state_id(), 123);
        assert!(entry.is_preferred());
        assert!(entry.is_lazy_evaluated());
        assert_eq!(entry.g_value, 2.0);
        assert_eq!(entry.h_value.into_inner(), 3.0);
        assert_eq!(entry.priority.into_inner(), 5.0);
    }

    #[test]
    fn tie_breaking_orders_by_priority_then_estimate_then_fifo() {
        let mut open = TieBreakingOpenList::default();
        for (order, (f, preferred, h)) in [
            (5.0, true, 3.0),
            (4.0, false, 2.0),
            (5.0, false, 1.0),
            (5.0, true, 2.0),
            (5.0, false, 2.0),
        ]
        .into_iter()
        .enumerate()
        {
            let mut entry = OpenEntry::new(order, 0.0, h, f, preferred);
            entry.insertion_order = order as u32;
            open.push(entry);
        }
        let ids = std::iter::from_fn(|| open.pop())
            .map(OpenEntry::state_id)
            .collect::<Vec<_>>();
        assert_eq!(ids, [1, 2, 3, 4, 0]);
    }

    #[test]
    fn preferred_first_ignores_regular_priority_until_preferred_heap_is_empty() {
        let mut open = PreferredFirstOpenList::default();
        open.push(OpenEntry::new(1, 0.0, 1.0, 1.0, false));
        open.push(OpenEntry::new(2, 0.0, 100.0, 100.0, true));
        assert_eq!(open.pop().unwrap().state_id(), 2);
        assert_eq!(open.pop().unwrap().state_id(), 1);
        assert!(open.pop().is_none());
    }
}
