use ordered_float::OrderedFloat;
use planforge_sas::state_registry::StateID;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Debug, Clone, Copy)]
pub(crate) struct OpenEntry {
    pub(crate) f_value: OrderedFloat<f64>,
    pub(crate) h_value: OrderedFloat<f64>,
    pub(crate) g_value: f64,
    pub(crate) insertion_order: u32,
    /// Compact state ID plus the preferred and fast/slow-second-pop flags in
    /// its two high bits. Keeping all priority values as `f64` while packing
    /// these booleans makes an entry 32 rather than 40 bytes.
    tagged_state_id: u32,
}

const PREFERRED_FLAG: u32 = 1 << 31;
const SECOND_FLAG: u32 = 1 << 30;
const STATE_ID_MASK: u32 = SECOND_FLAG - 1;

impl OpenEntry {
    pub(crate) fn state_id(self) -> StateID {
        (self.tagged_state_id & STATE_ID_MASK) as StateID
    }

    pub(crate) fn is_preferred(self) -> bool {
        self.tagged_state_id & PREFERRED_FLAG != 0
    }

    pub(crate) fn is_second(self) -> bool {
        self.tagged_state_id & SECOND_FLAG != 0
    }
}

impl PartialEq for OpenEntry {
    fn eq(&self, other: &Self) -> bool {
        self.f_value == other.f_value
            && self.h_value == other.h_value
            && self.is_preferred() == other.is_preferred()
            && self.insertion_order == other.insertion_order
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
        // BinaryHeap is a max-heap; we invert so smaller `f` pops first.
        // `is_preferred` is a *forward* comparison: `true > false`, so a
        // preferred entry compares greater (pops sooner) at equal `f`.
        other
            .f_value
            .cmp(&self.f_value)
            .then_with(|| self.is_preferred().cmp(&other.is_preferred()))
            .then_with(|| other.h_value.cmp(&self.h_value))
            .then_with(|| other.insertion_order.cmp(&self.insertion_order))
    }
}

/// Open list with a primary heap and an optional "preferred-first"
/// secondary heap.
///
/// When `use_preferred_first` is `true` (GBFS with a heuristic that emits
/// preferred operators), entries flagged as preferred go into
/// `preferred_heap` and `pop` drains that heap first. This is the
/// canonical FF/FD dual-queue lazy-greedy ordering: states reached via a
/// helpful action are expanded ahead of the rest, which empirically
/// dwarfs the speedup of a tie-break-only integration.
///
/// When `use_preferred_first` is `false` (A\*), only `regular_heap` is
/// used; preferred-ness still participates in `OpenEntry`'s `Ord` as a
/// tie-break between `f` and `h`, which is safe for admissibility since
/// it only reorders entries with identical `f`.
#[derive(Debug)]
pub(crate) struct DualQueueOpenList {
    regular_heap: BinaryHeap<OpenEntry>,
    preferred_heap: BinaryHeap<OpenEntry>,
    use_preferred_first: bool,
    next_insertion_order: u32,
}

impl DualQueueOpenList {
    pub(crate) fn new(use_preferred_first: bool) -> Self {
        Self {
            regular_heap: BinaryHeap::new(),
            preferred_heap: BinaryHeap::new(),
            use_preferred_first,
            next_insertion_order: 0,
        }
    }

    pub(crate) fn insert(
        &mut self,
        state_id: StateID,
        g_value: f64,
        h_value: f64,
        f_value: f64,
        is_preferred: bool,
    ) {
        self.insert_with_second(state_id, g_value, h_value, f_value, is_preferred, false);
    }

    pub(crate) fn insert_with_second(
        &mut self,
        state_id: StateID,
        g_value: f64,
        h_value: f64,
        f_value: f64,
        is_preferred: bool,
        second: bool,
    ) {
        let compact_state_id = u32::try_from(state_id).unwrap_or_else(|_| {
            panic!("open-list state id {state_id} exceeds the compact representation")
        });
        assert!(
            compact_state_id <= STATE_ID_MASK,
            "open-list state id {state_id} exceeds the 30-bit compact representation"
        );
        let tagged_state_id = compact_state_id
            | if is_preferred { PREFERRED_FLAG } else { 0 }
            | if second { SECOND_FLAG } else { 0 };
        let entry = OpenEntry {
            f_value: OrderedFloat(f_value),
            h_value: OrderedFloat(h_value),
            g_value,
            insertion_order: self.next_insertion_order,
            tagged_state_id,
        };
        self.next_insertion_order = self
            .next_insertion_order
            .checked_add(1)
            .expect("open-list insertion count exceeds u32");
        if self.use_preferred_first && is_preferred {
            self.preferred_heap.push(entry);
        } else {
            self.regular_heap.push(entry);
        }
    }

    pub(crate) fn pop(&mut self) -> Option<OpenEntry> {
        if self.use_preferred_first
            && let Some(entry) = self.preferred_heap.pop()
        {
            return Some(entry);
        }
        self.regular_heap.pop()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.regular_heap.is_empty() && self.preferred_heap.is_empty()
    }

    pub(crate) fn len(&self) -> usize {
        self.regular_heap.len() + self.preferred_heap.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_entry_is_32_bytes_and_preserves_flags_and_id() {
        assert_eq!(std::mem::size_of::<OpenEntry>(), 32);
        let mut open = DualQueueOpenList::new(false);
        open.insert_with_second(123, 2.0, 3.0, 5.0, true, true);
        let entry = open.pop().unwrap();
        assert_eq!(entry.state_id(), 123);
        assert!(entry.is_preferred());
        assert!(entry.is_second());
        assert_eq!(entry.g_value, 2.0);
        assert_eq!(entry.h_value.into_inner(), 3.0);
        assert_eq!(entry.f_value.into_inner(), 5.0);
    }
}
