use ordered_float::OrderedFloat;
use planforge_sas::numeric::state_registry::StateID;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Debug, Clone, Copy)]
pub(crate) struct OpenEntry {
    pub(crate) f_value: OrderedFloat<f64>,
    pub(crate) h_value: OrderedFloat<f64>,
    pub(crate) g_value: f64,
    pub(crate) state_id: StateID,
    pub(crate) insertion_order: usize,
    /// `true` iff the operator that generated this successor was reported
    /// as a preferred (helpful) action by the heuristic for the parent
    /// state. Used as a tie-break inside a single queue, and as the queue
    /// selector for dual-queue GBFS.
    pub(crate) is_preferred: bool,
    /// `true` iff this entry has already been popped once and the slow
    /// admissible heuristic recomputed and folded in. Used only by the
    /// fast/slow A* variant (`new_fast_slow`). On the first pop of a
    /// `second == false` entry, the slow heuristic is evaluated, the
    /// entry is reinserted with `f' = g + max(h_f, h_s)` and
    /// `second = true`, and the expansion is deferred to the next pop.
    /// For ordinary A*/GBFS this field is always `false` and the field
    /// does not affect `Ord`.
    pub(crate) second: bool,
}

impl PartialEq for OpenEntry {
    fn eq(&self, other: &Self) -> bool {
        self.f_value == other.f_value
            && self.h_value == other.h_value
            && self.is_preferred == other.is_preferred
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
            .then_with(|| self.is_preferred.cmp(&other.is_preferred))
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
    next_insertion_order: usize,
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
        let entry = OpenEntry {
            f_value: OrderedFloat(f_value),
            h_value: OrderedFloat(h_value),
            g_value,
            state_id,
            insertion_order: self.next_insertion_order,
            is_preferred,
            second,
        };
        self.next_insertion_order += 1;
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
}
