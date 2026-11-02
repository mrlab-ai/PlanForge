//! The open lists of best-first search.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::marker::PhantomData;

use planforge_sas::state_registry::StateID;

use super::NodeOrdering;

/// A queued search node: its path cost `g`, its heuristic estimate `h`, its
/// state and when it was queued. An open list orders entries by whatever it
/// computes from these; the built-in ones by the [`NodeOrdering`]'s key.
#[derive(Debug, Clone, Copy)]
pub struct OpenEntry {
    g: f64,
    h: f64,
    insertion_order: u32,
    /// The state ID plus the preferred and lazy-evaluated flags in its two
    /// high bits, which keeps an entry at 24 bytes.
    tagged_state_id: u32,
}

const PREFERRED_FLAG: u32 = 1 << 31;
const LAZY_EVALUATED_FLAG: u32 = 1 << 30;
const STATE_ID_MASK: u32 = LAZY_EVALUATED_FLAG - 1;

impl OpenEntry {
    pub(super) fn new(
        state_id: StateID,
        g: f64,
        h: f64,
        preferred: bool,
        insertion_order: u32,
    ) -> Self {
        let compact = u32::try_from(state_id).expect("open-list state id exceeds u32");
        assert!(
            compact <= STATE_ID_MASK,
            "open-list state id exceeds the 30-bit compact representation"
        );
        Self {
            g,
            h,
            insertion_order,
            tagged_state_id: compact | if preferred { PREFERRED_FLAG } else { 0 },
        }
    }

    pub fn state_id(self) -> StateID {
        (self.tagged_state_id & STATE_ID_MASK) as StateID
    }

    pub fn g(self) -> f64 {
        self.g
    }

    pub fn h(self) -> f64 {
        self.h
    }

    /// The number of entries queued before this one.
    pub fn insertion_order(self) -> u32 {
        self.insertion_order
    }

    /// Whether a preferred operator of its parent generated the node.
    pub fn is_preferred(self) -> bool {
        self.tagged_state_id & PREFERRED_FLAG != 0
    }

    pub(super) fn is_lazy_evaluated(self) -> bool {
        self.tagged_state_id & LAZY_EVALUATED_FLAG != 0
    }

    /// The entry requeued with the estimate `h` as the `insertion_order`th
    /// entry; `lazy_evaluated` marks it as evaluated by the lazy heuristic.
    pub(super) fn requeued(mut self, h: f64, insertion_order: u32, lazy_evaluated: bool) -> Self {
        self.h = h;
        self.insertion_order = insertion_order;
        if lazy_evaluated {
            self.tagged_state_id |= LAZY_EVALUATED_FLAG;
        }
        self
    }
}

/// The queue discipline of best-first search. Object safe: a registered,
/// user-written open list is a `Box<dyn OpenList>`, which works with every
/// [`NodeOrdering`]; the built-in lists are generic over the ordering, so the
/// search loop dispatches to them statically.
pub trait OpenList {
    /// Whether the list treats preferred nodes differently, and the search
    /// must therefore ask its heuristic for preferred operators.
    fn uses_preferred_operators(&self) -> bool;
    fn push(&mut self, entry: OpenEntry);
    fn pop(&mut self) -> Option<OpenEntry>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl OpenList for Box<dyn OpenList> {
    fn uses_preferred_operators(&self) -> bool {
        (**self).uses_preferred_operators()
    }
    fn push(&mut self, entry: OpenEntry) {
        (**self).push(entry);
    }
    fn pop(&mut self) -> Option<OpenEntry> {
        (**self).pop()
    }
    fn len(&self) -> usize {
        (**self).len()
    }
}

/// An entry in a heap ordered by `O`'s key, then first in, first out.
struct Keyed<O>(OpenEntry, PhantomData<fn() -> O>);

impl<O: NodeOrdering> Keyed<O> {
    #[inline]
    fn key(&self) -> (O::Key, u32) {
        (O::key(self.0.g, self.0.h), self.0.insertion_order)
    }
}

impl<O: NodeOrdering> PartialEq for Keyed<O> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl<O: NodeOrdering> Eq for Keyed<O> {}

impl<O: NodeOrdering> PartialOrd for Keyed<O> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<O: NodeOrdering> Ord for Keyed<O> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        // `BinaryHeap` is a max-heap: the smaller key compares greater.
        other.key().cmp(&self.key())
    }
}

/// One heap by `O`'s key, ties first in, first out; preferred nodes are not
/// distinguished.
pub struct TieBreakingOpenList<O>(BinaryHeap<Keyed<O>>);

impl<O: NodeOrdering> Default for TieBreakingOpenList<O> {
    fn default() -> Self {
        Self(BinaryHeap::new())
    }
}

impl<O: NodeOrdering> OpenList for TieBreakingOpenList<O> {
    fn uses_preferred_operators(&self) -> bool {
        false
    }
    #[inline]
    fn push(&mut self, entry: OpenEntry) {
        self.0.push(Keyed(entry, PhantomData));
    }
    #[inline]
    fn pop(&mut self) -> Option<OpenEntry> {
        self.0.pop().map(|keyed| keyed.0)
    }
    fn len(&self) -> usize {
        self.0.len()
    }
}

/// Two [`TieBreakingOpenList`]s; the one of preferred nodes is drained
/// strictly before the other.
pub struct PreferredFirstOpenList<O> {
    preferred: TieBreakingOpenList<O>,
    regular: TieBreakingOpenList<O>,
}

impl<O: NodeOrdering> Default for PreferredFirstOpenList<O> {
    fn default() -> Self {
        Self {
            preferred: TieBreakingOpenList::default(),
            regular: TieBreakingOpenList::default(),
        }
    }
}

impl<O: NodeOrdering> OpenList for PreferredFirstOpenList<O> {
    fn uses_preferred_operators(&self) -> bool {
        true
    }
    #[inline]
    fn push(&mut self, entry: OpenEntry) {
        if entry.is_preferred() {
            self.preferred.push(entry);
        } else {
            self.regular.push(entry);
        }
    }
    #[inline]
    fn pop(&mut self) -> Option<OpenEntry> {
        self.preferred.pop().or_else(|| self.regular.pop())
    }
    fn len(&self) -> usize {
        self.preferred.len() + self.regular.len()
    }
}

#[cfg(test)]
mod tests {
    use super::super::{AStar, Greedy};
    use super::*;

    #[test]
    fn an_open_entry_is_24_bytes_and_keeps_its_flags_and_id() {
        assert_eq!(std::mem::size_of::<OpenEntry>(), 24);
        let entry = OpenEntry::new(123, 2.0, 3.0, true, 0).requeued(4.0, 7, true);
        assert_eq!(entry.state_id(), 123);
        assert!(entry.is_preferred() && entry.is_lazy_evaluated());
        assert_eq!(
            (entry.g(), entry.h(), entry.insertion_order()),
            (2.0, 4.0, 7)
        );
    }

    fn popped<Q: OpenList>(mut open: Q, entries: &[(f64, f64, bool)]) -> Vec<StateID> {
        for (order, &(g, h, preferred)) in entries.iter().enumerate() {
            open.push(OpenEntry::new(order, g, h, preferred, order as u32));
        }
        std::iter::from_fn(|| open.pop())
            .map(OpenEntry::state_id)
            .collect()
    }

    #[test]
    fn astar_orders_by_f_then_h_then_first_in() {
        let entries = [
            (2.0, 3.0, true),
            (2.0, 2.0, false),
            (4.0, 1.0, false),
            (3.0, 2.0, true),
            (1.0, 4.0, false),
        ];
        let open = TieBreakingOpenList::<AStar>::default();
        assert_eq!(popped(open, &entries), [1, 2, 3, 0, 4]);
    }

    #[test]
    fn preferred_first_drains_the_preferred_nodes_first() {
        let entries = [(0.0, 1.0, false), (0.0, 100.0, true), (0.0, 50.0, true)];
        let open = PreferredFirstOpenList::<Greedy>::default();
        assert_eq!(popped(open, &entries), [2, 1, 0]);
    }
}
