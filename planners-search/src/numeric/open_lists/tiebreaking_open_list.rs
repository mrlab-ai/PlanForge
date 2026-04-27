use super::open_list::{OpenList, SearchNode};
use tracing::debug;
use ordered_float::OrderedFloat;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::env;
use std::error::Error;
use std::fmt;

#[cfg(test)]
mod tests;

type EvaluationKey = Vec<OrderedFloat<f64>>;

#[derive(Debug)]
struct HeapEntry {
    key: EvaluationKey,
    insertion_order: usize,
    ascending: bool,
    node: SearchNode,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key && self.insertion_order == other.insertion_order
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        debug_assert_eq!(self.ascending, other.ascending);
        let key_order = if self.ascending {
            other.key.cmp(&self.key)
        } else {
            self.key.cmp(&other.key)
        };

        if key_order == Ordering::Equal {
            other.insertion_order.cmp(&self.insertion_order)
        } else {
            key_order
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TieBreakingOpenListError {
    EmptyEvaluatorList,
}

impl fmt::Display for TieBreakingOpenListError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TieBreakingOpenListError::EmptyEvaluatorList => {
                write!(f, "tie-breaking open list requires at least one evaluator")
            }
        }
    }
}

impl Error for TieBreakingOpenListError {}

/// A tie-breaking open list that sorts lexicographically by evaluation values
/// and breaks exact ties using FIFO order.
///
/// The evaluator order passed to `TieBreakingOpenList::new` defines the
/// comparison order. For example, using `[f, h]` means nodes are ordered by
/// increasing `f = g + h`, and for equal `f` values the node with lower `h`
/// is preferred. If all evaluator values are equal, insertion order is kept.
#[derive(Debug)]
pub struct TieBreakingOpenList {
    /// Heap of nodes ordered by evaluation key and FIFO insertion order.
    heap: BinaryHeap<HeapEntry>,
    /// Total number of nodes stored across all buckets.
    size: usize,
    /// The names of evaluators used to compute keys.
    evaluator_names: Vec<String>,
    /// Whether the list is sorted in ascending order (`true`) or descending (`false`).
    ascending: bool,
    /// Monotonic insertion counter for FIFO tie-breaking.
    next_insertion_order: usize,
}

impl TieBreakingOpenList {
    /// Create a new tie-breaking open list with the given evaluator names.
    pub fn new(
        evaluator_names: Vec<String>,
        ascending: bool,
    ) -> Result<Self, TieBreakingOpenListError> {
        if evaluator_names.is_empty() {
            return Err(TieBreakingOpenListError::EmptyEvaluatorList);
        }

        Ok(Self {
            heap: BinaryHeap::new(),
            size: 0,
            evaluator_names,
            ascending,
            next_insertion_order: 0,
        })
    }

    /// Compute the lexicographic evaluation key for a given node.
    fn compute_key(&self, node: &SearchNode) -> EvaluationKey {
        let mut key = Vec::with_capacity(self.evaluator_names.len());

        for evaluator_name in &self.evaluator_names {
            let value = node.evaluation.get_heuristic_value(evaluator_name);
            key.push(OrderedFloat(value));
        }

        key
    }
}

impl OpenList for TieBreakingOpenList {
    fn insert(&mut self, node: SearchNode) {
        let key = self.compute_key(&node);
        if env::var_os("TRACE_OPEN_LIST_KEYS").is_some() {
            let key_str = key
                .iter()
                .map(|value| format!("{:.17}", value.into_inner()))
                .collect::<Vec<_>>()
                .join(",");
            debug!(
                "TRACE open-list-insert sid={} key=[{}]",
                node.state.get_id(),
                key_str
            );
        }
        self.heap.push(HeapEntry {
            key,
            insertion_order: self.next_insertion_order,
            ascending: self.ascending,
            node,
        });
        self.next_insertion_order += 1;
        self.size += 1;
    }

    fn pop(&mut self) -> Option<SearchNode> {
        let entry = self.heap.pop()?;
        self.size -= 1;
        Some(entry.node)
    }

    fn peek(&self) -> Option<&SearchNode> {
        self.heap.peek().map(|entry| &entry.node)
    }

    fn is_empty(&self) -> bool {
        self.size == 0
    }

    fn len(&self) -> usize {
        self.size
    }

    fn clear(&mut self) {
        self.heap.clear();
        self.size = 0;
    }

    fn required_evaluators(&self) -> Vec<String> {
        self.evaluator_names.clone()
    }
}
