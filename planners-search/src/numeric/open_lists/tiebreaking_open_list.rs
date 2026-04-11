use super::open_list::{OpenList, SearchNode};
use ordered_float::OrderedFloat;
use std::env;
use std::collections::{BTreeMap, VecDeque};
use std::error::Error;
use std::fmt;

#[cfg(test)]
mod tests;

type EvaluationKey = Vec<OrderedFloat<f64>>;

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
/// The evaluator order passed to [`TieBreakingOpenList::new`] defines the
/// comparison order. For example, using `[f, h]` means nodes are ordered by
/// increasing `f = g + h`, and for equal `f` values the node with lower `h`
/// is preferred. If all evaluator values are equal, insertion order is kept.
#[derive(Debug)]
pub struct TieBreakingOpenList {
    /// Maps evaluation keys to FIFO queues of nodes
    buckets: BTreeMap<EvaluationKey, VecDeque<SearchNode>>,
    /// Total number of nodes stored across all buckets
    size: usize,
    /// The names of evaluators used to compute keys
    evaluator_names: Vec<String>,
    /// Whether the list is sorted in ascending order (true) or descending (false)
    ascending: bool,
}

impl TieBreakingOpenList {
    /// Creates a new tie-breaking open list with the given evaluator names.
    pub fn new(
        evaluator_names: Vec<String>,
        ascending: bool,
    ) -> Result<Self, TieBreakingOpenListError> {
        if evaluator_names.is_empty() {
            return Err(TieBreakingOpenListError::EmptyEvaluatorList);
        }

        Ok(Self {
            buckets: BTreeMap::new(),
            size: 0,
            evaluator_names,
            ascending,
        })
    }

    /// Computes the lexicographic evaluation key for a given node.
    fn compute_key(&self, node: &SearchNode) -> EvaluationKey {
        let mut key = Vec::with_capacity(self.evaluator_names.len());

        for evaluator_name in &self.evaluator_names {
            let value = node.evaluation.get_heuristic_value(evaluator_name);
            key.push(OrderedFloat(value));
        }

        key
    }

    fn best_bucket(&self) -> Option<(&EvaluationKey, &VecDeque<SearchNode>)> {
        if self.ascending {
            self.buckets.first_key_value()
        } else {
            self.buckets.last_key_value()
        }
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
            println!(
                "TRACE open-list-insert sid={} key=[{}]",
                node.state.get_id(),
                key_str
            );
        }
        self.buckets
            .entry(key)
            .or_insert_with(VecDeque::new)
            .push_back(node);
        self.size += 1;
    }

    fn pop(&mut self) -> Option<SearchNode> {
        let mut best_bucket = if self.ascending {
            self.buckets.first_entry()?
        } else {
            self.buckets.last_entry()?
        };

        let (node, bucket_is_empty) = {
            let bucket = best_bucket.get_mut();
            let node = bucket
                .pop_front()
                .expect("best bucket in tie-breaking open list must not be empty");
            (node, bucket.is_empty())
        };
        self.size -= 1;

        if bucket_is_empty {
            best_bucket.remove_entry();
        }

        Some(node)
    }

    fn peek(&self) -> Option<&SearchNode> {
        self.best_bucket()?.1.front()
    }

    fn is_empty(&self) -> bool {
        self.size == 0
    }

    fn len(&self) -> usize {
        self.size
    }

    fn clear(&mut self) {
        self.buckets.clear();
        self.size = 0;
    }

    fn required_evaluators(&self) -> Vec<String> {
        self.evaluator_names.clone()
    }
}
