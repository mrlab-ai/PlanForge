use super::open_list::{OpenList, SearchNode};
use ordered_float::OrderedFloat;
use std::collections::{HashMap, VecDeque};

/// A tie-breaking open list that sorts lexicographically by evaluation values
/// and breaks exact ties using FIFO order.
///
/// The evaluator order passed to [`TieBreakingOpenList::new`] defines the
/// comparison order. For example, using `[f, h]` means nodes are ordered by
/// increasing `f = g + h`, and for equal `f` values the node with lower `h`
/// is preferred. If all evaluator values are equal, insertion order is kept.
pub struct TieBreakingOpenList {
    /// Maps evaluation keys to FIFO queues of nodes
    buckets: HashMap<Vec<OrderedFloat<f64>>, VecDeque<SearchNode>>,
    /// The names of evaluators used to compute keys
    evaluator_names: Vec<String>,
    /// Whether the list is sorted in ascending order (true) or descending (false)
    ascending: bool,
}

impl TieBreakingOpenList {
    /// Creates a new tie-breaking open list with the given evaluator names.
    pub fn new(evaluator_names: Vec<String>, ascending: bool) -> Self {
        Self {
            buckets: HashMap::new(),
            evaluator_names,
            ascending,
        }
    }

    /// Computes the lexicographic evaluation key for a given node.
    fn compute_key(&self, node: &SearchNode) -> Vec<OrderedFloat<f64>> {
        let mut key = Vec::with_capacity(self.evaluator_names.len());

        for evaluator_name in &self.evaluator_names {
            let value = node.evaluation.get_heuristic_value(evaluator_name);
            key.push(OrderedFloat(value));
        }

        key
    }

    /// Finds the best key according to the sort order.
    fn find_best_key(&self) -> Option<&Vec<OrderedFloat<f64>>> {
        if self.buckets.is_empty() {
            return None;
        }

        if self.ascending {
            self.buckets.keys().min()
        } else {
            self.buckets.keys().max()
        }
    }
}

impl OpenList for TieBreakingOpenList {
    fn insert(&mut self, node: SearchNode) {
        let key = self.compute_key(&node);
        self.buckets
            .entry(key)
            .or_insert_with(VecDeque::new)
            .push_back(node);
    }

    fn pop(&mut self) -> Option<SearchNode> {
        // Find the best key
        let best_key = self.find_best_key()?.clone();

        // Get the bucket for this key and pop from front (FIFO)
        if let Some(bucket) = self.buckets.get_mut(&best_key) {
            let node = bucket.pop_front();

            // Remove empty buckets
            if bucket.is_empty() {
                self.buckets.remove(&best_key);
            }

            node
        } else {
            None
        }
    }

    fn peek(&self) -> Option<&SearchNode> {
        let best_key = self.find_best_key()?;
        self.buckets.get(best_key)?.front()
    }

    fn is_empty(&self) -> bool {
        self.buckets.is_empty()
    }

    fn len(&self) -> usize {
        self.buckets.values().map(|bucket| bucket.len()).sum()
    }

    fn clear(&mut self) {
        self.buckets.clear();
    }

    fn required_evaluators(&self) -> Vec<String> {
        self.evaluator_names.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::numeric::evaluation::EvaluationResult;
    use crate::search::numeric::state_registry::ConcreteState;

    fn create_test_node(state_id: usize, g_value: f64) -> SearchNode {
        let state = ConcreteState::new(state_id);
        let mut evaluation = EvaluationResult::new_with_id(state.get_id(), g_value, false);
        evaluation.set_heuristic_value("g".to_string(), g_value);
        SearchNode::root(state, evaluation)
    }

    fn create_test_node_with_values(
        state_id: usize,
        g_value: f64,
        heuristic_values: &[(&str, f64)],
    ) -> SearchNode {
        let state = ConcreteState::new(state_id);
        let mut evaluation = EvaluationResult::new_with_id(state.get_id(), g_value, false);
        for (name, value) in heuristic_values {
            evaluation.set_heuristic_value((*name).to_string(), *value);
        }
        SearchNode::root(state, evaluation)
    }

    #[test]
    fn test_tiebreaking_empty() {
        let mut open_list = TieBreakingOpenList::new(vec!["g".to_string()], true);

        assert!(open_list.is_empty());
        assert_eq!(open_list.len(), 0);
        assert!(open_list.pop().is_none());
        assert!(open_list.peek().is_none());
    }

    #[test]
    fn test_tiebreaking_single_node() {
        let mut open_list = TieBreakingOpenList::new(vec!["g".to_string()], true);

        let node = create_test_node(1, 10.0);
        open_list.insert(node);

        assert!(!open_list.is_empty());
        assert_eq!(open_list.len(), 1);

        let popped = open_list.pop().unwrap();
        assert_eq!(popped.state.get_id(), 1);
        assert_eq!(popped.g_value(), 10.0);

        assert!(open_list.is_empty());
    }

    #[test]
    fn test_tiebreaking_g_value_ordering() {
        let mut open_list = TieBreakingOpenList::new(vec!["g".to_string()], true); // ascending

        // Insert nodes with different g-values
        open_list.insert(create_test_node(1, 30.0));
        open_list.insert(create_test_node(2, 10.0));
        open_list.insert(create_test_node(3, 20.0));

        // Should pop in g-value order (ascending)
        assert_eq!(open_list.pop().unwrap().g_value(), 10.0);
        assert_eq!(open_list.pop().unwrap().g_value(), 20.0);
        assert_eq!(open_list.pop().unwrap().g_value(), 30.0);
    }

    #[test]
    fn test_tiebreaking_descending_order() {
        let mut open_list = TieBreakingOpenList::new(vec!["g".to_string()], false); // descending

        // Insert nodes with different g-values
        open_list.insert(create_test_node(1, 10.0));
        open_list.insert(create_test_node(2, 30.0));
        open_list.insert(create_test_node(3, 20.0));

        // Should pop in reverse g-value order (descending)
        assert_eq!(open_list.pop().unwrap().g_value(), 30.0);
        assert_eq!(open_list.pop().unwrap().g_value(), 20.0);
        assert_eq!(open_list.pop().unwrap().g_value(), 10.0);
    }

    #[test]
    fn test_tiebreaking_fifo_order() {
        let mut open_list = TieBreakingOpenList::new(vec!["g".to_string()], true);

        // Insert nodes with same g-value (should use FIFO tie-breaking)
        open_list.insert(create_test_node(1, 10.0));
        open_list.insert(create_test_node(2, 10.0));
        open_list.insert(create_test_node(3, 10.0));

        // Should pop in insertion order (FIFO)
        assert_eq!(open_list.pop().unwrap().state.get_id(), 1);
        assert_eq!(open_list.pop().unwrap().state.get_id(), 2);
        assert_eq!(open_list.pop().unwrap().state.get_id(), 3);
    }

    #[test]
    fn test_tiebreaking_peek() {
        let mut open_list = TieBreakingOpenList::new(vec!["g".to_string()], true);

        open_list.insert(create_test_node(1, 30.0));
        open_list.insert(create_test_node(2, 10.0));

        // Peek should return the best node without removing it
        let peeked = open_list.peek().unwrap();
        assert_eq!(peeked.g_value(), 10.0);
        assert_eq!(open_list.len(), 2);

        // Pop should return the same node
        let popped = open_list.pop().unwrap();
        assert_eq!(popped.g_value(), 10.0);
        assert_eq!(open_list.len(), 1);
    }

    #[test]
    fn test_tiebreaking_complex_scenario() {
        let mut open_list = TieBreakingOpenList::new(vec!["g".to_string()], true);

        // Mixed g-values with ties
        open_list.insert(create_test_node(1, 20.0));
        open_list.insert(create_test_node(2, 10.0));
        open_list.insert(create_test_node(3, 20.0)); // tie with node 1
        open_list.insert(create_test_node(4, 15.0));
        open_list.insert(create_test_node(5, 10.0)); // tie with node 2

        // Should pop in g-value order, with FIFO tie-breaking
        assert_eq!(open_list.pop().unwrap().state.get_id(), 2); // g=10.0, first
        assert_eq!(open_list.pop().unwrap().state.get_id(), 5); // g=10.0, second
        assert_eq!(open_list.pop().unwrap().state.get_id(), 4); // g=15.0
        assert_eq!(open_list.pop().unwrap().state.get_id(), 1); // g=20.0, first
        assert_eq!(open_list.pop().unwrap().state.get_id(), 3); // g=20.0, second
    }

    #[test]
    fn test_tiebreaking_prefers_lower_h_for_equal_f() {
        let mut open_list =
            TieBreakingOpenList::new(vec!["f_h".to_string(), "h".to_string()], true);

        open_list.insert(create_test_node_with_values(
            1,
            4.0,
            &[("g", 4.0), ("h", 5.0), ("f_h", 9.0)],
        ));
        open_list.insert(create_test_node_with_values(
            2,
            6.0,
            &[("g", 6.0), ("h", 3.0), ("f_h", 9.0)],
        ));
        open_list.insert(create_test_node_with_values(
            3,
            2.0,
            &[("g", 2.0), ("h", 8.0), ("f_h", 10.0)],
        ));

        assert_eq!(open_list.pop().unwrap().state.get_id(), 2);
        assert_eq!(open_list.pop().unwrap().state.get_id(), 1);
        assert_eq!(open_list.pop().unwrap().state.get_id(), 3);
    }

    #[test]
    fn test_tiebreaking_uses_fifo_when_f_and_h_match() {
        let mut open_list =
            TieBreakingOpenList::new(vec!["f_h".to_string(), "h".to_string()], true);

        open_list.insert(create_test_node_with_values(
            1,
            4.0,
            &[("g", 4.0), ("h", 5.0), ("f_h", 9.0)],
        ));
        open_list.insert(create_test_node_with_values(
            2,
            4.0,
            &[("g", 4.0), ("h", 5.0), ("f_h", 9.0)],
        ));

        assert_eq!(open_list.pop().unwrap().state.get_id(), 1);
        assert_eq!(open_list.pop().unwrap().state.get_id(), 2);
    }

    #[test]
    fn test_required_evaluators() {
        let open_list = TieBreakingOpenList::new(vec!["g".to_string(), "h".to_string()], true);
        let required = open_list.required_evaluators();

        assert_eq!(required.len(), 2);
        assert!(required.contains(&"g".to_string()));
        assert!(required.contains(&"h".to_string()));
    }

    #[test]
    fn test_clear() {
        let mut open_list = TieBreakingOpenList::new(vec!["g".to_string()], true);

        open_list.insert(create_test_node(1, 10.0));
        open_list.insert(create_test_node(2, 20.0));
        assert_eq!(open_list.len(), 2);

        open_list.clear();
        assert!(open_list.is_empty());
        assert_eq!(open_list.len(), 0);
    }
}
