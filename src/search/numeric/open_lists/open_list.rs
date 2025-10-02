//! Modern open list interface for planning
//! 
//! This module defines a simplified, type-safe interface for open lists that works
//! with the modern evaluation system. It eliminates the complexity of the C++ version
//! while maintaining all essential functionality.

use crate::search::numeric::state_registry::ConcreteState;
use crate::search::numeric::numeric_task::Operator;
use crate::search::numeric::evaluation::{EvaluationResult, Evaluator};
use crate::search::numeric::evaluation::evaluator::EvaluatorRef;
use std::rc::Rc;

/// Represents a search node with state, path information, and evaluation results
/// 
/// This combines the functionality of the C++ StateOpenListEntry and EdgeOpenListEntry
/// into a single, flexible structure that works with any search algorithm.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchNode {
    /// The planning state
    pub state: ConcreteState,
    /// Optional parent node for path reconstruction
    pub parent: Option<Rc<SearchNode>>,
    /// The operator that was applied to reach this state
    pub operator: Option<Operator>,
    /// Evaluation results for this node
    pub evaluation: EvaluationResult,
}

impl SearchNode {
    /// Creates a new search node with evaluation
    pub fn new(
        state: ConcreteState,
        parent: Option<Rc<SearchNode>>,
        operator: Option<Operator>,
        evaluation: EvaluationResult,
    ) -> Self {
        Self {
            state,
            parent,
            operator,
            evaluation,
        }
    }

    /// Creates a root node (no parent or operator)
    pub fn root(state: ConcreteState, evaluation: EvaluationResult) -> Self {
        Self::new(state, None, None, evaluation)
    }

    /// Creates a successor node from this node
    pub fn successor(
        self: &Rc<Self>,
        state: ConcreteState,
        operator: Operator,
        evaluation: EvaluationResult,
    ) -> Self {
        Self::new(state, Some(self.clone()), Some(operator), evaluation)
    }

    /// Gets the g-value (path cost) for this node
    pub fn g_value(&self) -> f64 {
        self.evaluation.g_value
    }

    /// Gets the heuristic value for a specific evaluator
    pub fn h_value(&self, heuristic_name: &str) -> f64 {
        self.evaluation.get_heuristic_value(heuristic_name)
    }

    /// Gets the f-value (g + h) for a specific heuristic
    pub fn f_value(&self, heuristic_name: &str) -> f64 {
        self.evaluation.get_f_value(heuristic_name)
    }

    /// Checks if this node represents a dead end
    pub fn is_dead_end(&self) -> bool {
        self.evaluation.is_dead_end
    }

    /// Checks if this node represents a reliable dead end
    pub fn is_reliable_dead_end(&self) -> bool {
        self.evaluation.is_reliable_dead_end
    }

    /// Reconstructs the path from the root to this node
    pub fn path(&self) -> Vec<Operator> {
        let mut path = Vec::new();
        let mut current = Some(self);

        while let Some(node) = current {
            if let Some(op) = &node.operator {
                path.push(op.clone());
            }
            current = node.parent.as_ref().map(|p| p.as_ref());
        }

        path.reverse();
        path
    }

    /// Gets the depth of this node (number of operators from root)
    pub fn depth(&self) -> usize {
        self.path().len()
    }
}

/// Priority function for determining node ordering in open lists
/// 
/// This replaces the complex evaluator system in the C++ version with
/// a simple function-based approach.
pub type PriorityFunction = dyn Fn(&SearchNode) -> Vec<f64> + Send + Sync;

/// Simplified open list trait for search algorithms
/// 
/// This interface is much cleaner than the C++ version and focuses on the
/// essential operations needed by search algorithms.
pub trait OpenList {
    /// Inserts a node into the open list
    fn insert(&mut self, node: SearchNode);

    /// Removes and returns the node with the best priority
    /// Returns None if the open list is empty
    fn pop(&mut self) -> Option<SearchNode>;

    /// Peeks at the best node without removing it
    fn peek(&self) -> Option<&SearchNode>;

    /// Returns true if the open list is empty
    fn is_empty(&self) -> bool;

    /// Returns the number of nodes in the open list
    fn len(&self) -> usize;

    /// Clears all nodes from the open list
    fn clear(&mut self);

    /// Gets the names of evaluators used by this open list
    /// This is useful for ensuring all required evaluations are computed
    fn required_evaluators(&self) -> Vec<String>;

    /// Checks if this open list can handle nodes with dead end states
    /// Some open lists might want to reject dead ends immediately
    fn accepts_dead_ends(&self) -> bool {
        false
    }
}

/// A simple FIFO (First-In-First-Out) open list implementation
/// 
/// This is useful for breadth-first search.
pub struct FifoOpenList {
    nodes: std::collections::VecDeque<SearchNode>,
}

impl FifoOpenList {
    pub fn new() -> Self {
        Self {
            nodes: std::collections::VecDeque::new(),
        }
    }
}

impl Default for FifoOpenList {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenList for FifoOpenList {
    fn insert(&mut self, node: SearchNode) {
        self.nodes.push_back(node);
    }

    fn pop(&mut self) -> Option<SearchNode> {
        self.nodes.pop_front()
    }

    fn peek(&self) -> Option<&SearchNode> {
        self.nodes.front()
    }

    fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    fn len(&self) -> usize {
        self.nodes.len()
    }

    fn clear(&mut self) {
        self.nodes.clear();
    }

    fn required_evaluators(&self) -> Vec<String> {
        vec![] // FIFO doesn't need any evaluators
    }
}

/// A simple LIFO (Last-In-First-Out) open list implementation
/// 
/// This is useful for depth-first search.
pub struct LifoOpenList {
    nodes: Vec<SearchNode>,
}

impl LifoOpenList {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
        }
    }
}

impl Default for LifoOpenList {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenList for LifoOpenList {
    fn insert(&mut self, node: SearchNode) {
        self.nodes.push(node);
    }

    fn pop(&mut self) -> Option<SearchNode> {
        self.nodes.pop()
    }

    fn peek(&self) -> Option<&SearchNode> {
        self.nodes.last()
    }

    fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    fn len(&self) -> usize {
        self.nodes.len()
    }

    fn clear(&mut self) {
        self.nodes.clear();
    }

    fn required_evaluators(&self) -> Vec<String> {
        vec![] // LIFO doesn't need any evaluators
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_node(id: i32, g_value: f64) -> SearchNode {
        // Use simple pool offset for test states
        let state = ConcreteState::new(id as usize);
        let evaluation = EvaluationResult::new(state.clone(), g_value, false);
        SearchNode::root(state, evaluation)
    }

    fn create_test_operator() -> Operator {
        // Create a test operator using the public constructor
        Operator::new(
            "test_op".to_string(),
            vec![],
            vec![],
            vec![],
            3,
        )
    }

    #[test]
    fn test_search_node_basic() {
        let node = create_test_node(1, 10.0);
        
        assert_eq!(node.g_value(), 10.0);
        assert_eq!(node.depth(), 0);
        assert!(!node.is_dead_end());
        assert!(node.path().is_empty());
    }

    #[test]
    fn test_search_node_successor() {
        let parent = Rc::new(create_test_node(1, 5.0));
        let child_state = create_test_node(2, 8.0).state;
        let child_eval = EvaluationResult::new(child_state.clone(), 8.0, false);
        
        let operator = create_test_operator();
        
        let child = parent.successor(child_state, operator.clone(), child_eval);
        
        assert_eq!(child.g_value(), 8.0);
        assert_eq!(child.depth(), 1);
        assert_eq!(child.operator, Some(operator));
        assert!(child.parent.is_some());
    }

    #[test]
    fn test_fifo_open_list() {
        let mut open_list = FifoOpenList::new();
        
        assert!(open_list.is_empty());
        assert_eq!(open_list.len(), 0);
        assert!(open_list.pop().is_none());
        
        let node1 = create_test_node(1, 10.0);
        let node2 = create_test_node(2, 20.0);
        
        open_list.insert(node1);
        open_list.insert(node2);
        
        assert!(!open_list.is_empty());
        assert_eq!(open_list.len(), 2);
        
        // FIFO: first inserted should be first out
        let popped1 = open_list.pop().unwrap();
        assert_eq!(popped1.g_value(), 10.0);
        
        let popped2 = open_list.pop().unwrap();
        assert_eq!(popped2.g_value(), 20.0);
        
        assert!(open_list.is_empty());
    }

    #[test]
    fn test_lifo_open_list() {
        let mut open_list = LifoOpenList::new();
        
        let node1 = create_test_node(1, 10.0);
        let node2 = create_test_node(2, 20.0);
        
        open_list.insert(node1);
        open_list.insert(node2);
        
        // LIFO: last inserted should be first out
        let popped1 = open_list.pop().unwrap();
        assert_eq!(popped1.g_value(), 20.0);
        
        let popped2 = open_list.pop().unwrap();
        assert_eq!(popped2.g_value(), 10.0);
        
        assert!(open_list.is_empty());
    }

    #[test]
    fn test_open_list_peek() {
        let mut fifo = FifoOpenList::new();
        let node = create_test_node(1, 42.0);
        
        assert!(fifo.peek().is_none());
        
        fifo.insert(node);
        
        let peeked = fifo.peek().unwrap();
        assert_eq!(peeked.g_value(), 42.0);
        assert_eq!(fifo.len(), 1); // Peek shouldn't remove
        
        let popped = fifo.pop().unwrap();
        assert_eq!(popped.g_value(), 42.0);
        assert!(fifo.peek().is_none());
    }

    #[test]
    fn test_open_list_clear() {
        let mut open_list = FifoOpenList::new();
        
        open_list.insert(create_test_node(1, 10.0));
        open_list.insert(create_test_node(2, 20.0));
        
        assert_eq!(open_list.len(), 2);
        
        open_list.clear();
        
        assert!(open_list.is_empty());
        assert_eq!(open_list.len(), 0);
    }

    #[test]
    fn test_required_evaluators() {
        let fifo = FifoOpenList::new();
        let lifo = LifoOpenList::new();
        
        assert!(fifo.required_evaluators().is_empty());
        assert!(lifo.required_evaluators().is_empty());
    }
}
