//! Lightweight search engine implementation for numeric planning
//! 
//! This module provides a simplified search engine based on the C++ Fast Downward
//! implementation, focusing on A* search with minimal overhead.

use crate::search::numeric::{
    numeric_task::{AbstractNumericTask, Fact, Operator},
    state_registry::{ConcreteState, StateID, StateRegistry},
    evaluation::{Evaluator, EvaluationResult, EvaluationState, Heuristic},
    evaluation::heuristic::ZeroHeuristic,
    evaluation::g_evaluator::{GEvaluator, SumEvaluator},
    open_lists::{OpenList, SearchNode, TieBreakingOpenList},
    successor_generator::{GroundedSuccessorGenerator, Node},
};
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};
use std::rc::Rc;

/// Search status indicating the outcome of the search
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SearchStatus {
    InProgress,
    Solved,
    Failed,
    Timeout,
}

/// A plan is a sequence of operators
pub type Plan = Vec<Operator>;

/// Search result containing the outcome and optional plan
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub status: SearchStatus,
    pub plan: Option<Plan>,
    pub nodes_expanded: usize,
    pub nodes_generated: usize,
    pub search_time: Duration,
}

/// Simple search node information for tracking parent relationships
#[derive(Debug, Clone)]
struct SearchNodeInfo {
    parent_state: Option<StateID>,
    parent_operator: Option<Operator>,
    g_value: f64,
    status: NodeStatus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum NodeStatus {
    New,
    Open,
    Closed,
}

/// Base trait for search engines
pub trait SearchEngine {
    fn search(&mut self) -> SearchResult;
    fn print_initial_h_values(&mut self);
}

/// Lightweight A* search implementation
/// 
/// This provides a minimal A* search with:
/// - f = g + h evaluation with tie-breaking on h
/// - ZeroHeuristic as default heuristic
/// - No reopening of closed nodes
/// - Basic plan reconstruction
pub struct AStarSearch<'a> {
    task: &'a dyn AbstractNumericTask,
    state_registry: StateRegistry<'a>,
    successor_generator: Box<dyn Node<'a> + 'a>,
    
    // Search components
    open_list: TieBreakingOpenList,
    closed_set: HashSet<StateID>,
    search_nodes: HashMap<StateID, SearchNodeInfo>,
    
    // Evaluators
    heuristic: Box<dyn Heuristic>,
    g_evaluator: GEvaluator,
    f_evaluator: SumEvaluator,
    
    // Configuration
    time_limit: Duration,
    
    // Statistics
    nodes_expanded: usize,
    nodes_generated: usize,
}

impl<'a> AStarSearch<'a> {
    /// Creates a successor generator for the given task
    fn create_successor_generator(task: &'a dyn AbstractNumericTask) -> Box<dyn Node<'a> + 'a> {
        let mut queue = VecDeque::new();
        for (op_id, operator) in task.get_operators().iter().enumerate() {
            queue.push_back((operator, op_id as u32));
        }

        let mut generator = GroundedSuccessorGenerator::new(task);
        generator.construct(&mut 0, &mut queue).unwrap()
    }

    /// Creates a new A* search instance
    pub fn new(
        task: &'a dyn AbstractNumericTask,
        state_registry: StateRegistry<'a>,
        heuristic: Option<Box<dyn Heuristic>>,
        time_limit: Option<Duration>,
    ) -> Self {
        let successor_generator = Self::create_successor_generator(task);
        
        // Use ZeroHeuristic as default
        let heuristic = heuristic.unwrap_or_else(|| Box::new(ZeroHeuristic::new(None)));
        
        // Create evaluators for A*
        let g_evaluator = GEvaluator::new(None);
        let f_evaluator = SumEvaluator::f_evaluator(heuristic.name());
        
        // Create open list with f-value primary, h-value secondary (tie-breaking)
        let evaluator_names = vec![
            f_evaluator.name(),
            heuristic.name(),
        ];
        let open_list = TieBreakingOpenList::new(evaluator_names, true); // ascending order
        
        Self {
            task,
            state_registry,
            successor_generator,
            open_list,
            closed_set: HashSet::new(),
            search_nodes: HashMap::new(),
            heuristic,
            g_evaluator,
            f_evaluator,
            time_limit: time_limit.unwrap_or(Duration::from_secs(30 * 60)), // 30 minutes default
            nodes_expanded: 0,
            nodes_generated: 0,
        }
    }
    
    /// Checks if the given state satisfies all goal conditions
    fn is_goal_state(&self, state: &ConcreteState) -> bool {
        for i in 0..self.task.get_num_goals() {
            let goal_fact = self.task.get_goal_fact(i);
            if !self.state_satisfies_fact(state, goal_fact) {
                return false;
            }
        }
        true
    }
    
    /// Checks if a state satisfies a specific fact
    fn state_satisfies_fact(&self, state: &ConcreteState, fact: &Fact) -> bool {
    fact.is_true(state, &self.state_registry)
    }
    
    /// Traces back the path from goal state to initial state
    fn extract_plan(&self, goal_state: StateID) -> Plan {
        let mut plan = Vec::new();
        let mut current_state = goal_state;
        
        while let Some(node_info) = self.search_nodes.get(&current_state) {
            if let (Some(parent_state), Some(operator)) = (&node_info.parent_state, &node_info.parent_operator) {
                plan.push(operator.clone());
                current_state = *parent_state;                
            } else {
                break; // Reached initial state
            }
        }
        
        plan.reverse();
        plan
    }
    
    /// Evaluates a state and creates evaluation result
    fn evaluate_state(&self, state: &ConcreteState, g_value: f64) -> Result<EvaluationResult, Box<dyn std::error::Error>> {
        // Create evaluation state
        let mut eval_state = EvaluationState::new(state.clone(), g_value, false);
        
        // Evaluate g-value
        self.g_evaluator.evaluate_state(&mut eval_state)?;
        
        // Evaluate heuristic
        self.heuristic.evaluate_state(&mut eval_state)?;
        
        // Evaluate f-value
        self.f_evaluator.evaluate_state(&mut eval_state)?;
        
        Ok(eval_state.into_result())
    }
    
    /// Generates successor states for a given state
    fn generate_successors(&mut self, state: &ConcreteState) -> Vec<(ConcreteState, Operator, f64)> {
        // For now, let's use a simpler approach - iterate through all operators
        // and check preconditions manually. This is less efficient but works around
        // the lifetime issues with the successor generator.
        let mut successors = Vec::new();
        
        for op in self.task.get_operators() {
            // Check if all preconditions are satisfied
            let mut applicable = true;
            for precondition in op.preconditions() {
                if !self.state_satisfies_fact(state, precondition) {
                    applicable = false;
                    break;
                }
            }
            
            if applicable {
                match self.state_registry.get_successor_state(state, op) {
                    Ok(succ_state) => {
                        // Use operator cost (or default to 1.0)
                        let cost = 1.0; // TODO: Get actual operator cost from task
                        successors.push((succ_state, op.clone(), cost));
                    }
                    Err(_) => {
                        // Skip operators that can't be applied
                        continue;
                    }
                }
            }
        }
        
        successors
    }
    
    /// Performs one step of A* search
    fn step(&mut self) -> SearchStatus {
        if self.open_list.is_empty() {
            return SearchStatus::Failed;
        }
        
        // Get next node from open list
        let node = match self.open_list.pop() {
            Some(node) => node,
            None => return SearchStatus::Failed,
        };

        // Print debug info
        //println!("Expanding node: {:?}", node.state);
        //println!("{}", node.state.debug_with_registry(&self.state_registry));

        let state_id = node.state.get_id();
        
        // Check if already closed
        if self.closed_set.contains(&state_id) {
            if self.nodes_expanded <= 20 {
                println!("  Skipping already closed state: {}", state_id);
            }
            return SearchStatus::InProgress;
        }
        
        // Check if this node is stale (better path found since it was added to open list)
        if let Some(current_info) = self.search_nodes.get(&state_id) {
            if current_info.g_value < node.g_value() {
                if self.nodes_expanded <= 20 {
                    println!("  Skipping stale node: {} (current g: {} < node g: {})", 
                             state_id, current_info.g_value, node.g_value());
                }
                return SearchStatus::InProgress;
            }
        }
        
        // Close the node
        self.closed_set.insert(state_id);
        self.nodes_expanded += 1;
        
        // Debug: Print information about the expanded node
        if self.nodes_expanded <= 20 {
            let search_info = self.search_nodes.get(&state_id).unwrap();
            println!("Expanding node {} (g: {}, expanded: {})", 
                     state_id, search_info.g_value, self.nodes_expanded);
        }
        
        // Check if we're re-expanding a node (this should never happen in proper A*)
        if self.nodes_expanded > 20 && self.nodes_expanded % 10000 == 0 {
            println!("WARNING: Expanded {} nodes (this seems excessive)", self.nodes_expanded);
        }
        
        // Check if goal
        if self.is_goal_state(&node.state) {
            return SearchStatus::Solved;
        }
        
        // Generate successors
        let successors = self.generate_successors(&node.state);
        
        // Get the current best g-value for this state
        let current_g = if let Some(info) = self.search_nodes.get(&state_id) {
            info.g_value
        } else {
            0.0 // Initial state
        };
        
        for (succ_state, operator, op_cost) in successors {
            let succ_state_id = succ_state.get_id();
            
            // Skip if already closed
            if self.closed_set.contains(&succ_state_id) {
                continue;
            }
            
            let new_g_value = current_g + op_cost;
            
            // Debug: Print cost information for first few steps  
            if self.nodes_expanded <= 5 {
                println!("  -> Successor: {} (op: {}, cost: {}, g: {} -> {})", 
                         succ_state_id, operator.name(), op_cost, current_g, new_g_value);
            }
            
            // Check if we've seen this state before
            if let Some(existing_info) = self.search_nodes.get(&succ_state_id) {
                if existing_info.g_value <= new_g_value {
                    continue; // We already have a better or equal path
                }
            }
            
            // Create new search node info
            let node_info = SearchNodeInfo {
                parent_state: Some(state_id),
                parent_operator: Some(operator.clone()),
                g_value: new_g_value,
                status: NodeStatus::Open,
            };
            
            self.search_nodes.insert(succ_state_id, node_info);
            self.nodes_generated += 1;
            
            // Evaluate and add to open list
            if let Ok(evaluation) = self.evaluate_state(&succ_state, new_g_value) {
                let search_node = SearchNode::root(succ_state, evaluation);
                self.open_list.insert(search_node);
            }
        }
        
        SearchStatus::InProgress
    }
}

impl<'a> SearchEngine for AStarSearch<'a> {
    fn search(&mut self) -> SearchResult {
        let start_time = Instant::now();
        
        // Initialize search with initial state
    let initial_state = self.state_registry.get_initial_state();
        
        // Add initial state to open list
        if let Ok(initial_evaluation) = self.evaluate_state(&initial_state, 0.0) {
            let initial_node = SearchNode::root(initial_state.clone(), initial_evaluation);
            self.open_list.insert(initial_node);
        }
        
        // Initialize search node info for initial state
        let initial_info = SearchNodeInfo {
            parent_state: None,
            parent_operator: None,
            g_value: 0.0,
            status: NodeStatus::Open,
        };
        self.search_nodes.insert(initial_state.get_id(), initial_info);
        
        // Main search loop
        loop {
            // Check time limit
            if start_time.elapsed() > self.time_limit {
                return SearchResult {
                    status: SearchStatus::Timeout,
                    plan: None,
                    nodes_expanded: self.nodes_expanded,
                    nodes_generated: self.nodes_generated,
                    search_time: start_time.elapsed(),
                };
            }
            
            // Perform one search step
            match self.step() {
                SearchStatus::Solved => {
                    // Find the goal state (last closed state)
                    let goal_state_id = *self.closed_set.iter().last().unwrap();
                    let plan = self.extract_plan(goal_state_id);
                    
                    return SearchResult {
                        status: SearchStatus::Solved,
                        plan: Some(plan),
                        nodes_expanded: self.nodes_expanded,
                        nodes_generated: self.nodes_generated,
                        search_time: start_time.elapsed(),
                    };
                }
                SearchStatus::Failed => {
                    return SearchResult {
                        status: SearchStatus::Failed,
                        plan: None,
                        nodes_expanded: self.nodes_expanded,
                        nodes_generated: self.nodes_generated,
                        search_time: start_time.elapsed(),
                    };
                }
                SearchStatus::InProgress => continue,
                SearchStatus::Timeout => unreachable!(), // Handled above
            }
        }
    }
    
    fn print_initial_h_values(&mut self) {
        let initial_state = self.state_registry.get_initial_state();
        if let Ok(evaluation) = self.evaluate_state(&initial_state, 0.0) {
            println!("Initial heuristic value for {}: {}", 
                     self.heuristic.name(), 
                     evaluation.get_heuristic_value(&self.heuristic.name()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_search_status_enum() {
        // Test basic enum functionality
        assert_eq!(SearchStatus::InProgress, SearchStatus::InProgress);
        assert_ne!(SearchStatus::Solved, SearchStatus::Failed);
    }
    
    #[test]
    fn test_search_result_creation() {
        let result = SearchResult {
            status: SearchStatus::Failed,
            plan: None,
            nodes_expanded: 0,
            nodes_generated: 0,
            search_time: Duration::from_millis(100),
        };
        
        assert_eq!(result.status, SearchStatus::Failed);
        assert!(result.plan.is_none());
        assert_eq!(result.nodes_expanded, 0);
    }
}
