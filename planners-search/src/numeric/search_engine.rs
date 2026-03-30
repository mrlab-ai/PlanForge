//! Lightweight search engine implementation for numeric planning
//!
//! This module provides a simplified search engine based on the C++ Fast Downward
//! implementation, focusing on A* search with minimal overhead.

#[cfg(test)]
mod tests;

use crate::numeric::{
    evaluation::g_evaluator::{GEvaluator, SumEvaluator},
    evaluation::heuristic::BlindHeuristic,
    evaluation::{EvaluationResult, EvaluationState, Evaluator, Heuristic},
    open_lists::{OpenList, SearchNode, TieBreakingOpenList},
    successor_generator::{ApplicableOperator, GroundedSuccessorGenerator, Node},
};
use ordered_float::OrderedFloat;
use planners_sas::numeric::numeric_task::{AbstractNumericTask, Fact, Operator};
use planners_sas::numeric::state_registry::{ConcreteState, StateID, StateRegistry};
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};

fn min_action_cost_from_initial_metric_deltas<'a>(
    state_registry: &StateRegistry<'a>,
    initial_state: &ConcreteState,
    operators: &[Operator],
) -> f64 {
    let mut min_cost = f64::INFINITY;
    for op in operators {
        let delta = match state_registry.metric_delta_applying_operator(initial_state, op) {
            Ok(v) => v,
            Err(_) => continue,
        };
        min_cost = min_cost.min(delta);
    }

    if min_cost.is_finite() { min_cost } else { 0.0 }
}

/// Search status indicating the outcome of the search
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SearchStatus {
    InProgress,
    Solved(StateID), // Include the goal state ID
    Failed,
    Timeout,
    MemoryLimitReached,
}

/// A plan is a sequence of operators
pub type Plan = Vec<Operator>;

/// Search result containing the outcome and optional plan
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub status: SearchStatus,
    pub plan: Option<Plan>,
    /// Solution cost as used by the search engine (g-value of the goal state).
    ///
    /// When a metric is defined, this corresponds to the accumulated metric
    /// deltas.
    /// Otherwise, it corresponds to the sum of declared operator costs.
    pub solution_cost: Option<f64>,
    pub nodes_expanded: usize,
    pub nodes_reopened: usize,
    pub nodes_evaluated: usize,
    pub evaluations: usize,
    pub nodes_generated: usize,
    pub dead_ends: usize,
    pub nodes_expanded_until_last_jump: usize,
    pub nodes_reopened_until_last_jump: usize,
    pub nodes_evaluated_until_last_jump: usize,
    pub nodes_generated_until_last_jump: usize,
    pub registered_states: usize,
    pub search_time: Duration,
}

#[derive(Debug, Clone, Copy, Default)]
struct SearchCounters {
    expanded: usize,
    reopened: usize,
    evaluated: usize,
    generated: usize,
}

#[derive(Debug, Clone, Copy, Default)]
struct ProgressSnapshot {
    improved: bool,
}

/// Simple search node information for tracking parent relationships
#[derive(Debug, Clone)]
struct SearchNodeInfo {
    parent_state: Option<StateID>,
    parent_operator_id: Option<usize>,
    g_value: f64,
}

pub trait SearchEngine {
    fn search(&mut self) -> SearchResult;
    fn print_initial_h_values(&mut self);
}

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
    time_limit: Option<Duration>,
    max_memory_bytes: Option<u64>,
    initial_state: Option<ConcreteState>,

    // Statistics
    nodes_evaluated: usize,
    nodes_expanded: usize,
    nodes_reopened: usize,
    nodes_generated: usize,
    dead_ends: usize,
    counters_at_last_jump: SearchCounters,
    last_reported_f_layer: Option<OrderedFloat<f64>>,
    best_reported_heuristic_value: Option<OrderedFloat<f64>>,
    state_values_buffer: Vec<i32>,
    applicable_operators_buffer: Vec<ApplicableOperator<'a>>,
    successor_numeric_values_buffer: Vec<f64>,
    successor_cost_values_buffer: Vec<f64>,
}

impl<'a> AStarSearch<'a> {
    /// Creates a successor generator for the given task
    fn create_successor_generator(task: &'a dyn AbstractNumericTask) -> Box<dyn Node<'a> + 'a> {
        let mut queue = VecDeque::new();
        for (op_id, operator) in task.get_operators().iter().enumerate() {
            queue.push_back((operator, op_id));
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
        max_memory_bytes: Option<u64>,
    ) -> Self {
        let successor_generator = Self::create_successor_generator(task);

        // Build initial state early so numeric constants are initialized in the registry.
        // (Required to derive a correct min_action_cost under metric.)
        let mut state_registry = state_registry;
        let initial_state = state_registry.get_initial_state();

        // Determine min_action_cost.
        //
        // Numeric-fd mirror: in metric tasks, blind returns the minimum operator cost,
        // where each operator cost is the raw metric delta when applying the operator
        // in the initial state.
        let declared_min_cost = task
            .get_operators()
            .iter()
            .map(|op| op.cost() as f64)
            .fold(f64::INFINITY, |a, b| a.min(b));

        let uses_metric = task.metric().use_metric();
        let min_action_cost = if uses_metric {
            min_action_cost_from_initial_metric_deltas(
                &state_registry,
                &initial_state,
                task.get_operators(),
            )
        } else if declared_min_cost.is_finite() {
            declared_min_cost.max(0.0)
        } else {
            1.0
        };

        // Use BlindHeuristic as default, configured with min_action_cost
        let heuristic = heuristic.unwrap_or_else(|| {
            Box::new(BlindHeuristic::with_min_action_cost(min_action_cost, None))
        });

        // Create evaluators for A*
        let g_evaluator = GEvaluator::new(None);
        let f_evaluator = SumEvaluator::f_evaluator(heuristic.name());

        // Create open list with f-value primary, h-value secondary (tie-breaking)
        let evaluator_names = vec![f_evaluator.name(), heuristic.name()];
        let open_list = TieBreakingOpenList::new(evaluator_names, true)
            .expect("A* tie-breaking open list must have at least one evaluator");

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
            time_limit,
            max_memory_bytes,
            initial_state: Some(initial_state),
            nodes_evaluated: 0,
            nodes_expanded: 0,
            nodes_reopened: 0,
            nodes_generated: 0,
            dead_ends: 0,
            counters_at_last_jump: SearchCounters::default(),
            last_reported_f_layer: None,
            best_reported_heuristic_value: None,
            state_values_buffer: Vec::with_capacity(task.variables().len()),
            applicable_operators_buffer: Vec::new(),
            successor_numeric_values_buffer: Vec::with_capacity(task.numeric_variables().len()),
            successor_cost_values_buffer: Vec::new(),
        }
    }

    fn resource_limit_status(&self, start_time: &Instant) -> Option<SearchStatus> {
        if let Some(time_limit) = self.time_limit {
            if start_time.elapsed() > time_limit {
                return Some(SearchStatus::Timeout);
            }
        }

        if let Some(max_memory_bytes) = self.max_memory_bytes {
            let current_memory_bytes = current_memory_kb().saturating_mul(1024);
            if current_memory_bytes >= max_memory_bytes {
                return Some(SearchStatus::MemoryLimitReached);
            }
        }

        None
    }

    fn terminal_result(&self, status: SearchStatus, start_time: &Instant) -> SearchResult {
        SearchResult {
            status,
            plan: None,
            solution_cost: None,
            nodes_expanded: self.nodes_expanded,
            nodes_reopened: self.nodes_reopened,
            nodes_evaluated: self.nodes_evaluated,
            evaluations: self.nodes_evaluated,
            nodes_generated: self.nodes_generated,
            dead_ends: self.dead_ends,
            nodes_expanded_until_last_jump: self.counters_at_last_jump.expanded,
            nodes_reopened_until_last_jump: self.counters_at_last_jump.reopened,
            nodes_evaluated_until_last_jump: self.counters_at_last_jump.evaluated,
            nodes_generated_until_last_jump: self.counters_at_last_jump.generated,
            registered_states: self.state_registry.num_registered_states(),
            search_time: start_time.elapsed(),
        }
    }

    fn maybe_print_f_value(&mut self, f_value: f64, start_time: &Instant) {
        let f_value = OrderedFloat(f_value);
        if self.last_reported_f_layer == Some(f_value) {
            return;
        }

        self.last_reported_f_layer = Some(f_value);

        // Snapshot counters at the start of each new f-layer.
        // This mirrors Fast Downward's “until last jump” statistics.
        self.counters_at_last_jump = SearchCounters {
            expanded: self.nodes_expanded,
            reopened: self.nodes_reopened,
            evaluated: self.nodes_evaluated,
            generated: self.nodes_generated,
        };

        println!(
            "f = {} [{} evaluated, {} expanded, t={:.6}s, {} KB]",
            format_progress_value(f_value.into_inner()),
            self.nodes_evaluated,
            self.nodes_expanded,
            start_time.elapsed().as_secs_f64(),
            current_memory_kb(),
        );
    }

    fn maybe_print_f_layer(&mut self, node: &SearchNode, start_time: &Instant) {
        let f_value = node
            .evaluation
            .get_heuristic_value(&self.f_evaluator.name());
        self.maybe_print_f_value(f_value, start_time);
    }

    fn maybe_report_heuristic_progress(
        &mut self,
        evaluation: &EvaluationResult,
        start_time: &Instant,
    ) -> ProgressSnapshot {
        let h_value = OrderedFloat(evaluation.get_heuristic_value(&self.heuristic.name()));
        if self
            .best_reported_heuristic_value
            .is_some_and(|best| h_value >= best)
        {
            return ProgressSnapshot { improved: false };
        }

        self.best_reported_heuristic_value = Some(h_value);
        println!(
            "New best heuristic value for {}: {}",
            self.heuristic.name(),
            format_progress_value(h_value.into_inner()),
        );
        self.print_checkpoint_line(evaluation.g_value, start_time);

        ProgressSnapshot { improved: true }
    }

    fn print_checkpoint_line(&self, g_value: f64, start_time: &Instant) {
        println!(
            "[g={}, {} evaluated, {} expanded, t={:.6}s, {} KB]",
            format_progress_value(g_value),
            self.nodes_evaluated,
            self.nodes_expanded,
            start_time.elapsed().as_secs_f64(),
            current_memory_kb(),
        );
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
            if let (Some(parent_state), Some(operator_id)) =
                (node_info.parent_state, node_info.parent_operator_id)
            {
                plan.push(self.task.get_operators()[operator_id].clone());
                current_state = parent_state;
            } else {
                break; // Reached initial state
            }
        }

        plan.reverse();
        plan
    }

    /// Evaluates a state and creates evaluation result
    fn evaluate_state(
        &self,
        state: &ConcreteState,
        g_value: f64,
    ) -> Result<EvaluationResult, Box<dyn std::error::Error>> {
        let mut eval_state = EvaluationState::new_with_registry(
            state,
            g_value,
            false,
            self.task,
            &self.state_registry,
        );
        let is_goal = self.is_goal_state(state);
        eval_state.set_is_goal(is_goal);

        // Evaluate g-value
        self.g_evaluator.evaluate_state(&mut eval_state)?;

        // Evaluate heuristic (can use goal flag)
        self.heuristic.evaluate_state(&mut eval_state)?;

        // Evaluate f-value
        self.f_evaluator.evaluate_state(&mut eval_state)?;

        Ok(eval_state.into_result())
    }

    fn populate_applicable_operators(&mut self, state: &ConcreteState) {
        state.fill_state(&self.state_registry, &mut self.state_values_buffer);
        self.applicable_operators_buffer.clear();
        self.successor_generator.get_applicable_operators(
            &self.state_values_buffer,
            &mut self.applicable_operators_buffer,
        );
    }

    /// Performs one step of A* search
    fn step(&mut self, start_time: &Instant) -> SearchStatus {
        if self.open_list.is_empty() {
            return SearchStatus::Failed;
        }

        // Get next node from open list
        let node = match self.open_list.pop() {
            Some(node) => node,
            None => return SearchStatus::Failed,
        };

        let state_id = node.state.get_id();

        // Check if already closed
        if self.closed_set.contains(&state_id) {
            return SearchStatus::InProgress;
        }

        // Check if this node is stale (better path found since it was added to open list)
        if let Some(current_info) = self.search_nodes.get(&state_id) {
            if current_info.g_value < node.g_value() {
                return SearchStatus::InProgress;
            }
        }

        self.maybe_print_f_layer(&node, start_time);

        self.closed_set.insert(state_id);
        self.nodes_expanded += 1;

        if self.is_goal_state(&node.state) {
            return SearchStatus::Solved(state_id);
        }

        // Get the current best g-value for this state
        let current_g = if let Some(info) = self.search_nodes.get(&state_id) {
            info.g_value
        } else {
            0.0 // Initial state
        };

        self.populate_applicable_operators(&node.state);
        let mut applicable_operators = std::mem::take(&mut self.applicable_operators_buffer);

        for (operator, operator_id) in applicable_operators.iter().copied() {
            let succ_state = match self.state_registry.get_successor_state_with_buffers(
                &node.state,
                operator,
                &mut self.successor_numeric_values_buffer,
                &mut self.successor_cost_values_buffer,
            ) {
                Ok(succ_state) => succ_state,
                Err(_) => continue,
            };
            let succ_state_id = succ_state.get_id();

            // Count every successfully constructed successor state.
            self.nodes_generated += 1;
            let was_closed = self.closed_set.contains(&succ_state_id);

            let op_cost = if self.task.metric().use_metric() {
                self.state_registry
                    .transition_cost(&node.state, &succ_state)
                    .unwrap_or(1.0)
            } else {
                operator.cost() as f64
            };
            let new_g_value = current_g + op_cost;

            // Check if we've seen this state before
            if let Some(existing_info) = self.search_nodes.get(&succ_state_id) {
                if existing_info.g_value <= new_g_value {
                    continue; // We already have a better or equal path
                }
            }

            if was_closed {
                self.closed_set.remove(&succ_state_id);
                self.nodes_reopened += 1;
            }

            // Create new search node info
            let node_info = SearchNodeInfo {
                parent_state: Some(state_id),
                parent_operator_id: Some(operator_id),
                g_value: new_g_value,
            };

            // Record/update best g-value and parent pointers.
            self.search_nodes.insert(succ_state_id, node_info);

            // Evaluate and add to open list
            if let Ok(evaluation) = self.evaluate_state(&succ_state, new_g_value) {
                self.nodes_evaluated += 1;
                if evaluation.is_dead_end {
                    self.dead_ends += 1;
                    if !self.open_list.accepts_dead_ends() {
                        continue;
                    }
                }

                if !evaluation.is_dead_end {
                    let _ = self.maybe_report_heuristic_progress(&evaluation, start_time);
                }

                let search_node = SearchNode::root(succ_state, evaluation);
                self.open_list.insert(search_node);
            }
        }

        applicable_operators.clear();
        self.applicable_operators_buffer = applicable_operators;

        SearchStatus::InProgress
    }
}

impl<'a> SearchEngine for AStarSearch<'a> {
    fn search(&mut self) -> SearchResult {
        let start_time = Instant::now();

        // Initialize search with initial state (created in constructor)
        let initial_state = self
            .initial_state
            .as_ref()
            .cloned()
            .unwrap_or_else(|| self.state_registry.get_initial_state());

        // Add initial state to open list
        if let Ok(initial_evaluation) = self.evaluate_state(&initial_state, 0.0) {
            self.nodes_evaluated += 1;
            if initial_evaluation.is_dead_end {
                self.dead_ends += 1;
            } else {
                let progress =
                    self.maybe_report_heuristic_progress(&initial_evaluation, &start_time);
                if progress.improved {
                    self.maybe_print_f_value(
                        initial_evaluation.get_heuristic_value(&self.f_evaluator.name()),
                        &start_time,
                    );
                }
            }

            let initial_node = SearchNode::root(initial_state.clone(), initial_evaluation);
            if !initial_node.is_dead_end() || self.open_list.accepts_dead_ends() {
                self.open_list.insert(initial_node);
            }

            self.print_initial_h_values();
        }

        // Initialize search node info for initial state
        let initial_info = SearchNodeInfo {
            parent_state: None,
            parent_operator_id: None,
            g_value: 0.0,
        };
        self.search_nodes
            .insert(initial_state.get_id(), initial_info);

        // Main search loop
        loop {
            match self
                .resource_limit_status(&start_time)
                .unwrap_or_else(|| self.step(&start_time))
            {
                SearchStatus::Solved(goal_state_id) => {
                    // Use the goal state ID returned from step()
                    let plan = self.extract_plan(goal_state_id);
                    let solution_cost = self
                        .search_nodes
                        .get(&goal_state_id)
                        .map(|info| info.g_value);

                    return SearchResult {
                        status: SearchStatus::Solved(goal_state_id),
                        plan: Some(plan),
                        solution_cost,
                        nodes_expanded: self.nodes_expanded,
                        nodes_reopened: self.nodes_reopened,
                        nodes_evaluated: self.nodes_evaluated,
                        evaluations: self.nodes_evaluated,
                        nodes_generated: self.nodes_generated,
                        dead_ends: self.dead_ends,
                        nodes_expanded_until_last_jump: self.counters_at_last_jump.expanded,
                        nodes_reopened_until_last_jump: self.counters_at_last_jump.reopened,
                        nodes_evaluated_until_last_jump: self.counters_at_last_jump.evaluated,
                        nodes_generated_until_last_jump: self.counters_at_last_jump.generated,
                        registered_states: self.state_registry.num_registered_states(),
                        search_time: start_time.elapsed(),
                    };
                }
                SearchStatus::Failed => {
                    return self.terminal_result(SearchStatus::Failed, &start_time);
                }
                SearchStatus::InProgress => continue,
                SearchStatus::Timeout => {
                    return self.terminal_result(SearchStatus::Timeout, &start_time);
                }
                SearchStatus::MemoryLimitReached => {
                    return self.terminal_result(SearchStatus::MemoryLimitReached, &start_time);
                }
            }
        }
    }

    fn print_initial_h_values(&mut self) {
        let initial_state = self.state_registry.get_initial_state();
        if let Ok(evaluation) = self.evaluate_state(&initial_state, 0.0) {
            println!(
                "Initial heuristic value for {}: {}",
                self.heuristic.name(),
                format_progress_value(evaluation.get_heuristic_value(&self.heuristic.name()))
            );
        }
    }
}

fn format_progress_value(value: f64) -> String {
    if value.is_infinite() && value.is_sign_positive() {
        return "infinity".to_string();
    }

    if value.fract().abs() < 1e-9 {
        format!("{:.0}", value)
    } else {
        format!("{:.6}", value)
    }
}

#[cfg(target_os = "linux")]
fn current_memory_kb() -> u64 {
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if let Some(value) = line.strip_prefix("VmRSS:") {
                if let Some(kb) = value
                    .split_whitespace()
                    .next()
                    .and_then(|part| part.parse::<u64>().ok())
                {
                    return kb;
                }
            }
        }
    }

    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    let result = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if result != 0 {
        return 0;
    }
    let usage = unsafe { usage.assume_init() };
    usage.ru_maxrss.max(0) as u64
}

#[cfg(not(target_os = "linux"))]
fn current_memory_kb() -> u64 {
    0
}
