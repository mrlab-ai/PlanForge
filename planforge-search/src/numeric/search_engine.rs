//! Lightweight search engine implementation for numeric planning
//!
//! This module provides a simplified search engine based on the C++ Fast Downward
//! implementation, focusing on A* search with minimal overhead.

#[cfg(test)]
mod tests;

use crate::numeric::{
    evaluation::heuristic::BlindHeuristic,
    evaluation::{EvaluationError, EvaluationState, Heuristic},
    successor_generator::{ApplicableOperator, SuccessorTree},
};
use ordered_float::OrderedFloat;
use planforge_sas::numeric::numeric_task::{
    AbstractNumericTask, ExplicitFact, Operator, metric_operator_cost_from_initial_values,
};
use planforge_sas::numeric::state_registry::{
    ConcreteState, ExpansionContext, StateID, StateRegistry,
};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::env;
use std::time::{Duration, Instant};
use tracing::{debug, error, info};

const MEMORY_CHECK_EXPANSION_INTERVAL: usize = 1024;

pub fn compute_effective_operator_costs<'a>(
    task: &'a dyn AbstractNumericTask,
    _state_registry: &StateRegistry<'a>,
    _initial_state: &ConcreteState,
) -> Vec<f64> {
    task.get_operators()
        .iter()
        .map(|op| metric_operator_cost_from_initial_values(task, op))
        .collect()
}

/// Search status indicating the outcome of the search.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SearchStatus {
    InProgress,
    Solved(StateID), // Include the goal state ID.
    Failed,
    Timeout,
    MemoryLimitReached,
}

/// A plan is a sequence of operators.
pub type Plan = Vec<Operator>;

/// Search result containing the outcome and optional plan.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub status: SearchStatus,
    pub plan: Option<Plan>,
    /// Solution cost as used by the search engine (`g`-value of the goal state).
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

/// Simple search node information for tracking parent relationships.
#[derive(Debug, Clone, Copy)]
struct SearchNodeInfo {
    parent_state: Option<StateID>,
    parent_operator_id: Option<usize>,
    g_value: f64,
    is_dead_end: bool,
    is_closed: bool,
}

#[derive(Debug, Clone, Copy)]
struct SearchEvaluation {
    h_value: f64,
    f_value: f64,
    g_value: f64,
    is_dead_end: bool,
}

#[derive(Debug, Clone, Copy)]
struct TraceFlags {
    expanded_states: bool,
    initial_successors: bool,
    improved_duplicates: bool,
    generated_states: bool,
    evaluated_successors: bool,
}

impl TraceFlags {
    fn from_environment() -> Self {
        Self {
            expanded_states: env::var_os("TRACE_EXPANDED_STATES").is_some(),
            initial_successors: env::var_os("TRACE_INITIAL_SUCCESSORS").is_some(),
            improved_duplicates: env::var_os("TRACE_IMPROVED_DUPLICATES").is_some(),
            generated_states: env::var_os("TRACE_GENERATED_STATES").is_some(),
            evaluated_successors: env::var_os("TRACE_EVALUATED_SUCCESSORS").is_some(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct OpenEntry {
    f_value: OrderedFloat<f64>,
    h_value: OrderedFloat<f64>,
    g_value: f64,
    state_id: StateID,
    insertion_order: usize,
    /// `true` iff the operator that generated this successor was reported
    /// as a preferred (helpful) action by the heuristic for the parent
    /// state. Used as a tie-break inside a single queue, and as the queue
    /// selector for dual-queue GBFS.
    is_preferred: bool,
    /// `true` iff this entry has already been popped once and the slow
    /// admissible heuristic recomputed and folded in. Used only by the
    /// fast/slow A* variant (`new_fast_slow`). On the first pop of a
    /// `second == false` entry, the slow heuristic is evaluated, the
    /// entry is reinserted with `f' = g + max(h_f, h_s)` and
    /// `second = true`, and the expansion is deferred to the next pop.
    /// For ordinary A*/GBFS this field is always `false` and the field
    /// does not affect `Ord`.
    second: bool,
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
struct AStarOpenList {
    regular_heap: BinaryHeap<OpenEntry>,
    preferred_heap: BinaryHeap<OpenEntry>,
    use_preferred_first: bool,
    next_insertion_order: usize,
}

impl AStarOpenList {
    fn new(use_preferred_first: bool) -> Self {
        Self {
            regular_heap: BinaryHeap::new(),
            preferred_heap: BinaryHeap::new(),
            use_preferred_first,
            next_insertion_order: 0,
        }
    }

    fn insert(
        &mut self,
        state_id: StateID,
        g_value: f64,
        h_value: f64,
        f_value: f64,
        is_preferred: bool,
    ) {
        self.insert_with_second(state_id, g_value, h_value, f_value, is_preferred, false);
    }

    fn insert_with_second(
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

    fn pop(&mut self) -> Option<OpenEntry> {
        if self.use_preferred_first
            && let Some(entry) = self.preferred_heap.pop()
        {
            return Some(entry);
        }
        self.regular_heap.pop()
    }

    fn is_empty(&self) -> bool {
        self.regular_heap.is_empty() && self.preferred_heap.is_empty()
    }
}

pub trait SearchEngine {
    fn search(&mut self) -> SearchResult;
    fn print_initial_h_values(&mut self);
}

/// Priority-key construction for best-first search.
///
/// `Astar` uses `f = g + h`, the textbook admissible best-first key. `Gbfs`
/// drops `g` from the key — successors are popped strictly in order of `h`,
/// which is the greedy best-first variant. `g` is still accumulated for plan
/// cost reporting; only the open-list priority changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PriorityMode {
    Astar,
    Gbfs,
}

impl PriorityMode {
    #[inline]
    fn priority_value(self, g_value: f64, h_value: f64) -> f64 {
        match self {
            PriorityMode::Astar => g_value + h_value,
            PriorityMode::Gbfs => h_value,
        }
    }

    #[inline]
    fn priority_label(self) -> &'static str {
        match self {
            PriorityMode::Astar => "f",
            PriorityMode::Gbfs => "h",
        }
    }
}

pub struct AStarSearch<'a> {
    task: &'a dyn AbstractNumericTask,
    state_registry: StateRegistry<'a>,
    successor_generator: SuccessorTree<'a>,
    operator_costs: Vec<f64>,
    /// Cached `task.metric().use_metric()` so per-successor cost selection
    /// does not chase the trait vtable.
    use_metric: bool,

    // Search components.
    open_list: AStarOpenList,
    search_nodes: Vec<Option<SearchNodeInfo>>,
    /// Per-state cache of preferred operator IDs reported by the
    /// heuristic for that state, indexed by `state_id`. Populated right
    /// after `evaluate_state` returns `Ok` (so the snapshot is captured
    /// before the heuristic's internal cache is overwritten by the next
    /// state's evaluation). Read back when the state is *expanded* — we
    /// then mark each successor's open-list entry as preferred iff the
    /// operator that generated it is in this set.
    preferred_op_ids_by_state: Vec<Option<Box<[u32]>>>,

    // Evaluators.
    heuristic: Box<dyn Heuristic + 'a>,
    heuristic_name: String,
    /// Optional second admissible heuristic used by the fast/slow A*
    /// variant. When `Some`, every popped open-list entry with
    /// `second == false` is recomputed against this heuristic and
    /// reinserted with `f' = g + max(h_f, h_s)` and `second = true`. The
    /// first heuristic (`heuristic`) is the *fast* one used for initial
    /// ordering; `heuristic_slow` is the *slow* but more informative one
    /// computed lazily only when a state is actually about to be
    /// expanded. See `new_fast_slow` for construction.
    heuristic_slow: Option<Box<dyn Heuristic + 'a>>,
    heuristic_slow_name: Option<String>,
    priority_mode: PriorityMode,

    // Configuration.
    time_limit: Option<Duration>,
    max_memory_bytes: Option<u64>,
    initial_state: Option<ConcreteState>,
    next_memory_check_expanded: usize,
    trace_flags: TraceFlags,

    // Statistics.
    nodes_evaluated: usize,
    nodes_expanded: usize,
    nodes_reopened: usize,
    nodes_generated: usize,
    dead_ends: usize,
    counters_at_last_jump: SearchCounters,
    last_reported_f_layer: Option<i64>,
    best_reported_heuristic_value: Option<OrderedFloat<f64>>,
    state_values_buffer: Vec<usize>,
    applicable_operators_buffer: Vec<ApplicableOperator<'a>>,
    successor_numeric_values_buffer: Vec<f64>,
    successor_cost_values_buffer: Vec<f64>,
    expansion_context: ExpansionContext,
}

impl<'a> AStarSearch<'a> {
    /// Create a successor generator for the given task.
    fn create_successor_generator(task: &'a dyn AbstractNumericTask) -> SuccessorTree<'a> {
        SuccessorTree::new(task)
    }

    /// Create a new A* search instance.
    pub fn new(
        task: &'a dyn AbstractNumericTask,
        state_registry: StateRegistry<'a>,
        heuristic: Option<Box<dyn Heuristic + 'a>>,
        time_limit: Option<Duration>,
        max_memory_bytes: Option<u64>,
    ) -> Self {
        Self::with_priority_mode(
            task,
            state_registry,
            heuristic,
            time_limit,
            max_memory_bytes,
            PriorityMode::Astar,
        )
    }

    /// Create a new greedy best-first search instance. Identical to A* except
    /// the open-list priority is `h` only — `g` is still tracked for plan cost
    /// but not used in tie-breaking. GBFS is incomplete in pathological cases
    /// and not admissible, but it solves many tasks far faster than A* with
    /// the same heuristic.
    pub fn new_gbfs(
        task: &'a dyn AbstractNumericTask,
        state_registry: StateRegistry<'a>,
        heuristic: Option<Box<dyn Heuristic + 'a>>,
        time_limit: Option<Duration>,
        max_memory_bytes: Option<u64>,
    ) -> Self {
        Self::with_priority_mode(
            task,
            state_registry,
            heuristic,
            time_limit,
            max_memory_bytes,
            PriorityMode::Gbfs,
        )
    }

    /// A* with two admissible heuristics — a fast preliminary one
    /// (`heuristic_fast`, used to order the open list) and a slower but
    /// possibly tighter one (`heuristic_slow`, evaluated only when a state
    /// is about to be expanded).
    ///
    /// On the first pop of a state's open-list entry, the slow heuristic
    /// is computed, the entry is reinserted with priority
    /// `f' = g + max(h_f, h_s)`, and the expansion is deferred until the
    /// second pop. Because `max` of two admissible heuristics is
    /// admissible, the resulting search remains optimal. The benefit is
    /// that the slow heuristic is only evaluated on states A* actually
    /// considers expanding, not on every state generated.
    pub fn new_fast_slow(
        task: &'a dyn AbstractNumericTask,
        state_registry: StateRegistry<'a>,
        heuristic_fast: Box<dyn Heuristic + 'a>,
        heuristic_slow: Box<dyn Heuristic + 'a>,
        time_limit: Option<Duration>,
        max_memory_bytes: Option<u64>,
    ) -> Self {
        let slow_name = heuristic_slow.name();
        let mut search = Self::with_priority_mode(
            task,
            state_registry,
            Some(heuristic_fast),
            time_limit,
            max_memory_bytes,
            PriorityMode::Astar,
        );
        search.heuristic_slow = Some(heuristic_slow);
        search.heuristic_slow_name = Some(slow_name);
        search
    }

    fn with_priority_mode(
        task: &'a dyn AbstractNumericTask,
        state_registry: StateRegistry<'a>,
        heuristic: Option<Box<dyn Heuristic + 'a>>,
        time_limit: Option<Duration>,
        max_memory_bytes: Option<u64>,
        priority_mode: PriorityMode,
    ) -> Self {
        let successor_generator = Self::create_successor_generator(task);

        // Build initial state early so numeric constants are initialized in the registry.
        // Required to derive a correct min_action_cost under metric.
        let mut state_registry = state_registry;
        let initial_state = state_registry.get_initial_state();
        let operator_costs =
            compute_effective_operator_costs(task, &state_registry, &initial_state);

        // Determine `min_action_cost`.
        let min_action_cost = operator_costs
            .iter()
            .copied()
            .fold(f64::INFINITY, |a, b| a.min(b));

        let min_action_cost = if min_action_cost.is_finite() {
            min_action_cost.max(0.0)
        } else {
            1.0
        };

        // Use `BlindHeuristic` as default, configured with `min_action_cost`.
        let heuristic = heuristic.unwrap_or_else(|| {
            Box::new(BlindHeuristic::with_min_action_cost(min_action_cost, None))
        });
        let heuristic_name = heuristic.name();

        let use_metric = task.metric().use_metric();
        // Dual-queue preferred-first ordering is the FF default for GBFS.
        // The `PLANFORGE_NO_PREFERRED` environment variable forces it off
        // for A/B benchmarking; it doesn't affect correctness, only the
        // open-list pop order.
        let use_preferred_first = priority_mode == PriorityMode::Gbfs
            && env::var_os("PLANFORGE_NO_PREFERRED").is_none();
        Self {
            task,
            state_registry,
            successor_generator,
            operator_costs,
            use_metric,
            open_list: AStarOpenList::new(use_preferred_first),
            search_nodes: Vec::new(),
            preferred_op_ids_by_state: Vec::new(),
            heuristic,
            heuristic_name,
            heuristic_slow: None,
            heuristic_slow_name: None,
            priority_mode,
            time_limit,
            max_memory_bytes,
            initial_state: Some(initial_state),
            next_memory_check_expanded: 0,
            trace_flags: TraceFlags::from_environment(),
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
            expansion_context: ExpansionContext::default(),
        }
    }

    fn resource_limit_status(&mut self, start_time: &Instant) -> Option<SearchStatus> {
        if let Some(time_limit) = self.time_limit
            && start_time.elapsed() > time_limit
        {
            return Some(SearchStatus::Timeout);
        }

        if let Some(max_memory_bytes) = self.max_memory_bytes {
            if self.nodes_expanded < self.next_memory_check_expanded {
                return None;
            }
            self.next_memory_check_expanded = self.nodes_expanded + MEMORY_CHECK_EXPANSION_INTERVAL;
            let current_memory_bytes = current_memory_kb().saturating_mul(1024);
            if current_memory_bytes >= max_memory_bytes {
                return Some(SearchStatus::MemoryLimitReached);
            }
        }

        None
    }

    fn ensure_search_node_capacity(&mut self, state_id: StateID) {
        if state_id >= self.search_nodes.len() {
            self.search_nodes.resize(state_id + 1, None);
        }
    }

    fn search_node_info(&self, state_id: StateID) -> Option<&SearchNodeInfo> {
        self.search_nodes.get(state_id).and_then(Option::as_ref)
    }

    fn search_node_info_mut(&mut self, state_id: StateID) -> Option<&mut SearchNodeInfo> {
        self.search_nodes.get_mut(state_id).and_then(Option::as_mut)
    }

    fn set_search_node_info(&mut self, state_id: StateID, info: SearchNodeInfo) {
        self.ensure_search_node_capacity(state_id);
        self.search_nodes[state_id] = Some(info);
    }

    fn store_preferred_op_ids(&mut self, state_id: StateID, ids: Vec<usize>) {
        if state_id >= self.preferred_op_ids_by_state.len() {
            self.preferred_op_ids_by_state.resize_with(state_id + 1, || None);
        }
        if ids.is_empty() {
            self.preferred_op_ids_by_state[state_id] = None;
        } else {
            let packed: Box<[u32]> = ids.into_iter().map(|x| x as u32).collect();
            self.preferred_op_ids_by_state[state_id] = Some(packed);
        }
    }

    /// Remove and return the cached preferred-op IDs for `state_id`, if any.
    /// We `take` rather than borrow because the only consumer is the
    /// expansion step, after which the IDs aren't needed again unless the
    /// state is reopened — in which case `evaluate_state` will resnapshot.
    fn take_preferred_op_ids(&mut self, state_id: StateID) -> Option<Box<[u32]>> {
        self.preferred_op_ids_by_state.get_mut(state_id).and_then(Option::take)
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
        // For GBFS the priority is `h`, which is non-monotonic — the "next
        // layer" abstraction doesn't apply. Skip; per-improvement progress is
        // still reported via `maybe_report_heuristic_progress`.
        if self.priority_mode == PriorityMode::Gbfs {
            return;
        }
        let f_layer = f_value as i64;
        if self.last_reported_f_layer == Some(f_layer) {
            return;
        }
        match self.last_reported_f_layer {
            Some(last_layer) if f_layer <= last_layer => {
                return;
            }
            _ => {}
        }

        self.last_reported_f_layer = Some(f_layer);

        // Snapshot counters at the start of each new `f`-layer.
        // This mirrors Fast Downward's “until last jump” statistics.
        self.counters_at_last_jump = SearchCounters {
            expanded: self.nodes_expanded,
            reopened: self.nodes_reopened,
            evaluated: self.nodes_evaluated,
            generated: self.nodes_generated,
        };

        info!(
            "{} = {} [{} evaluated, {} expanded, t={:.6}s, {} KB]",
            self.priority_mode.priority_label(),
            f_layer,
            self.nodes_evaluated,
            self.nodes_expanded,
            start_time.elapsed().as_secs_f64(),
            current_memory_kb(),
        );
    }

    fn maybe_print_f_layer(&mut self, entry: OpenEntry, start_time: &Instant) {
        self.maybe_print_f_value(entry.f_value.into_inner(), start_time);
    }

    fn maybe_report_heuristic_progress(
        &mut self,
        evaluation: &SearchEvaluation,
        start_time: &Instant,
    ) -> ProgressSnapshot {
        let h_value = OrderedFloat(evaluation.h_value);
        if self
            .best_reported_heuristic_value
            .is_some_and(|best| h_value >= best)
        {
            return ProgressSnapshot { improved: false };
        }

        self.best_reported_heuristic_value = Some(h_value);
        info!(
            "New best heuristic value for {}: {}",
            self.heuristic_name,
            format_progress_value(h_value.into_inner()),
        );
        self.print_checkpoint_line(evaluation.g_value, start_time);

        ProgressSnapshot { improved: true }
    }

    fn print_checkpoint_line(&self, g_value: f64, start_time: &Instant) {
        info!(
            "[g={}, {} evaluated, {} expanded, t={:.6}s, {} KB]",
            format_progress_value(g_value),
            self.nodes_evaluated,
            self.nodes_expanded,
            start_time.elapsed().as_secs_f64(),
            current_memory_kb(),
        );
    }

    /// Check if the given state satisfies all goal conditions.
    fn is_goal_state(&self, state: &ConcreteState) -> bool {
        for i in 0..self.task.get_num_goals() {
            let goal_fact = self.task.get_goal_fact(i);
            if !self.state_satisfies_fact(state, goal_fact) {
                return false;
            }
        }
        true
    }

    /// Check if a state satisfies a specific fact.
    fn state_satisfies_fact(&self, state: &ConcreteState, fact: &ExplicitFact) -> bool {
        fact.is_hold(state, &self.state_registry)
    }

    /// Trace back the path from goal state to initial state.
    fn extract_plan(&self, goal_state: StateID) -> Plan {
        let mut plan = Vec::new();
        let mut current_state = goal_state;

        while let Some(node_info) = self.search_node_info(current_state) {
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

    /// Evaluate a state for A* without materializing named evaluator results.
    fn evaluate_state(
        &self,
        state: &ConcreteState,
        g_value: f64,
    ) -> Result<SearchEvaluation, Box<dyn std::error::Error>> {
        let mut eval_state = EvaluationState::new_with_registry(
            state,
            g_value,
            false,
            self.task,
            &self.state_registry,
        );
        let is_goal = self.is_goal_state(state);
        eval_state.set_is_goal(is_goal);

        let evaluation = match self.heuristic.compute_heuristic(&eval_state) {
            Ok(h_value) if h_value.is_infinite() && h_value.is_sign_positive() => {
                SearchEvaluation {
                    h_value,
                    f_value: f64::INFINITY,
                    g_value,
                    is_dead_end: true,
                }
            }
            Ok(h_value) => SearchEvaluation {
                h_value,
                f_value: self.priority_mode.priority_value(g_value, h_value),
                g_value,
                is_dead_end: false,
            },
            Err(EvaluationError::DeadEnd { reliable }) => {
                let _ = reliable;
                SearchEvaluation {
                    h_value: f64::INFINITY,
                    f_value: f64::INFINITY,
                    g_value,
                    is_dead_end: true,
                }
            }
            Err(err) => return Err(Box::new(err)),
        };
        Ok(evaluation)
    }

    /// Compute the slow heuristic for `state`, fold it into the entry via
    /// `max(h_f, h_s)`, and reinsert as a `second == true` entry. On
    /// dead-end (h_s = +infinity), mark the state dead in
    /// `search_nodes` instead of reinserting. The caller is responsible
    /// for `return`-ing immediately after this method so the existing
    /// pop is treated as a deferred expansion.
    fn evaluate_and_reinsert_for_slow(&mut self, entry: OpenEntry, state: &ConcreteState) {
        let Some(slow) = self.heuristic_slow.as_ref() else {
            // Caller should have checked `is_some()` first. Treat as a
            // no-op rather than panic.
            return;
        };
        let mut eval_state = EvaluationState::new_with_registry(
            state,
            entry.g_value,
            false,
            self.task,
            &self.state_registry,
        );
        eval_state.set_is_goal(self.is_goal_state(state));
        let slow_h = match slow.compute_heuristic(&eval_state) {
            Ok(h) => h,
            Err(EvaluationError::DeadEnd { .. }) => f64::INFINITY,
            Err(_) => {
                // Computation failure: behave conservatively by keeping
                // the fast value (no update).
                self.open_list.insert_with_second(
                    entry.state_id,
                    entry.g_value,
                    entry.h_value.into_inner(),
                    entry.f_value.into_inner(),
                    entry.is_preferred,
                    true,
                );
                return;
            }
        };
        if slow_h.is_infinite() && slow_h.is_sign_positive() {
            // h_s reports a dead end. Mark state and drop the entry.
            self.dead_ends = self.dead_ends.saturating_add(1);
            if let Some(info) = self.search_node_info_mut(entry.state_id) {
                info.is_dead_end = true;
            } else {
                self.set_search_node_info(
                    entry.state_id,
                    SearchNodeInfo {
                        parent_state: None,
                        parent_operator_id: None,
                        g_value: entry.g_value,
                        is_dead_end: true,
                        is_closed: false,
                    },
                );
            }
            return;
        }
        let combined_h = entry.h_value.into_inner().max(slow_h);
        let new_f = entry.g_value + combined_h;
        self.open_list.insert_with_second(
            entry.state_id,
            entry.g_value,
            combined_h,
            new_f,
            entry.is_preferred,
            true,
        );
    }

    fn populate_applicable_operators(&mut self, state: &ConcreteState) {
        state.fill_state(&self.state_registry, &mut self.state_values_buffer);
        self.applicable_operators_buffer.clear();
        self.successor_generator.get_applicable_operators(
            &self.state_values_buffer,
            &mut self.applicable_operators_buffer,
        );
    }

    /// Perform one step of A* search.
    fn step(&mut self, start_time: &Instant) -> SearchStatus {
        if self.open_list.is_empty() {
            return SearchStatus::Failed;
        }

        // Get next node from open list
        let entry = match self.open_list.pop() {
            Some(entry) => entry,
            None => return SearchStatus::Failed,
        };

        let state_id = entry.state_id;
        let state = match self.state_registry.lookup_state(state_id) {
            Ok(state) => state,
            Err(_) => return SearchStatus::InProgress,
        };

        // Check if already closed.
        if self
            .search_node_info(state_id)
            .is_some_and(|info| info.is_closed)
        {
            return SearchStatus::InProgress;
        }

        // Check if this node is stale (better path found since it was added to open list).
        if let Some(current_info) = self.search_node_info(state_id)
            && current_info.g_value < entry.g_value
        {
            return SearchStatus::InProgress;
        }

        // Fast/slow A* lazy slow-heuristic step. If a slow heuristic is
        // configured and this entry hasn't yet been re-evaluated against
        // it, compute h_s now, reinsert with `f' = g + max(h_f, h_s)` and
        // `second = true`, and defer the actual expansion to the next pop.
        // Mirrors the AAAI paper's algorithm: every popped entry is
        // either a "first pop" that triggers the slow evaluation, or a
        // "second pop" that proceeds to expand. Because max of admissible
        // heuristics is admissible, optimality is preserved.
        if self.heuristic_slow.is_some() && !entry.second {
            self.evaluate_and_reinsert_for_slow(entry, &state);
            return SearchStatus::InProgress;
        }

        self.maybe_print_f_layer(entry, start_time);

        if self.trace_flags.expanded_states {
            debug!(
                "TRACE expanded sid={} g={:.17} h={:.17} f={:.17}",
                state_id,
                entry.g_value,
                entry.h_value.into_inner(),
                entry.f_value.into_inner()
            );
        }

        if let Some(info) = self.search_node_info_mut(state_id) {
            info.is_closed = true;
        } else {
            self.set_search_node_info(
                state_id,
                SearchNodeInfo {
                    parent_state: None,
                    parent_operator_id: None,
                    g_value: entry.g_value,
                    is_dead_end: false,
                    is_closed: true,
                },
            );
        }
        self.nodes_expanded += 1;

        if self.is_goal_state(&state) {
            return SearchStatus::Solved(state_id);
        }

        // Get the current best `g`-value for this state.
        let current_g = if let Some(info) = self.search_node_info(state_id) {
            info.g_value
        } else {
            0.0 // Initial state.
        };

        // Snapshot of this parent's preferred-operator IDs. Reading via
        // `take` is intentional: once we've started expanding `state` we
        // won't need them again unless the node is reopened, in which case
        // `evaluate_state` will resnapshot. Using `take` also reclaims the
        // boxed slice's memory eagerly.
        let parent_preferred_ids = self.take_preferred_op_ids(state_id);

        self.populate_applicable_operators(&state);
        let mut applicable_operators = std::mem::take(&mut self.applicable_operators_buffer);
        let trace_initial_successors =
            self.nodes_expanded == 1 && self.trace_flags.initial_successors;
        let trace_improved_duplicates = self.trace_flags.improved_duplicates;
        let trace_generated_states = self.trace_flags.generated_states;
        let trace_evaluated_successors = self.trace_flags.evaluated_successors;

        // Fill the parent's numeric/cost/metric values once; reuse across all
        // successors below.
        let mut expansion_context = std::mem::take(&mut self.expansion_context);
        if let Err(_) = self
            .state_registry
            .build_expansion_context(&state, &mut expansion_context)
        {
            self.expansion_context = expansion_context;
            self.applicable_operators_buffer = applicable_operators;
            return SearchStatus::InProgress;
        }

        for (operator, operator_id) in applicable_operators.iter().copied() {
            let (succ_state, op_cost) = match self.state_registry.apply_operator_in_context(
                &state,
                operator,
                &expansion_context,
                &mut self.successor_numeric_values_buffer,
                &mut self.successor_cost_values_buffer,
            ) {
                Ok(result) => result,
                Err(_) => continue,
            };
            let succ_state_id = succ_state.get_id();
            let op_cost = if self.use_metric {
                op_cost
            } else {
                self.operator_costs
                    .get(operator_id)
                    .copied()
                    .unwrap_or(operator.cost() as f64)
            };
            let new_g_value = current_g + op_cost;

            // Count every successfully constructed successor state.
            self.nodes_generated += 1;
            if trace_generated_states {
                debug!(
                    "TRACE generated parent_sid={} succ_sid={} op={} g={}",
                    state_id,
                    succ_state_id,
                    operator.name(),
                    format_progress_value(new_g_value)
                );
            }

            // Check if we've seen this state before.
            let mut improved_duplicate = false;
            let mut was_closed = false;
            let mut old_g = None;
            if let Some(existing_info) = self.search_node_info(succ_state_id) {
                if existing_info.is_dead_end {
                    continue;
                }
                if existing_info.g_value <= new_g_value {
                    // We already have a better or equal path.
                    continue;
                }
                improved_duplicate = true;
                was_closed = existing_info.is_closed;
                old_g = Some(existing_info.g_value);
            }

            if was_closed {
                if let Some(existing_info) = self.search_node_info_mut(succ_state_id) {
                    existing_info.is_closed = false;
                }
                self.nodes_reopened += 1;
            }

            // Is this successor reached via one of the parent's
            // preferred (helpful) operators? We use the parent snapshot
            // taken above; per-successor it's a small linear scan, but
            // helpful-action lists from FF are typically tiny (single
            // digits), so this is cheap compared to evaluating the
            // successor.
            let is_preferred = parent_preferred_ids
                .as_deref()
                .is_some_and(|ids| ids.contains(&(operator_id as u32)));

            // Evaluate and add to open list.
            if let Ok(evaluation) = self.evaluate_state(&succ_state, new_g_value) {
                if !improved_duplicate {
                    self.nodes_evaluated += 1;
                }

                // Snapshot the heuristic's preferred-operator IDs for the
                // successor *now*, before any other state's evaluation
                // overwrites the heuristic's internal scratch. Stored on
                // `preferred_op_ids_by_state[succ_state_id]` and read back
                // when this successor is later expanded.
                let preferred_ids = self.heuristic.get_preferred_operator_ids();
                self.store_preferred_op_ids(succ_state_id, preferred_ids);

                if trace_evaluated_successors {
                    debug!(
                        "TRACE evaluated-successor parent_sid={} succ_sid={} op={} g={:.17} h={:.17} f={:.17} dead_end={}",
                        state_id,
                        succ_state_id,
                        operator.name(),
                        new_g_value,
                        evaluation.h_value,
                        evaluation.f_value,
                        evaluation.is_dead_end,
                    );
                }

                if improved_duplicate && trace_improved_duplicates {
                    debug!(
                        "TRACE improved-duplicate sid={} op={} old_g={} new_g={} h={} dead_end={}",
                        succ_state_id,
                        operator.name(),
                        old_g
                            .map(format_progress_value)
                            .unwrap_or_else(|| "<missing>".to_string()),
                        format_progress_value(new_g_value),
                        format_progress_value(evaluation.h_value),
                        evaluation.is_dead_end,
                    );
                }

                let node_info = SearchNodeInfo {
                    parent_state: Some(state_id),
                    parent_operator_id: Some(operator_id),
                    g_value: new_g_value,
                    is_dead_end: evaluation.is_dead_end,
                    is_closed: false,
                };

                // Record/update best `g`-value, parent pointers, and dead-end status.
                self.set_search_node_info(succ_state_id, node_info);

                if trace_initial_successors {
                    debug!(
                        "TRACE initial-successor op={} g={} h={} f={} dead_end={} state_id={}",
                        operator.name(),
                        format_progress_value(new_g_value),
                        format_progress_value(evaluation.h_value),
                        format_progress_value(evaluation.f_value),
                        evaluation.is_dead_end,
                        succ_state_id
                    );
                }
                if evaluation.is_dead_end {
                    self.dead_ends += 1;
                    continue;
                }

                let _ = self.maybe_report_heuristic_progress(&evaluation, start_time);
                self.open_list.insert(
                    succ_state_id,
                    new_g_value,
                    evaluation.h_value,
                    evaluation.f_value,
                    is_preferred,
                );
            }
        }

        applicable_operators.clear();
        self.applicable_operators_buffer = applicable_operators;
        self.expansion_context = expansion_context;

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
        match self.evaluate_state(&initial_state, 0.0) {
            Ok(initial_evaluation) => {
                self.nodes_evaluated += 1;
                if initial_evaluation.is_dead_end {
                    self.dead_ends += 1;
                } else {
                    let progress =
                        self.maybe_report_heuristic_progress(&initial_evaluation, &start_time);
                    if progress.improved {
                        self.maybe_print_f_value(initial_evaluation.f_value, &start_time);
                    }
                }

                if !initial_evaluation.is_dead_end {
                    // The initial state has no parent operator, so
                    // "preferred-via-parent" is vacuously false. Still
                    // snapshot the initial state's own preferred IDs so
                    // its successors can be classified.
                    let initial_id = initial_state.get_id();
                    let initial_preferred = self.heuristic.get_preferred_operator_ids();
                    self.store_preferred_op_ids(initial_id, initial_preferred);
                    self.open_list.insert(
                        initial_id,
                        0.0,
                        initial_evaluation.h_value,
                        initial_evaluation.f_value,
                        false,
                    );
                }

                self.print_initial_h_values();
            }
            Err(err) => {
                error!("Initial state evaluation failed: {err}");
            }
        }

        // Initialize search node info for initial state.
        let initial_info = SearchNodeInfo {
            parent_state: None,
            parent_operator_id: None,
            g_value: 0.0,
            is_dead_end: false,
            is_closed: false,
        };
        self.set_search_node_info(initial_state.get_id(), initial_info);

        // Main search loop.
        loop {
            match self
                .resource_limit_status(&start_time)
                .unwrap_or_else(|| self.step(&start_time))
            {
                SearchStatus::Solved(goal_state_id) => {
                    // Use the goal state ID returned from step()
                    let plan = self.extract_plan(goal_state_id);
                    let solution_cost = self
                        .search_node_info(goal_state_id)
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
            info!(
                "Initial heuristic value for {}: {}",
                self.heuristic_name,
                format_progress_value(evaluation.h_value)
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
            if let Some(value) = line.strip_prefix("VmRSS:")
                && let Some(kb) = value
                    .split_whitespace()
                    .next()
                    .and_then(|part| part.parse::<u64>().ok())
            {
                return kb;
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
