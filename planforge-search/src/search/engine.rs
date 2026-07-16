use super::config::{ExpansionScratch, SearchConfig};
use super::open_list::{DualQueueOpenList, OpenEntry};
use super::policy::SearchPolicy;
use super::space::{SearchNodeInfo, SearchSpace};
use super::stats::{ProgressSnapshot, SearchCounters, SearchStats, TraceFlags};
use super::{
    SearchEngine, SearchResult, SearchStatus, compute_effective_operator_costs, current_memory_kb,
    format_progress_value,
};
use crate::{
    evaluation::heuristic::BlindHeuristic,
    evaluation::{EvaluationError, EvaluationState, Heuristic},
    successor_generator::SuccessorTree,
};
use ordered_float::OrderedFloat;
use planforge_sas::numeric_task::{ExplicitFact, TaskRef};
use planforge_sas::state_registry::{ConcreteState, StateRegistry};
use std::env;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, error, info};

const MEMORY_CHECK_EXPANSION_INTERVAL: usize = 1024;

#[derive(Debug, Clone, Copy)]
struct SearchEvaluation {
    h_value: f64,
    f_value: f64,
    g_value: f64,
    is_dead_end: bool,
}

pub struct AStarSearch<'a> {
    task: TaskRef<'a>,
    state_registry: StateRegistry<'a>,
    successor_generator: SuccessorTree,

    // Search components.
    open_list: DualQueueOpenList,
    space: SearchSpace,

    // Evaluators.
    heuristic: Box<dyn Heuristic + 'a>,
    heuristic_name: String,
    initial_state_is_proven_optimal: bool,
    policy: SearchPolicy<'a>,

    config: SearchConfig,
    stats: SearchStats,
    scratch: ExpansionScratch,
    start_time: Option<Instant>,

    initial_state: Option<ConcreteState>,
    next_memory_check_expanded: usize,

    last_reported_f_layer: Option<i64>,
    best_reported_heuristic_value: Option<OrderedFloat<f64>>,
}

impl<'a> AStarSearch<'a> {
    /// Create a new A* search instance.
    pub fn new(
        task: TaskRef<'a>,
        state_registry: StateRegistry<'a>,
        heuristic: Option<Box<dyn Heuristic + 'a>>,
        time_limit: Option<Duration>,
        max_memory_bytes: Option<u64>,
    ) -> Self {
        Self::with_policy(
            task,
            state_registry,
            heuristic,
            time_limit,
            max_memory_bytes,
            SearchPolicy::AStar,
        )
    }

    /// Create a new greedy best-first search instance. Identical to A* except
    /// the open-list priority is `h` only — `g` is still tracked for plan cost
    /// but not used in tie-breaking. GBFS is incomplete in pathological cases
    /// and not admissible, but it solves many tasks far faster than A* with
    /// the same heuristic.
    pub fn new_gbfs(
        task: TaskRef<'a>,
        state_registry: StateRegistry<'a>,
        heuristic: Option<Box<dyn Heuristic + 'a>>,
        time_limit: Option<Duration>,
        max_memory_bytes: Option<u64>,
    ) -> Self {
        Self::with_policy(
            task,
            state_registry,
            heuristic,
            time_limit,
            max_memory_bytes,
            SearchPolicy::Gbfs,
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
        task: TaskRef<'a>,
        state_registry: StateRegistry<'a>,
        heuristic_fast: Box<dyn Heuristic + 'a>,
        heuristic_slow: Box<dyn Heuristic + 'a>,
        time_limit: Option<Duration>,
        max_memory_bytes: Option<u64>,
    ) -> Self {
        Self::with_policy(
            task,
            state_registry,
            Some(heuristic_fast),
            time_limit,
            max_memory_bytes,
            SearchPolicy::FastSlow {
                slow: heuristic_slow,
            },
        )
    }

    fn with_policy(
        task: TaskRef<'a>,
        state_registry: StateRegistry<'a>,
        heuristic: Option<Box<dyn Heuristic + 'a>>,
        time_limit: Option<Duration>,
        max_memory_bytes: Option<u64>,
        policy: SearchPolicy<'a>,
    ) -> Self {
        let successor_generator = SuccessorTree::new(&*task);

        // Build initial state early so numeric constants are initialized in the registry.
        // Required to derive a correct min_action_cost under metric.
        let mut state_registry = state_registry;
        let initial_state = state_registry.get_initial_state();
        let operator_costs =
            compute_effective_operator_costs(&*task, &state_registry, &initial_state);

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
        let initial_state_is_proven_optimal = heuristic.proves_initial_state_optimal();

        let use_metric = task.metric().use_metric();
        // Dual-queue preferred-first ordering is the FF default for GBFS.
        // The `PLANFORGE_NO_PREFERRED` environment variable forces it off
        // for A/B benchmarking; it doesn't affect correctness, only the
        // open-list pop order.
        let use_preferred_first =
            policy.is_gbfs() && env::var_os("PLANFORGE_NO_PREFERRED").is_none();
        let num_variables = task.variables().len();
        let num_numeric_variables = task.numeric_variables().len();
        Self {
            task,
            state_registry,
            successor_generator,
            open_list: DualQueueOpenList::new(use_preferred_first),
            space: SearchSpace::new(),
            heuristic,
            heuristic_name,
            initial_state_is_proven_optimal,
            policy,
            config: SearchConfig {
                operator_costs,
                use_metric,
                time_limit,
                max_memory_bytes,
                trace: TraceFlags::from_environment(),
            },
            stats: SearchStats::default(),
            scratch: ExpansionScratch::with_capacity(num_variables, num_numeric_variables),
            start_time: None,
            initial_state: Some(initial_state),
            next_memory_check_expanded: 0,
            last_reported_f_layer: None,
            best_reported_heuristic_value: None,
        }
    }

    fn resource_limit_status(&mut self, start_time: &Instant) -> Option<SearchStatus> {
        if let Some(time_limit) = self.config.time_limit
            && start_time.elapsed() > time_limit
        {
            return Some(SearchStatus::Timeout);
        }

        if let Some(max_memory_bytes) = self.config.max_memory_bytes {
            if self.stats.nodes_expanded < self.next_memory_check_expanded {
                return None;
            }
            self.next_memory_check_expanded =
                self.stats.nodes_expanded + MEMORY_CHECK_EXPANSION_INTERVAL;
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
            nodes_expanded: self.stats.nodes_expanded,
            nodes_reopened: self.stats.nodes_reopened,
            nodes_evaluated: self.stats.nodes_evaluated,
            evaluations: self.stats.nodes_evaluated,
            nodes_generated: self.stats.nodes_generated,
            dead_ends: self.stats.dead_ends,
            nodes_expanded_until_last_jump: self.stats.counters_at_last_jump.expanded,
            nodes_reopened_until_last_jump: self.stats.counters_at_last_jump.reopened,
            nodes_evaluated_until_last_jump: self.stats.counters_at_last_jump.evaluated,
            nodes_generated_until_last_jump: self.stats.counters_at_last_jump.generated,
            registered_states: self.state_registry.num_registered_states(),
            search_time: start_time.elapsed(),
        }
    }

    fn maybe_print_f_value(&mut self, f_value: f64, start_time: &Instant) {
        // For GBFS the priority is `h`, which is non-monotonic — the "next
        // layer" abstraction doesn't apply. Skip; per-improvement progress is
        // still reported via `maybe_report_heuristic_progress`.
        if !self.policy.reports_f_layers() {
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
        self.stats.counters_at_last_jump = SearchCounters {
            expanded: self.stats.nodes_expanded,
            reopened: self.stats.nodes_reopened,
            evaluated: self.stats.nodes_evaluated,
            generated: self.stats.nodes_generated,
        };

        info!(
            "{} = {} [{} evaluated, {} expanded, t={:.6}s, {} KB]",
            self.policy.priority_label(),
            f_layer,
            self.stats.nodes_evaluated,
            self.stats.nodes_expanded,
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
            self.stats.nodes_evaluated,
            self.stats.nodes_expanded,
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
            &*self.task,
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
                f_value: self.policy.priority_value(g_value, h_value),
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
    /// the search space instead of reinserting. The caller is responsible
    /// for `return`-ing immediately after this method so the existing
    /// pop is treated as a deferred expansion.
    fn evaluate_and_reinsert_for_slow(&mut self, entry: OpenEntry, state: &ConcreteState) {
        let SearchPolicy::FastSlow { slow } = &self.policy else {
            // Caller should have checked the policy variant first. Treat as a
            // no-op rather than panic.
            return;
        };
        let mut eval_state = EvaluationState::new_with_registry(
            state,
            entry.g_value,
            false,
            &*self.task,
            &self.state_registry,
        );
        eval_state.set_is_goal(self.is_goal_state(state));
        let slow_h = match slow.compute_heuristic(&eval_state) {
            Ok(h) => Some(h),
            Err(EvaluationError::DeadEnd { .. }) => Some(f64::INFINITY),
            Err(_) => None,
        };
        drop(eval_state);
        let Some(slow_h) = slow_h else {
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
        };
        if slow_h.is_infinite() && slow_h.is_sign_positive() {
            // h_s reports a dead end. Mark state and drop the entry.
            self.stats.dead_ends = self.stats.dead_ends.saturating_add(1);
            if let Some(info) = self.space.node_mut(entry.state_id) {
                info.is_dead_end = true;
            } else {
                self.space.set_node(
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
        state.fill_state(&self.state_registry, &mut self.scratch.state_values);
        self.scratch.applicable_operators.clear();
        self.successor_generator.get_applicable_operators(
            &self.scratch.state_values,
            &mut self.scratch.applicable_operators,
        );
    }

    pub fn initialize(&mut self) {
        debug_assert!(self.start_time.is_none());
        let start_time = Instant::now();
        self.start_time = Some(start_time);

        // Initialize search with initial state (created in constructor)
        let initial_state = self
            .initial_state
            .as_ref()
            .cloned()
            .unwrap_or_else(|| self.state_registry.get_initial_state());

        // Add initial state to open list
        match self.evaluate_state(&initial_state, 0.0) {
            Ok(initial_evaluation) => {
                self.stats.nodes_evaluated += 1;
                if initial_evaluation.is_dead_end {
                    self.stats.dead_ends += 1;
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
                    self.space.store_preferred(initial_id, initial_preferred);
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
        self.space.set_node(initial_state.get_id(), initial_info);
    }

    /// Perform one step of A* search.
    pub fn step(&mut self) -> SearchStatus {
        let start_time = *self
            .start_time
            .as_ref()
            .expect("step called before initialize");
        if let Some(status) = self.resource_limit_status(&start_time) {
            return status;
        }

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
        if self.space.node(state_id).is_some_and(|info| info.is_closed) {
            return SearchStatus::InProgress;
        }

        // Check if this node is stale (better path found since it was added to open list).
        if let Some(current_info) = self.space.node(state_id)
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
        if matches!(self.policy, SearchPolicy::FastSlow { .. }) && !entry.second {
            self.evaluate_and_reinsert_for_slow(entry, &state);
            return SearchStatus::InProgress;
        }

        self.maybe_print_f_layer(entry, &start_time);

        if self.config.trace.expanded_states {
            debug!(
                "TRACE expanded sid={} g={:.17} h={:.17} f={:.17}",
                state_id,
                entry.g_value,
                entry.h_value.into_inner(),
                entry.f_value.into_inner()
            );
        }

        if let Some(info) = self.space.node_mut(state_id) {
            info.is_closed = true;
        } else {
            self.space.set_node(
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
        self.stats.nodes_expanded += 1;

        if self.is_goal_state(&state) {
            return SearchStatus::Solved(state_id);
        }

        // Get the current best `g`-value for this state.
        let current_g = if let Some(info) = self.space.node(state_id) {
            info.g_value
        } else {
            0.0 // Initial state.
        };

        // Snapshot of this parent's preferred-operator IDs. Reading via
        // `take` is intentional: once we've started expanding `state` we
        // won't need them again unless the node is reopened, in which case
        // `evaluate_state` will resnapshot. Using `take` also reclaims the
        // boxed slice's memory eagerly.
        let parent_preferred_ids = self.space.take_preferred(state_id);

        self.populate_applicable_operators(&state);
        let mut applicable_operators = std::mem::take(&mut self.scratch.applicable_operators);
        let trace_initial_successors =
            self.stats.nodes_expanded == 1 && self.config.trace.initial_successors;
        let trace_improved_duplicates = self.config.trace.improved_duplicates;
        let trace_generated_states = self.config.trace.generated_states;
        let trace_evaluated_successors = self.config.trace.evaluated_successors;

        // Fill the parent's numeric/cost/metric values once; reuse across all
        // successors below.
        let mut expansion_context = std::mem::take(&mut self.scratch.expansion_context);
        if let Err(_) = self
            .state_registry
            .build_expansion_context(&state, &mut expansion_context)
        {
            self.scratch.expansion_context = expansion_context;
            self.scratch.applicable_operators = applicable_operators;
            return SearchStatus::InProgress;
        }

        // Clone the task handle so `operators` borrows the local `Arc`
        // rather than `self` (the loop body needs `&mut self`).
        let task = Arc::clone(&self.task);
        let operators = task.get_operators();
        for &op_id in applicable_operators.iter() {
            let operator_id = op_id as usize;
            let operator = &operators[operator_id];
            let (succ_state, op_cost) = match self.state_registry.apply_operator_in_context(
                &state,
                operator,
                &expansion_context,
                &mut self.scratch.successor_numeric,
                &mut self.scratch.successor_cost,
            ) {
                Ok(result) => result,
                Err(_) => continue,
            };
            let succ_state_id = succ_state.get_id();
            let op_cost = if self.config.use_metric {
                op_cost
            } else {
                self.config
                    .operator_costs
                    .get(operator_id)
                    .copied()
                    .unwrap_or(operator.cost() as f64)
            };
            let new_g_value = current_g + op_cost;

            // Count every successfully constructed successor state.
            self.stats.nodes_generated += 1;
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
            if let Some(existing_info) = self.space.node(succ_state_id) {
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
                if let Some(existing_info) = self.space.node_mut(succ_state_id) {
                    existing_info.is_closed = false;
                }
                self.stats.nodes_reopened += 1;
            }

            // Is this successor reached via one of the parent's
            // preferred (helpful) operators? We use the parent snapshot
            // taken above; per-successor it's a small linear scan, but
            // helpful-action lists from FF are typically tiny (single
            // digits), so this is cheap compared to evaluating the
            // successor.
            let is_preferred = parent_preferred_ids
                .as_deref()
                .is_some_and(|ids| ids.contains(&op_id));

            // Evaluate and add to open list.
            if let Ok(evaluation) = self.evaluate_state(&succ_state, new_g_value) {
                if !improved_duplicate {
                    self.stats.nodes_evaluated += 1;
                }

                // Snapshot the heuristic's preferred-operator IDs for the
                // successor *now*, before any other state's evaluation
                // overwrites the heuristic's internal scratch. Stored on
                // the search space and read back when this successor is
                // later expanded.
                let preferred_ids = self.heuristic.get_preferred_operator_ids();
                self.space.store_preferred(succ_state_id, preferred_ids);

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
                self.space.set_node(succ_state_id, node_info);

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
                    self.stats.dead_ends += 1;
                    continue;
                }

                let _ = self.maybe_report_heuristic_progress(&evaluation, &start_time);
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
        self.scratch.applicable_operators = applicable_operators;
        self.scratch.expansion_context = expansion_context;

        SearchStatus::InProgress
    }

    pub fn finish(&mut self, status: SearchStatus) -> SearchResult {
        let start_time = *self
            .start_time
            .as_ref()
            .expect("finish called before initialize");
        match status {
            SearchStatus::Solved(goal_state_id) => {
                // Use the goal state ID returned from step()
                let plan = self.space.extract_plan(goal_state_id, &*self.task);
                let solution_cost = self.space.node(goal_state_id).map(|info| info.g_value);

                assert!(
                    !self.initial_state_is_proven_optimal
                        || self.stats.counters_at_last_jump.expanded == 0,
                    "A* entered a higher f-layer after its heuristic proved h(init) = h*: {} nodes were expanded before the last jump",
                    self.stats.counters_at_last_jump.expanded
                );

                SearchResult {
                    status: SearchStatus::Solved(goal_state_id),
                    plan: Some(plan),
                    solution_cost,
                    nodes_expanded: self.stats.nodes_expanded,
                    nodes_reopened: self.stats.nodes_reopened,
                    nodes_evaluated: self.stats.nodes_evaluated,
                    evaluations: self.stats.nodes_evaluated,
                    nodes_generated: self.stats.nodes_generated,
                    dead_ends: self.stats.dead_ends,
                    nodes_expanded_until_last_jump: self.stats.counters_at_last_jump.expanded,
                    nodes_reopened_until_last_jump: self.stats.counters_at_last_jump.reopened,
                    nodes_evaluated_until_last_jump: self.stats.counters_at_last_jump.evaluated,
                    nodes_generated_until_last_jump: self.stats.counters_at_last_jump.generated,
                    registered_states: self.state_registry.num_registered_states(),
                    search_time: start_time.elapsed(),
                }
            }
            SearchStatus::Failed => self.terminal_result(SearchStatus::Failed, &start_time),
            SearchStatus::InProgress => unreachable!(),
            SearchStatus::Timeout => self.terminal_result(SearchStatus::Timeout, &start_time),
            SearchStatus::MemoryLimitReached => {
                self.terminal_result(SearchStatus::MemoryLimitReached, &start_time)
            }
        }
    }

    pub fn print_initial_h_values(&mut self) {
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

impl<'a> SearchEngine for AStarSearch<'a> {
    fn initialize(&mut self) {
        AStarSearch::initialize(self);
    }

    fn step(&mut self) -> SearchStatus {
        AStarSearch::step(self)
    }

    fn finish(&mut self, status: SearchStatus) -> SearchResult {
        AStarSearch::finish(self, status)
    }

    fn print_initial_h_values(&mut self) {
        AStarSearch::print_initial_h_values(self);
    }
}
