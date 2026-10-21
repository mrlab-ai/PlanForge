use super::config::{ExpansionScratch, SearchConfig};
use super::open_list::{OpenEntry, OpenList, PreferredFirstOpenList, TieBreakingOpenList};
use super::policy::{GPlusH, H, Priority};
use super::space::{SearchNodeInfo, SearchSpace};
use super::stats::{ProgressSnapshot, SearchCounters, SearchStats};
use super::{SearchEngine, SearchResult, SearchStatus, current_memory_kb, format_progress_value};
use crate::{
    evaluation::{EvaluationError, EvaluationState, Heuristic},
    successor_generator::SuccessorTree,
};
use anyhow::{Context, Result, anyhow};
use ordered_float::OrderedFloat;
use planforge_sas::numeric_task::{
    AbstractNumericTask, ExplicitFact, INF_VALUE, NumericValue, ONE_VALUE, Operator, OperatorIndex,
    ZERO_VALUE,
};
use planforge_sas::state_registry::{ConcreteState, StateID, StateRegistry};
use planforge_sas::utils::float_tolerance;
use std::time::{Duration, Instant};
use tracing::{debug, info};

const MEMORY_CHECK_EXPANSION_INTERVAL: usize = 1024;

/// Outcome of taking one entry off the open list.
enum PoppedNode {
    /// Expand this entry's state.
    Expand(OpenEntry, ConcreteState),
    /// The entry is closed, superseded or deferred; the search continues.
    Skipped,
    /// The open list holds nothing more.
    Exhausted,
}

/// The operator that produced one successor, in the three forms the search
/// needs it: the operator itself, its id in the task and in the successor
/// generator, and the metric cost its application incurred.
struct AppliedOperator<'a> {
    operator: &'a Operator,
    operator_id: usize,
    op_id: OperatorIndex,
    metric_op_cost: NumericValue,
}

/// The values every successor of one expansion shares.
struct ParentExpansion {
    state_id: StateID,
    g_value: f64,
    /// Operators the heuristic called helpful in the parent, or `None` when
    /// the heuristic does not report preferred operators.
    preferred_operator_range: Option<(OperatorIndex, OperatorIndex)>,
}

#[derive(Debug, Clone, Copy)]
struct SearchEvaluation {
    h_value: f64,
    f_value: f64,
    g_value: f64,
    is_dead_end: bool,
    heuristic_revision: u64,
}

/// Inputs shared by every best-first search instantiation.
pub struct SearchParams<'a> {
    pub heuristic: Box<dyn Heuristic + 'a>,
    pub lazy: Option<Box<dyn Heuristic + 'a>>,
    pub reevaluate_dynamic: bool,
    pub time_limit: Option<Duration>,
    pub max_memory_bytes: Option<u64>,
}

impl<'a> SearchParams<'a> {
    pub fn new(heuristic: Box<dyn Heuristic + 'a>) -> Self {
        Self {
            heuristic,
            lazy: None,
            reevaluate_dynamic: false,
            time_limit: None,
            max_memory_bytes: None,
        }
    }
}

#[derive(Clone, Copy, Default)]
struct HeuristicRecord {
    revision: u64,
    h: f64,
}

enum PopAction {
    Expand,
    Requeue(OpenEntry),
    Prune,
    Skip,
}

pub struct BestFirstSearch<'a, P: Priority, Q: OpenList> {
    task: &'a dyn AbstractNumericTask,
    operators: &'a [Operator],
    goals: &'a [ExplicitFact],
    state_registry: StateRegistry<'a>,
    successor_generator: SuccessorTree,

    // Search components.
    open_list: Q,
    space: SearchSpace,

    // Evaluators.
    heuristic: Box<dyn Heuristic + 'a>,
    heuristic_name: String,
    initial_state_is_proven_optimal: bool,
    lazy: Option<Box<dyn Heuristic + 'a>>,
    priority: std::marker::PhantomData<P>,
    next_insertion_order: u32,

    config: SearchConfig<'a>,
    stats: SearchStats,
    scratch: ExpansionScratch,
    start_time: Option<Instant>,

    initial_state: ConcreteState,
    next_memory_check_expanded: usize,

    last_reported_f_layer: Option<i64>,
    best_reported_heuristic_value: Option<OrderedFloat<f64>>,
    /// Per-state heuristic revision at the most recent evaluation. Allocated
    /// only for A* with `mpd=true`; ordinary static searches pay no per-state
    /// memory cost.
    heuristic_revisions: Option<Vec<HeuristicRecord>>,
}

pub type AStarSearch<'a> = BestFirstSearch<'a, GPlusH, TieBreakingOpenList>;
pub type GreedyBestFirstSearch<'a> = BestFirstSearch<'a, H, PreferredFirstOpenList>;

impl<'a, P: Priority, Q: OpenList> BestFirstSearch<'a, P, Q> {
    pub fn new(
        task: &'a dyn AbstractNumericTask,
        state_registry: StateRegistry<'a>,
        params: SearchParams<'a>,
    ) -> Self {
        let SearchParams {
            heuristic,
            lazy,
            reevaluate_dynamic,
            time_limit,
            max_memory_bytes,
        } = params;
        let successor_generator = SuccessorTree::new(task);

        // Build initial state early so numeric constants are initialized in the registry.
        // Required to derive a correct min_action_cost under metric.
        let mut state_registry = state_registry;
        let initial_state = state_registry.get_initial_state();
        let operator_costs = task.effective_operator_costs();

        // Determine `min_action_cost`.
        let min_action_cost = operator_costs
            .iter()
            .copied()
            .fold(INF_VALUE, NumericValue::min);

        let min_action_cost = if min_action_cost.is_finite() {
            min_action_cost.max(ZERO_VALUE)
        } else {
            ONE_VALUE
        };

        let heuristic_name = heuristic.heuristic_name().to_string();
        let initial_state_is_proven_optimal = heuristic.proves_initial_state_optimal();

        let use_metric = task.metric().use_metric();
        let num_variables = task.variables().len();
        let num_numeric_variables = task.numeric_variables().len();
        info!(
            "State representation: bins={}",
            state_registry.global_state_packer().num_bins()
        );
        Self {
            task,
            operators: task.get_operators(),
            goals: task.goals(),
            state_registry,
            successor_generator,
            open_list: Q::default(),
            space: SearchSpace::new(),
            heuristic,
            heuristic_name,
            initial_state_is_proven_optimal,
            lazy,
            priority: std::marker::PhantomData,
            next_insertion_order: 0,
            config: SearchConfig {
                operator_costs,
                min_action_cost: min_action_cost.value(),
                use_metric,
                time_limit,
                max_memory_bytes,
            },
            stats: SearchStats::default(),
            scratch: ExpansionScratch::with_capacity(num_variables, num_numeric_variables),
            start_time: None,
            initial_state,
            next_memory_check_expanded: 0,
            last_reported_f_layer: None,
            best_reported_heuristic_value: None,
            heuristic_revisions: reevaluate_dynamic.then(Vec::new),
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

    fn maybe_print_f_value(&mut self, f_value: f64, start_time: &Instant) {
        // For GBFS the priority is `h`, which is non-monotonic — the "next
        // layer" abstraction doesn't apply. Skip; per-improvement progress is
        // still reported via `maybe_report_heuristic_progress`.
        if !P::MONOTONE {
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
            "{} = {} [{} evaluated, {} expanded, {} states, {} open, t={:.6}s, {} KB]",
            P::LABEL,
            f_layer,
            self.stats.nodes_evaluated,
            self.stats.nodes_expanded,
            self.state_registry.num_registered_states(),
            self.open_list.len(),
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
        self.state_registry.view(state).holds_all(self.goals)
    }

    /// Evaluate a state for A* without materializing named evaluator results.
    fn evaluate_state(
        &self,
        heuristic: &dyn Heuristic,
        state: &ConcreteState,
        g_value: f64,
        is_goal: bool,
    ) -> Result<SearchEvaluation> {
        let mut eval_state = EvaluationState::new(state, self.task, &self.state_registry);
        eval_state.set_is_goal(is_goal);

        let evaluation = match heuristic.compute_heuristic(&eval_state) {
            Ok(h_value) if h_value.is_infinite() && h_value.is_sign_positive() => {
                SearchEvaluation {
                    h_value,
                    f_value: f64::INFINITY,
                    g_value,
                    is_dead_end: true,
                    heuristic_revision: heuristic.revision(),
                }
            }
            Ok(h_value) => {
                // Every plan from a non-goal state applies an action, so it
                // costs at least the cheapest one: the maximum with the blind
                // heuristic stays admissible and consistent, and lifts a
                // heuristic that is 0 on a non-goal state, as an abstraction
                // is on every state of an abstract goal state.
                let h_value = if is_goal {
                    h_value
                } else {
                    h_value.max(self.config.min_action_cost)
                };
                SearchEvaluation {
                    h_value,
                    f_value: P::key(g_value, h_value),
                    g_value,
                    is_dead_end: false,
                    heuristic_revision: heuristic.revision(),
                }
            }
            Err(EvaluationError::DeadEnd { reliable }) => {
                // Reliable and unreliable dead ends both prune, matching Fast
                // Downward's `OpenList::is_dead_end`: an unreliable detection
                // prunes as soon as *every* evaluator agrees, and this engine
                // runs a single heuristic, so agreement is trivial. The
                // `Ok(+infinity)` arm above marks the state dead on the same
                // grounds. Reliability is still worth seeing in a trace,
                // because it is the one case where pruning rests on the
                // heuristic rather than on a proof.
                if !reliable {
                    debug!(
                        "pruning state on an unreliable dead-end report from {}",
                        heuristic.heuristic_name()
                    );
                }
                SearchEvaluation {
                    h_value: f64::INFINITY,
                    f_value: f64::INFINITY,
                    g_value,
                    is_dead_end: true,
                    heuristic_revision: heuristic.revision(),
                }
            }
            Err(err) => return Err(anyhow!(err)),
        };
        Ok(evaluation)
    }

    fn record_heuristic_revision(&mut self, state_id: usize, revision: u64, h: f64) {
        let Some(revisions) = &mut self.heuristic_revisions else {
            return;
        };
        if revisions.len() <= state_id {
            revisions.resize(state_id + 1, HeuristicRecord::default());
        }
        revisions[state_id] = HeuristicRecord { revision, h };
    }

    fn recorded_heuristic_revision(&self, state_id: usize) -> HeuristicRecord {
        self.heuristic_revisions
            .as_ref()
            .expect("revision tracking is enabled for mpd")
            .get(state_id)
            .copied()
            .expect("every queued state has a recorded heuristic revision")
    }

    /// Closed and superseded paths, MPD and lazy evaluation all consume a pop
    /// here. Queue insertion and dead-end bookkeeping are handled by the caller.
    fn on_pop(&mut self, entry: OpenEntry, state: &ConcreteState) -> Result<PopAction> {
        let state_id = entry.state_id();
        let info = self
            .space
            .node(state_id)
            .expect("open-list entry must have a search node");
        if info.is_closed || info.g_value < entry.g_value {
            return Ok(PopAction::Skip);
        }
        let is_goal = self.space.is_goal(state_id);
        if self.heuristic_revisions.is_some() {
            let previous = self.recorded_heuristic_revision(state_id);
            if self.heuristic.reevaluate_on_every_pop()
                || previous.revision < self.heuristic.revision()
            {
                let evaluation = self
                    .evaluate_state(&*self.heuristic, state, entry.g_value, is_goal)
                    .with_context(|| {
                        format!("pop-time heuristic re-evaluation failed for state {state_id}")
                    })?;
                self.stats.evaluations += 1;
                self.record_heuristic_revision(
                    state_id,
                    evaluation.heuristic_revision,
                    evaluation.h_value,
                );
                assert!(
                    evaluation.h_value + float_tolerance::SEARCH_EPSILON >= previous.h,
                    "dynamic heuristic decreased for state {state_id} at revision {}: {} -> {}",
                    evaluation.heuristic_revision,
                    previous.h,
                    evaluation.h_value
                );
                if evaluation.is_dead_end {
                    return Ok(PopAction::Prune);
                }
                if evaluation.h_value > entry.h_value.into_inner() {
                    return Ok(PopAction::Requeue(
                        entry.with_estimate(evaluation.h_value, evaluation.f_value),
                    ));
                }
            }
        }
        if let Some(lazy) = &self.lazy
            && !entry.is_lazy_evaluated()
        {
            let evaluation = self
                .evaluate_state(&**lazy, state, entry.g_value, is_goal)
                .with_context(|| {
                    format!("slow heuristic evaluation failed for state {state_id}")
                })?;
            // Lazy evaluations retain the historical counter until the separate
            // evaluation-accounting change.
            if evaluation.is_dead_end {
                return Ok(PopAction::Prune);
            }
            let combined = entry.h_value.into_inner().max(evaluation.h_value);
            let mut entry = entry.with_estimate(combined, P::key(entry.g_value, combined));
            entry.mark_lazy_evaluated();
            return Ok(PopAction::Requeue(entry));
        }
        Ok(PopAction::Expand)
    }

    fn push_entry(&mut self, mut entry: OpenEntry) {
        entry.insertion_order = self.next_insertion_order;
        self.next_insertion_order = self
            .next_insertion_order
            .checked_add(1)
            .expect("open-list insertion count exceeds u32");
        self.open_list.push(entry);
    }

    fn populate_applicable_operators(&mut self, state: &ConcreteState) {
        self.state_registry
            .view(state)
            .fill_propositional(&mut self.scratch.state_values);
        self.scratch.applicable_operators.clear();
        self.successor_generator.get_applicable_operators(
            &self.scratch.state_values,
            &mut self.scratch.applicable_operators,
        );
    }

    pub fn initialize(&mut self) -> Result<()> {
        debug_assert!(self.start_time.is_none());
        let start_time = Instant::now();
        self.start_time = Some(start_time);

        // Initialize search with initial state (created in constructor)
        let initial_state = self.initial_state.clone();

        // Add initial state to open list
        let initial_is_goal = self.is_goal_state(&initial_state);
        let initial_evaluation = self
            .evaluate_state(&*self.heuristic, &initial_state, 0.0, initial_is_goal)
            .context("initial state evaluation failed")?;
        self.stats.nodes_evaluated += 1;
        self.stats.evaluations += 1;
        self.record_heuristic_revision(
            initial_state.get_id(),
            initial_evaluation.heuristic_revision,
            initial_evaluation.h_value,
        );
        if initial_evaluation.is_dead_end {
            self.stats.dead_ends += 1;
        } else {
            let progress = self.maybe_report_heuristic_progress(&initial_evaluation, &start_time);
            if progress.improved {
                self.maybe_print_f_value(initial_evaluation.f_value, &start_time);
            }
        }
        info!(
            "Initial heuristic value for {}: {}",
            self.heuristic_name,
            format_progress_value(initial_evaluation.h_value)
        );

        if !initial_evaluation.is_dead_end {
            // The initial state has no parent operator, so
            // "preferred-via-parent" is vacuously false. Still snapshot the
            // initial state's own preferred IDs so its successors can be
            // classified.
            let initial_id = initial_state.get_id();
            self.heuristic
                .copy_preferred_operator_ids(&mut self.scratch.preferred_ids);
            self.space
                .store_preferred(initial_id, &self.scratch.preferred_ids);
            self.push_entry(OpenEntry::new(
                initial_id,
                0.0,
                initial_evaluation.h_value,
                initial_evaluation.f_value,
                false,
            ));
        }

        // Initialize search node info for initial state.
        let initial_info = SearchNodeInfo {
            parent_state: None,
            parent_operator_id: None,
            g_value: 0.0,
            is_dead_end: initial_evaluation.is_dead_end,
            is_closed: false,
            is_goal: initial_is_goal,
        };
        self.space.set_node(initial_state.get_id(), initial_info);
        Ok(())
    }

    /// Perform one step of A* search.
    pub fn step(&mut self) -> Result<SearchStatus> {
        let start_time = *self
            .start_time
            .as_ref()
            .expect("step called before initialize");
        if let Some(status) = self.resource_limit_status(&start_time) {
            return Ok(status);
        }

        let (entry, state) = match self.pop_next_to_expand()? {
            PoppedNode::Expand(entry, state) => (entry, state),
            PoppedNode::Skipped => return Ok(SearchStatus::InProgress),
            PoppedNode::Exhausted => return Ok(SearchStatus::Failed),
        };
        let state_id = entry.state_id();

        self.maybe_print_f_layer(entry, &start_time);
        self.close_expanded_node(state_id);

        if self.space.is_goal(state_id) {
            return Ok(SearchStatus::Solved(state_id));
        }

        self.expand_successors(&state, state_id, &start_time)?;
        Ok(SearchStatus::InProgress)
    }

    /// Take the cheapest open entry that is actually worth expanding.
    ///
    /// An entry is skipped when its state is already closed, when a cheaper
    /// path to it has been found since it was queued, or when it is a stale
    /// re-evaluation. With a lazy heuristic the first pop of an entry only
    /// triggers the slow heuristic and reinserts it; the second pop expands.
    fn pop_next_to_expand(&mut self) -> Result<PoppedNode> {
        let Some(entry) = self.open_list.pop() else {
            return Ok(PoppedNode::Exhausted);
        };

        let state_id = entry.state_id();
        let state = self
            .state_registry
            .lookup_state(state_id)
            .map_err(|error| anyhow!("open list references missing state {state_id}: {error:?}"))?;

        match self.on_pop(entry, &state)? {
            PopAction::Expand => Ok(PoppedNode::Expand(entry, state)),
            PopAction::Requeue(entry) => {
                self.push_entry(entry);
                Ok(PoppedNode::Skipped)
            }
            PopAction::Prune => {
                self.stats.dead_ends += 1;
                self.space.mark_dead_end(state_id);
                Ok(PoppedNode::Skipped)
            }
            PopAction::Skip => Ok(PoppedNode::Skipped),
        }
    }

    /// Close the node the search is about to expand and count the expansion.
    #[inline]
    fn close_expanded_node(&mut self, state_id: StateID) {
        assert!(
            self.space.contains_node(state_id),
            "expanded open-list entry must have a search node"
        );
        self.space.mark_closed(state_id);
        self.stats.nodes_expanded += 1;
    }

    /// Generate, evaluate and queue every successor of an expanded state.
    fn expand_successors(
        &mut self,
        state: &ConcreteState,
        state_id: StateID,
        start_time: &Instant,
    ) -> Result<()> {
        let parent = ParentExpansion {
            state_id,
            g_value: self
                .space
                .node(state_id)
                .expect("expanded node is closed before its successors are generated")
                .g_value,
            // Reading via `take` is intentional: once we've started expanding
            // `state` we won't need the preferred operators again unless the
            // node is reopened, in which case `evaluate_state` resnapshots.
            // It also reclaims the boxed slice's memory eagerly.
            preferred_operator_range: self.space.take_preferred(state_id),
        };

        self.populate_applicable_operators(state);
        let mut applicable_operators = std::mem::take(&mut self.scratch.applicable_operators);

        // Fill the parent's numeric/cost/metric values once; reuse across all
        // successors below.
        let mut expansion_context = std::mem::take(&mut self.scratch.expansion_context);
        if let Err(error) = self
            .state_registry
            .build_expansion_context(state, &mut expansion_context)
        {
            self.scratch.expansion_context = expansion_context;
            self.scratch.applicable_operators = applicable_operators;
            return Err(anyhow!(
                "failed to build expansion context for state {state_id}: {error:?}"
            ));
        }

        for &op_id in applicable_operators.iter() {
            let operator_id = op_id.index();
            let operator = self
                .operators
                .get(operator_id)
                .expect("successor generator returned an invalid operator id");
            let (succ_state, metric_op_cost) = self
                .state_registry
                .apply_operator_in_context(
                    state,
                    operator,
                    &expansion_context,
                    &mut self.scratch.successor_numeric,
                    &mut self.scratch.successor_cost,
                )
                .map_err(|error| {
                    anyhow!(
                        "failed to apply operator {operator_id} ({}) to state {state_id}: {error:?}",
                        operator.name()
                    )
                })?;
            self.process_successor(
                &parent,
                AppliedOperator {
                    operator,
                    operator_id,
                    op_id,
                    metric_op_cost,
                },
                &succ_state,
                start_time,
            )?;
        }

        applicable_operators.clear();
        self.scratch.applicable_operators = applicable_operators;
        self.scratch.expansion_context = expansion_context;
        Ok(())
    }

    /// Account for one generated successor: skip it if the search space
    /// already holds it on an at-least-as-good path, otherwise evaluate it,
    /// record the improved path, and queue it unless it is a dead end.
    #[inline]
    fn process_successor(
        &mut self,
        parent: &ParentExpansion,
        applied: AppliedOperator<'_>,
        succ_state: &ConcreteState,
        start_time: &Instant,
    ) -> Result<()> {
        let AppliedOperator {
            operator,
            operator_id,
            op_id,
            metric_op_cost,
        } = applied;
        let succ_state_id = succ_state.get_id();
        let new_g_value = parent.g_value
            + self
                .operator_cost(OperatorIndex::from_usize(operator_id), metric_op_cost)
                .value();

        // Count every successfully constructed successor state.
        self.stats.nodes_generated += 1;
        let (improved_duplicate, was_closed) = match self.space.node(succ_state_id) {
            // A dead end stays a dead end, and an at-least-as-good path is
            // already recorded — either way there is nothing to do.
            Some(info) if info.is_dead_end || info.g_value <= new_g_value => return Ok(()),
            Some(info) => (true, info.is_closed),
            None => (false, false),
        };
        if was_closed {
            self.stats.nodes_reopened += 1;
        }

        // Is this successor reached via one of the parent's preferred
        // (helpful) operators? Per-successor it's a small linear scan, but
        // helpful-action lists from FF are typically tiny (single digits), so
        // this is cheap compared to evaluating the successor.
        let is_preferred = parent
            .preferred_operator_range
            .is_some_and(|range| self.space.preferred_contains(range, op_id));

        let is_goal = self.is_goal_state(succ_state);
        let evaluation = self
            .evaluate_state(&*self.heuristic, succ_state, new_g_value, is_goal)
            .with_context(|| {
                format!(
                    "heuristic evaluation failed for successor state {succ_state_id} generated by operator {operator_id} ({})",
                    operator.name()
                )
            })?;
        self.stats.evaluations += 1;
        if !improved_duplicate {
            self.stats.nodes_evaluated += 1;
        }
        self.record_heuristic_revision(
            succ_state_id,
            evaluation.heuristic_revision,
            evaluation.h_value,
        );

        // Snapshot the heuristic's preferred-operator IDs for the successor
        // *now*, before any other state's evaluation overwrites the
        // heuristic's internal scratch. Stored on the search space and read
        // back when this successor is later expanded.
        self.heuristic
            .copy_preferred_operator_ids(&mut self.scratch.preferred_ids);
        self.space
            .store_preferred(succ_state_id, &self.scratch.preferred_ids);

        // Record/update best `g`-value, parent pointers, and dead-end status.
        self.space.set_node(
            succ_state_id,
            SearchNodeInfo {
                parent_state: Some(parent.state_id),
                parent_operator_id: Some(OperatorIndex::from_usize(operator_id)),
                g_value: new_g_value,
                is_dead_end: evaluation.is_dead_end,
                is_closed: false,
                is_goal,
            },
        );

        if evaluation.is_dead_end {
            self.stats.dead_ends += 1;
            return Ok(());
        }

        let _ = self.maybe_report_heuristic_progress(&evaluation, start_time);
        self.push_entry(OpenEntry::new(
            succ_state_id,
            new_g_value,
            evaluation.h_value,
            evaluation.f_value,
            is_preferred,
        ));
        Ok(())
    }

    /// Cost charged for applying `operator`: the task metric when the search
    /// optimises it, otherwise the configured (unit or per-operator) cost.
    #[inline]
    fn operator_cost(
        &self,
        operator_id: OperatorIndex,
        metric_op_cost: NumericValue,
    ) -> NumericValue {
        if self.config.use_metric {
            metric_op_cost
        } else {
            self.config.operator_costs[operator_id.index()]
        }
    }

    pub fn finish(&mut self, status: SearchStatus) -> SearchResult {
        let start_time = *self
            .start_time
            .as_ref()
            .expect("finish called before initialize");
        let (plan, solution_cost) = match status {
            SearchStatus::Solved(goal_state_id) => {
                debug_assert!(
                    !self.initial_state_is_proven_optimal
                        || self.stats.counters_at_last_jump.expanded == 0,
                    "A* entered a higher f-layer after its heuristic proved h(init) = h*"
                );
                (
                    Some(self.space.extract_plan(goal_state_id, self.task)),
                    Some(
                        self.space
                            .node(goal_state_id)
                            .expect("a solved state has a search node")
                            .g_value,
                    ),
                )
            }
            SearchStatus::InProgress => unreachable!("cannot finish an in-progress search"),
            _ => (None, None),
        };
        SearchResult {
            status,
            plan,
            solution_cost,
            nodes_expanded: self.stats.nodes_expanded,
            nodes_reopened: self.stats.nodes_reopened,
            nodes_evaluated: self.stats.nodes_evaluated,
            evaluations: self.stats.evaluations,
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
}

impl<P: Priority, Q: OpenList> SearchEngine for BestFirstSearch<'_, P, Q> {
    fn initialize(&mut self) -> Result<()> {
        BestFirstSearch::initialize(self)
    }

    fn step(&mut self) -> Result<SearchStatus> {
        BestFirstSearch::step(self)
    }

    fn finish(&mut self, status: SearchStatus) -> SearchResult {
        BestFirstSearch::finish(self, status)
    }
}
