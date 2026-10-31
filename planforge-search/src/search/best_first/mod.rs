//! Best-first search: A*, greedy best-first search and uniform-cost search,
//! one loop instantiated per [`NodeOrdering`] and [`OpenList`].

#[cfg(test)]
mod tests;

mod open_list;
mod ordering;
mod search_space;

pub use open_list::{OpenEntry, OpenList, PreferredFirstOpenList, TieBreakingOpenList};
pub use ordering::{AStar, Greedy, NodeOrdering, UniformCost};

use std::marker::PhantomData;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, ensure};
use planforge_sas::numeric_task::{AbstractNumericTask, OperatorIndex, TaskRef};
use planforge_sas::state_registry::{ConcreteState, StateID, StateRegistry};
use planforge_sas::utils::float_tolerance;
use tracing::info;

use self::search_space::{PreferredRange, SearchNodeInfo, SearchSpace};
use super::{SearchOutcome, SearchResult, SearchStatistics};
use crate::evaluation::heuristic::minimum_action_cost;
use crate::evaluation::{EvaluationState, Heuristic};
use crate::resource_limits::resident_memory_bytes;
use crate::successor_generator::RegistryExpander;

pub type AStarSearch<'a> = BestFirstSearch<'a, AStar, TieBreakingOpenList>;
pub type GreedyBestFirstSearch<'a> = BestFirstSearch<'a, Greedy, PreferredFirstOpenList>;

/// The resident memory is measured once per this many expansions.
const MEMORY_CHECK_EXPANSION_INTERVAL: usize = 1024;

/// What a best-first search runs and within which limits.
pub struct SearchParams<'a> {
    pub heuristic: Box<dyn Heuristic + 'a>,
    /// A second heuristic, evaluated when a node is first popped; the node is
    /// then requeued under the larger of both estimates.
    pub lazy: Option<Box<dyn Heuristic + 'a>>,
    /// Multi-path dependence: re-evaluate a popped node whose estimate
    /// predates the heuristic's current revision.
    pub mpd: bool,
    pub time_limit: Option<Duration>,
    pub max_memory_bytes: Option<u64>,
}

impl<'a> SearchParams<'a> {
    /// `heuristic` alone, without limits.
    pub fn new(heuristic: Box<dyn Heuristic + 'a>) -> Self {
        Self {
            heuristic,
            lazy: None,
            mpd: false,
            time_limit: None,
            max_memory_bytes: None,
        }
    }
}

/// Best-first search, expanding open nodes in the order `O` gives their
/// priorities and `Q` queues them.
pub struct BestFirstSearch<'a, O: NodeOrdering, Q: OpenList> {
    /// Kept apart from the rest of the search so that an expansion can borrow
    /// it while the search processes the successors.
    expander: RegistryExpander,
    core: SearchCore<'a, O, Q>,
}

/// Everything of a search but its successor generator.
struct SearchCore<'a, O: NodeOrdering, Q: OpenList> {
    task: TaskRef<'a>,
    registry: StateRegistry<'a>,
    initial_state: ConcreteState,
    /// Per-operator costs, for a task whose costs do not depend on the state.
    operator_costs: Vec<f64>,
    /// Whether a metric prices each transition instead.
    use_metric: bool,
    heuristic: Box<dyn Heuristic + 'a>,
    lazy: Option<Box<dyn Heuristic + 'a>>,
    /// The cheapest operator cost, a lower bound on the estimate of every
    /// non-goal state.
    min_action_cost: f64,
    /// The latest evaluation of each state; tracked only with `mpd`.
    evaluation_records: Option<Vec<Option<EvaluationRecord>>>,
    preferred_operators: Vec<OperatorIndex>,
    open_list: Q,
    next_insertion_order: u32,
    space: SearchSpace,
    statistics: SearchStatistics,
    progress: Progress,
    time_limit: Option<Duration>,
    max_memory_bytes: Option<u64>,
    next_memory_check: usize,
    ordering: PhantomData<O>,
}

/// One heuristic evaluation of a node.
#[derive(Debug, Clone, Copy)]
struct Evaluation {
    h_value: f64,
    priority: f64,
    /// The heuristic's revision after the evaluation.
    revision: u64,
}

impl Evaluation {
    fn is_dead_end(&self) -> bool {
        self.h_value == f64::INFINITY
    }
}

#[derive(Debug, Clone, Copy)]
struct EvaluationRecord {
    revision: u64,
    h_value: f64,
}

/// How a search ended, before its plan is extracted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Termination {
    Solved(StateID),
    Unsolvable,
    Timeout,
    OutOfMemory,
}

/// What one step of the search did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Step {
    Continue,
    Terminate(Termination),
}

/// What taking one entry off the open list yields.
enum Popped {
    /// A node to expand.
    Expand(StateID, ConcreteState),
    /// The entry was closed, superseded, deferred or pruned.
    Skipped,
    /// The open list is empty.
    Exhausted,
}

/// The parent's values every successor of one expansion shares.
struct Parent {
    state_id: StateID,
    g_value: f64,
    preferred_operators: PreferredRange,
}

impl<'a, O: NodeOrdering, Q: OpenList> BestFirstSearch<'a, O, Q> {
    /// A search of `task`; its clock starts here.
    pub fn new(task: TaskRef<'a>, params: SearchParams<'a>) -> Self {
        let SearchParams {
            heuristic,
            lazy,
            mpd,
            time_limit,
            max_memory_bytes,
        } = params;
        assert!(
            max_memory_bytes.is_none() || resident_memory_bytes().is_some(),
            "a search memory limit needs resident-memory measurement, which only Linux provides"
        );
        let mut registry = StateRegistry::for_task(task.clone());
        // Registering the initial state fixes the numeric constants the
        // metric's operator costs read.
        let initial_state = registry.get_initial_state();
        let operator_costs = task.effective_operator_costs().into_owned();
        info!(
            "State representation: bins={}",
            registry.global_state_packer().num_bins()
        );
        Self {
            expander: RegistryExpander::new(&*task),
            core: SearchCore {
                use_metric: task.metric().use_metric(),
                min_action_cost: minimum_action_cost(&operator_costs),
                operator_costs,
                task,
                registry,
                initial_state,
                progress: Progress::new(heuristic.name().to_owned()),
                heuristic,
                lazy,
                evaluation_records: mpd.then(Vec::new),
                preferred_operators: Vec::new(),
                open_list: Q::default(),
                next_insertion_order: 0,
                space: SearchSpace::default(),
                statistics: SearchStatistics::default(),
                time_limit,
                max_memory_bytes,
                next_memory_check: 0,
                ordering: PhantomData,
            },
        }
    }

    /// Search until a goal is expanded, the open list runs empty or a limit
    /// is reached.
    pub fn run(mut self) -> Result<SearchResult> {
        self.core.initialize()?;
        loop {
            if let Step::Terminate(termination) = self.step()? {
                return Ok(self.core.finish(termination));
            }
        }
    }

    /// Expand at most one node.
    fn step(&mut self) -> Result<Step> {
        let core = &mut self.core;
        if let Some(termination) = core.limit_reached() {
            return Ok(Step::Terminate(termination));
        }
        let (state_id, state) = match core.pop()? {
            Popped::Expand(state_id, state) => (state_id, state),
            Popped::Skipped => return Ok(Step::Continue),
            Popped::Exhausted => return Ok(Step::Terminate(Termination::Unsolvable)),
        };
        core.space.mark_closed(state_id);
        core.statistics.nodes.expanded += 1;
        if core.space.is_goal(state_id) {
            return Ok(Step::Terminate(Termination::Solved(state_id)));
        }

        let parent = Parent {
            state_id,
            g_value: core.space.g_value(state_id),
            preferred_operators: core.space.take_preferred(state_id),
        };
        let mut expansion = self
            .expander
            .expand(&core.registry, &state)
            .map_err(|error| anyhow!("failed to expand state {state_id}: {error:?}"))?;
        let task = core.task.clone();
        for &operator_id in expansion.applicable_operators() {
            let operator = &task.get_operators()[operator_id.index()];
            let (successor, metric_cost) = expansion
                .successor(&mut core.registry, operator)
                .map_err(|error| {
                    anyhow!(
                        "failed to apply operator {operator_id} ({}) to state {state_id}: {error:?}",
                        operator.name()
                    )
                })?;
            core.process_successor(&parent, operator_id, metric_cost, &successor)
                .with_context(|| {
                    format!(
                        "processing the successor of state {state_id} under operator {operator_id} ({}) failed",
                        operator.name()
                    )
                })?;
        }
        Ok(Step::Continue)
    }
}

impl<O: NodeOrdering, Q: OpenList> SearchCore<'_, O, Q> {
    fn initialize(&mut self) -> Result<()> {
        let state = self.initial_state.clone();
        let state_id = state.get_id();
        let is_goal = self.is_goal(&state);
        let evaluation = self
            .evaluate_primary(state_id, &state, 0.0, is_goal)
            .context("initial state evaluation failed")?;
        self.statistics.nodes.evaluated += 1;
        self.space.set_node(
            state_id,
            SearchNodeInfo {
                parent_state: None,
                parent_operator_id: None,
                g_value: 0.0,
                is_dead_end: evaluation.is_dead_end(),
                is_closed: false,
                is_goal,
            },
        );
        info!(
            "Initial heuristic value for {}: {}",
            self.progress.heuristic_name,
            format_progress_value(evaluation.h_value)
        );
        if evaluation.is_dead_end() {
            self.statistics.dead_ends += 1;
            return Ok(());
        }
        if self.progress.improves_best_h(evaluation.h_value) {
            self.report_best_h(evaluation.h_value, 0.0);
            self.report_layer(evaluation.priority);
        }
        self.push(OpenEntry::new(
            state_id,
            0.0,
            evaluation.h_value,
            evaluation.priority,
            false,
        ));
        Ok(())
    }

    fn limit_reached(&mut self) -> Option<Termination> {
        if self
            .time_limit
            .is_some_and(|limit| self.progress.start_time.elapsed() > limit)
        {
            return Some(Termination::Timeout);
        }
        let max_memory_bytes = self.max_memory_bytes?;
        if self.statistics.nodes.expanded < self.next_memory_check {
            return None;
        }
        self.next_memory_check = self.statistics.nodes.expanded + MEMORY_CHECK_EXPANSION_INTERVAL;
        let memory = resident_memory_bytes()
            .expect("a memory limit is only accepted where memory is measured");
        (memory >= max_memory_bytes).then_some(Termination::OutOfMemory)
    }

    /// Take the next open entry and decide what it is worth.
    ///
    /// An entry is skipped when its state is closed or reached more cheaply
    /// since it was queued. With `mpd`, a state whose estimate predates the
    /// heuristic's revision is re-evaluated and requeued if its estimate grew.
    /// With a lazy heuristic, the first pop of an entry evaluates it and
    /// requeues it; the second pop expands it.
    fn pop(&mut self) -> Result<Popped> {
        let Some(entry) = self.open_list.pop() else {
            return Ok(Popped::Exhausted);
        };
        let state_id = entry.state_id();
        let node = self
            .space
            .node(state_id)
            .expect("an open-list entry has a search node");
        if node.is_closed || node.g_value < entry.g_value {
            return Ok(Popped::Skipped);
        }
        let state = self
            .registry
            .lookup_state(state_id)
            .map_err(|error| anyhow!("open list references missing state {state_id}: {error:?}"))?;

        if let Some(records) = &self.evaluation_records {
            let previous = records[state_id].expect("every queued state has an evaluation record");
            if self.heuristic.reevaluate_on_every_pop()
                || previous.revision < self.heuristic.revision()
            {
                let evaluation = self
                    .evaluate_primary(state_id, &state, entry.g_value, node.is_goal)
                    .with_context(|| {
                        format!("pop-time heuristic re-evaluation failed for state {state_id}")
                    })?;
                assert!(
                    evaluation.h_value + float_tolerance::SEARCH_EPSILON >= previous.h_value,
                    "dynamic heuristic decreased for state {state_id} at revision {}: {} -> {}",
                    evaluation.revision,
                    previous.h_value,
                    evaluation.h_value
                );
                if evaluation.is_dead_end() {
                    return Ok(self.prune(state_id));
                }
                if evaluation.h_value > entry.h_value.into_inner() {
                    return Ok(
                        self.requeue(entry.with_estimate(evaluation.h_value, evaluation.priority))
                    );
                }
            }
        }

        if let Some(lazy) = self.lazy.as_deref_mut()
            && !entry.is_lazy_evaluated()
        {
            let evaluation = evaluate::<O>(
                lazy,
                &evaluation_state(&*self.task, &self.registry, &state, node.is_goal),
                entry.g_value,
                self.min_action_cost,
                None,
            )
            .with_context(|| format!("slow heuristic evaluation failed for state {state_id}"))?;
            self.statistics.evaluations += 1;
            if evaluation.is_dead_end() {
                return Ok(self.prune(state_id));
            }
            let h_value = entry.h_value.into_inner().max(evaluation.h_value);
            let mut entry = entry.with_estimate(h_value, O::priority(entry.g_value, h_value));
            entry.mark_lazy_evaluated();
            return Ok(self.requeue(entry));
        }

        self.report_layer(entry.priority.into_inner());
        Ok(Popped::Expand(state_id, state))
    }

    /// Account for one generated successor: skip it if it is a dead end or
    /// already reached at most as expensively, otherwise evaluate it, record
    /// the cheaper path and queue it.
    fn process_successor(
        &mut self,
        parent: &Parent,
        operator_id: OperatorIndex,
        metric_cost: f64,
        successor: &ConcreteState,
    ) -> Result<()> {
        let state_id = successor.get_id();
        let cost = if self.use_metric {
            metric_cost
        } else {
            self.operator_costs[operator_id.index()]
        };
        let g_value = parent.g_value + cost;
        self.statistics.nodes.generated += 1;
        let is_new = match self.space.node(state_id) {
            Some(node) if node.is_dead_end || node.g_value <= g_value => return Ok(()),
            Some(node) => {
                if node.is_closed {
                    self.statistics.nodes.reopened += 1;
                }
                false
            }
            None => true,
        };

        let is_goal = self.is_goal(successor);
        let evaluation = self.evaluate_primary(state_id, successor, g_value, is_goal)?;
        if is_new {
            self.statistics.nodes.evaluated += 1;
        }
        self.space.set_node(
            state_id,
            SearchNodeInfo {
                parent_state: Some(parent.state_id),
                parent_operator_id: Some(operator_id),
                g_value,
                is_dead_end: evaluation.is_dead_end(),
                is_closed: false,
                is_goal,
            },
        );
        if evaluation.is_dead_end() {
            self.statistics.dead_ends += 1;
            return Ok(());
        }
        if self.progress.improves_best_h(evaluation.h_value) {
            self.report_best_h(evaluation.h_value, g_value);
        }
        let is_preferred = self
            .space
            .preferred_contains(parent.preferred_operators, operator_id);
        self.push(OpenEntry::new(
            state_id,
            g_value,
            evaluation.h_value,
            evaluation.priority,
            is_preferred,
        ));
        Ok(())
    }

    /// Evaluate `state` with the primary heuristic, recording its preferred
    /// operators when the open list uses them and the evaluation when `mpd`
    /// tracks it.
    fn evaluate_primary(
        &mut self,
        state_id: StateID,
        state: &ConcreteState,
        g_value: f64,
        is_goal: bool,
    ) -> Result<Evaluation> {
        let preferred = Q::USES_PREFERRED_OPERATORS.then(|| {
            self.preferred_operators.clear();
            &mut self.preferred_operators
        });
        let evaluation = evaluate::<O>(
            &mut *self.heuristic,
            &evaluation_state(&*self.task, &self.registry, state, is_goal),
            g_value,
            self.min_action_cost,
            preferred,
        )?;
        self.statistics.evaluations += 1;
        if Q::USES_PREFERRED_OPERATORS {
            self.space
                .store_preferred(state_id, &self.preferred_operators);
        }
        if let Some(records) = &mut self.evaluation_records {
            if records.len() <= state_id {
                records.resize(state_id + 1, None);
            }
            records[state_id] = Some(EvaluationRecord {
                revision: evaluation.revision,
                h_value: evaluation.h_value,
            });
        }
        Ok(evaluation)
    }

    fn is_goal(&self, state: &ConcreteState) -> bool {
        self.registry.view(state).holds_all(self.task.goals())
    }

    fn requeue(&mut self, entry: OpenEntry) -> Popped {
        self.push(entry);
        Popped::Skipped
    }

    fn prune(&mut self, state_id: StateID) -> Popped {
        self.statistics.dead_ends += 1;
        self.space.mark_dead_end(state_id);
        Popped::Skipped
    }

    fn push(&mut self, mut entry: OpenEntry) {
        entry.insertion_order = self.next_insertion_order;
        self.next_insertion_order = self
            .next_insertion_order
            .checked_add(1)
            .expect("open-list insertion count exceeds u32");
        self.open_list.push(entry);
    }

    /// Report the first expansion in a higher priority layer, and count the
    /// nodes up to it (Fast Downward's "until last jump"). Only a monotone
    /// priority has layers.
    fn report_layer(&mut self, priority: f64) {
        if !O::MONOTONE {
            return;
        }
        let layer = priority as i64;
        if self.progress.last_layer.is_some_and(|last| layer <= last) {
            return;
        }
        self.progress.last_layer = Some(layer);
        self.statistics.nodes_until_last_jump = self.statistics.nodes;
        info!(
            "{} = {} [{} evaluated, {} expanded, {} states, {} open, t={:.6}s{}]",
            O::LABEL,
            layer,
            self.statistics.nodes.evaluated,
            self.statistics.nodes.expanded,
            self.registry.num_registered_states(),
            self.open_list.len(),
            self.progress.start_time.elapsed().as_secs_f64(),
            memory_report(),
        );
    }

    fn report_best_h(&self, h_value: f64, g_value: f64) {
        info!(
            "New best heuristic value for {}: {}",
            self.progress.heuristic_name,
            format_progress_value(h_value),
        );
        info!(
            "[g={}, {} evaluated, {} expanded, t={:.6}s{}]",
            format_progress_value(g_value),
            self.statistics.nodes.evaluated,
            self.statistics.nodes.expanded,
            self.progress.start_time.elapsed().as_secs_f64(),
            memory_report(),
        );
    }

    fn finish(self, termination: Termination) -> SearchResult {
        let outcome = match termination {
            Termination::Solved(goal) => {
                debug_assert!(
                    !self.heuristic.proves_initial_state_optimal()
                        || self.statistics.nodes_until_last_jump.expanded == 0,
                    "A* entered a higher f-layer after its heuristic proved h(init) = h*"
                );
                SearchOutcome::Solved {
                    plan: self.space.extract_plan(goal, &*self.task),
                    cost: self.space.g_value(goal),
                }
            }
            Termination::Unsolvable => SearchOutcome::Unsolvable,
            Termination::Timeout => SearchOutcome::Timeout,
            Termination::OutOfMemory => SearchOutcome::OutOfMemory,
        };
        SearchResult {
            outcome,
            statistics: SearchStatistics {
                registered_states: self.registry.num_registered_states(),
                ..self.statistics
            },
            search_time: self.progress.start_time.elapsed(),
        }
    }
}

/// Evaluate `state` with `heuristic`. A non-goal state's estimate is raised to
/// the cheapest operator cost: every plan from it applies an operator, so the
/// maximum with that bound stays admissible and consistent, and lifts a
/// heuristic that is 0 on non-goal states, as an abstraction is on the states
/// of an abstract goal.
fn evaluate<O: NodeOrdering>(
    heuristic: &mut dyn Heuristic,
    eval_state: &EvaluationState<'_, '_>,
    g_value: f64,
    min_action_cost: f64,
    preferred: Option<&mut Vec<OperatorIndex>>,
) -> Result<Evaluation> {
    let h_value = match preferred {
        Some(preferred) => heuristic.evaluate_with_preferred(eval_state, preferred),
        None => heuristic.evaluate(eval_state),
    }?;
    ensure!(
        !h_value.is_nan(),
        "heuristic {} estimated NaN for state {}",
        heuristic.name(),
        eval_state.state().get_id()
    );
    let h_value = if eval_state.is_goal() {
        h_value
    } else {
        h_value.max(min_action_cost)
    };
    Ok(Evaluation {
        h_value,
        priority: O::priority(g_value, h_value),
        revision: heuristic.revision(),
    })
}

fn evaluation_state<'state, 'task>(
    task: &'task dyn AbstractNumericTask,
    registry: &'state StateRegistry<'task>,
    state: &'state ConcreteState,
    is_goal: bool,
) -> EvaluationState<'state, 'task> {
    let mut eval_state = EvaluationState::new(state, task, registry);
    eval_state.set_is_goal(is_goal);
    eval_state
}

/// The progress reports of a search.
struct Progress {
    start_time: Instant,
    heuristic_name: String,
    /// The highest priority layer expanded so far.
    last_layer: Option<i64>,
    best_h: f64,
}

impl Progress {
    fn new(heuristic_name: String) -> Self {
        Self {
            start_time: Instant::now(),
            heuristic_name,
            last_layer: None,
            best_h: f64::INFINITY,
        }
    }

    /// Whether `h_value` is below every estimate reported so far; it is then
    /// the new best.
    fn improves_best_h(&mut self, h_value: f64) -> bool {
        let improves = h_value < self.best_h;
        if improves {
            self.best_h = h_value;
        }
        improves
    }
}

/// `value` as a progress report prints it: integral values without decimals.
fn format_progress_value(value: f64) -> String {
    if value == f64::INFINITY {
        "infinity".to_owned()
    } else if value.fract().abs() < 1e-9 {
        format!("{value:.0}")
    } else {
        format!("{value:.6}")
    }
}

/// The progress-line suffix for resident memory, empty where it is not
/// measured.
fn memory_report() -> String {
    resident_memory_bytes().map_or_else(String::new, |bytes| format!(", {} KB", bytes / 1024))
}
