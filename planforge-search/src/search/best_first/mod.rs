//! Best-first search: A*, greedy best-first search and uniform-cost search,
//! one loop instantiated per [`NodeOrdering`] and [`OpenList`], built on the
//! [`SearchSpace`] and [`SearchProgress`] blocks.

#[cfg(test)]
mod tests;

mod open_list;
mod ordering;
mod preferred_operators;

pub use open_list::{OpenEntry, OpenList, PreferredFirstOpenList, TieBreakingOpenList};
pub use ordering::{AStar, Greedy, NodeOrdering, UniformCost};

use std::marker::PhantomData;
use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use planforge_sas::numeric_task::{OperatorIndex, Task};
use planforge_sas::state_registry::{ConcreteState, StateID};
use planforge_sas::utils::float_tolerance;

use self::preferred_operators::PreferredOperators;
use super::progress::SearchProgress;
use super::search_space::{Reached, SearchSpace, Successor};
use super::{SearchLimits, SearchOutcome, SearchResult};
use crate::evaluation::heuristic::minimum_action_cost;
use crate::evaluation::{EvaluationState, Heuristic};
use crate::successor_generator::SuccessorGeneratorConfig;

pub type AStarSearch<'a> = BestFirstSearch<'a, AStar, TieBreakingOpenList>;
pub type GreedyBestFirstSearch<'a> = BestFirstSearch<'a, Greedy, PreferredFirstOpenList>;

/// What a best-first search runs and within which limits.
pub struct SearchParams<'a> {
    pub heuristic: Box<dyn Heuristic + 'a>,
    /// A second heuristic, evaluated when a node is first popped; the node is
    /// then requeued under the larger of both estimates.
    pub lazy: Option<Box<dyn Heuristic + 'a>>,
    /// Multi-path dependence: re-evaluate a popped node whose estimate
    /// predates the heuristic's current revision.
    pub mpd: bool,
    pub successor_generator: SuccessorGeneratorConfig,
    pub limits: SearchLimits,
}

impl<'a> SearchParams<'a> {
    /// `heuristic` alone, without limits.
    pub fn new(heuristic: Box<dyn Heuristic + 'a>) -> Self {
        Self {
            heuristic,
            lazy: None,
            mpd: false,
            successor_generator: SuccessorGeneratorConfig::default(),
            limits: SearchLimits::default(),
        }
    }
}

/// Best-first search, expanding open nodes in the order `O` gives their
/// priorities and `Q` queues them.
pub struct BestFirstSearch<'a, O: NodeOrdering, Q: OpenList> {
    space: SearchSpace,
    progress: SearchProgress,
    heuristic: Box<dyn Heuristic + 'a>,
    lazy: Option<Box<dyn Heuristic + 'a>>,
    /// The cheapest operator cost, a lower bound on the estimate of every
    /// non-goal state.
    min_action_cost: f64,
    /// The latest evaluation of each state; tracked only with `mpd`.
    evaluation_records: Option<Vec<Option<EvaluationRecord>>>,
    preferred: PreferredOperators,
    /// The preferred operators of the latest evaluation.
    preferred_scratch: Vec<OperatorIndex>,
    open_list: Q,
    next_insertion_order: u32,
    /// The successors of the current expansion and their evaluations.
    successors: Vec<Successor>,
    evaluations: Vec<Evaluation>,
    ordering: PhantomData<O>,
}

/// One heuristic evaluation of a node.
#[derive(Debug, Clone, Copy)]
struct Evaluation {
    h: f64,
    priority: f64,
    /// The heuristic's revision after the evaluation.
    revision: u64,
}

impl Evaluation {
    fn is_dead_end(&self) -> bool {
        self.h == f64::INFINITY
    }
}

#[derive(Debug, Clone, Copy)]
struct EvaluationRecord {
    revision: u64,
    h: f64,
}

/// What taking one entry off the open list yields.
enum Popped {
    /// A node to expand.
    Expand(ConcreteState),
    /// The entry was deferred or pruned.
    Skipped,
    /// The open list is empty.
    Exhausted,
}

impl<'a, O: NodeOrdering, Q: OpenList> BestFirstSearch<'a, O, Q> {
    /// A search of `task`; its clock starts here.
    pub fn new(task: Arc<Task>, params: SearchParams<'a>) -> Self {
        let SearchParams {
            heuristic,
            lazy,
            mpd,
            successor_generator,
            limits,
        } = params;
        let space = SearchSpace::new(task.clone(), successor_generator.build(&task));
        Self {
            min_action_cost: minimum_action_cost(space.task().effective_operator_costs()),
            space,
            progress: SearchProgress::start(limits, heuristic.name()),
            heuristic,
            lazy,
            evaluation_records: mpd.then(Vec::new),
            preferred: PreferredOperators::default(),
            preferred_scratch: Vec::new(),
            open_list: Q::default(),
            next_insertion_order: 0,
            successors: Vec::new(),
            evaluations: Vec::new(),
            ordering: PhantomData,
        }
    }

    /// Search until a goal is expanded, the open list runs empty or a limit
    /// is reached.
    pub fn run(mut self) -> Result<SearchResult> {
        self.initialize()?;
        loop {
            if let Some(outcome) = self.step()? {
                return Ok(self.progress.finish(outcome, &self.space));
            }
        }
    }

    fn initialize(&mut self) -> Result<()> {
        let state = self.space.initial_state();
        let state_id = state.get_id();
        assert_eq!(self.space.reach(&state, None, 0.0), Reached::New);
        let evaluation = self
            .evaluate(&state, 0.0)
            .context("initial state evaluation failed")?;
        self.progress.statistics.nodes.evaluated += 1;
        self.progress.report_initial_h(evaluation.h);
        if evaluation.is_dead_end() {
            self.space.mark_dead_end(state_id);
            self.progress.statistics.dead_ends += 1;
            return Ok(());
        }
        if self.progress.report_best_h(evaluation.h, 0.0) && O::MONOTONE {
            self.progress.report_layer(
                O::LABEL,
                evaluation.priority,
                &self.space,
                self.open_list.len(),
            );
        }
        self.push(OpenEntry::new(
            state_id,
            0.0,
            evaluation.h,
            evaluation.priority,
            false,
        ));
        Ok(())
    }

    /// Expand at most one node; the outcome once the search ends.
    fn step(&mut self) -> Result<Option<SearchOutcome>> {
        if let Some(outcome) = self.progress.limit_reached() {
            return Ok(Some(outcome));
        }
        let state = match self.pop()? {
            Popped::Expand(state) => state,
            Popped::Skipped => return Ok(None),
            Popped::Exhausted => return Ok(Some(SearchOutcome::Unsolvable)),
        };
        let state_id = state.get_id();
        self.space.close(state_id);
        self.progress.statistics.nodes.expanded += 1;
        if self.space.is_goal(state_id) {
            debug_assert!(
                !self.heuristic.proves_initial_state_optimal()
                    || self.progress.statistics.nodes_until_last_jump.expanded == 0,
                "A* entered a higher f-layer after its heuristic proved h(init) = h*"
            );
            return Ok(Some(self.space.solution(state_id)));
        }
        self.expand(&state)?;
        Ok(None)
    }

    /// Take the next open entry and decide what it is worth.
    ///
    /// An entry is skipped when its state is closed or reached more cheaply
    /// since it was queued. With `mpd`, a state whose estimate predates the
    /// heuristic's revision is re-evaluated and requeued if its estimate grew.
    /// With a lazy heuristic, the first pop of an entry evaluates it and
    /// requeues it; the second pop expands it.
    fn pop(&mut self) -> Result<Popped> {
        // Stale entries -- their state was closed or reached more cheaply
        // since -- are dropped here, so the caller's limit check runs once
        // per pop that does work rather than once per entry.
        let entry = loop {
            let Some(entry) = self.open_list.pop() else {
                return Ok(Popped::Exhausted);
            };
            let node = self
                .space
                .node(entry.state_id())
                .expect("an open-list entry has a search node");
            if !node.is_closed && node.g >= entry.g_value {
                break entry;
            }
        };
        let state_id = entry.state_id();
        let state = self.space.state(state_id);

        if let Some(records) = &self.evaluation_records {
            let previous = records[state_id].expect("every queued state has an evaluation record");
            if self.heuristic.reevaluate_on_every_pop()
                || previous.revision < self.heuristic.revision()
            {
                let evaluation = self.evaluate(&state, entry.g_value).with_context(|| {
                    format!("pop-time heuristic re-evaluation failed for state {state_id}")
                })?;
                assert!(
                    evaluation.h + float_tolerance::SEARCH_EPSILON >= previous.h,
                    "dynamic heuristic decreased for state {state_id} at revision {}: {} -> {}",
                    evaluation.revision,
                    previous.h,
                    evaluation.h
                );
                if evaluation.is_dead_end() {
                    return Ok(self.prune(state_id));
                }
                if evaluation.h > entry.h_value.into_inner() {
                    return Ok(self.requeue(entry.with_estimate(evaluation.h, evaluation.priority)));
                }
            }
        }

        if let Some(lazy) = self.lazy.as_deref_mut()
            && !entry.is_lazy_evaluated()
        {
            let evaluation = evaluate::<O>(
                lazy,
                &self.space.evaluation_state(&state),
                entry.g_value,
                self.min_action_cost,
                None,
            )
            .with_context(|| format!("slow heuristic evaluation failed for state {state_id}"))?;
            self.progress.statistics.evaluations += 1;
            if evaluation.is_dead_end() {
                return Ok(self.prune(state_id));
            }
            let h = entry.h_value.into_inner().max(evaluation.h);
            let mut entry = entry.with_estimate(h, O::priority(entry.g_value, h));
            entry.mark_lazy_evaluated();
            return Ok(self.requeue(entry));
        }

        if O::MONOTONE {
            self.progress.report_layer(
                O::LABEL,
                entry.priority.into_inner(),
                &self.space,
                self.open_list.len(),
            );
        }
        Ok(Popped::Expand(state))
    }

    /// Expand `state` in three phases: generate its successors and drop the
    /// duplicates, evaluate the others, and queue them in generation order.
    ///
    /// Duplicate detection records each path as it is generated, so a state
    /// generated twice in one expansion is evaluated and queued twice if the
    /// second path is cheaper, and dropped otherwise: the same as processing
    /// each successor completely before generating the next.
    fn expand(&mut self, state: &ConcreteState) -> Result<()> {
        let parent = state.get_id();
        let parent_g = self.space.g(parent);
        let preferred = self.preferred.take(parent);
        let mut successors = std::mem::take(&mut self.successors);
        let mut evaluations = std::mem::take(&mut self.evaluations);

        let statistics = &mut self.progress.statistics.nodes;
        self.space
            .reach_successors(state, &mut successors, |reached| {
                statistics.generated += 1;
                match reached {
                    Reached::New => statistics.evaluated += 1,
                    Reached::Cheaper { reopened } => statistics.reopened += usize::from(reopened),
                    Reached::Duplicate => {}
                }
            })?;

        evaluations.clear();
        for successor in &successors {
            let evaluation = self
                .evaluate(&successor.state, parent_g + successor.cost)
                .with_context(|| {
                    format!(
                        "evaluating the successor of state {parent} under operator {} failed",
                        successor.operator
                    )
                })?;
            evaluations.push(evaluation);
        }

        for (successor, evaluation) in successors.iter().zip(&evaluations) {
            let state_id = successor.state.get_id();
            let g = parent_g + successor.cost;
            if evaluation.is_dead_end() {
                // A state reached twice is counted once.
                if !self
                    .space
                    .node(state_id)
                    .is_some_and(|node| node.is_dead_end)
                {
                    self.space.mark_dead_end(state_id);
                    self.progress.statistics.dead_ends += 1;
                }
                continue;
            }
            self.progress.report_best_h(evaluation.h, g);
            let is_preferred = Q::USES_PREFERRED_OPERATORS
                && self.preferred.contains(preferred, successor.operator);
            self.push(OpenEntry::new(
                state_id,
                g,
                evaluation.h,
                evaluation.priority,
                is_preferred,
            ));
        }
        self.successors = successors;
        self.evaluations = evaluations;
        Ok(())
    }

    /// Evaluate `state`, reached on a path of cost `g`, with the primary
    /// heuristic, recording its preferred operators when the open list uses
    /// them and the evaluation when `mpd` tracks it.
    #[inline]
    fn evaluate(&mut self, state: &ConcreteState, g: f64) -> Result<Evaluation> {
        let state_id = state.get_id();
        let preferred = Q::USES_PREFERRED_OPERATORS.then(|| {
            self.preferred_scratch.clear();
            &mut self.preferred_scratch
        });
        let evaluation = evaluate::<O>(
            &mut *self.heuristic,
            &self.space.evaluation_state(state),
            g,
            self.min_action_cost,
            preferred,
        )?;
        self.progress.statistics.evaluations += 1;
        if Q::USES_PREFERRED_OPERATORS {
            self.preferred.store(state_id, &self.preferred_scratch);
        }
        if let Some(records) = &mut self.evaluation_records {
            if records.len() <= state_id {
                records.resize(state_id + 1, None);
            }
            records[state_id] = Some(EvaluationRecord {
                revision: evaluation.revision,
                h: evaluation.h,
            });
        }
        Ok(evaluation)
    }

    fn requeue(&mut self, entry: OpenEntry) -> Popped {
        self.push(entry);
        Popped::Skipped
    }

    fn prune(&mut self, state_id: StateID) -> Popped {
        self.progress.statistics.dead_ends += 1;
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
}

/// Evaluate `state` with `heuristic`. A non-goal state's estimate is raised to
/// the cheapest operator cost: every plan from it applies an operator, so the
/// maximum with that bound stays admissible and consistent, and lifts a
/// heuristic that is 0 on non-goal states, as an abstraction is on the states
/// of an abstract goal.
#[inline]
fn evaluate<O: NodeOrdering>(
    heuristic: &mut dyn Heuristic,
    eval_state: &EvaluationState<'_, '_>,
    g: f64,
    min_action_cost: f64,
    preferred: Option<&mut Vec<OperatorIndex>>,
) -> Result<Evaluation> {
    let h = match preferred {
        Some(preferred) => heuristic.evaluate_with_preferred(eval_state, preferred),
        None => heuristic.evaluate(eval_state),
    }?;
    ensure!(
        !h.is_nan(),
        "heuristic {} estimated NaN for state {}",
        heuristic.name(),
        eval_state.state().get_id()
    );
    let h = if eval_state.is_goal() {
        h
    } else {
        h.max(min_action_cost)
    };
    Ok(Evaluation {
        h,
        priority: O::priority(g, h),
        revision: heuristic.revision(),
    })
}
