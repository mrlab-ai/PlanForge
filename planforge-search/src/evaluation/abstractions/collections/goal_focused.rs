//! Goal-focused abstraction collections, shared by every abstraction backend.
//!
//! A goal-focused collection builds `variants_per_goal` members per task goal,
//! each an abstraction of the single-goal view of the task. The schedule decides
//! which goal and variant comes next, how each variant is constructed (refinement
//! direction, tie rank, random seed), from which concrete state it refines, and
//! when the collection stops. Backends only build one member at a time, so domain
//! and Cartesian abstractions follow the identical member sequence.
//!
//! Scheduling: goals are visited round-robin until every goal has `min(V, 2)`
//! variants; further variants go to the goal whose best member has the largest
//! `h(s0)`, ties to the smaller goal id. A goal-free task gets exactly one
//! abstraction of the full task.
//!
//! Progress states: every variant owns a concrete *progress state*, which its
//! members refine from. It starts as the task initial state and walks forward
//! through the validated concrete plans the variant's members produce. A
//! variant's progress is complete once its progress state reaches the complete
//! task goal, or once it turns out to be an abstract dead end; the variant then
//! refines from the task initial state again. After the scheduled members,
//! goals that no member built from the initial state covers get an
//! initial-state specialist, and goals a variant has re-opened are retried from
//! that variant's progress state.
//!
//! Changing the refinement root only changes which counterexamples refinement
//! sees. Every member still abstracts the full state space of its goal task, so
//! all members remain admissible components for collection heuristics.

use std::collections::VecDeque;
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use planforge_sas::explicit_state::{ExplicitState, ExplicitStateSpace};
use planforge_sas::numeric_task::{AbstractNumericTask, OperatorIndex};
use tracing::{debug, info};

use super::budget::{CollectionBudget, CollectionLimits, CollectionStopReason};
use super::config::SplitKinds;
use super::member::{CollectionMember, MemberConstruction};
use super::strategy::CollectionStrategy;
use crate::evaluation::abstractions::common::abstraction::CollectionMembership;
use crate::evaluation::abstractions::common::cegar::initial_splits::RefinementScope;
use crate::evaluation::abstractions::common::task::SingleGoalTask;
use crate::evaluation::seed::{DEFAULT_RANDOM_SEED, derive_variant_seed};

/// A goal-focused collection stops once fewer states than this are left: a
/// member with fewer is the trivial abstraction, which tells search nothing.
/// A standard collection stops only once no state is left.
const MIN_MEMBER_STATES: usize = 2;

/// The variants every goal gets before further variants go to the goals
/// whose best members are strongest: one of each split kind when the
/// variants alternate.
const ROUND_ROBIN_VARIANTS: usize = 2;

/// Why the collection is building a member.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MemberKind {
    /// Variant `variant_id` of its goal: one of the `G * V` members the
    /// configuration asks for.
    Scheduled { variant_id: usize },
    /// A goal that no member built from the task initial state covers yet:
    /// built like variant 0, from the task initial state.
    InitialStateSpecialist,
    /// A goal variant `variant_id` already attempted, retried from the
    /// variant's advanced progress state.
    Continuation { variant_id: usize },
}

impl MemberKind {
    /// The variant whose progress state the member refines from; none for an
    /// initial-state specialist.
    pub(crate) fn variant_id(self) -> Option<usize> {
        match self {
            Self::Scheduled { variant_id } | Self::Continuation { variant_id } => Some(variant_id),
            Self::InitialStateSpecialist => None,
        }
    }
}

/// One abstraction backend of a goal-focused collection.
pub(crate) trait GoalFocusedBackend {
    type Member: CollectionMember;

    /// Backend name used in log and error messages.
    const NAME: &'static str;

    /// Build the member `construction` describes for `task` (the member's
    /// single-goal view, or the full task), refining from `root`, or from the
    /// task initial state when `root` is `None`. A root that is an abstract
    /// dead end stops refinement and is reported in
    /// [`BuiltReport::root_is_dead_end`](super::member::BuiltReport::root_is_dead_end).
    fn build(
        &self,
        task: &dyn AbstractNumericTask,
        root: Option<&ExplicitState>,
        construction: &MemberConstruction,
    ) -> Result<Self::Member>;
}

/// Why `single_goal_tasks=false` and `progress_states=true` are
/// rejected together.
pub(crate) const FULL_TASK_PROGRESS_STATES_ERROR: &str = "`progress_states=true` cannot be combined with `single_goal_tasks=false`: a progress state only advances on a concrete plan for the full task, which completes the variant, so progress states would never move";

/// The schedule-level configuration of a goal-focused collection.
#[derive(Debug, Clone)]
pub(crate) struct GoalFocusedSchedule {
    pub(crate) strategy: CollectionStrategy,
    pub(crate) variants_per_goal: usize,
    /// The configured random seed, which variants derive their own from.
    pub(crate) random_seed: Option<u64>,
    pub(crate) limits: CollectionLimits,
    pub(crate) progress_states: bool,
    /// Goal-focused collections may alternate split kinds by variant; ranks
    /// and seeds do not depend on it.
    pub(crate) split_kinds: SplitKinds,
    /// The members' configured rank among tied best splits.
    pub(crate) split_selection_rank: Option<usize>,
    /// Goal-focused collections: build every member on its goal's
    /// single-goal task. When false, every member is built on the full task;
    /// goals are then only labels that schedule the `G * V` members
    /// round-robin and derive their seeds and ranks, and members carry no
    /// collection membership. Excludes progress states, which would be
    /// a no-op: a progress state only advances on a plan for the full task,
    /// and that completes the variant.
    pub(crate) single_goal_tasks: bool,
}

impl GoalFocusedSchedule {
    /// How `member` is constructed within `budget`.
    ///
    /// The variant's position in the split kind alternation, if any, is its
    /// id. A goal-focused collection ranks ties by `variant / 2` and derives
    /// a seed for every variant but the first, which is unseeded; a per-goal
    /// collection keeps the configured rank and derives a seed for every
    /// variant but the first, which keeps the configured seed. An
    /// initial-state specialist reuses variant 0's construction.
    fn construction(
        &self,
        member: &ScheduledMember,
        budget: &CollectionBudget,
    ) -> MemberConstruction {
        assert!(
            self.split_kinds != SplitKinds::Alternating || self.strategy.is_goal_focused(),
            "only goal-focused collections choose split kinds per variant"
        );
        let variant = member.kind.variant_id().unwrap_or(0);
        let derived_seed = || {
            derive_variant_seed(
                self.random_seed.unwrap_or(DEFAULT_RANDOM_SEED),
                member.goal_id,
                variant - 1,
            )
        };
        let (split_selection_rank, random_seed) = if self.strategy.is_goal_focused() {
            (Some(variant / 2), (variant > 0).then(derived_seed))
        } else if variant > 0 {
            (self.split_selection_rank, Some(derived_seed()))
        } else {
            (self.split_selection_rank, self.random_seed)
        };
        MemberConstruction {
            max_states: budget.member_max_states(),
            max_time: budget.member_max_time(),
            random_seed,
            split_kind: self.split_kinds.of(variant),
            split_selection_rank,
            scope: RefinementScope::default(),
        }
    }

    /// How the one member of a goal-free task is constructed within
    /// `budget`: like a first variant that keeps the configured seed and
    /// rank.
    fn goal_free_construction(&self, budget: &CollectionBudget) -> MemberConstruction {
        MemberConstruction {
            max_states: budget.member_max_states(),
            max_time: budget.member_max_time(),
            random_seed: self.random_seed,
            split_kind: self.split_kinds.of(0),
            split_selection_rank: self.split_selection_rank,
            scope: RefinementScope::default(),
        }
    }
}

/// Build the goal-focused collection of `task` with `backend`.
pub(crate) fn build_goal_focused_collection<B: GoalFocusedBackend>(
    task: &dyn AbstractNumericTask,
    schedule: &GoalFocusedSchedule,
    backend: &B,
) -> Result<Vec<B::Member>> {
    assert!(
        schedule.variants_per_goal > 0,
        "goal-focused collections need at least one variant per goal"
    );
    assert!(
        schedule.strategy != CollectionStrategy::Standard,
        "standard collections refine full-task members by random restarts"
    );
    assert!(
        schedule.single_goal_tasks
            || (schedule.strategy.is_goal_focused() && !schedule.progress_states),
        "full-task members require a goal-focused collection without progress states"
    );
    let start = Instant::now();
    let mut budget = CollectionBudget::new(schedule.limits, start)?;
    if task.get_num_goals() == 0 {
        return build_goal_free_member(task, schedule, backend, budget, start);
    }

    let mut state = CollectionState::new(task, schedule)?;
    let mut members = Vec::with_capacity(state.abstraction_count);
    let mut stop_reason = CollectionStopReason::RequestedMembersBuilt;
    while state.has_work_left() {
        if budget.remaining_states() < MIN_MEMBER_STATES && !members.is_empty() {
            stop_reason = CollectionStopReason::SizeLimit;
            break;
        }
        let member = match state.select_next_member() {
            NextMember::Build(member) => member,
            NextMember::Stop(reason) => {
                stop_reason = reason;
                break;
            }
        };
        // An empty collection is useless, so the first member is built even
        // with no time left -- with a zero budget, which stops it immediately.
        if budget
            .remaining_time()
            .is_some_and(|remaining| remaining.is_zero())
            && !members.is_empty()
        {
            stop_reason = CollectionStopReason::TimeLimit;
            break;
        }

        let construction = schedule.construction(&member, &budget);
        let BuiltMember {
            member: built,
            root,
        } = build_member(
            task,
            backend,
            &state,
            &member,
            &construction,
            members.len() + 1,
            !schedule.single_goal_tasks,
        )?;
        let report = built.report();
        budget.charge(report.num_states);
        state.record_built_member(&member, root, report.initial_h);
        state.advance_progress_state::<B>(&member, root, built.concrete_plan())?;

        members.push(built);
        if !crate::resource_limits::poll_and_release_if_exceeded() {
            stop_reason = CollectionStopReason::MemoryLimit;
            break;
        }
    }

    if stop_reason == CollectionStopReason::RequestedMembersBuilt
        && schedule.progress_states
        && state
            .goal_covered_from_initial_state
            .iter()
            .all(|covered| *covered)
    {
        stop_reason = CollectionStopReason::RequestedMembersAndInitialStateCoverage;
    }
    state.log_summary::<B>(
        members.len(),
        budget.used_states(),
        start.elapsed().as_secs_f64(),
        stop_reason,
    );
    Ok(members)
}

/// The one member of a goal-free task: an abstraction of the full task from
/// its initial state.
fn build_goal_free_member<B: GoalFocusedBackend>(
    task: &dyn AbstractNumericTask,
    schedule: &GoalFocusedSchedule,
    backend: &B,
    mut budget: CollectionBudget,
    start: Instant,
) -> Result<Vec<B::Member>> {
    let mut member = backend
        .build(task, None, &schedule.goal_free_construction(&budget))
        .with_context(|| format!("failed to build the {} collection abstraction", B::NAME))?;
    budget.charge(member.report().num_states);
    member.record(1, None);
    info!(
        "{} collection: abstractions=1, states={}, elapsed={:.3}s, stop_reason={}",
        B::NAME,
        budget.used_states(),
        start.elapsed().as_secs_f64(),
        CollectionStopReason::RequestedMembersBuilt
    );
    Ok(vec![member])
}

/// Where a member's refinement started.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MemberRoot {
    /// The task initial state, as for an initial-state specialist, without
    /// progress states, or as a progress state that has not advanced or is
    /// complete. The member covers its goal for every state.
    Initial,
    /// The variant's advanced progress state.
    AdvancedProgress,
    /// The variant's progress state, which was an abstract dead end, so the
    /// member was rebuilt from the task initial state.
    DeadProgress,
}

/// One finished member, and where its refinement started.
struct BuiltMember<M> {
    member: M,
    root: MemberRoot,
}

/// Restrict the task to the member's goal unless members abstract the
/// `full_task`, refine from its variant's progress state, and tag the result
/// as build `iteration` so collection heuristics can tell members apart.
fn build_member<B: GoalFocusedBackend>(
    task: &dyn AbstractNumericTask,
    backend: &B,
    state: &CollectionState<'_>,
    member: &ScheduledMember,
    construction: &MemberConstruction,
    iteration: usize,
    full_task: bool,
) -> Result<BuiltMember<B::Member>> {
    let goal_task =
        (!full_task).then(|| SingleGoalTask::new(task, *task.get_goal_fact(member.goal_id)));
    let member_task = goal_task
        .as_ref()
        .map_or(task, |goal_task| goal_task as &dyn AbstractNumericTask);
    debug!(
        "{} collection: building abstraction {iteration}, goal={}, kind={:?}, split_kind={}, tie_rank={:?}, max_states={}, seed={:?}",
        B::NAME,
        member.goal_id,
        member.kind,
        construction.split_kind,
        construction.split_selection_rank,
        construction.max_states,
        construction.random_seed
    );

    // A specialist and a variant whose progress is complete both refine from
    // the task initial state, which the backend takes as `None`.
    let progress = state.progress(member).filter(|progress| !progress.complete);
    let refinement_root = progress.map(|progress| &progress.state);
    let built = backend
        .build(member_task, refinement_root, construction)
        .with_context(|| {
            format!(
                "failed to build {} collection abstraction {iteration}",
                B::NAME
            )
        })?;
    // A dead progress state says nothing about the initial state, so the
    // member is rebuilt from there; a dead initial state is a result.
    let (mut built, root) = if refinement_root.is_some() && built.report().root_is_dead_end {
        info!(
            "{} collection: progress state is an abstract dead end for goal {}, kind {:?}; rebuilding this member from the task initial state",
            B::NAME,
            member.goal_id,
            member.kind
        );
        let rebuilt = backend
            .build(member_task, None, construction)
            .with_context(|| {
                format!(
                    "failed to rebuild {} collection abstraction {iteration} from the task initial state",
                    B::NAME
                )
            })?;
        (rebuilt, MemberRoot::DeadProgress)
    } else if progress.is_some_and(|progress| progress.advanced) {
        (built, MemberRoot::AdvancedProgress)
    } else {
        (built, MemberRoot::Initial)
    };

    let membership = (!full_task).then_some(CollectionMembership {
        goal_id: member.goal_id,
        variant_id: member.kind.variant_id(),
        from_progress_state: root == MemberRoot::AdvancedProgress,
    });
    built.record(iteration, membership);
    Ok(BuiltMember {
        member: built,
        root,
    })
}

/// One member the collection has decided to build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ScheduledMember {
    goal_id: usize,
    kind: MemberKind,
}

enum NextMember {
    Build(ScheduledMember),
    Stop(CollectionStopReason),
}

/// The progress state of one variant; see the module documentation.
struct VariantProgress {
    state: ExplicitState,
    satisfied_goals: usize,
    /// `state` is past the task initial state.
    advanced: bool,
    /// The variant refines from the task initial state again: its progress
    /// state reached the complete task goal or turned out to be a dead end.
    complete: bool,
}

/// Mutable state of one collection run on a task with goals.
///
/// With progress states enabled the collection keeps `variants_per_goal`
/// independent progress states; see the module documentation.
struct CollectionState<'task> {
    goal_count: usize,
    variants_per_goal: usize,
    /// The `G * V` scheduled members.
    abstraction_count: usize,
    progress_states: bool,
    single_goal_tasks: bool,
    /// The concrete states progress states move through.
    states: ExplicitStateSpace<'task>,
    /// The task initial state, which progress states start from and return
    /// to; `None` when progress states are off.
    initial_state: Option<ExplicitState>,
    /// One per variant; empty when progress states are off.
    progress: Vec<VariantProgress>,
    goal_covered_from_initial_state: Vec<bool>,
    variants_built_by_goal: Vec<usize>,
    best_initial_h_by_goal: Vec<f64>,
    /// `(goal_id, variant_id)` pairs to retry from a variant's advanced
    /// progress state.
    continuation_queue: VecDeque<(usize, usize)>,
    initial_abstractions_built: usize,
}

impl<'task> CollectionState<'task> {
    fn new(task: &'task dyn AbstractNumericTask, schedule: &GoalFocusedSchedule) -> Result<Self> {
        let goal_count = task.get_num_goals();
        assert!(goal_count > 0, "a goal-free task has a single member");
        let variants_per_goal = schedule.variants_per_goal;
        let abstraction_count = goal_count
            .checked_mul(variants_per_goal)
            .context("goal-focused collection abstraction count overflow")?;
        let states = ExplicitStateSpace::new(task);
        let initial_state = schedule
            .progress_states
            .then(|| states.initial_state())
            .transpose()?;
        let progress = initial_state
            .as_ref()
            .map_or_else(Vec::new, |initial_state| {
                let satisfied_goals = states.satisfied_goal_count(initial_state);
                (0..variants_per_goal)
                    .map(|_| VariantProgress {
                        state: initial_state.clone(),
                        satisfied_goals,
                        advanced: false,
                        complete: false,
                    })
                    .collect()
            });
        Ok(Self {
            goal_count,
            variants_per_goal,
            abstraction_count,
            progress_states: schedule.progress_states,
            single_goal_tasks: schedule.single_goal_tasks,
            states,
            initial_state,
            progress,
            goal_covered_from_initial_state: vec![false; goal_count],
            variants_built_by_goal: vec![0usize; goal_count],
            best_initial_h_by_goal: vec![0.0f64; goal_count],
            continuation_queue: VecDeque::new(),
            initial_abstractions_built: 0,
        })
    }

    /// The progress state `member` refines from, `None` for an
    /// initial-state specialist and when progress states are off.
    fn progress(&self, member: &ScheduledMember) -> Option<&VariantProgress> {
        member
            .kind
            .variant_id()
            .filter(|_| !self.progress.is_empty())
            .map(|variant_id| &self.progress[variant_id])
    }

    /// More scheduled members are due, a variant queued a retry, or some goal
    /// still lacks a member built from the task initial state.
    fn has_work_left(&self) -> bool {
        self.initial_abstractions_built < self.abstraction_count
            || !self.continuation_queue.is_empty()
            || (self.progress_states
                && self
                    .goal_covered_from_initial_state
                    .iter()
                    .any(|covered| !covered))
    }

    /// The scheduled members first, then with progress states the
    /// initial-state specialists, then the retries.
    fn select_next_member(&mut self) -> NextMember {
        if self.initial_abstractions_built < self.abstraction_count {
            let next_goal = if self.single_goal_tasks {
                select_next_collection_goal(
                    &self.variants_built_by_goal,
                    &self.best_initial_h_by_goal,
                    self.variants_per_goal,
                )
            } else {
                // Full-task members share one task: the goal is only a
                // label, so there is no goal to focus on.
                select_round_robin_goal(&self.variants_built_by_goal, self.variants_per_goal)
            };
            let Some(goal_id) = next_goal else {
                return NextMember::Stop(CollectionStopReason::RequestedMembersBuilt);
            };
            return NextMember::Build(ScheduledMember {
                goal_id,
                kind: MemberKind::Scheduled {
                    variant_id: self.variants_built_by_goal[goal_id],
                },
            });
        }
        if self.progress_states {
            if let Some(goal_id) = self
                .goal_covered_from_initial_state
                .iter()
                .position(|covered| !covered)
            {
                return NextMember::Build(ScheduledMember {
                    goal_id,
                    kind: MemberKind::InitialStateSpecialist,
                });
            }
            if let Some((goal_id, variant_id)) = self.pop_unsatisfied_continuation() {
                return NextMember::Build(ScheduledMember {
                    goal_id,
                    kind: MemberKind::Continuation { variant_id },
                });
            }
        }
        NextMember::Stop(CollectionStopReason::RequestedMembersAndInitialStateCoverage)
    }

    /// Drop queued retries whose goal the variant's progress state now
    /// satisfies anyway,
    /// and return the first one that still has work to do.
    fn pop_unsatisfied_continuation(&mut self) -> Option<(usize, usize)> {
        while let Some((goal_id, variant_id)) = self.continuation_queue.pop_front() {
            if !self
                .states
                .satisfies_goal(&self.progress[variant_id].state, goal_id)
            {
                return Some((goal_id, variant_id));
            }
        }
        None
    }

    fn record_built_member(&mut self, member: &ScheduledMember, root: MemberRoot, initial_h: f64) {
        if root != MemberRoot::AdvancedProgress {
            self.goal_covered_from_initial_state[member.goal_id] = true;
        }
        // Specialists and retries fill gaps; only scheduled members count
        // against the requested abstraction count.
        if let MemberKind::Scheduled { .. } = member.kind {
            self.variants_built_by_goal[member.goal_id] += 1;
            self.initial_abstractions_built += 1;
            self.best_initial_h_by_goal[member.goal_id] =
                self.best_initial_h_by_goal[member.goal_id].max(initial_h);
        }
    }

    /// Replay the member's concrete plan onto its variant's progress state, so
    /// the next member of that variant refines from further along.
    fn advance_progress_state<B: GoalFocusedBackend>(
        &mut self,
        member: &ScheduledMember,
        root: MemberRoot,
        concrete_plan: Option<&[OperatorIndex]>,
    ) -> Result<()> {
        if self
            .progress(member)
            .is_none_or(|progress| progress.complete)
        {
            return Ok(());
        }
        let variant_id = member
            .kind
            .variant_id()
            .expect("a member with a progress state has a variant");
        if root == MemberRoot::DeadProgress {
            self.complete_progress(variant_id);
            debug!(
                "{} collection: dead progress state completed variant {variant_id} after rebuilding goal {} from the initial state",
                B::NAME,
                member.goal_id
            );
            return Ok(());
        }
        let Some(operator_ids) = concrete_plan else {
            debug!(
                "{} collection: goal {} variant {variant_id} produced no concrete plan; progress state remains unchanged",
                B::NAME,
                member.goal_id
            );
            return Ok(());
        };

        let progress = &mut self.progress[variant_id];
        let previous_satisfied_goals = progress.satisfied_goals;
        let state = self.states.replay(&progress.state, operator_ids)?;
        ensure!(
            self.states.satisfies_goal(&state, member.goal_id),
            "{} collection: the concrete plan of goal {} variant {variant_id} does not reach its goal",
            B::NAME,
            member.goal_id
        );
        let satisfied_goals = self.states.satisfied_goal_count(&state);
        progress.state = state;
        progress.satisfied_goals = satisfied_goals;
        debug!(
            "{} collection: advanced the progress state of variant {variant_id} through {} concrete operators; satisfied_goals={satisfied_goals}/{}",
            B::NAME,
            operator_ids.len(),
            self.goal_count,
        );

        if satisfied_goals == self.goal_count {
            self.complete_progress(variant_id);
            debug!(
                "{} collection: full goal reached for variant {variant_id}; completed its progress",
                B::NAME
            );
            return Ok(());
        }
        self.progress[variant_id].advanced = true;
        if satisfied_goals > previous_satisfied_goals {
            self.queue_reopened_goals(variant_id);
        }
        Ok(())
    }

    /// Return a variant's progress state to the task initial state and stop
    /// advancing it, because it is a dead end or has reached the complete task
    /// goal. Retries queued for the variant go with it.
    fn complete_progress(&mut self, variant_id: usize) {
        let initial_state = self
            .initial_state
            .as_ref()
            .expect("progress states start at the initial state");
        self.progress[variant_id] = VariantProgress {
            state: initial_state.clone(),
            satisfied_goals: self.states.satisfied_goal_count(initial_state),
            advanced: false,
            complete: true,
        };
        self.continuation_queue
            .retain(|(_, queued_variant_id)| *queued_variant_id != variant_id);
    }

    /// The variant's progress state now satisfies goals it did not before, so
    /// goals this variant already attempted may be worth another abstraction
    /// from here.
    fn queue_reopened_goals(&mut self, variant_id: usize) {
        let state = &self.progress[variant_id].state;
        for (retry_goal_id, &variants_built) in self.variants_built_by_goal.iter().enumerate() {
            let was_already_attempted = variants_built > variant_id;
            if was_already_attempted
                && !self.states.satisfies_goal(state, retry_goal_id)
                && !self
                    .continuation_queue
                    .contains(&(retry_goal_id, variant_id))
            {
                self.continuation_queue
                    .push_back((retry_goal_id, variant_id));
            }
        }
    }

    fn log_summary<B: GoalFocusedBackend>(
        &self,
        member_count: usize,
        states_used: usize,
        elapsed_secs: f64,
        stop_reason: CollectionStopReason,
    ) {
        let goal_count = self.goal_count;
        info!(
            "{} collection: abstractions={member_count}, states={states_used}, elapsed={elapsed_secs:.3}s, stop_reason={stop_reason}",
            B::NAME
        );
        if !self.progress.is_empty() {
            info!(
                "{} collection: progress-state goal coverage={:?}/{goal_count}",
                B::NAME,
                self.progress
                    .iter()
                    .map(|progress| progress.satisfied_goals)
                    .collect::<Vec<_>>()
            );
        }
        info!(
            "{} collection: initial-state goal coverage={}/{goal_count}",
            B::NAME,
            self.goal_covered_from_initial_state
                .iter()
                .filter(|covered| **covered)
                .count(),
        );
    }
}

/// The first goal label with the fewest variants, while some has fewer than
/// `V`.
fn select_round_robin_goal(
    variants_built_by_goal: &[usize],
    variants_per_goal: usize,
) -> Option<usize> {
    let minimum_built = variants_built_by_goal.iter().copied().min()?;
    (minimum_built < variants_per_goal).then(|| {
        variants_built_by_goal
            .iter()
            .position(|&count| count == minimum_built)
            .expect("the minimum is attained")
    })
}

/// Round-robin until every goal has `min(V, ROUND_ROBIN_VARIANTS)` variants,
/// then give the next variant to the goal with the largest best `h(s0)`,
/// ties to the smaller id.
fn select_next_collection_goal(
    variants_built_by_goal: &[usize],
    best_initial_h_by_goal: &[f64],
    variants_per_goal: usize,
) -> Option<usize> {
    assert_eq!(
        variants_built_by_goal.len(),
        best_initial_h_by_goal.len(),
        "collection goal statistics must have equal lengths"
    );
    let guaranteed_variants = variants_per_goal.min(ROUND_ROBIN_VARIANTS);
    if let Some(minimum_built) = variants_built_by_goal
        .iter()
        .copied()
        .filter(|&count| count < guaranteed_variants)
        .min()
    {
        return variants_built_by_goal
            .iter()
            .position(|&count| count == minimum_built && count < guaranteed_variants);
    }

    variants_built_by_goal
        .iter()
        .enumerate()
        .filter(|(_, count)| **count < variants_per_goal)
        .max_by(|(left_id, _), (right_id, _)| {
            best_initial_h_by_goal[*left_id]
                .total_cmp(&best_initial_h_by_goal[*right_id])
                .then_with(|| right_id.cmp(left_id))
        })
        .map(|(goal_id, _)| goal_id)
}

#[cfg(test)]
mod tests;
