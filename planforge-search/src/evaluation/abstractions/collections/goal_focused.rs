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
//! `h(s0)`, ties to the smaller goal id.
//!
//! Progress states: every variant owns a concrete *progress state*, its
//! refinement root, which starts as the task initial state and walks forward
//! through the validated concrete plans the variant's members produce. A
//! variant's progress is complete once its progress state reaches the complete
//! task goal, or once it turns out to be an abstract dead end; the variant then
//! refines from the task initial state again. After the scheduled members,
//! goals that no member built from the initial state covers get an
//! initial-root specialist, and goals a variant has re-opened are retried from
//! that variant's progress state.
//!
//! Changing the refinement root only changes which counterexamples refinement
//! sees. Every member still abstracts the full state space of its goal task, so
//! all members remain admissible components for collection heuristics.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
use planforge_sas::numeric_task::{AbstractNumericTask, OperatorIndex};
use tracing::{debug, info};

use super::portfolio::{CollectionStrategy, derive_variant_seed};
use crate::evaluation::abstractions::common::cegar::SplitKind;
use crate::evaluation::abstractions::common::concrete_state::ConcreteState;
use crate::evaluation::abstractions::common::task::SingleGoalTask;

/// Why the collection is building a member.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MemberKind {
    /// One of the `G * V` members the configuration asks for.
    Scheduled,
    /// A goal that no member built from the task initial state covers yet.
    InitialRootSpecialist,
    /// A goal some variant already attempted, retried from that variant's
    /// advanced progress state.
    Continuation,
}

/// How the collection wants one member constructed.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct GoalVariant {
    /// The goal the member is scheduled under; `None` for a goal-free task.
    pub(crate) goal_id: Option<usize>,
    /// The member abstracts the full task: a goal-free task, or a collection
    /// with `single_goal_tasks` off, where `goal_id` is only a schedule label.
    pub(crate) full_task: bool,
    pub(crate) variant_id: usize,
    pub(crate) kind: MemberKind,
    /// `None` keeps the backend's configured split kind.
    pub(crate) split_kind: Option<SplitKind>,
    /// Which of the tied best refinement candidates to take; `None` keeps the
    /// backend's configured choice.
    pub(crate) tie_rank: Option<usize>,
    /// The member's random seed, replacing the configured one.
    pub(crate) seed: Option<u64>,
}

/// What is left of the collection's state and time budget when one member is
/// about to be built.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MemberBudget {
    pub(crate) remaining_states: usize,
    pub(crate) remaining_time: Option<Duration>,
}

/// What the collection needs to know about a built member.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BuiltReport {
    /// `h(s0)` of the member, computed on the un-rooted goal task.
    pub(crate) initial_h: f64,
    pub(crate) num_states: usize,
    /// Operator ids of the concrete plan that solved the member's goal task
    /// from its refinement root, if refinement found one.
    pub(crate) concrete_plan: Option<Vec<OperatorIndex>>,
    /// Refinement stopped because the root is an abstract dead end, and so
    /// a dead end of the member's goal task.
    pub(crate) root_is_dead_end: bool,
}

/// The collection-level facts recorded on every goal-focused member.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CollectionMembership {
    pub goal_id: usize,
    pub variant_id: usize,
    /// The member refined from an advanced (non-initial) progress state.
    pub from_progress_state: bool,
}

/// One abstraction backend of a goal-focused collection.
pub(crate) trait GoalFocusedBackend {
    type Member;

    /// Backend name used in log and error messages.
    const NAME: &'static str;

    /// Build one member for `task` (the member's single-goal view, or the full
    /// task when it has no goal), refining from `root`, or from the task
    /// initial state when `root` is `None`. A root that is an abstract dead end
    /// stops refinement and is reported in [`BuiltReport::root_is_dead_end`].
    fn build(
        &self,
        task: &dyn AbstractNumericTask,
        root: Option<&ConcreteState>,
        variant: &GoalVariant,
        budget: MemberBudget,
    ) -> Result<(Self::Member, BuiltReport)>;

    /// Tag a finished member as part of the collection. `membership` is `None`
    /// for the single member of a goal-free task.
    fn record_membership(
        &self,
        member: &mut Self::Member,
        membership: Option<CollectionMembership>,
    );
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
    pub(crate) max_collection_states: usize,
    pub(crate) total_max_time: Option<Duration>,
    pub(crate) progress_states: bool,
    /// Complementary collections: alternate progression and regression by
    /// variant parity. When false, every variant keeps the backend's
    /// configured direction; ranks and seeds are unaffected.
    pub(crate) alternate_split_kinds: bool,
    /// Complementary collections: build every member on its goal's
    /// single-goal task. When false, every member is built on the full task;
    /// goals are then only labels that schedule the `G * V` members
    /// round-robin and derive their seeds and ranks, and members carry no
    /// collection membership. Excludes progress states, which would be
    /// a no-op: a progress state only advances on a plan for the full task,
    /// and that completes the variant.
    pub(crate) single_goal_tasks: bool,
}

impl GoalFocusedSchedule {
    /// How member `member` of a task with `goal_count` goals is constructed.
    ///
    /// A complementary collection alternates progression and regression by
    /// variant parity (unless `alternate_split_kinds` is off), ranks ties by
    /// `variant / 2`, and derives a seed for every variant but the first;
    /// otherwise only the seed varies, and only when more than one variant
    /// per goal is requested. An initial-root specialist reuses variant 0's
    /// construction.
    fn goal_variant(&self, member: &ScheduledMember, goal_count: usize) -> GoalVariant {
        if goal_count == 0 {
            return GoalVariant {
                goal_id: None,
                full_task: true,
                variant_id: member.variant_id,
                kind: member.kind,
                split_kind: None,
                tie_rank: None,
                seed: self.random_seed,
            };
        }
        let construction_variant = match member.kind {
            MemberKind::InitialRootSpecialist => 0,
            MemberKind::Scheduled | MemberKind::Continuation => member.variant_id,
        };
        let base_seed = self.random_seed.unwrap_or(0);
        let derived_seed =
            || derive_variant_seed(base_seed, member.goal_id, construction_variant - 1);
        assert!(
            self.alternate_split_kinds || self.strategy.is_complementary(),
            "only complementary collections choose split kinds per variant"
        );
        let (split_kind, tie_rank, seed) = if self.strategy.is_complementary() {
            let split_kind = self.alternate_split_kinds.then(|| {
                if construction_variant.is_multiple_of(2) {
                    SplitKind::Concrete
                } else {
                    SplitKind::Wanted
                }
            });
            (
                split_kind,
                Some(construction_variant / 2),
                (construction_variant > 0).then(derived_seed),
            )
        } else if self.variants_per_goal > 1 && construction_variant > 0 {
            (None, None, Some(derived_seed()))
        } else {
            (None, None, self.random_seed)
        };
        GoalVariant {
            goal_id: Some(member.goal_id),
            full_task: !self.single_goal_tasks,
            variant_id: member.variant_id,
            kind: member.kind,
            split_kind,
            tie_rank,
            seed,
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
        schedule.max_collection_states > 0,
        "goal-focused collections need a positive state budget"
    );
    assert!(
        schedule.variants_per_goal > 0,
        "goal-focused collections need at least one variant per goal"
    );
    assert!(
        schedule.single_goal_tasks
            || (schedule.strategy.is_complementary() && !schedule.progress_states),
        "full-task members require a complementary collection without progress states"
    );
    let goal_count = task.get_num_goals();
    let variants_per_goal = if goal_count == 0 {
        1
    } else {
        schedule.variants_per_goal
    };
    let abstraction_count = goal_count
        .max(1)
        .checked_mul(variants_per_goal)
        .with_context(|| format!("{} collection abstraction count overflow", B::NAME))?;
    let progress_states = schedule.progress_states;

    let start = Instant::now();
    let mut remaining_states = schedule.max_collection_states;
    let mut members = Vec::with_capacity(abstraction_count);
    let mut state = CollectionState::new(task, goal_count, variants_per_goal, progress_states)?;

    let mut stop_reason = "requested abstraction count reached";
    while state.has_work_left(abstraction_count, progress_states) {
        if remaining_states < 2 && !members.is_empty() {
            stop_reason = "collection size limit";
            break;
        }
        let member = match state.select_next_member(
            task,
            goal_count,
            abstraction_count,
            variants_per_goal,
            progress_states,
            schedule.single_goal_tasks,
        )? {
            NextMember::Build(member) => member,
            NextMember::Stop(reason) => {
                stop_reason = reason;
                break;
            }
        };
        let remaining_time =
            match remaining_collection_time(schedule.total_max_time, start, members.is_empty()) {
                CollectionBudget::Remaining(remaining) => remaining,
                CollectionBudget::Exhausted => {
                    stop_reason = "collection time limit";
                    break;
                }
            };

        let variant = schedule.goal_variant(&member, goal_count);
        let built = build_member(
            task,
            backend,
            &state,
            &member,
            &variant,
            members.len(),
            MemberBudget {
                remaining_states,
                remaining_time,
            },
        )?;

        let state_count = built.report.num_states;
        ensure!(
            state_count <= remaining_states,
            "{} goal abstraction used {state_count} states with only {remaining_states} remaining",
            B::NAME
        );
        remaining_states -= state_count;

        state.record_built_member(
            &member,
            goal_count,
            built.built_from_initial_root,
            built.reset_progress_state,
            built.report.initial_h,
        );
        state.advance_progress_state::<B>(
            task,
            goal_count,
            &member,
            built.report.concrete_plan.as_deref(),
            built.reset_progress_state,
        )?;

        members.push(built.member);
        if !crate::resource_limits::poll_and_release_if_exceeded() {
            stop_reason = "memory limit";
            break;
        }
    }

    if stop_reason == "requested abstraction count reached"
        && progress_states
        && state
            .initial_root_goal_covered
            .iter()
            .all(|covered| *covered)
    {
        stop_reason = "requested abstractions and initial-root goal coverage reached";
    }
    state.log_summary::<B>(
        members.len(),
        goal_count,
        schedule.max_collection_states - remaining_states,
        start.elapsed().as_secs_f64(),
        stop_reason,
    );
    Ok(members)
}

/// One finished member, with the facts the collection bookkeeping needs about
/// how it was built.
struct BuiltMember<M> {
    member: M,
    report: BuiltReport,
    /// The member refined from the task initial state, so it covers its goal
    /// for every state.
    built_from_initial_root: bool,
    /// The variant's progress state was an abstract dead end and the member had
    /// to be rebuilt from the task initial state.
    reset_progress_state: bool,
}

/// Restrict the task to the member's goal, refine from its variant's progress
/// state, and tag
/// the result so collection heuristics can tell members apart.
fn build_member<B: GoalFocusedBackend>(
    task: &dyn AbstractNumericTask,
    backend: &B,
    state: &CollectionState,
    member: &ScheduledMember,
    variant: &GoalVariant,
    abstraction_id: usize,
    budget: MemberBudget,
) -> Result<BuiltMember<B::Member>> {
    let goal_task = (!variant.full_task)
        .then(|| SingleGoalTask::new(task, *task.get_goal_fact(member.goal_id)));
    let member_task = goal_task
        .as_ref()
        .map_or(task, |goal_task| goal_task as &dyn AbstractNumericTask);
    debug!(
        "{} collection: building abstraction {}, goal={}, variant={}, kind={:?}, split_kind={:?}, tie_rank={:?}, remaining_states={}, seed={:?}",
        B::NAME,
        abstraction_id + 1,
        member.goal_id,
        member.variant_id,
        member.kind,
        variant.split_kind,
        variant.tie_rank,
        budget.remaining_states,
        variant.seed
    );

    // A specialist and a variant whose progress is complete both refine from
    // the task initial state,
    // which the backend takes as `None`.
    let is_specialist = member.kind == MemberKind::InitialRootSpecialist;
    let progress_is_complete = state.progress_is_complete(member.variant_id);
    let refinement_root = (!is_specialist && !progress_is_complete)
        .then(|| state.refinement_roots.get(member.variant_id))
        .flatten();
    let built_from_initial_root = is_specialist
        || refinement_root.is_none()
        || !state.progress_state_is_advanced(member.variant_id);

    let (built, report) = backend
        .build(member_task, refinement_root, variant, budget)
        .with_context(|| {
            format!(
                "failed to build {} collection abstraction {abstraction_id}",
                B::NAME
            )
        })?;
    // A dead progress state says nothing about the initial state, so the
    // member is rebuilt from there; a dead initial state is a result.
    let reset_progress_state = refinement_root.is_some() && report.root_is_dead_end;
    let (mut built, report) = if reset_progress_state {
        info!(
            "{} collection: progress state is an abstract dead end for goal {}, variant {}; rebuilding this member from the task initial state",
            B::NAME,
            member.goal_id,
            member.variant_id
        );
        backend
            .build(member_task, None, variant, budget)
            .with_context(|| {
                format!(
                    "failed to rebuild {} collection abstraction {abstraction_id} from the task initial state",
                    B::NAME
                )
            })?
    } else {
        (built, report)
    };

    let membership = (!variant.full_task).then(|| CollectionMembership {
        goal_id: member.goal_id,
        variant_id: member.variant_id,
        from_progress_state: !is_specialist
            && !progress_is_complete
            && state.progress_state_is_advanced(member.variant_id)
            && !reset_progress_state,
    });
    backend.record_membership(&mut built, membership);

    Ok(BuiltMember {
        member: built,
        report,
        built_from_initial_root,
        reset_progress_state,
    })
}

/// One member the collection has decided to build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ScheduledMember {
    /// `0` for the single member of a goal-free task.
    goal_id: usize,
    /// For an initial-root specialist, `variants_per_goal`: it sits past the
    /// scheduled variants and has no progress state.
    variant_id: usize,
    kind: MemberKind,
}

enum NextMember {
    Build(ScheduledMember),
    Stop(&'static str),
}

/// What is left of the collection's total time budget.
enum CollectionBudget {
    /// Time limit for the next member, `None` when the collection is
    /// unbounded.
    Remaining(Option<Duration>),
    /// The budget is spent and the collection already has a member.
    Exhausted,
}

fn remaining_collection_time(
    total_max_time: Option<Duration>,
    start: Instant,
    collection_is_empty: bool,
) -> CollectionBudget {
    let Some(total_max_time) = total_max_time else {
        return CollectionBudget::Remaining(None);
    };
    let elapsed = start.elapsed();
    if elapsed < total_max_time {
        return CollectionBudget::Remaining(Some(total_max_time - elapsed));
    }
    if collection_is_empty {
        // An empty collection is useless, so the first member is built even
        // with no time left -- with a zero budget, which stops it immediately.
        CollectionBudget::Remaining(Some(Duration::ZERO))
    } else {
        CollectionBudget::Exhausted
    }
}

/// Mutable state of one collection run.
///
/// With progress states enabled the collection keeps `variants_per_goal`
/// independent progress states; see the module documentation.
struct CollectionState {
    /// `None` when progress states are off, in which case there are none
    /// at all.
    initial_refinement_root: Option<ConcreteState>,
    refinement_roots: Vec<ConcreteState>,
    satisfied_goals_by_root: Vec<usize>,
    progress_state_advanced: Vec<bool>,
    progress_complete: Vec<bool>,
    initial_root_goal_covered: Vec<bool>,
    variants_built_by_goal: Vec<usize>,
    best_initial_h_by_goal: Vec<f64>,
    /// `(goal_id, variant_id)` pairs to retry from a variant's advanced
    /// progress state.
    continuation_queue: VecDeque<(usize, usize)>,
    initial_abstractions_built: usize,
}

impl CollectionState {
    fn new(
        task: &dyn AbstractNumericTask,
        goal_count: usize,
        variants_per_goal: usize,
        progress_states: bool,
    ) -> Result<Self> {
        let initial_refinement_root = (progress_states && goal_count > 0)
            .then(|| ConcreteState::initial(task))
            .transpose()?;
        let refinement_roots = initial_refinement_root
            .as_ref()
            .map_or_else(Vec::new, |root| vec![root.clone(); variants_per_goal]);
        let satisfied_goals_by_root = refinement_roots
            .iter()
            .map(|root| root.satisfied_goal_count(task))
            .collect::<Result<Vec<_>>>()?;
        let variant_count = refinement_roots.len();
        Ok(Self {
            initial_refinement_root,
            refinement_roots,
            satisfied_goals_by_root,
            progress_state_advanced: vec![false; variant_count],
            progress_complete: vec![false; variant_count],
            initial_root_goal_covered: vec![false; goal_count],
            variants_built_by_goal: vec![0usize; goal_count],
            best_initial_h_by_goal: vec![0.0f64; goal_count],
            continuation_queue: VecDeque::new(),
            initial_abstractions_built: 0,
        })
    }

    /// `false` for a variant without a progress state: progress states are off, or
    /// the variant is an initial-root specialist's.
    fn progress_state_is_advanced(&self, variant_id: usize) -> bool {
        self.progress_state_advanced
            .get(variant_id)
            .copied()
            .unwrap_or(false)
    }

    /// `false` for a variant without a progress state: progress states are off, or
    /// the variant is an initial-root specialist's.
    fn progress_is_complete(&self, variant_id: usize) -> bool {
        self.progress_complete
            .get(variant_id)
            .copied()
            .unwrap_or(false)
    }

    /// More scheduled members are due, a variant queued a retry, or some goal
    /// still lacks a member built from the task initial state.
    fn has_work_left(&self, abstraction_count: usize, progress_states: bool) -> bool {
        self.initial_abstractions_built < abstraction_count
            || !self.continuation_queue.is_empty()
            || (progress_states
                && self
                    .initial_root_goal_covered
                    .iter()
                    .any(|covered| !covered))
    }

    fn select_next_member(
        &mut self,
        task: &dyn AbstractNumericTask,
        goal_count: usize,
        abstraction_count: usize,
        variants_per_goal: usize,
        progress_states: bool,
        single_goal_tasks: bool,
    ) -> Result<NextMember> {
        let scheduled_member_pending = self.initial_abstractions_built < abstraction_count;
        let initial_root_specialist_goal = (!scheduled_member_pending && progress_states)
            .then(|| {
                self.initial_root_goal_covered
                    .iter()
                    .position(|covered| !covered)
            })
            .flatten();
        let continuation = if progress_states
            && !scheduled_member_pending
            && initial_root_specialist_goal.is_none()
        {
            self.pop_unsatisfied_continuation(task)?
        } else {
            None
        };

        if goal_count == 0 {
            // A goal-free task gets exactly one abstraction of the full task.
            return Ok(NextMember::Build(ScheduledMember {
                goal_id: 0,
                variant_id: 0,
                kind: MemberKind::Scheduled,
            }));
        }
        if scheduled_member_pending {
            let next_goal = if single_goal_tasks {
                select_next_collection_goal(
                    &self.variants_built_by_goal,
                    &self.best_initial_h_by_goal,
                    variants_per_goal,
                )
            } else {
                // Full-task members share one task: the goal is only a
                // label, so there is no goal to focus on.
                select_round_robin_goal(&self.variants_built_by_goal, variants_per_goal)
            };
            let Some(goal_id) = next_goal else {
                return Ok(NextMember::Stop("requested abstraction count reached"));
            };
            return Ok(NextMember::Build(ScheduledMember {
                goal_id,
                variant_id: self.variants_built_by_goal[goal_id],
                kind: MemberKind::Scheduled,
            }));
        }
        if let Some(goal_id) = initial_root_specialist_goal {
            return Ok(NextMember::Build(ScheduledMember {
                goal_id,
                variant_id: variants_per_goal,
                kind: MemberKind::InitialRootSpecialist,
            }));
        }
        if let Some((goal_id, variant_id)) = continuation {
            return Ok(NextMember::Build(ScheduledMember {
                goal_id,
                variant_id,
                kind: MemberKind::Continuation,
            }));
        }
        Ok(NextMember::Stop(
            "requested abstractions and initial-root goal coverage reached",
        ))
    }

    /// Drop queued retries whose goal the variant's progress state now
    /// satisfies anyway,
    /// and return the first one that still has work to do.
    fn pop_unsatisfied_continuation(
        &mut self,
        task: &dyn AbstractNumericTask,
    ) -> Result<Option<(usize, usize)>> {
        while let Some((goal_id, variant_id)) = self.continuation_queue.pop_front() {
            let root = self
                .refinement_roots
                .get(variant_id)
                .expect("a progress-state continuation references a missing root");
            if !root.satisfies_goal(task, goal_id)? {
                return Ok(Some((goal_id, variant_id)));
            }
        }
        Ok(None)
    }

    fn record_built_member(
        &mut self,
        member: &ScheduledMember,
        goal_count: usize,
        built_from_initial_root: bool,
        reset_progress_state: bool,
        initial_h: f64,
    ) {
        if goal_count == 0 {
            self.initial_abstractions_built += 1;
            return;
        }
        if built_from_initial_root || reset_progress_state {
            self.initial_root_goal_covered[member.goal_id] = true;
        }
        // Specialists and retries fill gaps; only scheduled members count
        // against the requested abstraction count.
        if member.kind == MemberKind::Scheduled {
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
        task: &dyn AbstractNumericTask,
        goal_count: usize,
        member: &ScheduledMember,
        concrete_plan: Option<&[OperatorIndex]>,
        reset_progress_state: bool,
    ) -> Result<()> {
        let variant_id = member.variant_id;
        if member.kind == MemberKind::InitialRootSpecialist
            || self.progress_is_complete(variant_id)
            || variant_id >= self.refinement_roots.len()
        {
            return Ok(());
        }
        if reset_progress_state {
            self.complete_progress(task, variant_id)?;
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

        let previous_satisfied_goals = self.satisfied_goals_by_root[variant_id];
        let root = self.refinement_roots[variant_id].replay(task, operator_ids)?;
        ensure!(
            root.satisfies_goal(task, member.goal_id)?,
            "{} collection: the concrete plan of goal {} variant {variant_id} does not reach its goal",
            B::NAME,
            member.goal_id
        );
        let satisfied_goals = root.satisfied_goal_count(task)?;
        self.refinement_roots[variant_id] = root;
        self.satisfied_goals_by_root[variant_id] = satisfied_goals;
        debug!(
            "{} collection: advanced the progress state of variant {variant_id} through {} concrete operators; satisfied_goals={satisfied_goals}/{goal_count}",
            B::NAME,
            operator_ids.len(),
        );

        if satisfied_goals == goal_count {
            self.complete_progress(task, variant_id)?;
            debug!(
                "{} collection: full goal reached for variant {variant_id}; completed its progress",
                B::NAME
            );
            return Ok(());
        }
        self.progress_state_advanced[variant_id] = true;
        if satisfied_goals > previous_satisfied_goals {
            self.queue_reopened_goals(task, variant_id)?;
        }
        Ok(())
    }

    /// Return a variant's progress state to the task initial state and stop
    /// advancing it, because it is a dead end or has reached the complete task
    /// goal. Retries queued for the variant go with it.
    fn complete_progress(
        &mut self,
        task: &dyn AbstractNumericTask,
        variant_id: usize,
    ) -> Result<()> {
        let initial_root = self
            .initial_refinement_root
            .as_ref()
            .expect("a progress state requires an initial root")
            .clone();
        self.satisfied_goals_by_root[variant_id] = initial_root.satisfied_goal_count(task)?;
        self.refinement_roots[variant_id] = initial_root;
        self.progress_state_advanced[variant_id] = false;
        self.progress_complete[variant_id] = true;
        self.continuation_queue
            .retain(|(_, queued_variant_id)| *queued_variant_id != variant_id);
        Ok(())
    }

    /// The variant's progress state now satisfies goals it did not before, so
    /// goals this variant already attempted may be worth another abstraction
    /// from here.
    fn queue_reopened_goals(
        &mut self,
        task: &dyn AbstractNumericTask,
        variant_id: usize,
    ) -> Result<()> {
        let root = &self.refinement_roots[variant_id];
        for (retry_goal_id, &variants_built) in self.variants_built_by_goal.iter().enumerate() {
            let was_already_attempted = variants_built > variant_id;
            if was_already_attempted
                && !root.satisfies_goal(task, retry_goal_id)?
                && !self
                    .continuation_queue
                    .contains(&(retry_goal_id, variant_id))
            {
                self.continuation_queue
                    .push_back((retry_goal_id, variant_id));
            }
        }
        Ok(())
    }

    fn log_summary<B: GoalFocusedBackend>(
        &self,
        member_count: usize,
        goal_count: usize,
        states_used: usize,
        elapsed_secs: f64,
        stop_reason: &str,
    ) {
        info!(
            "{} collection: abstractions={member_count}, states={states_used}, elapsed={elapsed_secs:.3}s, stop_reason={stop_reason}",
            B::NAME
        );
        if !self.satisfied_goals_by_root.is_empty() {
            info!(
                "{} collection: progress-state goal coverage={:?}/{goal_count}",
                B::NAME,
                self.satisfied_goals_by_root
            );
        }
        if !self.initial_root_goal_covered.is_empty() {
            info!(
                "{} collection: initial-root goal coverage={}/{goal_count}",
                B::NAME,
                self.initial_root_goal_covered
                    .iter()
                    .filter(|covered| **covered)
                    .count(),
            );
        }
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

/// Round-robin until every goal has `min(V, 2)` variants, then give the next
/// variant to the goal with the largest best `h(s0)`, ties to the smaller id.
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
    let guaranteed_variants = variants_per_goal.min(2);
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
