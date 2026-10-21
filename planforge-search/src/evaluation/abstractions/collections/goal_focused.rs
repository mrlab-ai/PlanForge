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

mod progress;
mod schedule;
#[cfg(test)]
mod tests;

use std::time::Instant;

use anyhow::{Context, Result};
use planforge_sas::explicit_state::ExplicitState;
use planforge_sas::numeric_task::AbstractNumericTask;
use tracing::{debug, info};

pub(crate) use self::schedule::GoalFocusedSchedule;
use self::schedule::{CollectionState, NextMember, ScheduledMember};
use super::budget::{CollectionBudget, CollectionStopReason};
use super::member::{CollectionMember, MemberConstruction};
use super::strategy::CollectionStrategy;
use crate::evaluation::abstractions::common::abstraction::CollectionMembership;
use crate::evaluation::abstractions::common::task::SingleGoalTask;

/// A goal-focused collection stops once fewer states than this are left: a
/// member with fewer is the trivial abstraction, which tells search nothing.
/// A standard collection stops only once no state is left.
const MIN_MEMBER_STATES: usize = 2;

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

        let BuiltMember {
            member: built,
            root,
        } = build_member(
            task,
            backend,
            schedule,
            &budget,
            &state,
            &member,
            members.len() + 1,
        )?;
        let report = built.report();
        budget.charge(report.num_states);
        state.record_built_member::<B>(&member, root, report.initial_h, built.concrete_plan())?;

        members.push(built);
        if !crate::resource_limits::poll_and_release_if_exceeded() {
            stop_reason = CollectionStopReason::MemoryLimit;
            break;
        }
    }

    if stop_reason == CollectionStopReason::RequestedMembersBuilt
        && schedule.progress_states
        && state.goals_covered_from_initial_state()
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

/// Restrict the task to the member's goal unless members abstract the full
/// task, refine from its variant's progress state within what is left of
/// `budget`, and tag the result as build `iteration` so collection
/// heuristics can tell members apart.
fn build_member<B: GoalFocusedBackend>(
    task: &dyn AbstractNumericTask,
    backend: &B,
    schedule: &GoalFocusedSchedule,
    budget: &CollectionBudget,
    state: &CollectionState<'_>,
    member: &ScheduledMember,
    iteration: usize,
) -> Result<BuiltMember<B::Member>> {
    let full_task = !schedule.single_goal_tasks;
    let construction = schedule.construction(member, budget);
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
        .build(member_task, refinement_root, &construction)
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
        // The first build took time the rebuild no longer has.
        let rebuilt = backend
            .build(member_task, None, &schedule.construction(member, budget))
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
