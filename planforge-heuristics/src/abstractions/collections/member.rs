//! How collections build their members, and what they need of them.

use std::time::Duration;

use planforge_sas::numeric_task::OperatorIndex;

use super::budget::CollectionStopReason;
use crate::abstractions::common::abstraction::CegarAbstraction;
use crate::abstractions::common::cegar::initial_splits::RefinementScope;
use crate::abstractions::common::cegar::{CegarStopReason, SplitKind};

/// How a collection builds one member: the member options it sets, the rest
/// being the configured ones.
#[derive(Debug, Clone, PartialEq)]
pub struct MemberConstruction {
    pub(crate) max_states: usize,
    pub(crate) max_time: Option<Duration>,
    pub(crate) random_seed: Option<u64>,
    pub(crate) split_kind: SplitKind,
    pub(crate) split_selection_rank: Option<usize>,
    pub(crate) scope: RefinementScope,
}

/// A built member with its place in the collection.
#[derive(Debug, Clone, PartialEq)]
pub struct Member<A> {
    pub abstraction: A,
    pub membership: CollectionMembership,
}

/// Where a member sits in its collection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CollectionMembership {
    /// The one-based number of the build that produced the member; a
    /// standard collection counts the restarts it dropped as duplicates too.
    pub iteration: usize,
    /// Set exactly for a member of a goal-focused collection built on its
    /// goal's single-goal task.
    pub goal: Option<GoalMembership>,
}

/// Where a member of a goal-focused collection sits among the members built
/// on its goal's task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GoalMembership {
    pub goal_id: usize,
    /// The variant whose progress state the member refined from; none for an
    /// initial-state specialist.
    pub variant_id: Option<usize>,
    /// The member refined from an advanced (non-initial) progress state.
    pub from_progress_state: bool,
}

/// What a collection reads from a built member.
pub(crate) trait CollectionMember {
    fn report(&self) -> BuiltReport;

    /// Operator ids of the concrete plan that solved the member's task from
    /// its refinement root, if refinement found one.
    fn concrete_plan(&self) -> Option<&[OperatorIndex]>;

    /// Why the member settles the task for a standard collection, if it does:
    /// its abstract plan is a concrete plan, or the initial state is an
    /// abstract dead end, and no further member can help the search.
    fn settles(&self) -> Option<CollectionStopReason> {
        if self.concrete_plan().is_some() {
            Some(CollectionStopReason::ConcretePlan)
        } else if self.report().root_is_dead_end {
            Some(CollectionStopReason::InitialStateDeadEnd)
        } else {
            None
        }
    }
}

/// What the collection needs to know about a built member.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct BuiltReport {
    /// `h(s0)` of the member, computed on the un-rooted goal task.
    pub(crate) initial_h: f64,
    pub(crate) num_states: usize,
    /// Refinement stopped because the root is an abstract dead end, and so a
    /// dead end of the member's task.
    pub(crate) root_is_dead_end: bool,
}

impl<A: CegarAbstraction> CollectionMember for A {
    fn report(&self) -> BuiltReport {
        BuiltReport {
            initial_h: self.initial_h(),
            num_states: self.num_states(),
            root_is_dead_end: self.metadata().stop_reason == CegarStopReason::DeadEnd,
        }
    }

    fn concrete_plan(&self) -> Option<&[OperatorIndex]> {
        self.metadata().concrete_plan_operator_ids.as_deref()
    }
}
