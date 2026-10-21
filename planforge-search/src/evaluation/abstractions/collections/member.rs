//! How collections build their members, and what they need of them.

use std::time::Duration;

use planforge_sas::numeric_task::OperatorIndex;

use super::budget::CollectionStopReason;
use crate::evaluation::abstractions::common::abstraction::{
    AbstractionUse, CegarAbstraction, CollectionMembership,
};
use crate::evaluation::abstractions::common::cegar::initial_splits::RefinementScope;
use crate::evaluation::abstractions::common::cegar::{CegarStopReason, SplitKind};

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

/// A member of a collection: what the collection reads from it once built,
/// and how it is marked as a member.
pub(crate) trait CollectionMember {
    fn report(&self) -> BuiltReport;

    /// Operator ids of the concrete plan that solved the member's task from
    /// its refinement root, if refinement found one.
    fn concrete_plan(&self) -> Option<&[OperatorIndex]>;

    /// Marks the member as built by the collection's build `iteration`, and
    /// with `membership` for a member of a goal-focused collection built on
    /// its goal's task.
    fn record(&mut self, iteration: usize, membership: Option<CollectionMembership>);

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

    fn record(&mut self, iteration: usize, membership: Option<CollectionMembership>) {
        self.metadata_mut().abstraction_use = AbstractionUse::CollectionMember {
            iteration,
            membership,
        };
    }
}
