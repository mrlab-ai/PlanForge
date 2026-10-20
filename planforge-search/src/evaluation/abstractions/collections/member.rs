//! What collections need of the abstractions they collect.

use planforge_sas::numeric_task::OperatorIndex;

use super::goal_focused::CollectionMembership;
use crate::evaluation::abstractions::common::abstraction::{
    AbstractionUse, CegarAbstraction, CollectionRecord,
};
use crate::evaluation::abstractions::common::cegar::CegarStopReason;

/// A member of a collection: what the collection reads from it once built,
/// and how it is marked as a member.
pub(crate) trait CollectionMember {
    fn report(&self) -> BuiltReport;

    /// Marks the member as a collection's: at `record`, and with `membership`
    /// for a member of a goal-focused collection built on its goal's task.
    fn record(&mut self, record: CollectionRecord, membership: Option<CollectionMembership>);
}

/// What the collection needs to know about a built member.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BuiltReport {
    /// `h(s0)` of the member, computed on the un-rooted goal task.
    pub(crate) initial_h: f64,
    pub(crate) num_states: usize,
    /// Operator ids of the concrete plan that solved the member's task from
    /// its refinement root, if refinement found one.
    pub(crate) concrete_plan: Option<Vec<OperatorIndex>>,
    /// Refinement stopped because the root is an abstract dead end, and so a
    /// dead end of the member's task.
    pub(crate) root_is_dead_end: bool,
}

impl BuiltReport {
    /// Why the member settles the task for a standard collection, if it does:
    /// its abstract plan is a concrete plan, or the initial state is an
    /// abstract dead end, and no further member can help the search.
    pub(crate) fn settles(&self) -> Option<&'static str> {
        if self.concrete_plan.is_some() {
            Some("concrete plan")
        } else if self.root_is_dead_end {
            Some("initial state is a dead end")
        } else {
            None
        }
    }
}

impl<A: CegarAbstraction> CollectionMember for A {
    fn report(&self) -> BuiltReport {
        let metadata = self.metadata();
        BuiltReport {
            initial_h: self.initial_h(),
            num_states: self.num_states(),
            concrete_plan: metadata.concrete_plan_operator_ids.clone(),
            root_is_dead_end: metadata.stop_reason == CegarStopReason::DeadEnd,
        }
    }

    fn record(&mut self, record: CollectionRecord, membership: Option<CollectionMembership>) {
        let metadata = self.metadata_mut();
        metadata.abstraction_use = AbstractionUse::CollectionMember;
        metadata.collection = Some(record);
        metadata.collection_member = membership;
    }
}
