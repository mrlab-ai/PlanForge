//! What a collection may spend on its members, and why it stopped.

use std::fmt;
use std::time::{Duration, Instant};

use anyhow::Result;

use crate::resource_limits::{Deadline, min_time_limit};

/// The limits of a collection: for all members together and for each.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CollectionLimits {
    /// The most abstract states of all members together.
    pub(crate) max_collection_states: usize,
    /// How long building the collection may take; unbounded when `None`.
    pub(crate) total_max_time: Option<Duration>,
    /// The most abstract states of one member.
    pub(crate) member_max_states: usize,
    /// How long building one member may take; unbounded when `None`.
    pub(crate) member_max_time: Option<Duration>,
}

/// What is left of a collection's limits while it builds its members.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CollectionBudget {
    limits: CollectionLimits,
    remaining_states: usize,
    deadline: Deadline,
}

impl CollectionBudget {
    /// The full budget of a collection started at `start`.
    pub(crate) fn new(limits: CollectionLimits, start: Instant) -> Result<Self> {
        assert!(
            limits.max_collection_states > 0 && limits.member_max_states > 0,
            "a collection and its members need a positive state budget"
        );
        Ok(Self {
            limits,
            remaining_states: limits.max_collection_states,
            deadline: Deadline::after(start, limits.total_max_time)?,
        })
    }

    /// The states the members may still have together.
    pub(crate) fn remaining_states(&self) -> usize {
        self.remaining_states
    }

    /// The states all members have together.
    pub(crate) fn used_states(&self) -> usize {
        self.limits.max_collection_states - self.remaining_states
    }

    /// The most states the next member may have.
    pub(crate) fn member_max_states(&self) -> usize {
        self.limits.member_max_states.min(self.remaining_states)
    }

    /// How long the collection may still take; `None` when unbounded.
    pub(crate) fn remaining_time(&self) -> Option<Duration> {
        self.deadline.remaining()
    }

    /// How long the next member may take; `None` when unbounded.
    pub(crate) fn member_max_time(&self) -> Option<Duration> {
        min_time_limit(self.limits.member_max_time, self.remaining_time())
    }

    pub(crate) fn deadline_passed(&self) -> bool {
        self.deadline.passed()
    }

    /// Charges a member with `states` states, which its construction kept
    /// within [`member_max_states`](Self::member_max_states).
    pub(crate) fn charge(&mut self, states: usize) {
        assert!(
            states <= self.member_max_states(),
            "a member has {states} states, over its limit of {}",
            self.member_max_states()
        );
        self.remaining_states -= states;
    }
}

/// Why a collection stopped building members.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CollectionStopReason {
    /// The members used up the collection's states.
    SizeLimit,
    /// The collection's time is up.
    TimeLimit,
    /// The member under construction hit the deadline of the whole
    /// abstraction generation.
    GenerationDeadline,
    /// A member's abstract plan is a concrete plan.
    ConcretePlan,
    /// A member proves the initial state a dead end.
    InitialStateDeadEnd,
    /// No new member appeared for the stagnation limit.
    StagnationLimit,
    /// The process reached its memory limit.
    MemoryLimit,
    /// Every requested member is built.
    RequestedMembersBuilt,
    /// Every requested member is built, and every goal has a member built
    /// from the initial state.
    RequestedMembersAndInitialStateCoverage,
}

impl fmt::Display for CollectionStopReason {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::SizeLimit => "collection size limit",
            Self::TimeLimit => "collection time limit",
            Self::GenerationDeadline => "abstraction generation deadline",
            Self::ConcretePlan => "concrete plan",
            Self::InitialStateDeadEnd => "initial state is a dead end",
            Self::StagnationLimit => "stagnation limit",
            Self::MemoryLimit => "memory limit",
            Self::RequestedMembersBuilt => "requested abstraction count reached",
            Self::RequestedMembersAndInitialStateCoverage => {
                "requested abstractions and initial-state goal coverage reached"
            }
        })
    }
}
