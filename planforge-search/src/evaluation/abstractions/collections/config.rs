//! The configuration of a collection of abstractions, the same for domain and
//! Cartesian abstractions.

use std::time::Duration;

use anyhow::{Result, ensure};

use super::goal_focused::{GoalFocusedSchedule, GoalVariant, MemberBudget};
use super::standard::{StandardMember, StandardSchedule};
use super::strategy::CollectionStrategy;
use crate::config::ApplyOptions;
use crate::evaluation::abstractions::common::cegar::options::CegarOptions;
use crate::evaluation::abstractions::common::cegar::{FlawKind, SplitKind};
use crate::resource_limits::min_time_limit;

/// The configuration of one member of a collection.
pub trait MemberConfig: ApplyOptions + Clone {
    fn cegar(&self) -> &CegarOptions;

    fn cegar_mut(&mut self) -> &mut CegarOptions;
}

/// How a collection specialises its member configuration.
pub trait MemberSpecialization: MemberConfig {
    /// This configuration for restart `member` of a standard collection.
    fn for_restart(&self, member: &StandardMember) -> Self;

    /// This configuration for a goal-focused variant; see
    /// [`CegarOptions::for_variant`].
    fn for_variant(
        &self,
        variant: &GoalVariant,
        fixed_split_kind: Option<SplitKind>,
        budget: MemberBudget,
    ) -> Self;
}

/// A collection of abstractions configured by `M`. The member options bound
/// every member: `max_states` and `max_time` are per member.
#[derive(Debug, Clone, PartialEq, planforge_search::config::ApplyOptions)]
pub struct CollectionConfig<M: ApplyOptions> {
    pub collection_strategy: CollectionStrategy,
    /// Members per goal of a `per_goal` or `goal_focused` collection; 1 for
    /// `standard`.
    pub variants_per_goal: usize,
    /// The most abstract states of all members together.
    pub max_collection_size: usize,
    /// How long building the collection may take; unbounded when `None`.
    pub total_max_time: Option<Duration>,
    /// Let every variant of a goal-focused collection refine from the
    /// concrete states its earlier members' plans reached.
    pub progress_states: bool,
    /// Alternate concrete and wanted splits between members: by iteration in
    /// a `standard` collection, by variant parity in a `goal_focused` one.
    /// Unset, exactly goal-focused collections alternate.
    pub alternate_split_kinds: Option<bool>,
    /// Build every member of a goal-focused collection on its goal's
    /// single-goal task. When false, every member abstracts the full task and
    /// goals only label the `G * V` members, which are scheduled round-robin
    /// and carry no collection membership.
    pub single_goal_tasks: bool,
    /// The split kind of every member when the members do not alternate; the
    /// members' own when unset.
    pub split_kind: Option<SplitKind>,
    /// The seed the members derive theirs from.
    pub random_seed: Option<u64>,
    /// Standard collections: seconds without a new member after which the
    /// collection stagnates.
    pub stagnation_limit: f64,
    /// Standard collections: the share of `total_max_time` after which
    /// restarts blacklist variables.
    pub blacklist_trigger_percentage: f64,
    /// Standard collections: start blacklisting at stagnation instead of
    /// stopping.
    pub enable_blacklist_on_stagnation: bool,
    #[option(flatten)]
    pub member: M,
}

impl<M: MemberConfig> CollectionConfig<M> {
    /// Rejects options the strategy does not follow.
    pub(crate) fn validate(&self) -> Result<()> {
        let strategy = self.collection_strategy;
        ensure!(
            self.max_collection_size > 0,
            "`max_collection_size` must be > 0"
        );
        ensure!(
            self.variants_per_goal > 0,
            "`variants_per_goal` must be > 0"
        );
        ensure!(
            self.variants_per_goal == 1 || strategy != CollectionStrategy::Standard,
            "`variants_per_goal` requires `collection_strategy=per_goal` or `goal_focused`"
        );
        ensure!(
            !self.progress_states || strategy != CollectionStrategy::Standard,
            "`progress_states=true` requires `collection_strategy=per_goal` or `goal_focused`"
        );
        let flaw_kind = self.member.cegar().flaw_kind;
        ensure!(
            !self.progress_states || flaw_kind == FlawKind::Progression,
            "progress states advance along executed concrete plans and need flaw_kind=progression, not flaw_kind={flaw_kind}"
        );
        ensure!(
            self.single_goal_tasks || strategy.is_goal_focused(),
            "`single_goal_tasks=false` requires `collection_strategy=goal_focused`"
        );
        ensure!(
            self.single_goal_tasks || !self.progress_states,
            "{}",
            super::goal_focused::FULL_TASK_PROGRESS_STATES_ERROR
        );
        ensure!(
            self.alternate_split_kinds != Some(true) || strategy != CollectionStrategy::PerGoal,
            "`alternate_split_kinds=true` requires `collection_strategy=standard` or `goal_focused`"
        );
        ensure!(
            !self.alternates_split_kinds() || self.split_kind.is_none(),
            "`alternate_split_kinds` chooses the split kind per member; it cannot be combined with a fixed `split_kind`"
        );
        Ok(())
    }

    fn alternates_split_kinds(&self) -> bool {
        self.collection_strategy
            .alternates_split_kinds(self.alternate_split_kinds)
    }

    /// Bounds the collection and each member by `limit`.
    pub(crate) fn cap_construction_time(&mut self, limit: Duration) {
        self.total_max_time = min_time_limit(self.total_max_time, Some(limit));
        let cegar = self.member.cegar_mut();
        cegar.max_time = min_time_limit(cegar.max_time, Some(limit));
    }

    pub(crate) fn standard_schedule(&self) -> StandardSchedule {
        let seconds = |limit: Option<Duration>| limit.map_or(f64::INFINITY, |d| d.as_secs_f64());
        let cegar = self.member.cegar();
        StandardSchedule {
            max_states: cegar.max_states,
            max_collection_size: self.max_collection_size,
            max_time: seconds(cegar.max_time),
            total_max_time: seconds(self.total_max_time),
            stagnation_limit: self.stagnation_limit,
            blacklist_trigger_percentage: self.blacklist_trigger_percentage,
            enable_blacklist_on_stagnation: self.enable_blacklist_on_stagnation,
            random_seed: self.random_seed,
            alternate_split_kinds: self.alternates_split_kinds(),
            split_kind: self.split_kind,
        }
    }

    pub(crate) fn goal_focused_schedule(&self) -> GoalFocusedSchedule {
        GoalFocusedSchedule {
            strategy: self.collection_strategy,
            variants_per_goal: self.variants_per_goal,
            random_seed: self.random_seed,
            max_collection_states: self.max_collection_size,
            total_max_time: self.total_max_time,
            progress_states: self.progress_states,
            alternate_split_kinds: self.alternates_split_kinds(),
            single_goal_tasks: self.single_goal_tasks,
        }
    }
}
