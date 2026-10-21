//! The configuration of a collection of abstractions, the same for domain and
//! Cartesian abstractions.

use std::time::Duration;

use anyhow::{Result, ensure};

use super::goal_focused::{GoalFocusedSchedule, GoalVariant, MemberBudget};
use super::standard::{StandardMember, StandardSchedule};
use super::strategy::CollectionStrategy;
use crate::config::ApplyOptions;
use crate::evaluation::abstractions::common::cegar::initial_splits::RefinementScope;
use crate::evaluation::abstractions::common::cegar::options::CegarOptions;
use crate::evaluation::abstractions::common::cegar::{FlawKind, SplitKind};
use crate::evaluation::seed::DEFAULT_RANDOM_SEED;
use crate::resource_limits::min_time_limit;

/// The configuration of one member of a collection.
pub trait MemberConfig: ApplyOptions + Clone {
    fn cegar(&self) -> &CegarOptions;

    fn cegar_mut(&mut self) -> &mut CegarOptions;

    fn scope_mut(&mut self) -> &mut RefinementScope;

    /// This configuration for restart `member` of a standard collection,
    /// which splits the restart's initial variable and keeps away from its
    /// blacklisted variables.
    fn for_restart(&self, member: &StandardMember) -> Self {
        let mut config = self.clone();
        *config.cegar_mut() = self.cegar().for_restart(member);
        *config.scope_mut() = RefinementScope {
            initial_split_var: member.initial_split_var,
            blacklisted_prop_var_ids: member.blacklisted_prop_var_ids.clone(),
            blacklisted_numeric_var_ids: member.blacklisted_numeric_var_ids.clone(),
        };
        config
    }

    /// This configuration for a goal-focused variant, which starts from the
    /// trivial abstraction; see [`CegarOptions::for_variant`].
    fn for_variant(&self, variant: &GoalVariant, budget: MemberBudget) -> Self {
        let mut config = self.clone();
        *config.cegar_mut() = self.cegar().for_variant(variant, budget);
        *config.scope_mut() = RefinementScope::default();
        config
    }
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
    /// Alternate concrete and wanted splits between members, starting with
    /// concrete: by restart in a `standard` collection, by variant in a
    /// `goal_focused` one. Unset, exactly goal-focused collections alternate.
    pub alternate_split_kinds: Option<bool>,
    /// Build every member of a goal-focused collection on its goal's
    /// single-goal task. When false, every member abstracts the full task and
    /// goals only label the `G * V` members, which are scheduled round-robin
    /// and carry no collection membership.
    pub single_goal_tasks: bool,
    /// The split kind of every member when the members do not alternate;
    /// concrete when unset. The collection chooses its members' split kinds,
    /// so the members' own `split_kind` keeps its default. The members'
    /// `random_seed` is the seed of the collection, which the members derive
    /// theirs from.
    pub split_kind: Option<SplitKind>,
    /// Standard collections: how long without a new member the collection
    /// stagnates after; never when `None`.
    pub stagnation_limit: Option<Duration>,
    /// Standard collections: the share of `total_max_time`, between 0 and 1,
    /// after which restarts blacklist variables; never without a
    /// `total_max_time`.
    pub blacklist_trigger_percentage: f64,
    /// Standard collections: start blacklisting at stagnation instead of
    /// stopping.
    pub enable_blacklist_on_stagnation: bool,
    #[option(flatten)]
    pub member: M,
}

const DEFAULT_STAGNATION_LIMIT: Option<Duration> = Some(Duration::from_secs(20));
const DEFAULT_BLACKLIST_TRIGGER_PERCENTAGE: f64 = 0.75;
const DEFAULT_ENABLE_BLACKLIST_ON_STAGNATION: bool = true;

/// How a collection chooses its members' split kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SplitKinds {
    /// Every member splits with this kind.
    Fixed(SplitKind),
    /// The members alternate concrete and wanted splits, starting with
    /// concrete.
    Alternating,
}

impl SplitKinds {
    /// The split kind of the member at zero-based position `index` of the
    /// alternation.
    pub(crate) fn of(self, index: usize) -> SplitKind {
        match self {
            Self::Fixed(kind) => kind,
            Self::Alternating if index.is_multiple_of(2) => SplitKind::Concrete,
            Self::Alternating => SplitKind::Wanted,
        }
    }
}

impl<M: MemberConfig + Default> Default for CollectionConfig<M> {
    fn default() -> Self {
        let mut member = M::default();
        member.cegar_mut().max_states = 1_000_000;
        Self {
            collection_strategy: CollectionStrategy::Standard,
            variants_per_goal: 1,
            max_collection_size: 10_000_000,
            total_max_time: Some(Duration::from_secs(10)),
            progress_states: false,
            alternate_split_kinds: None,
            single_goal_tasks: true,
            split_kind: None,
            stagnation_limit: DEFAULT_STAGNATION_LIMIT,
            blacklist_trigger_percentage: DEFAULT_BLACKLIST_TRIGGER_PERCENTAGE,
            enable_blacklist_on_stagnation: DEFAULT_ENABLE_BLACKLIST_ON_STAGNATION,
            member,
        }
    }
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
        ensure!(
            self.member.cegar().split_kind == SplitKind::default(),
            "a collection chooses its members' split kinds with its own `split_kind`"
        );
        ensure!(
            (0.0..=1.0).contains(&self.blacklist_trigger_percentage),
            "`blacklist_trigger_percentage` must be between 0 and 1, got {}",
            self.blacklist_trigger_percentage
        );
        if strategy != CollectionStrategy::Standard {
            ensure!(
                self.stagnation_limit == DEFAULT_STAGNATION_LIMIT
                    && self.blacklist_trigger_percentage == DEFAULT_BLACKLIST_TRIGGER_PERCENTAGE
                    && self.enable_blacklist_on_stagnation
                        == DEFAULT_ENABLE_BLACKLIST_ON_STAGNATION,
                "`stagnation_limit`, `blacklist_trigger_percentage` and `enable_blacklist_on_stagnation` require `collection_strategy=standard`"
            );
        }
        Ok(())
    }

    /// Whether members alternate split kinds: as configured, or by default
    /// exactly in goal-focused collections.
    fn alternates_split_kinds(&self) -> bool {
        self.alternate_split_kinds
            .unwrap_or(self.collection_strategy.is_goal_focused())
    }

    fn split_kinds(&self) -> SplitKinds {
        if self.alternates_split_kinds() {
            SplitKinds::Alternating
        } else {
            SplitKinds::Fixed(self.split_kind.unwrap_or_default())
        }
    }

    /// The configured seed of the collection; members that take a seed of
    /// their own derive it from this one, or from [`DEFAULT_RANDOM_SEED`]
    /// when unset.
    fn random_seed(&self) -> Option<u64> {
        self.member.cegar().random_seed
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
            stagnation_limit: seconds(self.stagnation_limit),
            blacklist_trigger_percentage: self.blacklist_trigger_percentage,
            enable_blacklist_on_stagnation: self.enable_blacklist_on_stagnation,
            random_seed: self.random_seed().unwrap_or(DEFAULT_RANDOM_SEED),
            split_kinds: self.split_kinds(),
        }
    }

    pub(crate) fn goal_focused_schedule(&self) -> GoalFocusedSchedule {
        GoalFocusedSchedule {
            strategy: self.collection_strategy,
            variants_per_goal: self.variants_per_goal,
            random_seed: self.random_seed(),
            max_collection_states: self.max_collection_size,
            total_max_time: self.total_max_time,
            progress_states: self.progress_states,
            split_kinds: self.split_kinds(),
            single_goal_tasks: self.single_goal_tasks,
        }
    }
}
