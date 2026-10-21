//! Which member a goal-focused collection builds next, and how; see the
//! [module documentation](super).

use anyhow::{Context, Result};
use planforge_sas::numeric_task::{AbstractNumericTask, OperatorIndex};
use tracing::info;

use super::progress::{ProgressStates, VariantProgress};
use super::{GoalFocusedBackend, MemberRoot};
use crate::evaluation::abstractions::collections::budget::{
    CollectionBudget, CollectionLimits, CollectionStopReason,
};
use crate::evaluation::abstractions::collections::config::SplitKinds;
use crate::evaluation::abstractions::collections::member::MemberConstruction;
use crate::evaluation::abstractions::collections::strategy::CollectionStrategy;
use crate::evaluation::abstractions::common::cegar::initial_splits::RefinementScope;
use crate::evaluation::seed::{DEFAULT_RANDOM_SEED, derive_variant_seed};

/// The variants every goal gets before further variants go to the goals
/// whose best members are strongest: one of each split kind when the
/// variants alternate.
const ROUND_ROBIN_VARIANTS: usize = 2;

/// Why the collection is building a member.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MemberKind {
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
    pub(super) fn variant_id(self) -> Option<usize> {
        match self {
            Self::Scheduled { variant_id } | Self::Continuation { variant_id } => Some(variant_id),
            Self::InitialStateSpecialist => None,
        }
    }
}

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
    pub(super) fn construction(
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
    pub(super) fn goal_free_construction(&self, budget: &CollectionBudget) -> MemberConstruction {
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

/// One member the collection has decided to build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ScheduledMember {
    pub(super) goal_id: usize,
    pub(super) kind: MemberKind,
}

pub(super) enum NextMember {
    Build(ScheduledMember),
    Stop(CollectionStopReason),
}

/// The scheduling state of one collection run on a task with goals, with the
/// variants' progress states when the collection has them.
pub(super) struct CollectionState<'task> {
    goal_count: usize,
    variants_per_goal: usize,
    /// The `G * V` scheduled members.
    pub(super) abstraction_count: usize,
    single_goal_tasks: bool,
    /// `None` when progress states are off.
    pub(super) progress: Option<ProgressStates<'task>>,
    goal_covered_from_initial_state: Vec<bool>,
    variants_built_by_goal: Vec<usize>,
    best_initial_h_by_goal: Vec<f64>,
    initial_abstractions_built: usize,
}

impl<'task> CollectionState<'task> {
    pub(super) fn new(
        task: &'task dyn AbstractNumericTask,
        schedule: &GoalFocusedSchedule,
    ) -> Result<Self> {
        let goal_count = task.get_num_goals();
        assert!(goal_count > 0, "a goal-free task has a single member");
        let variants_per_goal = schedule.variants_per_goal;
        let abstraction_count = goal_count
            .checked_mul(variants_per_goal)
            .context("goal-focused collection abstraction count overflow")?;
        let progress = schedule
            .progress_states
            .then(|| ProgressStates::new(task, variants_per_goal))
            .transpose()?;
        Ok(Self {
            goal_count,
            variants_per_goal,
            abstraction_count,
            single_goal_tasks: schedule.single_goal_tasks,
            progress,
            goal_covered_from_initial_state: vec![false; goal_count],
            variants_built_by_goal: vec![0usize; goal_count],
            best_initial_h_by_goal: vec![0.0f64; goal_count],
            initial_abstractions_built: 0,
        })
    }

    /// The progress state `member` refines from, `None` for an
    /// initial-state specialist and when progress states are off.
    pub(super) fn progress(&self, member: &ScheduledMember) -> Option<&VariantProgress> {
        let progress = self.progress.as_ref()?;
        member
            .kind
            .variant_id()
            .map(|variant_id| progress.variant(variant_id))
    }

    /// Every goal has a member built from the task initial state.
    pub(super) fn goals_covered_from_initial_state(&self) -> bool {
        self.goal_covered_from_initial_state
            .iter()
            .all(|covered| *covered)
    }

    /// More scheduled members are due, a variant queued a retry, or some goal
    /// still lacks a member built from the task initial state.
    pub(super) fn has_work_left(&self) -> bool {
        self.initial_abstractions_built < self.abstraction_count
            || self.progress.as_ref().is_some_and(|progress| {
                progress.has_continuations() || !self.goals_covered_from_initial_state()
            })
    }

    /// The scheduled members first, then with progress states the
    /// initial-state specialists, then the retries.
    pub(super) fn select_next_member(&mut self) -> NextMember {
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
        if let Some(progress) = &mut self.progress {
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
            if let Some((goal_id, variant_id)) = progress.pop_unsatisfied_continuation() {
                return NextMember::Build(ScheduledMember {
                    goal_id,
                    kind: MemberKind::Continuation { variant_id },
                });
            }
        }
        NextMember::Stop(CollectionStopReason::RequestedMembersAndInitialStateCoverage)
    }

    /// Records `member`, which refined from `root` and has `initial_h`, and
    /// advances its variant's progress state along its `concrete_plan`.
    pub(super) fn record_built_member<B: GoalFocusedBackend>(
        &mut self,
        member: &ScheduledMember,
        root: MemberRoot,
        initial_h: f64,
        concrete_plan: Option<&[OperatorIndex]>,
    ) -> Result<()> {
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
        if let (Some(progress), Some(variant_id)) = (&mut self.progress, member.kind.variant_id()) {
            progress.advance::<B>(
                member.goal_id,
                variant_id,
                root,
                concrete_plan,
                &self.variants_built_by_goal,
            )?;
        }
        Ok(())
    }

    pub(super) fn log_summary<B: GoalFocusedBackend>(
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
        if let Some(progress) = &self.progress {
            info!(
                "{} collection: progress-state goal coverage={:?}/{goal_count}",
                B::NAME,
                progress.satisfied_goals()
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
pub(super) fn select_next_collection_goal(
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
