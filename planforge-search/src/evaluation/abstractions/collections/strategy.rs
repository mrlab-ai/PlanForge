//! How a collection chooses its members, and the seeds its members derive.

use std::fmt;

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

use crate::evaluation::abstractions::common::cegar::FlawKind;

/// How a collection chooses its members.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CollectionStrategy {
    /// Full-task members by random restarts until the budget, stagnation,
    /// or a member that settles the task (SoCS 2026): each restart varies its
    /// seed and initial split, and blacklists variables once new members
    /// stop appearing.
    Standard,
    /// One member per goal on that goal's single-goal task, `variants_per_goal`
    /// times with derived seeds (Seipp and Helmert 2018).
    PerGoal,
    /// Single-goal members with variants that alternate split kinds and may
    /// refine from progress states (AAMAS 2027); see
    /// [`goal_focused`](super::goal_focused).
    GoalFocused,
}

impl CollectionStrategy {
    pub fn is_goal_focused(self) -> bool {
        matches!(self, Self::GoalFocused)
    }

    /// Whether members alternate concrete and wanted splits: as configured, or
    /// by default exactly for goal-focused collections.
    pub fn alternates_split_kinds(self, alternate_split_kinds: Option<bool>) -> bool {
        alternate_split_kinds.unwrap_or(self.is_goal_focused())
    }
}

impl fmt::Display for CollectionStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Standard => "standard",
            Self::PerGoal => "per_goal",
            Self::GoalFocused => "goal_focused",
        })
    }
}

impl crate::config::FromOptionValue for CollectionStrategy {
    fn from_option_value(value: &crate::config::ConfigValue) -> Result<Self, String> {
        match crate::config::atom(value)? {
            "standard" => Ok(Self::Standard),
            "per_goal" => Ok(Self::PerGoal),
            "goal_focused" => Ok(Self::GoalFocused),
            other => Err(format!(
                "invalid collection_strategy `{other}`; expected one of standard, per_goal, goal_focused"
            )),
        }
    }
}

/// The collection options both representations share.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CollectionOptions {
    pub(crate) strategy: CollectionStrategy,
    pub(crate) variants_per_goal: usize,
    pub(crate) max_collection_size: usize,
    pub(crate) progress_states: bool,
    pub(crate) alternate_split_kinds: Option<bool>,
    pub(crate) single_goal_tasks: bool,
    /// Whether the configuration fixes the split kind of every member.
    pub(crate) fixed_split_kind: bool,
    pub(crate) flaw_kind: FlawKind,
}

/// Rejects options the strategy does not follow.
pub(crate) fn validate_collection_options(options: CollectionOptions) -> Result<()> {
    let CollectionOptions {
        strategy,
        variants_per_goal,
        max_collection_size,
        progress_states,
        alternate_split_kinds,
        single_goal_tasks,
        fixed_split_kind,
        flaw_kind,
    } = options;
    ensure!(max_collection_size > 0, "`max_collection_size` must be > 0");
    ensure!(variants_per_goal > 0, "`variants_per_goal` must be > 0");
    ensure!(
        variants_per_goal == 1 || strategy != CollectionStrategy::Standard,
        "`variants_per_goal` requires `collection_strategy=per_goal` or `goal_focused`"
    );
    ensure!(
        !progress_states || strategy != CollectionStrategy::Standard,
        "`progress_states=true` requires `collection_strategy=per_goal` or `goal_focused`"
    );
    ensure!(
        !progress_states || flaw_kind == FlawKind::Progression,
        "progress states advance along executed concrete plans and need flaw_kind=progression, not flaw_kind={flaw_kind}"
    );
    ensure!(
        single_goal_tasks || strategy.is_goal_focused(),
        "`single_goal_tasks=false` requires `collection_strategy=goal_focused`"
    );
    ensure!(
        single_goal_tasks || !progress_states,
        "{}",
        super::goal_focused::FULL_TASK_PROGRESS_STATES_ERROR
    );
    ensure!(
        alternate_split_kinds != Some(true) || strategy != CollectionStrategy::PerGoal,
        "`alternate_split_kinds=true` requires `collection_strategy=standard` or `goal_focused`"
    );
    ensure!(
        !strategy.alternates_split_kinds(alternate_split_kinds) || !fixed_split_kind,
        "`alternate_split_kinds` chooses the split kind per member; it cannot be combined with a fixed `split_kind`"
    );
    Ok(())
}

pub(crate) fn mix_seed(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

pub(crate) fn stable_text_seed(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

pub(crate) fn derive_variant_seed(base_seed: u64, goal_id: usize, variant_id: usize) -> u64 {
    let goal = u64::try_from(goal_id).expect("goal id does not fit u64");
    let variant = u64::try_from(variant_id).expect("variant id does not fit u64");
    mix_seed(base_seed ^ goal.rotate_left(21) ^ variant.rotate_left(43))
}

#[cfg(test)]
mod tests {
    use super::{derive_variant_seed, stable_text_seed};

    #[test]
    fn variant_coordinates_produce_distinct_deterministic_seeds() {
        let seeds = [
            derive_variant_seed(7, 0, 0),
            derive_variant_seed(7, 0, 1),
            derive_variant_seed(7, 1, 0),
        ];
        assert_ne!(seeds[0], seeds[1]);
        assert_ne!(seeds[0], seeds[2]);
        assert_eq!(seeds[1], derive_variant_seed(7, 0, 1));
    }

    #[test]
    fn text_seeds_are_stable_and_distinguish_variable_names() {
        assert_eq!(stable_text_seed("x(b0)"), stable_text_seed("x(b0)"));
        assert_ne!(stable_text_seed("x(b0)"), stable_text_seed("x(b1)"));
    }
}
