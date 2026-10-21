//! How a collection chooses its members, and the seeds its members derive.

use serde::{Deserialize, Serialize};

use crate::config::option_enum;

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

option_enum!("collection_strategy", CollectionStrategy {
    Standard => "standard",
    PerGoal => "per_goal",
    GoalFocused => "goal_focused",
});

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
