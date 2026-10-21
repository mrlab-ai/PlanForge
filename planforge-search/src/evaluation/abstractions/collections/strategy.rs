//! How a collection chooses its members.

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
}

option_enum!("collection_strategy", CollectionStrategy {
    Standard => "standard",
    PerGoal => "per_goal",
    GoalFocused => "goal_focused",
});
