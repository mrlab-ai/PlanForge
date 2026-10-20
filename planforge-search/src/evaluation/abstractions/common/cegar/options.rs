//! The options of one CEGAR run, the same for domain and Cartesian
//! abstractions, and how collections specialise them for their members.

use std::time::Duration;

use super::{FlawKind, FlawScope, SplitKind, SplitSelection};
use crate::evaluation::abstractions::collections::goal_focused::{GoalVariant, MemberBudget};
use crate::evaluation::abstractions::collections::standard::StandardMember;

#[derive(Debug, Clone, PartialEq, planforge_search::config::ApplyOptions)]
pub struct CegarOptions {
    /// The most abstract states the abstraction may have.
    pub max_states: usize,
    /// The most refinements.
    pub max_iterations: usize,
    /// How long refinement may take; unbounded when `None`.
    pub max_time: Option<Duration>,
    /// Merge the operators that induce the same abstract transitions into one
    /// label.
    pub combine_labels: bool,
    /// Seeds split selection; without one, ties go to the first candidate.
    pub random_seed: Option<u64>,
    pub flaw_kind: FlawKind,
    pub flaw_scope: FlawScope,
    pub split_kind: SplitKind,
    pub split_selection: SplitSelection,
    /// Which of the tied best splits to take: `split_selection` scores the
    /// candidates, the best-scoring ones are sorted by a deterministic key,
    /// and the one at `rank % ties` is taken. `None` breaks ties with the
    /// random seed.
    pub split_selection_rank: Option<usize>,
    /// Log the refined abstraction.
    pub debug: bool,
}

impl CegarOptions {
    /// The options of restart `member` of a standard collection.
    pub(crate) fn for_restart(&self, member: &StandardMember) -> Self {
        Self {
            max_states: member.max_states,
            max_time: member
                .max_time
                .is_finite()
                .then(|| Duration::from_secs_f64(member.max_time.max(0.0))),
            random_seed: Some(member.seed),
            split_kind: member.split_kind.unwrap_or(self.split_kind),
            ..self.clone()
        }
    }

    /// The options of a goal-focused member: capped by what is left of the
    /// collection's budgets, with the variant's seed and tie rank, and its
    /// split kind, else the collection's `fixed_split_kind`.
    pub(crate) fn for_variant(
        &self,
        variant: &GoalVariant,
        fixed_split_kind: Option<SplitKind>,
        budget: MemberBudget,
    ) -> Self {
        Self {
            max_states: self.max_states.min(budget.remaining_states),
            // Whichever of the two budgets exist bind; the tighter one wins.
            max_time: self.max_time.into_iter().chain(budget.remaining_time).min(),
            random_seed: variant.seed,
            split_kind: variant
                .split_kind
                .or(fixed_split_kind)
                .unwrap_or(self.split_kind),
            split_selection_rank: variant.tie_rank.or(self.split_selection_rank),
            ..self.clone()
        }
    }
}
