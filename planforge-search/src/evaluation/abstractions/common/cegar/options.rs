//! The options of one CEGAR run, the same for domain and Cartesian
//! abstractions.

use crate::config::TimeLimit;

use super::{FlawKind, FlawScope, SplitKind, SplitSelection};
use crate::evaluation::seed::RANDOM_SPLIT_SEED;

#[derive(Debug, Clone, PartialEq, planforge_search::config::ApplyOptions)]
pub struct CegarOptions {
    /// The most abstract states the abstraction may have.
    pub max_states: usize,
    /// The most refinements.
    pub max_iterations: usize,
    /// How long refinement may take; unbounded for `infinity`.
    pub max_time: TimeLimit,
    /// Merge the operators that induce the same abstract transitions into one
    /// label.
    pub combine_labels: bool,
    /// Seeds split selection. Without a seed, split selection is unseeded:
    /// ties go to the first candidate, and random split selection draws from
    /// the fixed [`RANDOM_SPLIT_SEED`]. In a collection, the seed of the
    /// collection.
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
    /// Keep the transition system after refinement, which cost partitioning
    /// reads; the distances alone answer lookups.
    #[option(skip)]
    pub keep_transitions: bool,
}

impl Default for CegarOptions {
    fn default() -> Self {
        Self {
            max_states: 100_000,
            max_iterations: usize::MAX,
            max_time: TimeLimit::UNLIMITED,
            combine_labels: true,
            random_seed: Some(RANDOM_SPLIT_SEED),
            flaw_kind: FlawKind::Progression,
            flaw_scope: FlawScope::First,
            split_kind: SplitKind::Concrete,
            split_selection: SplitSelection::Random,
            split_selection_rank: None,
            debug: false,
            keep_transitions: true,
        }
    }
}

impl CegarOptions {
    /// Rejects limits no CEGAR run can follow.
    pub(crate) fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.max_states > 0, "`max_states` must be > 0");
        anyhow::ensure!(self.max_iterations > 0, "`max_iterations` must be > 0");
        self.max_time.duration().map_err(anyhow::Error::msg)?;
        Ok(())
    }
}
