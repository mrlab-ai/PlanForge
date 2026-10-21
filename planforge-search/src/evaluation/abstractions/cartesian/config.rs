use super::*;

use crate::evaluation::abstractions::collections::config::{CollectionConfig, MemberConfig};
use crate::evaluation::abstractions::common::cegar::initial_splits::RefinementScope;
use crate::evaluation::abstractions::common::cegar::options::CegarOptions;

#[derive(Debug, Clone, PartialEq, planforge_search::config::ApplyOptions)]
pub struct CartesianAbstractionConfig {
    #[option(flatten)]
    pub cegar: CegarOptions,
    #[option(skip)]
    pub scope: RefinementScope,
}

impl Default for CartesianAbstractionConfig {
    fn default() -> Self {
        Self {
            cegar: CegarOptions {
                max_states: 10_000,
                max_iterations: usize::MAX,
                max_time: None,
                combine_labels: false,
                random_seed: None,
                flaw_kind: FlawKind::Progression,
                flaw_scope: FlawScope::First,
                split_kind: SplitKind::Concrete,
                split_selection: SplitSelection::MinGrowth,
                split_selection_rank: None,
                debug: false,
                keep_transition_system: true,
            },
            scope: RefinementScope::default(),
        }
    }
}

impl MemberConfig for CartesianAbstractionConfig {
    fn cegar(&self) -> &CegarOptions {
        &self.cegar
    }

    fn cegar_mut(&mut self) -> &mut CegarOptions {
        &mut self.cegar
    }

    fn scope_mut(&mut self) -> &mut RefinementScope {
        &mut self.scope
    }
}

/// A collection of Cartesian abstractions.
pub type CartesianAbstractionCollectionConfig = CollectionConfig<CartesianAbstractionConfig>;

impl Default for CartesianAbstractionCollectionConfig {
    fn default() -> Self {
        let member = CartesianAbstractionConfig::default();
        Self {
            collection_strategy: CollectionStrategy::Standard,
            variants_per_goal: 1,
            max_collection_size: 10_000_000,
            total_max_time: None,
            progress_states: false,
            alternate_split_kinds: None,
            single_goal_tasks: true,
            split_kind: None,
            random_seed: member.cegar.random_seed,
            stagnation_limit: 20.0,
            blacklist_trigger_percentage: 0.75,
            enable_blacklist_on_stagnation: true,
            member,
        }
    }
}
