//! Configuration of domain-abstraction CEGAR.

use crate::evaluation::abstractions::collections::config::{CollectionConfig, MemberConfig};
use crate::evaluation::abstractions::collections::strategy::CollectionStrategy;
use crate::evaluation::abstractions::common::cegar::initial_splits::RefinementScope;
use crate::evaluation::abstractions::common::cegar::options::CegarOptions;
use crate::evaluation::abstractions::common::cegar::{
    FlawKind, FlawScope, SplitKind, SplitSelection,
};
use crate::evaluation::seed::RANDOM_SPLIT_SEED;

#[derive(Debug, Clone, PartialEq, planforge_search::config::ApplyOptions)]
pub struct DomainAbstractionConfig {
    #[option(flatten)]
    pub cegar: CegarOptions,
    /// Extract abstract plans whose steps leave the variables they do not
    /// touch open.
    pub use_wildcard_plans: bool,
    #[option(skip)]
    pub scope: RefinementScope,
}

impl Default for DomainAbstractionConfig {
    fn default() -> Self {
        Self {
            cegar: CegarOptions {
                max_states: 100_000,
                max_iterations: usize::MAX,
                max_time: None,
                combine_labels: true,
                random_seed: Some(RANDOM_SPLIT_SEED),
                flaw_kind: FlawKind::Progression,
                flaw_scope: FlawScope::First,
                split_kind: SplitKind::Concrete,
                split_selection: SplitSelection::Random,
                split_selection_rank: None,
                debug: false,
                keep_transition_system: true,
            },
            use_wildcard_plans: false,
            scope: RefinementScope::default(),
        }
    }
}

impl MemberConfig for DomainAbstractionConfig {
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

/// A collection of domain abstractions.
pub type DomainAbstractionCollectionConfig = CollectionConfig<DomainAbstractionConfig>;

impl Default for DomainAbstractionCollectionConfig {
    fn default() -> Self {
        let member = DomainAbstractionConfig::default();
        Self {
            collection_strategy: CollectionStrategy::Standard,
            variants_per_goal: 1,
            max_collection_size: 10_000_000,
            total_max_time: Some(std::time::Duration::from_secs(10)),
            progress_states: false,
            alternate_split_kinds: None,
            single_goal_tasks: true,
            split_kind: None,
            random_seed: member.cegar.random_seed,
            stagnation_limit: 20.0,
            blacklist_trigger_percentage: 0.75,
            enable_blacklist_on_stagnation: true,
            member: DomainAbstractionConfig {
                cegar: CegarOptions {
                    max_states: 1_000_000,
                    ..member.cegar
                },
                ..member
            },
        }
    }
}
