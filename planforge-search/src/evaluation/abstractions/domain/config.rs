//! Configuration of domain-abstraction CEGAR.

use std::collections::HashSet;

use planforge_sas::numeric_task::VariableIndex;

use crate::evaluation::abstractions::collections::config::{
    CollectionConfig, MemberConfig, MemberSpecialization,
};
use crate::evaluation::abstractions::collections::goal_focused::{GoalVariant, MemberBudget};
use crate::evaluation::abstractions::collections::standard::StandardMember;
use crate::evaluation::abstractions::collections::strategy::CollectionStrategy;
use crate::evaluation::abstractions::common::cegar::options::CegarOptions;
use crate::evaluation::abstractions::common::cegar::{
    FlawKind, FlawScope, SplitKind, SplitSelection,
};

#[derive(Debug, Clone, PartialEq, planforge_search::config::ApplyOptions)]
pub struct DomainAbstractionConfig {
    #[option(flatten)]
    pub cegar: CegarOptions,
    /// Extract abstract plans whose steps leave the variables they do not
    /// touch open.
    pub use_wildcard_plans: bool,
    /// The variables the trivial abstraction is split on before CEGAR; the
    /// goal variables when `None`.
    #[option(skip)]
    pub init_split_var_ids: Option<HashSet<VariableIndex>>,
    /// Variables CEGAR may not refine.
    #[option(skip)]
    pub blacklisted_prop_var_ids: HashSet<VariableIndex>,
    #[option(skip)]
    pub blacklisted_numeric_var_ids: HashSet<VariableIndex>,
    /// When false, `DomainAbstractionGenerator::generate` skips building the
    /// abstract operators' source sets. Only saturated cost partitioning
    /// reads them; heuristics that read only the distance table, such as the
    /// canonical and maximum heuristics, turn this off, since the source sets
    /// can be large (~12 GB of `CartesianSet`s on
    /// minecraft-sword-advanced/prob_30x30_5).
    #[option(skip)]
    pub compute_source_sets: bool,
}

impl Default for DomainAbstractionConfig {
    fn default() -> Self {
        Self {
            cegar: CegarOptions {
                max_states: 100_000,
                max_iterations: usize::MAX,
                max_time: None,
                combine_labels: true,
                random_seed: Some(2011),
                flaw_kind: FlawKind::Progression,
                flaw_scope: FlawScope::First,
                split_kind: SplitKind::Concrete,
                split_selection: SplitSelection::Random,
                split_selection_rank: None,
                debug: false,
            },
            use_wildcard_plans: false,
            init_split_var_ids: None,
            blacklisted_prop_var_ids: HashSet::new(),
            blacklisted_numeric_var_ids: HashSet::new(),
            compute_source_sets: true,
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
}

impl MemberSpecialization for DomainAbstractionConfig {
    /// A restart splits its initial variable alone and keeps away from its
    /// blacklisted variables.
    fn for_restart(&self, member: &StandardMember) -> Self {
        Self {
            cegar: self.cegar.for_restart(member),
            init_split_var_ids: Some(member.initial_split_var.into_iter().collect()),
            blacklisted_prop_var_ids: member.blacklisted_prop_var_ids.clone(),
            blacklisted_numeric_var_ids: member.blacklisted_numeric_var_ids.clone(),
            ..self.clone()
        }
    }

    /// A goal-focused member starts from the trivial abstraction: no initial
    /// splits and no blacklisted variables.
    fn for_variant(
        &self,
        variant: &GoalVariant,
        fixed_split_kind: Option<SplitKind>,
        budget: MemberBudget,
    ) -> Self {
        Self {
            cegar: self.cegar.for_variant(variant, fixed_split_kind, budget),
            init_split_var_ids: Some(HashSet::new()),
            blacklisted_prop_var_ids: HashSet::new(),
            blacklisted_numeric_var_ids: HashSet::new(),
            ..self.clone()
        }
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
