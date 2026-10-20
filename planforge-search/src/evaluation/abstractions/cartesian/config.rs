use super::*;

use crate::evaluation::abstractions::collections::config::{
    CollectionConfig, CollectionMember, MemberSpecialization,
};
use crate::evaluation::abstractions::common::cegar::options::CegarOptions;

#[derive(Debug, Clone, PartialEq, planforge_search::config::ApplyOptions)]
pub struct CartesianAbstractionConfig {
    #[option(flatten)]
    pub cegar: CegarOptions,
    #[option(skip)]
    pub compute_source_sets: bool,
    /// Retain the explicit graph required by cost partitioning.
    #[option(skip)]
    pub retain_transition_system: bool,
    /// The variable to split the trivial abstraction on before CEGAR, in the
    /// encoding of
    /// [`initial_split_candidates`](crate::evaluation::abstractions::collections::standard::initial_split_candidates):
    /// a propositional variable is
    /// split into its initial value and the rest, a goal condition at the
    /// initial values of its numeric operands.
    #[option(skip)]
    pub initial_split_var: Option<VariableIndex>,
    /// Variables CEGAR may not refine.
    #[option(skip)]
    pub blacklisted_prop_var_ids: HashSet<VariableIndex>,
    #[option(skip)]
    pub blacklisted_numeric_var_ids: HashSet<VariableIndex>,
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
            },
            compute_source_sets: true,
            retain_transition_system: true,
            initial_split_var: None,
            blacklisted_prop_var_ids: HashSet::new(),
            blacklisted_numeric_var_ids: HashSet::new(),
        }
    }
}

impl CollectionMember for CartesianAbstractionConfig {
    fn cegar(&self) -> &CegarOptions {
        &self.cegar
    }

    fn cegar_mut(&mut self) -> &mut CegarOptions {
        &mut self.cegar
    }
}

impl MemberSpecialization for CartesianAbstractionConfig {
    /// A restart splits its initial variable and keeps away from its
    /// blacklisted variables.
    fn for_restart(&self, member: &StandardMember) -> Self {
        Self {
            cegar: self.cegar.for_restart(member),
            initial_split_var: member.initial_split_var,
            blacklisted_prop_var_ids: member.blacklisted_prop_var_ids.clone(),
            blacklisted_numeric_var_ids: member.blacklisted_numeric_var_ids.clone(),
            ..self.clone()
        }
    }

    fn for_variant(
        &self,
        variant: &GoalVariant,
        fixed_split_kind: Option<SplitKind>,
        budget: MemberBudget,
    ) -> Self {
        Self {
            cegar: self.cegar.for_variant(variant, fixed_split_kind, budget),
            ..self.clone()
        }
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
