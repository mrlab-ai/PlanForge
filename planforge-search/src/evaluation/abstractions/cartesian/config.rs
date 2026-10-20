use super::*;

#[derive(Debug, Clone)]
pub struct CartesianAbstractionConfig {
    pub max_states: usize,
    pub max_time: Option<Duration>,
    pub combine_labels: bool,
    pub compute_source_sets: bool,
    /// Retain the explicit graph required by cost partitioning.
    pub retain_transition_system: bool,
    pub random_seed: Option<u64>,
    pub flaw_kind: FlawKind,
    pub flaw_scope: FlawScope,
    pub split_kind: SplitKind,
    pub split_selection_rank: Option<usize>,
    pub split_selection: SplitSelection,
    /// The variable to split the trivial abstraction on before CEGAR, in the
    /// encoding of
    /// [`initial_split_candidates`](crate::evaluation::abstractions::collections::standard::initial_split_candidates):
    /// a propositional variable is
    /// split into its initial value and the rest, a goal condition at the
    /// initial values of its numeric operands.
    pub initial_split_var: Option<VariableIndex>,
    /// Variables CEGAR may not refine.
    pub blacklisted_prop_var_ids: HashSet<VariableIndex>,
    pub blacklisted_numeric_var_ids: HashSet<VariableIndex>,
    pub debug: bool,
}

impl Default for CartesianAbstractionConfig {
    fn default() -> Self {
        Self {
            max_states: 10_000,
            max_time: None,
            combine_labels: false,
            compute_source_sets: true,
            retain_transition_system: true,
            random_seed: None,
            flaw_kind: FlawKind::Progression,
            flaw_scope: FlawScope::First,
            split_kind: SplitKind::Concrete,
            split_selection_rank: None,
            split_selection: SplitSelection::MinGrowth,
            initial_split_var: None,
            blacklisted_prop_var_ids: HashSet::new(),
            blacklisted_numeric_var_ids: HashSet::new(),
            debug: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CartesianAbstractionCollectionConfig {
    pub abstraction: CartesianAbstractionConfig,
    pub collection_strategy: CollectionStrategy,
    pub variants_per_goal: usize,
    pub max_collection_states: usize,
    pub total_max_time: Option<Duration>,
    pub progress_states: bool,
    /// Alternate concrete and wanted splits between members; unset, exactly
    /// goal-focused collections alternate (see
    /// [`CollectionStrategy::alternates_split_kinds`]).
    pub alternate_split_kinds: Option<bool>,
    /// Whether the configuration fixes the members' `split_kind`.
    pub fixed_split_kind: bool,
    /// Goal-focused collections only: build every member on its goal's
    /// single-goal task. When false, every member abstracts the full task and
    /// goals only label the `G * V` members, which are scheduled round-robin
    /// and carry no collection membership. Excludes `progress_states`.
    pub single_goal_tasks: bool,
    /// Standard collections: seconds without a new member after which the
    /// collection stagnates.
    pub stagnation_limit: f64,
    /// Standard collections: the share of `total_max_time` after which
    /// restarts blacklist variables.
    pub blacklist_trigger_percentage: f64,
    /// Standard collections: start blacklisting at stagnation instead of
    /// stopping.
    pub enable_blacklist_on_stagnation: bool,
}

impl Default for CartesianAbstractionCollectionConfig {
    fn default() -> Self {
        Self {
            abstraction: CartesianAbstractionConfig::default(),
            collection_strategy: CollectionStrategy::Standard,
            variants_per_goal: 1,
            max_collection_states: 10_000_000,
            total_max_time: None,
            progress_states: false,
            alternate_split_kinds: None,
            fixed_split_kind: false,
            single_goal_tasks: true,
            stagnation_limit: 20.0,
            blacklist_trigger_percentage: 0.75,
            enable_blacklist_on_stagnation: true,
        }
    }
}
