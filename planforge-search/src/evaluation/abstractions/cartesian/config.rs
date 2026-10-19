use super::*;

/// Rejects the flaw kinds Cartesian CEGAR does not implement.
pub fn validate_cartesian_flaw_kind(flaw_kind: FlawKind) -> Result<(), String> {
    match flaw_kind {
        FlawKind::Progression | FlawKind::Regression | FlawKind::ExecuteEntirePlan => Ok(()),
        FlawKind::SequenceProgression
        | FlawKind::SequenceRegression
        | FlawKind::SequenceBidirectional
        | FlawKind::TargetCentered => Err(format!(
            "Cartesian abstractions do not support flaw_kind={flaw_kind}; expected progression, regression or execute_entire_plan"
        )),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CartesianRefinementDirection {
    Progression,
    Regression,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CartesianSplitSelection {
    MinTransitionGrowth,
    Random,
}

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
    pub refinement_direction: CartesianRefinementDirection,
    pub split_selection_rank: Option<usize>,
    pub split_selection: CartesianSplitSelection,
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
            refinement_direction: CartesianRefinementDirection::Progression,
            split_selection_rank: None,
            split_selection: CartesianSplitSelection::MinTransitionGrowth,
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
    pub progressive_goal_roots: bool,
    /// Complementary collections only: alternate progression and regression
    /// by variant parity. When false, every variant refines in the configured
    /// `refinement_direction`.
    pub alternate_split_kinds: bool,
    /// Complementary collections only: build every member on its goal's
    /// single-goal task. When false, every member abstracts the full task and
    /// goals only label the `G * V` members, which are scheduled round-robin
    /// and carry no collection membership. Excludes `progressive_goal_roots`.
    pub single_goal_tasks: bool,
}

impl Default for CartesianAbstractionCollectionConfig {
    fn default() -> Self {
        Self {
            abstraction: CartesianAbstractionConfig::default(),
            collection_strategy: CollectionStrategy::Standard,
            variants_per_goal: 1,
            max_collection_states: 10_000_000,
            total_max_time: None,
            progressive_goal_roots: false,
            alternate_split_kinds: true,
            single_goal_tasks: true,
        }
    }
}
