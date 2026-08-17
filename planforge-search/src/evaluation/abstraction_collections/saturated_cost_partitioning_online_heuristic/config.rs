#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ScoringFunction {
    MaxHeuristic,
    MinStolenCosts,
    MaxHeuristicPerStolenCosts,
}

impl fmt::Display for ScoringFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScoringFunction::MaxHeuristic => write!(f, "max_heuristic"),
            ScoringFunction::MinStolenCosts => write!(f, "min_stolen_costs"),
            ScoringFunction::MaxHeuristicPerStolenCosts => {
                write!(f, "max_heuristic_per_stolen_costs")
            }
        }
    }
}

impl crate::config::FromOptionValue for ScoringFunction {
    fn from_option_value(value: &crate::config::ConfigValue) -> Result<Self, String> {
        match crate::config::atom(value)? {
            "max_heuristic" => Ok(Self::MaxHeuristic),
            "min_stolen_costs" => Ok(Self::MinStolenCosts),
            "max_heuristic_per_stolen_costs" => Ok(Self::MaxHeuristicPerStolenCosts),
            other => Err(format!("invalid ScoringFunction `{other}`")),
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OrderGenerator {
    Greedy,
    DynamicGreedy,
    Random,
    Diverse,
}

impl fmt::Display for OrderGenerator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrderGenerator::Greedy => write!(f, "greedy_orders"),
            OrderGenerator::DynamicGreedy => write!(f, "dynamic_greedy_orders"),
            OrderGenerator::Random => write!(f, "random_orders"),
            OrderGenerator::Diverse => write!(f, "diverse_orders"),
        }
    }
}

impl crate::config::FromOptionValue for OrderGenerator {
    fn from_option_value(value: &crate::config::ConfigValue) -> Result<Self, String> {
        match crate::config::atom(value)? {
            "greedy_orders" | "greedy_orders()" => Ok(Self::Greedy),
            "dynamic_greedy_orders" | "dynamic_greedy_orders()" => Ok(Self::DynamicGreedy),
            "random_orders" | "random_orders()" => Ok(Self::Random),
            "diverse_orders" | "diverse_orders()" => Ok(Self::Diverse),
            other => Err(format!("invalid OrderGenerator `{other}`")),
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Saturator {
    All,
    Perim,
    Perimstar,
}

impl fmt::Display for Saturator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Saturator::All => write!(f, "all"),
            Saturator::Perim => write!(f, "perim"),
            Saturator::Perimstar => write!(f, "perimstar"),
        }
    }
}

impl Saturator {
    pub(super) fn cap_sequence(
        self,
        current_state_id: Option<usize>,
    ) -> impl Iterator<Item = (&'static str, Option<usize>)> {
        let (steps, len) = match self {
            Self::All => ([("all", None), ("all", None)], 1),
            Self::Perim => ([("perim", current_state_id), ("all", None)], 1),
            Self::Perimstar => (
                [
                    ("perimstar/perim", current_state_id),
                    ("perimstar/all", None),
                ],
                2,
            ),
        };
        steps.into_iter().take(len)
    }
}

impl crate::config::FromOptionValue for Saturator {
    fn from_option_value(value: &crate::config::ConfigValue) -> Result<Self, String> {
        match crate::config::atom(value)? {
            "all" => Ok(Self::All),
            "perim" => Ok(Self::Perim),
            "perimstar" => Ok(Self::Perimstar),
            other => Err(format!("invalid Saturator `{other}`")),
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CostPartitioningMethod {
    Label,
    Region,
}

impl CostPartitioningMethod {
    pub fn uses_regions(self) -> bool {
        matches!(self, Self::Region)
    }
}

impl fmt::Display for CostPartitioningMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Label => write!(f, "label"),
            Self::Region => write!(f, "region"),
        }
    }
}

impl crate::config::FromOptionValue for CostPartitioningMethod {
    fn from_option_value(value: &crate::config::ConfigValue) -> Result<Self, String> {
        match crate::config::atom(value)? {
            "label" => Ok(Self::Label),
            "region" => Ok(Self::Region),
            other => Err(format!("invalid cost partitioning method `{other}`")),
        }
    }
}

#[derive(
    Debug, Clone, Deserialize, Serialize, PartialEq, planforge_search::config::ApplyOptions,
)]
pub struct ScpOnlineConfig {
    /// Whether to rebuild cost partitions during search. When false, all cost
    /// partitions are built before search and construction-only abstraction
    /// data is released immediately afterwards.
    pub online: bool,
    pub max_time: f64,
    pub table_construction_max_time: f64,
    pub max_size: usize,
    /// Build a Scorpion-style offline portfolio from random-walk samples.
    /// This option is only valid with `online=false`.
    pub diversify: bool,
    /// Number of reachable concrete states used to judge whether a candidate
    /// cost partition adds value to the offline portfolio.
    pub samples: usize,
    /// Maximum number of diversified cost partitions retained offline.
    pub max_orders: usize,
    pub interval: usize,
    /// Mirrored into `collection_config.combine_labels` so `combine_labels=true`
    /// sets both. To set them independently, use the nested `collection=…`
    /// form: `scp_online(combine_labels=true, collection=…(combine_labels=false))`.
    #[option(also_sets = "collection_config.combine_labels")]
    pub combine_labels: bool,
    /// Catch-all: flat collection keys (`scp_online(max_collection_size=…)`)
    /// route here. Explicit `collection=multi_domain_abstractions(…)` form
    /// also routes here via the `nested` arm.
    #[option(flatten, nested = "collection")]
    pub collection_config: DomainAbstractionCollectionGeneratorMultipleCegarConfig,
    pub use_numeric_pdbs: bool,
    pub max_pdb_states: usize,
    pub max_pattern_size: usize,
    pub only_interesting_patterns: bool,
    pub pdb_exploration_heuristic: PdbInternalHeuristic,
    pub pdb_frontier_heuristic: PdbInternalHeuristic,
    pub pdb_failed_lookup_heuristic: PdbInternalHeuristic,
    pub scoring_function: ScoringFunction,
    #[option(rename = "orders")]
    pub order_generator: OrderGenerator,
    /// Time reserved for the bounded initial order portfolio when offline
    /// diversification is enabled. This is independent of hill climbing.
    pub initial_order_generation_max_time: f64,
    pub order_optimization_max_time: f64,
    pub saturator: Saturator,
    /// Additional traversals over the same abstraction order using the
    /// remaining regional transition costs.
    pub residual_sweeps: usize,
    #[option(also_sets = "collection_config.random_seed")]
    pub random_seed: Option<u64>,
    pub partitioning: CostPartitioningMethod,
}

#[derive(
    Debug, Clone, Deserialize, Serialize, PartialEq, planforge_search::config::ApplyOptions,
)]
pub struct FillScpConfig {
    pub table_construction_max_time: f64,
    #[option(also_sets = "collection_config.combine_labels")]
    pub combine_labels: bool,
    #[option(flatten, nested = "collection")]
    pub collection_config: DomainAbstractionCollectionGeneratorMultipleCegarConfig,
    pub scoring_function: ScoringFunction,
    #[option(rename = "orders")]
    pub order_generator: OrderGenerator,
    pub order_optimization_max_time: f64,
    pub saturator: Saturator,
    #[option(also_sets = "collection_config.random_seed")]
    pub random_seed: Option<u64>,
    pub partitioning: CostPartitioningMethod,
    /// Flattened so `precision`, `epsilon`, etc. reach the nested LMcut config.
    /// SCP/fillSCP both flatten collection_config, but this `flatten` only
    /// applies if `collection_config` does not — only one flatten per struct.
    /// Here we use `nested = "lmcut"` instead, plus per-key forwarding via the
    /// hand-written wrapper (see `apply_fill_scp_options`). Actually — since
    /// `collection_config` is the catch-all, the LMcut fields must be named
    /// explicitly. `nested = "lmcut"` lets `lmcut=lmcutnumeric(precision=…)`
    /// work cleanly.
    #[option(nested = "lmcut")]
    pub lmcut_config: LmCutNumericConfig,
}

impl Default for FillScpConfig {
    fn default() -> Self {
        let collection_config = DomainAbstractionCollectionGeneratorMultipleCegarConfig {
            combine_labels: false,
            collection_strategy: CollectionStrategy::Standard,
            ..Default::default()
        };
        let random_seed = collection_config.random_seed;
        Self {
            table_construction_max_time: 30.0,
            combine_labels: false,
            collection_config,
            scoring_function: ScoringFunction::MaxHeuristicPerStolenCosts,
            order_generator: OrderGenerator::Greedy,
            order_optimization_max_time: 5.0,
            saturator: Saturator::All,
            random_seed,
            partitioning: CostPartitioningMethod::Label,
            lmcut_config: LmCutNumericConfig::default(),
        }
    }
}

impl FillScpConfig {
    pub fn force_full_goal_tasks(&mut self) {
        self.collection_config.collection_strategy = CollectionStrategy::Standard;
        self.collection_config.combine_labels = self.combine_labels;
        self.random_seed = self.collection_config.random_seed;
        // Label-mode fillSCP only consumes per-abstraction distance tables — it never
        // touches `ConcreteOperatorFootprint`. Building those footprints during CEGAR
        // is pure memory bloat (the same per-concrete-op `StateRegion` cost that
        // canonical/max already skip via 468f06a). Disable it unconditionally for
        // the label-CP path.
        if !self.partitioning.uses_regions() {
            self.collection_config
                .set_compute_operator_footprints(false);
        }
    }

    pub(super) fn as_scp_online_config(&self) -> ScpOnlineConfig {
        ScpOnlineConfig {
            online: false,
            max_time: 0.0,
            table_construction_max_time: self.table_construction_max_time,
            max_size: usize::MAX,
            diversify: false,
            samples: 1_000,
            max_orders: usize::MAX,
            interval: usize::MAX,
            combine_labels: self.combine_labels,
            collection_config: self.collection_config.clone(),
            use_numeric_pdbs: false,
            max_pdb_states: 0,
            max_pattern_size: 0,
            only_interesting_patterns: true,
            pdb_exploration_heuristic: PdbInternalHeuristic::Blind,
            pdb_frontier_heuristic: PdbInternalHeuristic::Zero,
            pdb_failed_lookup_heuristic: PdbInternalHeuristic::Zero,
            scoring_function: self.scoring_function,
            order_generator: self.order_generator,
            initial_order_generation_max_time: 10.0,
            order_optimization_max_time: self.order_optimization_max_time,
            saturator: self.saturator,
            residual_sweeps: 0,
            random_seed: self.random_seed,
            partitioning: self.partitioning,
        }
    }
}

impl Default for ScpOnlineConfig {
    fn default() -> Self {
        let collection_config = DomainAbstractionCollectionGeneratorMultipleCegarConfig {
            combine_labels: false,
            ..Default::default()
        };
        let random_seed = collection_config.random_seed;
        Self {
            online: true,
            max_time: 200.0,
            table_construction_max_time: 30.0,
            max_size: usize::MAX,
            diversify: false,
            samples: 1_000,
            max_orders: usize::MAX,
            // Default: build the SCP heuristic once at evaluation 0 and never
            // rebuild during search. Periodic rebuilds proved expensive enough
            // to dominate per-state cost on label and regional SCP alike.
            // Configure a finite `interval` only when targeted state-specific
            // re-orderings are worth the rebuild time (rarely, in practice).
            interval: usize::MAX,
            combine_labels: false,
            collection_config,
            use_numeric_pdbs: false,
            max_pdb_states: 50_000,
            max_pattern_size: 2,
            only_interesting_patterns: true,
            pdb_exploration_heuristic: PdbInternalHeuristic::Blind,
            pdb_frontier_heuristic: PdbInternalHeuristic::Zero,
            pdb_failed_lookup_heuristic: PdbInternalHeuristic::Zero,
            scoring_function: ScoringFunction::MaxHeuristicPerStolenCosts,
            order_generator: OrderGenerator::Greedy,
            initial_order_generation_max_time: 10.0,
            // Improve the best initial order after constructing the bounded
            // initial candidate portfolio. The table-construction deadline
            // still bounds the complete preprocessing phase.
            order_optimization_max_time: 5.0,
            saturator: Saturator::All,
            residual_sweeps: 0,
            random_seed,
            partitioning: CostPartitioningMethod::Label,
        }
    }
}

impl ScpOnlineConfig {
    /// Bound every phase of SCP construction by the caller's remaining time.
    pub(crate) fn cap_construction_time(&mut self, max_seconds: f64) {
        self.table_construction_max_time = self.table_construction_max_time.min(max_seconds);
        self.initial_order_generation_max_time =
            self.initial_order_generation_max_time.min(max_seconds);
        self.order_optimization_max_time = self.order_optimization_max_time.min(max_seconds);
    }

    pub fn pdb_heuristic_config(&self) -> PdbHeuristicConfig {
        PdbHeuristicConfig {
            exploration_heuristic: self.pdb_exploration_heuristic,
            frontier_heuristic: self.pdb_frontier_heuristic,
            failed_lookup_heuristic: self.pdb_failed_lookup_heuristic,
        }
    }
}

// ---------------------------------------------------------------------------
// Lookup tables and CP heuristic
// ---------------------------------------------------------------------------
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::evaluation::domain_abstractions::domain_abstraction_collection_generator_multiple_cegar::DomainAbstractionCollectionGeneratorMultipleCegarConfig;
use crate::evaluation::numeric_landmarks::lm_cut_numeric_heuristic::LmCutNumericConfig;
use crate::evaluation::pattern_databases::pattern_database::{
    PdbHeuristicConfig, PdbInternalHeuristic,
};

use super::super::portfolio::CollectionStrategy;
