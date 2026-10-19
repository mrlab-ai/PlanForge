use std::fmt;

use serde::{Deserialize, Serialize};

use crate::evaluation::domain_abstractions::domain_abstraction_collection_generator_multiple_cegar::DomainAbstractionCollectionGeneratorMultipleCegarConfig;
use crate::evaluation::pattern_databases::pattern_database::{
    PdbHeuristicConfig, PdbInternalHeuristic,
};

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
    /// The saturation passes over one abstraction, each with its log label and
    /// whether it caps saturation at the evaluated state's abstract goal
    /// distance (the perimeter of Seipp et al.).
    pub(super) fn passes(self) -> &'static [(&'static str, bool)] {
        match self {
            Self::All => &[("all", false)],
            Self::Perim => &[("perim", true)],
            Self::Perimstar => &[("perimstar/perim", true), ("perimstar/all", false)],
        }
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
pub struct ScpConfig {
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
    pub combine_labels: bool,
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
    pub random_seed: Option<u64>,
    pub partitioning: CostPartitioningMethod,
}

/// Parser-only compatibility surface for the flat `scp_online*` registry
/// names. New hierarchical configurations use [`ScpConfig`] plus
/// explicit abstraction sources.
#[derive(
    Debug, Clone, Deserialize, Serialize, PartialEq, planforge_search::config::ApplyOptions,
)]
pub(crate) struct LegacyScpConfig {
    pub online: bool,
    pub max_time: f64,
    pub table_construction_max_time: f64,
    pub max_size: usize,
    pub diversify: bool,
    pub samples: usize,
    pub max_orders: usize,
    pub interval: usize,
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

impl Default for ScpConfig {
    fn default() -> Self {
        let random_seed =
            DomainAbstractionCollectionGeneratorMultipleCegarConfig::default().random_seed;
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

impl ScpConfig {
    /// Rejects inconsistent settings. Diverse orders need random-walk
    /// samples, so `has_sampling_task` tells whether the heuristic has a task
    /// to sample from.
    pub(super) fn validate(&self, has_sampling_task: bool) -> Result<(), String> {
        let fail = |message: &str| Err(message.to_string());
        if self.online && self.interval == 0 {
            return fail("online SCP interval must be greater than zero");
        }
        if self.online && self.diversify {
            return fail("offline SCP diversification requires online=false");
        }
        if self.order_generator == OrderGenerator::Diverse && !self.diversify {
            return fail("diverse SCP orders require diversify=true");
        }
        if self.diversify && self.samples == 0 {
            return fail("offline SCP diversification requires samples > 0");
        }
        if self.diversify && self.max_orders == 0 {
            return fail("offline SCP diversification requires max_orders > 0");
        }
        if self.diversify
            && (self.initial_order_generation_max_time.is_nan()
                || self.initial_order_generation_max_time < 0.0)
        {
            return fail(
                "offline SCP diversification requires initial_order_generation_max_time >= 0",
            );
        }
        if self.diversify && !has_sampling_task {
            return fail(
                "offline SCP diversification requires an owned task reference for sampling",
            );
        }
        if self.max_size == 0 {
            return fail("SCP max_size must be greater than zero");
        }
        Ok(())
    }

    /// Bound every phase of SCP construction by the caller's remaining time.
    pub(crate) fn cap_construction_time(&mut self, max_seconds: f64) {
        self.table_construction_max_time = self.table_construction_max_time.min(max_seconds);
        self.initial_order_generation_max_time =
            self.initial_order_generation_max_time.min(max_seconds);
        self.order_optimization_max_time = self.order_optimization_max_time.min(max_seconds);
    }
}

impl Default for LegacyScpConfig {
    fn default() -> Self {
        let collection = ScpConfig::default();
        let collection_config = DomainAbstractionCollectionGeneratorMultipleCegarConfig {
            combine_labels: collection.combine_labels,
            ..Default::default()
        };
        Self {
            online: collection.online,
            max_time: collection.max_time,
            table_construction_max_time: collection.table_construction_max_time,
            max_size: collection.max_size,
            diversify: collection.diversify,
            samples: collection.samples,
            max_orders: collection.max_orders,
            interval: collection.interval,
            combine_labels: collection.combine_labels,
            collection_config,
            use_numeric_pdbs: false,
            max_pdb_states: 50_000,
            max_pattern_size: 2,
            only_interesting_patterns: true,
            pdb_exploration_heuristic: PdbInternalHeuristic::Blind,
            pdb_frontier_heuristic: PdbInternalHeuristic::Zero,
            pdb_failed_lookup_heuristic: PdbInternalHeuristic::Zero,
            scoring_function: collection.scoring_function,
            order_generator: collection.order_generator,
            initial_order_generation_max_time: collection.initial_order_generation_max_time,
            order_optimization_max_time: collection.order_optimization_max_time,
            saturator: collection.saturator,
            residual_sweeps: collection.residual_sweeps,
            random_seed: collection.random_seed,
            partitioning: collection.partitioning,
        }
    }
}

impl LegacyScpConfig {
    pub(crate) fn scp_collection_config(&self) -> ScpConfig {
        ScpConfig {
            online: self.online,
            max_time: self.max_time,
            table_construction_max_time: self.table_construction_max_time,
            max_size: self.max_size,
            diversify: self.diversify,
            samples: self.samples,
            max_orders: self.max_orders,
            interval: self.interval,
            combine_labels: self.combine_labels,
            scoring_function: self.scoring_function,
            order_generator: self.order_generator,
            initial_order_generation_max_time: self.initial_order_generation_max_time,
            order_optimization_max_time: self.order_optimization_max_time,
            saturator: self.saturator,
            residual_sweeps: self.residual_sweeps,
            random_seed: self.random_seed,
            partitioning: self.partitioning,
        }
    }

    pub(crate) fn pdb_heuristic_config(&self) -> PdbHeuristicConfig {
        PdbHeuristicConfig {
            exploration_heuristic: self.pdb_exploration_heuristic,
            frontier_heuristic: self.pdb_frontier_heuristic,
            failed_lookup_heuristic: self.pdb_failed_lookup_heuristic,
        }
    }
}
