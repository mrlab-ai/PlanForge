use std::fmt;

use serde::{Deserialize, Serialize};

use crate::evaluation::abstractions::domain::collection_generator::DomainAbstractionCollectionGeneratorMultipleCegarConfig;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OrderGenerator {
    /// The greedy order of the evaluated state (Seipp 2017): abstractions by
    /// h / max(1, stolen costs).
    Greedy,
    /// Greedy orders, goal-cover orders and random orders over the samples
    /// (Seipp et al. 2020).
    Diverse,
}

impl fmt::Display for OrderGenerator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrderGenerator::Greedy => write!(f, "greedy_orders"),
            OrderGenerator::Diverse => write!(f, "diverse_orders"),
        }
    }
}

impl crate::config::FromOptionValue for OrderGenerator {
    fn from_option_value(value: &crate::config::ConfigValue) -> Result<Self, String> {
        match crate::config::atom(value)? {
            "greedy_orders" | "greedy_orders()" => Ok(Self::Greedy),
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

/// Configuration of SCP over a collection of abstractions. All cost
/// partitionings are built at the first evaluated state, before search, and
/// construction-only abstraction data is released afterwards.
#[derive(
    Debug, Clone, Deserialize, Serialize, PartialEq, planforge_search::config::ApplyOptions,
)]
pub struct ScpConfig {
    /// Time for choosing diverse orders over the samples.
    pub max_time: f64,
    pub table_construction_max_time: f64,
    /// Choose orders over random-walk samples (Seipp et al. 2020).
    pub diversify: bool,
    /// Number of sampled states that judge whether an order adds value.
    pub samples: usize,
    /// Maximum number of diverse orders retained.
    pub max_orders: usize,
    pub combine_labels: bool,
    #[option(rename = "orders")]
    pub order_generator: OrderGenerator,
    /// Time for the candidate orders built for every sample.
    pub initial_order_generation_max_time: f64,
    pub saturator: Saturator,
    pub random_seed: Option<u64>,
    pub partitioning: CostPartitioningMethod,
}

impl Default for ScpConfig {
    fn default() -> Self {
        let random_seed =
            DomainAbstractionCollectionGeneratorMultipleCegarConfig::default().random_seed;
        Self {
            max_time: 200.0,
            table_construction_max_time: 30.0,
            diversify: false,
            samples: 1_000,
            max_orders: usize::MAX,
            combine_labels: false,
            order_generator: OrderGenerator::Greedy,
            initial_order_generation_max_time: 10.0,
            saturator: Saturator::All,
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
        if self.order_generator == OrderGenerator::Diverse && !self.diversify {
            return fail("diverse SCP orders require diversify=true");
        }
        if self.diversify && self.samples == 0 {
            return fail("SCP diversification requires samples > 0");
        }
        if self.diversify && self.max_orders == 0 {
            return fail("SCP diversification requires max_orders > 0");
        }
        if self.diversify
            && (self.initial_order_generation_max_time.is_nan()
                || self.initial_order_generation_max_time < 0.0)
        {
            return fail("SCP diversification requires initial_order_generation_max_time >= 0");
        }
        if self.diversify && !has_sampling_task {
            return fail("SCP diversification requires an owned task reference for sampling");
        }
        Ok(())
    }

    /// Bound every phase of SCP construction by the caller's remaining time.
    pub(crate) fn cap_construction_time(&mut self, max_seconds: f64) {
        self.table_construction_max_time = self.table_construction_max_time.min(max_seconds);
        self.initial_order_generation_max_time =
            self.initial_order_generation_max_time.min(max_seconds);
    }
}
