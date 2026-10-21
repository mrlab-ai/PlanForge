use serde::{Deserialize, Serialize};

use crate::config::option_enum;
use crate::evaluation::abstractions::domain::config::DomainAbstractionCollectionConfig;
use crate::resource_limits::time_limit_from_secs;

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

option_enum!("orders", OrderGenerator {
    Greedy => "greedy_orders" | "greedy_orders()",
    Diverse => "diverse_orders" | "diverse_orders()",
});

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Saturator {
    All,
    Perim,
    Perimstar,
}

option_enum!("saturator", Saturator {
    All => "all",
    Perim => "perim",
    Perimstar => "perimstar",
});

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

option_enum!("partitioning", CostPartitioningMethod {
    Label => "label",
    Region => "region",
});

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
    #[option(rename = "orders")]
    pub order_generator: OrderGenerator,
    /// Time for the candidate orders built for every sample.
    pub initial_order_generation_max_time: f64,
    /// Goal-cover orders per goal that every diversified collection keeps
    /// (AAMAS 2027), in addition to the best order.
    pub goal_cover_orders_per_goal: usize,
    /// Seeded random orders among the candidate orders of the evaluated
    /// state.
    pub random_orders: usize,
    pub saturator: Saturator,
    pub random_seed: Option<u64>,
    pub partitioning: CostPartitioningMethod,
}

impl Default for ScpConfig {
    fn default() -> Self {
        let random_seed = DomainAbstractionCollectionConfig::default().random_seed;
        Self {
            max_time: 200.0,
            table_construction_max_time: 30.0,
            diversify: false,
            samples: 1_000,
            max_orders: usize::MAX,
            order_generator: OrderGenerator::Greedy,
            initial_order_generation_max_time: 10.0,
            goal_cover_orders_per_goal: 4,
            random_orders: 3,
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
        for (option, seconds) in [
            ("max_time", self.max_time),
            (
                "table_construction_max_time",
                self.table_construction_max_time,
            ),
            (
                "initial_order_generation_max_time",
                self.initial_order_generation_max_time,
            ),
        ] {
            time_limit_from_secs(seconds).map_err(|error| format!("SCP {option}: {error}"))?;
        }
        if self.diversify && !has_sampling_task {
            return fail("SCP diversification requires an owned task reference for sampling");
        }
        Ok(())
    }

    /// Bound every phase of SCP construction by the caller's remaining time.
    /// An invalid limit stays as it is, for [`Self::validate`] to reject.
    pub(crate) fn cap_construction_time(&mut self, max_seconds: f64) {
        for limit in [
            &mut self.table_construction_max_time,
            &mut self.initial_order_generation_max_time,
        ] {
            if *limit > max_seconds {
                *limit = max_seconds;
            }
        }
    }
}
