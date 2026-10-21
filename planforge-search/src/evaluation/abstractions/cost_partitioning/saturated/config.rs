use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::config::option_enum;
use crate::evaluation::abstractions::domain::config::DomainAbstractionCollectionConfig;

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
    /// One cost per operator.
    Label,
    /// One cost per source set of every abstract transition, for the
    /// abstractions CEGAR builds. A pattern database has no source sets and
    /// takes one cost per operator of what remains, as in label cost
    /// partitioning.
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
///
/// Building them has one budget, [`Self::table_construction_max_time`]
/// together with the heuristic's construction deadline; the phases
/// limited by [`Self::initial_order_generation_max_time`] and
/// [`Self::max_time`] end at the budget's end at the latest. A cost
/// partitioning the budget cuts short keeps the components saturated so far.
/// Time limits are in seconds, `infinity` for none.
#[derive(
    Debug, Clone, Deserialize, Serialize, PartialEq, planforge_search::config::ApplyOptions,
)]
pub struct ScpConfig {
    /// Time for diversification, from its start: sampling and the orders
    /// built for the samples (Seipp et al. 2020).
    pub max_time: Option<Duration>,
    /// Time for building every cost partitioning, from the first evaluated
    /// state on.
    pub table_construction_max_time: Option<Duration>,
    /// Choose orders over random-walk samples (Seipp et al. 2020).
    pub diversify: bool,
    /// Number of sampled states that judge whether an order adds value.
    pub samples: usize,
    /// Maximum number of diverse orders retained.
    pub max_orders: usize,
    #[option(rename = "orders")]
    pub order_generator: OrderGenerator,
    /// Time for the candidate orders of the evaluated state, from their
    /// start.
    pub initial_order_generation_max_time: Option<Duration>,
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
        let random_seed = DomainAbstractionCollectionConfig::default()
            .member
            .cegar
            .random_seed;
        Self {
            max_time: Some(Duration::from_secs(200)),
            table_construction_max_time: None,
            diversify: false,
            samples: 1_000,
            max_orders: usize::MAX,
            order_generator: OrderGenerator::Greedy,
            initial_order_generation_max_time: Some(Duration::from_secs(10)),
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
        if self.diversify && !has_sampling_task {
            return fail("SCP diversification requires an owned task reference for sampling");
        }
        Ok(())
    }
}
