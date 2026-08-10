use serde::{Deserialize, Serialize};

use crate::config::{ConfigValue, FromOptionValue};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum OptimizeFor {
    InitialState,
    AllStates,
    Samples,
    DiverseSamples,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum DiverseFallback {
    LargestGap,
    Random,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum BoundsProvider {
    None,
    Monotone,
    Aibr,
    All,
}

impl BoundsProvider {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Monotone => "monotone",
            Self::Aibr => "aibr",
            Self::All => "all",
        }
    }
}

macro_rules! option_enum {
    ($type:ty, {$($text:literal => $value:expr),+ $(,)?}) => {
        impl FromOptionValue for $type {
            fn from_option_value(value: &ConfigValue) -> Result<Self, String> {
                match value.as_atom()? {
                    $($text => Ok($value),)+
                    other => Err(format!(
                        "invalid {} value `{other}`",
                        stringify!($type)
                    )),
                }
            }
        }
    };
}

option_enum!(OptimizeFor, {
    "initial_state" => Self::InitialState,
    "all_states" => Self::AllStates,
    "samples" => Self::Samples,
    "diverse_samples" => Self::DiverseSamples,
});
option_enum!(DiverseFallback, {
    "largest_gap" => Self::LargestGap,
    "random" => Self::Random,
});
option_enum!(BoundsProvider, {
    "none" => Self::None,
    "monotone" => Self::Monotone,
    "aibr" => Self::Aibr,
    "all" => Self::All,
});

#[derive(
    Debug, Clone, PartialEq, Deserialize, Serialize, planforge_search::config::ApplyOptions,
)]
pub struct NumericPotentialConfig {
    pub opt: OptimizeFor,
    pub num_samples: usize,
    pub num_heuristics: usize,
    pub max_diverse_generation_time: f64,
    pub include_initial_state_potential: bool,
    pub include_all_states_potential: bool,
    pub diverse_fallback: DiverseFallback,
    pub rays: usize,
    pub max_ray_generation_time: f64,
    pub ray_epsilon: f64,
    pub ray_certificate_file: String,
    pub max_potential: f64,
    pub ignore_numeric_variables: bool,
    pub bounds: BoundsProvider,
    pub simple_action_bounds: bool,
    pub goal_conditioned: bool,
    pub goal_cost_partitioning: bool,
    pub num_goal_cost_partitions: usize,
    pub num_goal_conditioned_heuristics: usize,
    pub num_goal_conditioned_samples: usize,
    pub max_conditioned_generation_time: f64,
    #[option(rename = "max_online_heuristics")]
    pub max_online_functions: usize,
    pub online_reoptimization_interval: usize,
    pub max_consecutive_online_misses: usize,
    pub max_online_misses: usize,
    pub max_online_lp_solves: usize,
    pub invalidate_online_cache_on_growth: bool,
    pub online_reoptimization_on_new_states_only: bool,
    /// C++ Heuristic base option. Rust stores open-list estimates rather than
    /// a separate heuristic cache, but this flag still controls whether
    /// online growth activates cache invalidation/revision tracking.
    pub cache_estimates: bool,
    pub precision: f64,
    pub epsilon: f64,
    pub dump_lp: bool,
    pub validate_duality: bool,
}

impl Default for NumericPotentialConfig {
    fn default() -> Self {
        Self {
            opt: OptimizeFor::InitialState,
            num_samples: 1000,
            num_heuristics: 4,
            max_diverse_generation_time: 30.0,
            include_initial_state_potential: true,
            include_all_states_potential: false,
            diverse_fallback: DiverseFallback::LargestGap,
            rays: 0,
            max_ray_generation_time: 30.0,
            ray_epsilon: 0.000_001,
            ray_certificate_file: "numeric_potential_ray_certificate.json".to_string(),
            max_potential: 1e8,
            ignore_numeric_variables: false,
            bounds: BoundsProvider::None,
            simple_action_bounds: false,
            goal_conditioned: true,
            goal_cost_partitioning: true,
            num_goal_cost_partitions: 4,
            num_goal_conditioned_heuristics: 1,
            num_goal_conditioned_samples: 100,
            max_conditioned_generation_time: 120.0,
            max_online_functions: 100,
            online_reoptimization_interval: 50,
            max_consecutive_online_misses: 20,
            max_online_misses: 12,
            max_online_lp_solves: 1000,
            invalidate_online_cache_on_growth: false,
            online_reoptimization_on_new_states_only: false,
            cache_estimates: true,
            precision: 0.000_001,
            epsilon: 0.0,
            dump_lp: false,
            validate_duality: false,
        }
    }
}

impl NumericPotentialConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.num_samples == 0 {
            return Err("numeric_potential num_samples must be at least 1".to_string());
        }
        if self.num_heuristics == 0 {
            return Err("numeric_potential num_heuristics must be at least 1".to_string());
        }
        if self.num_goal_cost_partitions == 0
            || self.num_goal_conditioned_heuristics == 0
            || self.num_goal_conditioned_samples == 0
        {
            return Err("numeric_potential portfolio counts must be at least 1".to_string());
        }
        for (name, value) in [
            (
                "max_diverse_generation_time",
                self.max_diverse_generation_time,
            ),
            ("max_ray_generation_time", self.max_ray_generation_time),
            ("ray_epsilon", self.ray_epsilon),
            ("max_potential", self.max_potential),
            (
                "max_conditioned_generation_time",
                self.max_conditioned_generation_time,
            ),
            ("precision", self.precision),
            ("epsilon", self.epsilon),
        ] {
            if value.is_nan() || value < 0.0 {
                return Err(format!("numeric_potential {name} must be non-negative"));
            }
        }
        if self.online_reoptimization_interval == 0 {
            return Err(
                "numeric_potential online_reoptimization_interval must be at least 1".to_string(),
            );
        }
        Ok(())
    }
}
