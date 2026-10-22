use crate::config::TimeLimit;

use crate::config::option_enum;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizeFor {
    InitialState,
    AllStates,
    Samples,
    DiverseSamples,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiverseFallback {
    LargestGap,
    Random,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundsProvider {
    None,
    Monotone,
    Aibr,
    All,
}

option_enum!("opt", OptimizeFor {
    InitialState => "initial_state",
    AllStates => "all_states",
    Samples => "samples",
    DiverseSamples => "diverse_samples",
});
option_enum!("diverse_fallback", DiverseFallback {
    LargestGap => "largest_gap",
    Random => "random",
});
option_enum!("bounds", BoundsProvider {
    None => "none",
    Monotone => "monotone",
    Aibr => "aibr",
    All => "all",
});

#[derive(Debug, Clone, PartialEq, planforge_search::config::ApplyOptions)]
pub struct NumericPotentialConfig {
    #[option(positional)]
    pub opt: OptimizeFor,
    #[option(positional)]
    pub num_samples: usize,
    #[option(positional)]
    pub num_heuristics: usize,
    #[option(positional)]
    pub max_diverse_generation_time: TimeLimit,
    #[option(positional)]
    pub include_initial_state_potential: bool,
    #[option(positional)]
    pub include_all_states_potential: bool,
    #[option(positional)]
    pub diverse_fallback: DiverseFallback,
    #[option(positional)]
    pub rays: usize,
    #[option(positional)]
    pub max_ray_generation_time: TimeLimit,
    #[option(positional)]
    pub ray_epsilon: f64,
    #[option(positional)]
    pub ray_certificate_file: String,
    #[option(positional)]
    pub max_potential: f64,
    #[option(positional)]
    pub ignore_numeric_variables: bool,
    #[option(positional)]
    pub bounds: BoundsProvider,
    #[option(positional)]
    pub simple_action_bounds: bool,
    #[option(positional)]
    pub goal_conditioned: bool,
    #[option(positional)]
    pub goal_cost_partitioning: bool,
    #[option(positional)]
    pub num_goal_cost_partitions: usize,
    #[option(positional)]
    pub num_goal_conditioned_heuristics: usize,
    #[option(positional)]
    pub num_goal_conditioned_samples: usize,
    #[option(positional)]
    pub max_conditioned_generation_time: TimeLimit,
    #[option(rename = "max_online_heuristics")]
    #[option(positional)]
    pub max_online_functions: usize,
    #[option(positional)]
    pub online_reoptimization_interval: usize,
    #[option(positional)]
    pub max_consecutive_online_misses: usize,
    #[option(positional)]
    pub max_online_misses: usize,
    #[option(positional)]
    pub max_online_lp_solves: usize,
    #[option(positional)]
    pub invalidate_online_cache_on_growth: bool,
    #[option(positional)]
    pub online_reoptimization_on_new_states_only: bool,
    /// C++ Heuristic base option. Rust stores open-list estimates rather than
    /// a separate heuristic cache, but this flag still controls whether
    /// online growth activates cache invalidation/revision tracking.
    #[option(positional)]
    pub cache_estimates: bool,
    #[option(positional)]
    pub precision: f64,
    #[option(positional)]
    pub epsilon: f64,
    #[option(positional)]
    pub dump_lp: bool,
    #[option(positional)]
    pub validate_duality: bool,
}

impl Default for NumericPotentialConfig {
    fn default() -> Self {
        Self {
            opt: OptimizeFor::InitialState,
            num_samples: 1000,
            num_heuristics: 4,
            max_diverse_generation_time: TimeLimit(30.0),
            include_initial_state_potential: true,
            include_all_states_potential: false,
            diverse_fallback: DiverseFallback::LargestGap,
            rays: 0,
            max_ray_generation_time: TimeLimit(30.0),
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
            max_conditioned_generation_time: TimeLimit(120.0),
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
                self.max_diverse_generation_time.seconds(),
            ),
            (
                "max_ray_generation_time",
                self.max_ray_generation_time.seconds(),
            ),
            ("ray_epsilon", self.ray_epsilon),
            ("max_potential", self.max_potential),
            (
                "max_conditioned_generation_time",
                self.max_conditioned_generation_time.seconds(),
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

/// Registry feature, available as metadata in builds without CPLEX.
pub struct NumericPotentialFeature;
impl crate::heuristic_factory::HeuristicFeature for NumericPotentialFeature {
    const NAME: &'static str = "numeric_potential";
    const BACKEND: crate::heuristic_factory::RequiredBackend =
        crate::heuristic_factory::RequiredBackend::Cplex;
    type Config = NumericPotentialConfig;
    fn requirements(_: &Self::Config) -> crate::heuristic_factory::TaskRequirements {
        crate::heuristic_factory::TaskRequirements::ABSTRACTABLE_GOAL
    }
    fn validate(config: &Self::Config) -> Result<(), crate::config::ConfigError> {
        config.validate().map_err(Into::into)
    }
    fn build<'a>(
        config: &Self::Config,
        context: &crate::heuristic_factory::BuildContext<'a>,
    ) -> Result<
        Box<dyn crate::evaluation::Heuristic + 'a>,
        crate::heuristic_factory::HeuristicBuildError,
    > {
        #[cfg(feature = "cplex")]
        {
            Ok(Box::new(
                super::numeric_potentials::NumericPotentialHeuristic::from_config(
                    context.task.clone(),
                    config.clone(),
                )?,
            ))
        }
        #[cfg(not(feature = "cplex"))]
        {
            let _ = (config, context);
            unreachable!("the registry checks backend availability before construction")
        }
    }
}
