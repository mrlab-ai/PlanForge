#[cfg(test)]
mod tests;

use std::cell::RefCell;
use std::fmt;

use planforge_sas::explicit_state::ExplicitState;
use planforge_sas::numeric_task::AbstractNumericTask;
use serde::{Deserialize, Serialize};

use crate::evaluation::evaluator::{EvaluationError, EvaluationState};
use crate::evaluation::heuristic::Heuristic;
use crate::evaluation::state_value_cache::StateValueCache;

use super::numeric_lm_cut_landmarks::LandmarkCutLandmarks;

// PARITY(numeric-fd): `lmcutnumeric()` in search strings goes through the option
// parser, so `LmCutNumericConfig::default()` must match the parser defaults rather
// than the direct C++ constructor defaults.
pub const DEFAULT_CEILING_LESS_THAN_ONE: bool = false;
pub const DEFAULT_IGNORE_NUMERIC: bool = false;
pub const DEFAULT_RANDOM_PCF: bool = false;
pub const DEFAULT_IRMAX: bool = false;
pub const DEFAULT_DISABLE_MA: bool = false;
pub const DEFAULT_USE_SECOND_ORDER_SIMPLE: bool = false;
pub const DEFAULT_USE_CONSTANT_ASSIGNMENT: bool = false;
pub const DEFAULT_BOUND_ITERATIONS: usize = 0;
pub const DEFAULT_PRECISION: f64 = 0.000001;
pub const DEFAULT_EPSILON: f64 = 0.0;

#[derive(
    Debug, Clone, Copy, Deserialize, Serialize, PartialEq, planforge_search::config::ApplyOptions,
)]
pub struct LmCutNumericConfig {
    #[option(positional)]
    pub ceiling_less_than_one: bool,
    #[option(positional)]
    pub ignore_numeric: bool,
    #[option(positional)]
    pub random_pcf: bool,
    #[option(positional)]
    pub irmax: bool,
    #[option(positional)]
    pub disable_ma: bool,
    #[option(positional)]
    pub use_second_order_simple: bool,
    #[option(positional)]
    pub use_constant_assignment: bool,
    #[option(positional)]
    pub bound_iterations: usize,
    #[option(positional)]
    pub precision: f64,
    #[option(positional)]
    pub epsilon: f64,
}

impl Default for LmCutNumericConfig {
    fn default() -> Self {
        Self {
            ceiling_less_than_one: DEFAULT_CEILING_LESS_THAN_ONE,
            ignore_numeric: DEFAULT_IGNORE_NUMERIC,
            random_pcf: DEFAULT_RANDOM_PCF,
            irmax: DEFAULT_IRMAX,
            disable_ma: DEFAULT_DISABLE_MA,
            use_second_order_simple: DEFAULT_USE_SECOND_ORDER_SIMPLE,
            use_constant_assignment: DEFAULT_USE_CONSTANT_ASSIGNMENT,
            bound_iterations: DEFAULT_BOUND_ITERATIONS,
            precision: DEFAULT_PRECISION,
            epsilon: DEFAULT_EPSILON,
        }
    }
}

impl fmt::Display for LmCutNumericConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ceiling_less_than_one={}, ignore_numeric={}, random_pcf={}, irmax={}, disable_ma={}, use_second_order_simple={}, use_constant_assignment={}, bound_iterations={}, precision={}, epsilon={}",
            self.ceiling_less_than_one,
            self.ignore_numeric,
            self.random_pcf,
            self.irmax,
            self.disable_ma,
            self.use_second_order_simple,
            self.use_constant_assignment,
            self.bound_iterations,
            self.precision,
            self.epsilon,
        )
    }
}

pub struct LandmarkCutNumericHeuristic<'task> {
    name: String,
    task: &'task dyn AbstractNumericTask,
    config: LmCutNumericConfig,
    landmark_generator: RefCell<LandmarkCutLandmarks<'task>>,
    state_scratch: RefCell<ExplicitState>,
    state_value_cache: RefCell<StateValueCache>,
}

impl LmCutNumericConfig {
    fn validate(&self) -> Result<(), crate::config::ConfigError> {
        for (name, value) in [("precision", self.precision), ("epsilon", self.epsilon)] {
            if value.is_nan() || value < 0.0 {
                return Err(crate::config::ConfigError::from(format!(
                    "lmcutnumeric {name} must be non-negative, got {value}"
                ))
                .at_option(name));
            }
        }
        if self.random_pcf {
            return Err(crate::config::ConfigError::from(
                "lmcutnumeric random_pcf=true is not implemented yet",
            )
            .at_option("random_pcf"));
        }
        Ok(())
    }
}

impl<'task> LandmarkCutNumericHeuristic<'task> {
    pub fn from_config(
        task: &'task dyn AbstractNumericTask,
        config: LmCutNumericConfig,
    ) -> Result<Self, String> {
        config.validate().map_err(|error| error.to_string())?;
        Ok(Self {
            name: "lmcutnumeric".to_string(),
            task,
            config,
            landmark_generator: RefCell::new(LandmarkCutLandmarks::new(task, config)),
            state_scratch: RefCell::new(ExplicitState::default()),
            state_value_cache: RefCell::new(StateValueCache::default()),
        })
    }

    pub fn config(&self) -> LmCutNumericConfig {
        self.config
    }
}

impl<'task> Heuristic for LandmarkCutNumericHeuristic<'task> {
    fn compute_heuristic(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError> {
        let state_id = eval_state.state().get_id();
        if let Some(value) = self.state_value_cache.borrow().get(state_id) {
            return Ok(value);
        }

        let registry = eval_state.state_registry();
        let state_buffer_len = eval_state.state().buffer(registry).len();
        let mut state = self.state_scratch.borrow_mut();
        eval_state.fill_values(&mut state)?;

        if state.holds_all(self.task.goals()) {
            self.state_value_cache.borrow_mut().insert(state_id, 0.0);
            return Ok(0.0);
        }

        let (dead_end, total_cost) = self
            .landmark_generator
            .borrow_mut()
            .compute_landmark_cost(&state.propositional, state_buffer_len, &state.numeric)
            .map_err(EvaluationError::ComputationFailed)?;

        if dead_end {
            return Ok(f64::INFINITY);
        }

        assert!(
            total_cost >= 0.0,
            "lmcutnumeric returned negative heuristic value"
        );
        self.state_value_cache
            .borrow_mut()
            .insert(state_id, total_cost);
        Ok(total_cost)
    }

    fn heuristic_name(&self) -> &str {
        &self.name
    }
}

/// Registry feature for numeric LM-cut.
pub struct LmCutNumericFeature;
impl crate::heuristic_factory::HeuristicFeature for LmCutNumericFeature {
    const NAME: &'static str = "lmcutnumeric";
    type Config = LmCutNumericConfig;
    fn validate(config: &Self::Config) -> Result<(), crate::config::ConfigError> {
        config.validate()
    }
    fn build<'a>(
        config: &Self::Config,
        context: &crate::heuristic_factory::BuildContext<'a>,
    ) -> Result<Box<dyn Heuristic + 'a>, crate::heuristic_factory::HeuristicBuildError> {
        let heuristic = LandmarkCutNumericHeuristic::from_config(&**context.task, *config)
            .map_err(|error| format!("failed to build lmcutnumeric heuristic: {error}"))?;
        Ok(Box::new(heuristic))
    }
}
