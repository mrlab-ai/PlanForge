#[cfg(test)]
mod tests;

use planforge_sas::numeric_task::Task;
use std::fmt;
use std::sync::Arc;

use planforge_sas::explicit_state::ExplicitState;

use planforge_search::evaluation::Heuristic;
use planforge_search::evaluation::state_value_cache::StateValueCache;
use planforge_search::evaluation::{EvaluationError, EvaluationState};

use super::numeric_lm_cut_landmarks::LandmarkCutLandmarks;
use super::operator_conditions::ConditionTableLayout;

/// The options of numeric LM-cut, with the defaults of numeric Fast
/// Downward's option parser.
#[derive(Debug, Clone, Copy, PartialEq, planforge_search::config::ApplyOptions)]
pub struct LmCutNumericConfig {
    pub ceiling_less_than_one: bool,
    pub ignore_numeric: bool,
    pub irmax: bool,
    pub disable_ma: bool,
    pub use_second_order_simple: bool,
    pub use_constant_assignment: bool,
    pub bound_iterations: usize,
    pub precision: f64,
    pub epsilon: f64,
    /// How the operator-condition table is stored; `sparse` saves memory on
    /// tasks with many operators and numeric conditions.
    pub condition_table: ConditionTableLayout,
}

impl Default for LmCutNumericConfig {
    fn default() -> Self {
        Self {
            ceiling_less_than_one: false,
            ignore_numeric: false,
            irmax: false,
            disable_ma: false,
            use_second_order_simple: false,
            use_constant_assignment: false,
            bound_iterations: 0,
            precision: 0.000_001,
            epsilon: 0.0,
            condition_table: ConditionTableLayout::Dense,
        }
    }
}

impl fmt::Display for LmCutNumericConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ceiling_less_than_one={}, ignore_numeric={}, irmax={}, disable_ma={}, use_second_order_simple={}, use_constant_assignment={}, bound_iterations={}, precision={}, epsilon={}, condition_table={}",
            self.ceiling_less_than_one,
            self.ignore_numeric,
            self.irmax,
            self.disable_ma,
            self.use_second_order_simple,
            self.use_constant_assignment,
            self.bound_iterations,
            self.precision,
            self.epsilon,
            self.condition_table,
        )
    }
}

pub struct LandmarkCutNumericHeuristic<'task> {
    name: String,
    task: &'task Task,
    config: LmCutNumericConfig,
    landmark_generator: LandmarkCutLandmarks,
    state_scratch: ExplicitState,
    /// Estimates by the state ids of the search's registry.
    state_value_cache: StateValueCache,
}

impl LmCutNumericConfig {
    fn validate(&self) -> Result<(), planforge_search::config::ConfigError> {
        for (name, value) in [("precision", self.precision), ("epsilon", self.epsilon)] {
            if value.is_nan() || value < 0.0 {
                return Err(planforge_search::config::ConfigError::from(format!(
                    "lmcutnumeric {name} must be non-negative, got {value}"
                ))
                .at_option(name));
            }
        }
        Ok(())
    }
}

impl<'task> LandmarkCutNumericHeuristic<'task> {
    pub fn from_config(task: &'task Task, config: LmCutNumericConfig) -> Result<Self, String> {
        config.validate().map_err(|error| error.to_string())?;
        Ok(Self {
            name: "lmcutnumeric".to_string(),
            task,
            config,
            landmark_generator: LandmarkCutLandmarks::new(Arc::new(task.clone()), config),
            state_scratch: ExplicitState::default(),
            state_value_cache: StateValueCache::default(),
        })
    }

    pub fn config(&self) -> LmCutNumericConfig {
        self.config
    }
}

impl<'task> Heuristic for LandmarkCutNumericHeuristic<'task> {
    fn evaluate(&mut self, eval_state: &EvaluationState<'_, '_>) -> Result<f64, EvaluationError> {
        let state_id = eval_state.state().get_id();
        if let Some(value) = self.state_value_cache.get(state_id) {
            return Ok(value);
        }

        let state = &mut self.state_scratch;
        eval_state.fill_values(state)?;

        if state.holds_all(self.task.goals()) {
            self.state_value_cache.insert(state_id, 0.0);
            return Ok(0.0);
        }

        let (dead_end, total_cost) = self
            .landmark_generator
            .compute_landmark_cost(&state.propositional, &state.numeric)
            .map_err(EvaluationError::ComputationFailed)?;

        if dead_end {
            return Ok(f64::INFINITY);
        }

        assert!(
            total_cost >= 0.0,
            "lmcutnumeric returned negative heuristic value"
        );
        self.state_value_cache.insert(state_id, total_cost);
        Ok(total_cost)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Registry feature for numeric LM-cut.
pub struct LmCutNumericFeature;
impl planforge_search::heuristic_factory::HeuristicFeature for LmCutNumericFeature {
    const NAME: &'static str = "lmcutnumeric";
    type Config = LmCutNumericConfig;
    fn validate(config: &Self::Config) -> Result<(), planforge_search::config::ConfigError> {
        config.validate()
    }
    fn build<'a>(
        config: &Self::Config,
        context: &planforge_search::heuristic_factory::BuildContext<'a>,
    ) -> Result<Box<dyn Heuristic + 'a>, planforge_search::heuristic_factory::HeuristicBuildError>
    {
        let heuristic = LandmarkCutNumericHeuristic::from_config(context.task, *config)
            .map_err(|error| format!("failed to build lmcutnumeric heuristic: {error}"))?;
        Ok(Box::new(heuristic))
    }
}
