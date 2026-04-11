use std::cell::RefCell;
use std::env;
use std::fmt;

use planners_sas::numeric::numeric_task::AbstractNumericTask;
use planners_sas::numeric::state_registry::{StateID, StateRegistry};
use serde::{Deserialize, Serialize};

use crate::numeric::evaluation::evaluator::{EvaluationError, EvaluationState};
use crate::numeric::evaluation::heuristic::Heuristic;

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

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
pub struct LmCutNumericConfig {
    pub ceiling_less_than_one: bool,
    pub ignore_numeric: bool,
    pub random_pcf: bool,
    pub irmax: bool,
    pub disable_ma: bool,
    pub use_second_order_simple: bool,
    pub use_constant_assignment: bool,
    pub bound_iterations: usize,
    pub precision: f64,
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
    prop_scratch: RefCell<Vec<i32>>,
    state_value_cache: RefCell<Vec<Option<f64>>>,
}

impl<'task> LandmarkCutNumericHeuristic<'task> {
    pub fn from_config(
        task: &'task dyn AbstractNumericTask,
        config: LmCutNumericConfig,
    ) -> Result<Self, String> {
        if config.precision < 0.0 {
            return Err("lmcutnumeric precision must be non-negative".to_string());
        }
        if config.epsilon < 0.0 {
            return Err("lmcutnumeric epsilon must be non-negative".to_string());
        }
        if config.random_pcf {
            return Err("lmcutnumeric random_pcf=true is not implemented yet".to_string());
        }
        Ok(Self {
            name: "lmcutnumeric".to_string(),
            task,
            config,
            landmark_generator: RefCell::new(LandmarkCutLandmarks::new(task, config)),
            prop_scratch: RefCell::new(Vec::new()),
            state_value_cache: RefCell::new(Vec::new()),
        })
    }

    fn cached_state_value(&self, state_id: StateID) -> Option<f64> {
        self.state_value_cache
            .borrow()
            .get(state_id)
            .and_then(|value| *value)
    }

    fn cache_state_value(&self, state_id: StateID, value: f64) {
        let mut cache = self.state_value_cache.borrow_mut();
        if cache.len() <= state_id {
            cache.resize(state_id + 1, None);
        }
        cache[state_id] = Some(value);
    }

    fn is_goal_state(&self, propositional_values: &[i32]) -> bool {
        (0..usize::try_from(self.task.get_num_goals().max(0)).unwrap_or(0)).all(|goal_index| {
            let goal = self.task.get_goal_fact(goal_index as i32);
            propositional_values.get(goal.var() as usize).copied() == Some(goal.value())
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
        if let Some(value) = self.cached_state_value(state_id) {
            return Ok(value);
        }

        let registry = eval_state.state_registry().ok_or_else(|| {
            EvaluationError::InvalidState(
                "LandmarkCutNumericHeuristic requires StateRegistry in EvaluationState".to_string(),
            )
        })?;
        let state_buffer_len = eval_state.state().buffer(registry).len();
        let mut numeric_values = Vec::new();
        registry
            .fill_numeric_vars(eval_state.state(), &mut numeric_values)
            .map_err(|err| {
                EvaluationError::ComputationFailed(format!(
                    "failed to prepare LM-cut numeric values: {err:?}"
                ))
            })?;
        let mut propositional_values = self.prop_scratch.borrow_mut();
        eval_state
            .state()
            .fill_state(registry, &mut propositional_values);

        if self.is_goal_state(&propositional_values) {
            self.cache_state_value(state_id, 0.0);
            return Ok(0.0);
        }

        let debug_state_id = env::var("LMCUT_DEBUG_STATE_ID")
            .ok()
            .and_then(|value| value.parse::<usize>().ok());
        let debug_state = debug_state_id == Some(state_id);

        let generator_result = if debug_state {
            let (dead_end, total_cost, landmarks) = self
                .landmark_generator
                .borrow_mut()
                .compute_landmarks(
                    &propositional_values,
                    state_buffer_len,
                    &numeric_values,
                    true,
                )
                .map_err(EvaluationError::ComputationFailed)?;
            (dead_end, total_cost, Some(landmarks))
        } else {
            let (dead_end, total_cost) = self
                .landmark_generator
                .borrow_mut()
                .compute_landmark_cost(
                    &propositional_values,
                    state_buffer_len,
                    &numeric_values,
                    false,
                )
                .map_err(EvaluationError::ComputationFailed)?;
            (dead_end, total_cost, None)
        };
        let (dead_end, total_cost, landmarks) = generator_result;

        if debug_state {
            let generator = self.landmark_generator.borrow();
            for (iteration, landmark) in landmarks.unwrap_or_default().iter().enumerate() {
                let details = landmark
                    .iter()
                    .map(|(multiplier, operator_id)| {
                        let operator_name = generator
                            .relaxed_operators()
                            .iter()
                            .find(|operator| {
                                operator.original_op_id_1 == Some(*operator_id)
                                    || operator.original_op_id_2 == Some(*operator_id)
                            })
                            .map(|operator| operator.name.as_str())
                            .unwrap_or("<unknown>");
                        format!("op={} mult={}", operator_name, multiplier)
                    })
                    .collect::<Vec<_>>()
                    .join(" | ");
                eprintln!(
                    "LMCUT_DEBUG_STATE state_id={} iteration={} landmark=[{}]",
                    state_id,
                    iteration + 1,
                    details,
                );
            }
            eprintln!(
                "LMCUT_DEBUG_STATE state_id={} total_cost={}",
                state_id, total_cost
            );
        }

        if dead_end {
            return Ok(f64::INFINITY);
        }

        assert!(
            total_cost >= 0.0,
            "lmcutnumeric returned negative heuristic value"
        );
        self.cache_state_value(state_id, total_cost);
        Ok(total_cost)
    }

    fn heuristic_name(&self) -> String {
        self.name.clone()
    }

    fn dead_ends_are_reliable(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use planners_sas::numeric::numeric_task::{ExplicitVariable, Fact, Metric, NumericRootTask};

    fn simple_var(name: &str, values: &[&str], axiom_layer: i32) -> ExplicitVariable {
        ExplicitVariable::new(
            values.len() as u32,
            name.to_string(),
            values.iter().map(|value| value.to_string()).collect(),
            axiom_layer,
            0,
        )
    }

    #[test]
    fn lmcutnumeric_config_defaults_match_fd_parser_defaults() {
        let config = LmCutNumericConfig::default();
        assert!(!config.ceiling_less_than_one);
        assert!(!config.ignore_numeric);
        assert!(!config.random_pcf);
        assert!(!config.irmax);
        assert!(!config.disable_ma);
        assert!(!config.use_second_order_simple);
        assert!(!config.use_constant_assignment);
        assert_eq!(config.bound_iterations, 0);
        assert_eq!(config.precision, 0.000001);
        assert_eq!(config.epsilon, 0.0);
    }

    #[test]
    fn from_config_accepts_second_order_simple_flag() {
        let task = NumericRootTask::new(
            3,
            Metric::new(true, -1),
            vec![simple_var("v0", &["zero", "one"], -1)],
            vec![],
            vec![Fact::new(0, 1)],
            vec![],
            vec![0],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            (0, 0),
        );
        let mut config = LmCutNumericConfig::default();
        config.use_second_order_simple = true;

        let heuristic = LandmarkCutNumericHeuristic::from_config(&task, config)
            .expect("supported SOSE flag should construct the heuristic");
        assert!(heuristic.config().use_second_order_simple);
    }

    #[test]
    fn from_config_accepts_irmax() {
        let task = NumericRootTask::new(
            3,
            Metric::new(true, -1),
            vec![simple_var("v0", &["zero", "one"], -1)],
            vec![],
            vec![Fact::new(0, 1)],
            vec![],
            vec![0],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            (0, 0),
        );
        let mut config = LmCutNumericConfig::default();
        config.irmax = true;

        let heuristic = LandmarkCutNumericHeuristic::from_config(&task, config)
            .expect("irmax flag should construct the heuristic");

        assert!(heuristic.config().irmax);
    }

    #[test]
    fn from_config_rejects_unimplemented_random_pcf() {
        let task = NumericRootTask::new(
            3,
            Metric::new(true, -1),
            vec![simple_var("v0", &["zero", "one"], -1)],
            vec![],
            vec![Fact::new(0, 1)],
            vec![],
            vec![0],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            (0, 0),
        );
        let mut config = LmCutNumericConfig::default();
        config.random_pcf = true;

        let error = match LandmarkCutNumericHeuristic::from_config(&task, config) {
            Err(error) => error,
            Ok(_) => panic!("random_pcf should be rejected until it is implemented"),
        };

        assert!(error.contains("random_pcf=true"));
    }

    #[test]
    fn from_config_accepts_disable_ma() {
        let task = NumericRootTask::new(
            3,
            Metric::new(true, -1),
            vec![simple_var("v0", &["zero", "one"], -1)],
            vec![],
            vec![Fact::new(0, 1)],
            vec![],
            vec![0],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            (0, 0),
        );
        let mut config = LmCutNumericConfig::default();
        config.disable_ma = true;

        let heuristic = LandmarkCutNumericHeuristic::from_config(&task, config)
            .expect("disable_ma flag should construct the heuristic");

        assert!(heuristic.config().disable_ma);
    }

    #[test]
    fn from_config_accepts_constant_assignment() {
        let task = NumericRootTask::new(
            3,
            Metric::new(true, -1),
            vec![simple_var("v0", &["zero", "one"], -1)],
            vec![],
            vec![Fact::new(0, 1)],
            vec![],
            vec![0],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            (0, 0),
        );
        let mut config = LmCutNumericConfig::default();
        config.use_constant_assignment = true;

        let heuristic = LandmarkCutNumericHeuristic::from_config(&task, config)
            .expect("constant assignment flag should construct the heuristic");

        assert!(heuristic.config().use_constant_assignment);
    }

    #[test]
    fn from_config_accepts_bound_iterations() {
        let task = NumericRootTask::new(
            3,
            Metric::new(true, -1),
            vec![simple_var("v0", &["zero", "one"], -1)],
            vec![],
            vec![Fact::new(0, 1)],
            vec![],
            vec![0],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            (0, 0),
        );
        let mut config = LmCutNumericConfig::default();
        config.bound_iterations = 1;

        let heuristic = LandmarkCutNumericHeuristic::from_config(&task, config)
            .expect("bound iterations should construct the heuristic once bounds are implemented");

        assert_eq!(heuristic.config().bound_iterations, 1);
    }
}
