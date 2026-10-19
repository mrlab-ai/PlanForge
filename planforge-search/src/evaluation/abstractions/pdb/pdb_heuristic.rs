#[cfg(test)]
mod tests;

use std::cell::RefCell;

use crate::evaluation::evaluator::{EvaluationError, EvaluationState};
use crate::evaluation::heuristic::Heuristic;
use crate::evaluation::state_value_cache::StateValueCache;

use planforge_sas::numeric_task::{AbstractNumericTask, ExplicitValueIndex, NumericValue};

use super::pattern_database::PatternDatabase;
use super::pattern_generator_greedy::{GreedyPatternGeneratorConfig, generate_greedy_pattern};
use super::projected_task::ProjectedTask;
use super::utils;
use super::validate_restricted_task;

pub struct GreedyNumericPdbHeuristic<'task> {
    name: String,
    pdb: PatternDatabase<'task>,
    state_value_cache: RefCell<StateValueCache>,
    prop_scratch: RefCell<Vec<ExplicitValueIndex>>,
    numeric_scratch: RefCell<Vec<NumericValue>>,
}

impl<'task> GreedyNumericPdbHeuristic<'task> {
    pub fn new(
        task: &'task dyn AbstractNumericTask,
        config: GreedyPatternGeneratorConfig,
    ) -> Result<Self, String> {
        validate_restricted_task(task)?;
        let pattern = generate_greedy_pattern(task, config);
        let projected_task = ProjectedTask::new(task, &pattern).map_err(|err| err.to_string())?;
        utils::print_projection_summary(task, &pattern, &projected_task);
        let pdb = PatternDatabase::with_heuristic_config(
            projected_task,
            config.max_pdb_states,
            config.pdb_heuristic_config(),
        )?;

        Ok(Self {
            name: "greedy_numeric_pdb".to_string(),
            pdb,
            state_value_cache: RefCell::new(StateValueCache::default()),
            prop_scratch: RefCell::new(Vec::new()),
            numeric_scratch: RefCell::new(Vec::new()),
        })
    }
}

impl Heuristic for GreedyNumericPdbHeuristic<'_> {
    fn compute_heuristic(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError> {
        let state_id = eval_state.state().get_id();
        if let Some(value) = self.state_value_cache.borrow().get(state_id) {
            return Ok(value);
        }

        if eval_state.is_goal() {
            self.state_value_cache.borrow_mut().insert(state_id, 0.0);
            return Ok(0.0);
        }

        let registry = eval_state.state_registry();
        let mut prop = self.prop_scratch.borrow_mut();
        let mut numeric = self.numeric_scratch.borrow_mut();
        let view = registry.view(eval_state.state());
        view.fill_propositional(&mut prop);
        view.fill_numeric(&mut numeric)
            .map_err(|error| EvaluationError::InvalidState(format!("{error:?}")))?;
        let heuristic_value = self
            .pdb
            .lookup_projected_or_fallback_from_source_state_values(&prop, &numeric)
            .map_err(EvaluationError::ComputationFailed)?;
        let heuristic_value = heuristic_value.max(self.pdb.min_operator_cost());
        self.state_value_cache
            .borrow_mut()
            .insert(state_id, heuristic_value);
        Ok(heuristic_value)
    }

    fn heuristic_name(&self) -> &str {
        &self.name
    }
}
