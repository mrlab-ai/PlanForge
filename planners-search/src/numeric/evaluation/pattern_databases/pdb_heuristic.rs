use std::cell::RefCell;

use crate::numeric::evaluation::evaluator::{EvaluationError, EvaluationState};
use crate::numeric::evaluation::heuristic::Heuristic;

use planners_sas::numeric::numeric_task::AbstractNumericTask;
use planners_sas::numeric::state_registry::StateRegistry;

use super::pattern_database::PatternDatabase;
use super::pattern_generator_greedy::{
    DEFAULT_MAX_PDB_STATES, GreedyPatternGeneratorConfig, generate_greedy_pattern,
};
use super::projected_task::ProjectedTask;
use super::utils;

pub struct GreedyNumericPdbHeuristic<'task> {
    name: String,
    pdb: PatternDatabase<'task>,
    prop_scratch: RefCell<Vec<usize>>,
    numeric_scratch: RefCell<Vec<f64>>,
}

impl<'task> GreedyNumericPdbHeuristic<'task> {
    pub fn new(task: &'task dyn AbstractNumericTask) -> Result<Self, String> {
        let pattern = generate_greedy_pattern(task, GreedyPatternGeneratorConfig::default());
        let projected_task = ProjectedTask::new(task, &pattern).map_err(|err| err.to_string())?;
        utils::print_projection_summary(task, &pattern, &projected_task);
        let pdb = PatternDatabase::new(projected_task, DEFAULT_MAX_PDB_STATES)?;

        Ok(Self {
            name: "greedy_numeric_pdb".to_string(),
            pdb,
            prop_scratch: RefCell::new(Vec::new()),
            numeric_scratch: RefCell::new(Vec::new()),
        })
    }

    fn require_task_and_registry<'s, 't>(
        eval_state: &'s EvaluationState<'s, 't>,
    ) -> Result<(&'t dyn AbstractNumericTask, &'s StateRegistry<'t>), EvaluationError> {
        let task = eval_state.task().ok_or_else(|| {
            EvaluationError::InvalidState(
                "GreedyNumericPdbHeuristic requires task in EvaluationState".to_string(),
            )
        })?;
        let registry = eval_state.state_registry().ok_or_else(|| {
            EvaluationError::InvalidState(
                "GreedyNumericPdbHeuristic requires StateRegistry in EvaluationState".to_string(),
            )
        })?;
        Ok((task, registry))
    }
}

impl Heuristic for GreedyNumericPdbHeuristic<'_> {
    fn compute_heuristic(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError> {
        let (_task, registry) = Self::require_task_and_registry(eval_state)?;

        let mut propositional_values = self.prop_scratch.borrow_mut();
        eval_state
            .state()
            .fill_state(registry, &mut propositional_values);

        let mut numeric_values = self.numeric_scratch.borrow_mut();
        registry
            .fill_numeric_vars(eval_state.state(), &mut numeric_values)
            .map_err(|err| {
                EvaluationError::ComputationFailed(format!("failed to read numeric state: {err:?}"))
            })?;

        let (projected_prop, projected_num) = self
            .pdb
            .project_state_values(&propositional_values, &numeric_values)
            .map_err(EvaluationError::ComputationFailed)?;

        Ok(self.pdb.lookup_or_fallback(&projected_prop, &projected_num))
    }

    fn heuristic_name(&self) -> String {
        self.name.clone()
    }
}
