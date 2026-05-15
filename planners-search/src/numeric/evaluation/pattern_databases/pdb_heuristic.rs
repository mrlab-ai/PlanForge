use std::cell::RefCell;

use crate::numeric::evaluation::evaluator::{EvaluationError, EvaluationState};
use crate::numeric::evaluation::heuristic::Heuristic;

use planners_sas::numeric::numeric_task::AbstractNumericTask;
use planners_sas::numeric::state_registry::{StateID, StateRegistry};

use super::pattern_database::PatternDatabase;
use super::pattern_generator_greedy::{GreedyPatternGeneratorConfig, generate_greedy_pattern};
use super::projected_task::ProjectedTask;
use super::utils;

#[allow(unused)]
pub struct GreedyNumericPdbHeuristic<'task> {
    name: String,
    task: &'task dyn AbstractNumericTask,
    pdb: PatternDatabase<'task>,
    prop_scratch: RefCell<Vec<usize>>,
    numeric_scratch: RefCell<Vec<f64>>,
    expanded_numeric_scratch: RefCell<Vec<f64>>,
    state_value_cache: RefCell<Vec<Option<f64>>>,
}

impl<'task> GreedyNumericPdbHeuristic<'task> {
    pub fn new(
        task: &'task dyn AbstractNumericTask,
        config: GreedyPatternGeneratorConfig,
    ) -> Result<Self, String> {
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
            task,
            prop_scratch: RefCell::new(Vec::new()),
            numeric_scratch: RefCell::new(Vec::new()),
            expanded_numeric_scratch: RefCell::new(Vec::new()),
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

    #[allow(unused)]
    fn is_goal_state(&self, propositional_values: &[usize]) -> bool {
        (0..self.task.get_num_goals()).all(|goal_index| {
            let goal = self.task.get_goal_fact(goal_index);
            propositional_values.get(goal.var()).copied() == Some(goal.value())
        })
    }
}

impl Heuristic for GreedyNumericPdbHeuristic<'_> {
    fn compute_heuristic(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError> {
        let state_id = eval_state.state().get_id();
        if let Some(value) = self.cached_state_value(state_id) {
            return Ok(value);
        }

        if eval_state.is_goal() {
            self.cache_state_value(state_id, 0.0);
            return Ok(0.0);
        }

        let (_task, registry) = Self::require_task_and_registry(eval_state)?;
        let heuristic_value = self
            .pdb
            .lookup_or_fallback_from_concrete_state(eval_state.state(), registry)
            .map_err(EvaluationError::ComputationFailed)?;
        let heuristic_value = heuristic_value.max(self.pdb.min_operator_cost());
        self.cache_state_value(state_id, heuristic_value);
        Ok(heuristic_value)
    }

    fn heuristic_name(&self) -> String {
        self.name.clone()
    }
}

#[cfg(test)]
mod tests {
    use planners_sas::numeric::axioms::{AssignmentAxiom, AxiomEvaluator, CalOperator};
    use planners_sas::numeric::numeric_task::{
        Effect, ExplicitFact, ExplicitVariable, Metric, NumericRootTask, NumericType,
        NumericVariable, Operator,
    };
    use planners_sas::numeric::state_registry::StateRegistry;
    use planners_sas::numeric::utils::int_packer::IntDoublePacker;

    use crate::numeric::evaluation::evaluator::{EvaluationState, Evaluator};
    use crate::numeric::evaluation::pattern_databases::pattern_generator_greedy::GreedyPatternGeneratorConfig;

    use super::GreedyNumericPdbHeuristic;

    fn initial_goal_task() -> NumericRootTask {
        NumericRootTask::new(
            1,
            Metric::new(true, None),
            vec![ExplicitVariable::new(
                2,
                "p".to_string(),
                vec!["p=0".to_string(), "p=1".to_string()],
                None,
                0,
            )],
            vec![NumericVariable::new(
                "x".to_string(),
                NumericType::Regular,
                None,
            )],
            vec![ExplicitFact::new(0, 1)],
            vec![],
            vec![1],
            vec![0.0],
            vec![Operator::new(
                "leave-goal".to_string(),
                vec![ExplicitFact::new(0, 1)],
                vec![Effect::new(vec![], 0, Some(1), 0)],
                vec![],
                1,
            )],
            vec![],
            vec![],
            vec![AssignmentAxiom::new(0, CalOperator::Sum, 0, 0)],
            ExplicitFact::new(0, 0),
        )
    }

    #[test]
    fn greedy_numeric_pdb_returns_zero_for_concrete_goal_state() {
        let task = initial_goal_task();
        let state_packer = IntDoublePacker::from_task(&task);
        let axiom_evaluator = AxiomEvaluator::new(&task, &state_packer);
        let mut state_registry = StateRegistry::new(&task, &state_packer, &axiom_evaluator);
        let initial_state = state_registry.get_initial_state();
        let heuristic = GreedyNumericPdbHeuristic::new(
            &task,
            GreedyPatternGeneratorConfig {
                max_pdb_states: 16,
                ..GreedyPatternGeneratorConfig::default()
            },
        )
        .expect("greedy numeric PDB should build for simple goal task");

        let mut eval_state =
            EvaluationState::new_with_registry(&initial_state, 0.0, false, &task, &state_registry);
        eval_state.set_is_goal(true);
        let value = heuristic
            .evaluate_state(&mut eval_state)
            .expect("goal evaluation should succeed");

        assert_eq!(value, 0.0);
    }
}
