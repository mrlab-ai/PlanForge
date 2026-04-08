use std::cell::RefCell;
use std::fmt;

use planners_sas::numeric::numeric_task::AbstractNumericTask;
use planners_sas::numeric::state_registry::StateRegistry;
use serde::{Deserialize, Serialize};

use crate::numeric::evaluation::evaluator::{EvaluationError, EvaluationState};
use crate::numeric::evaluation::heuristic::Heuristic;

use super::max_additive_subsets::{compute_additive_vars, compute_max_additive_subsets};
use super::pattern_collection::PatternCollection;
use super::pattern_generator_systematic::{
    SystematicPatternGeneratorConfig, generate_systematic_patterns,
};
use super::pdb_collection::PdbCollection;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
pub struct CanonicalNumericPdbConfig {
    pub max_pdb_states: usize,
    pub max_pattern_size: usize,
    pub only_interesting_patterns: bool,
    pub random_seed: i32,
    pub variable_order_type: super::variable_order_finder::GreedyVariableOrderType,
}

impl Default for CanonicalNumericPdbConfig {
    fn default() -> Self {
        let config = SystematicPatternGeneratorConfig::default();
        Self {
            max_pdb_states: config.max_pdb_states,
            max_pattern_size: config.max_pattern_size,
            only_interesting_patterns: config.only_interesting_patterns,
            random_seed: config.random_seed,
            variable_order_type: config.variable_order_type,
        }
    }
}

impl fmt::Display for CanonicalNumericPdbConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "max_pdb_states={}, max_pattern_size={}, only_interesting_patterns={}, random_seed={}, variable_order_type={}",
            self.max_pdb_states,
            self.max_pattern_size,
            self.only_interesting_patterns,
            self.random_seed,
            self.variable_order_type,
        )
    }
}

impl From<CanonicalNumericPdbConfig> for SystematicPatternGeneratorConfig {
    fn from(config: CanonicalNumericPdbConfig) -> Self {
        Self {
            max_pdb_states: config.max_pdb_states,
            max_pattern_size: config.max_pattern_size,
            only_interesting_patterns: config.only_interesting_patterns,
            random_seed: config.random_seed,
            variable_order_type: config.variable_order_type,
        }
    }
}

pub struct CanonicalPdbCollectionInformation<'task> {
    pdb_collection: PdbCollection<'task>,
    max_additive_subsets: Vec<Vec<usize>>,
}

impl<'task> CanonicalPdbCollectionInformation<'task> {
    pub fn new(
        task: &'task dyn AbstractNumericTask,
        patterns: PatternCollection,
        max_pdb_states: usize,
    ) -> Result<Self, String> {
        let pdb_collection = PdbCollection::new(task, patterns, max_pdb_states)?;
        let are_additive = compute_additive_vars(task);
        let max_additive_subsets =
            compute_max_additive_subsets(pdb_collection.patterns(), &are_additive);
        Ok(Self {
            pdb_collection,
            max_additive_subsets,
        })
    }

    pub fn with_explicit_subsets(
        pdb_collection: PdbCollection<'task>,
        max_additive_subsets: Vec<Vec<usize>>,
    ) -> Self {
        Self {
            pdb_collection,
            max_additive_subsets,
        }
    }

    pub fn pdb_collection(&self) -> &PdbCollection<'task> {
        &self.pdb_collection
    }

    pub fn max_additive_subsets(&self) -> &[Vec<usize>] {
        &self.max_additive_subsets
    }

    pub fn evaluate_projected_state_values(
        &self,
        propositional_values: &[i32],
        numeric_values: &[f64],
    ) -> Result<f64, String> {
        let mut best_value = 0.0_f64;

        for subset in &self.max_additive_subsets {
            let mut subset_value = 0.0;
            for &pdb_id in subset {
                let Some(pdb) = self.pdb_collection.pdb(pdb_id) else {
                    return Err(format!("invalid canonical subset pdb index {pdb_id}"));
                };
                let (projected_prop, projected_num) =
                    pdb.abstract_state_values(propositional_values, numeric_values)?;
                subset_value += pdb.lookup_or_fallback(&projected_prop, &projected_num);
            }
            best_value = best_value.max(subset_value);
        }

        Ok(best_value)
    }
}

pub struct CanonicalNumericPdbHeuristic<'task> {
    name: String,
    task: &'task dyn AbstractNumericTask,
    collection_information: CanonicalPdbCollectionInformation<'task>,
    prop_scratch: RefCell<Vec<i32>>,
    numeric_scratch: RefCell<Vec<f64>>,
}

impl<'task> CanonicalNumericPdbHeuristic<'task> {
    pub fn from_config(
        task: &'task dyn AbstractNumericTask,
        config: CanonicalNumericPdbConfig,
    ) -> Result<Self, String> {
        let generator_config: SystematicPatternGeneratorConfig = config.into();
        let patterns = generate_systematic_patterns(task, generator_config);
        Ok(Self::new(
            task,
            CanonicalPdbCollectionInformation::new(task, patterns, config.max_pdb_states)?,
        ))
    }

    pub fn new(
        task: &'task dyn AbstractNumericTask,
        collection_information: CanonicalPdbCollectionInformation<'task>,
    ) -> Self {
        Self {
            name: "canonical_numeric_pdb".to_string(),
            task,
            collection_information,
            prop_scratch: RefCell::new(Vec::new()),
            numeric_scratch: RefCell::new(Vec::new()),
        }
    }

    fn require_task_and_registry<'s, 't>(
        eval_state: &'s EvaluationState<'s, 't>,
    ) -> Result<(&'t dyn AbstractNumericTask, &'s StateRegistry<'t>), EvaluationError> {
        let task = eval_state.task().ok_or_else(|| {
            EvaluationError::InvalidState(
                "CanonicalNumericPdbHeuristic requires task in EvaluationState".to_string(),
            )
        })?;
        let registry = eval_state.state_registry().ok_or_else(|| {
            EvaluationError::InvalidState(
                "CanonicalNumericPdbHeuristic requires StateRegistry in EvaluationState"
                    .to_string(),
            )
        })?;
        Ok((task, registry))
    }

    fn is_goal_state(&self, propositional_values: &[i32]) -> bool {
        (0..usize::try_from(self.task.get_num_goals().max(0)).unwrap_or(0)).all(|goal_index| {
            let goal = self.task.get_goal_fact(goal_index as i32);
            propositional_values.get(goal.var() as usize).copied() == Some(goal.value())
        })
    }
}

impl Heuristic for CanonicalNumericPdbHeuristic<'_> {
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

        if self.is_goal_state(&propositional_values) {
            return Ok(0.0);
        }

        self.collection_information
            .evaluate_projected_state_values(&propositional_values, &numeric_values)
            .map_err(EvaluationError::ComputationFailed)
    }

    fn heuristic_name(&self) -> String {
        self.name.clone()
    }
}

#[cfg(test)]
mod tests {
    use planners_sas::numeric::numeric_task::{
        Effect, ExplicitVariable, Fact, Metric, NumericRootTask, NumericType, NumericVariable,
        Operator,
    };

    use super::*;
    use crate::numeric::evaluation::pattern_databases::projected_task::Pattern;

    fn simple_var(name: &str) -> ExplicitVariable {
        ExplicitVariable::new(
            2,
            name.to_string(),
            vec![format!("{name}=0"), format!("{name}=1")],
            -1,
            1,
        )
    }

    fn canonical_sample_task() -> NumericRootTask {
        NumericRootTask::new(
            1,
            Metric::new(true, -1),
            vec![simple_var("p"), simple_var("q")],
            vec![NumericVariable::new(
                "x".to_string(),
                NumericType::Regular,
                -1,
            )],
            vec![Fact::new(0, 1), Fact::new(1, 1)],
            vec![],
            vec![0, 0],
            vec![0.0],
            vec![
                Operator::new(
                    "set-p".to_string(),
                    vec![],
                    vec![Effect::new(vec![], 0, 0, 1)],
                    vec![],
                    2,
                ),
                Operator::new(
                    "set-q".to_string(),
                    vec![],
                    vec![Effect::new(vec![], 1, 0, 1)],
                    vec![],
                    3,
                ),
            ],
            vec![],
            vec![],
            vec![],
            (0, 0),
        )
    }

    #[test]
    fn canonical_collection_information_uses_explicit_subsets() {
        let task = canonical_sample_task();
        let patterns = PatternCollection::new(vec![
            Pattern::new(vec![0], vec![]),
            Pattern::new(vec![1], vec![]),
        ]);
        let pdb_collection = PdbCollection::new(&task, patterns, 32).unwrap();
        let collection_information = CanonicalPdbCollectionInformation::with_explicit_subsets(
            pdb_collection,
            vec![vec![0, 1], vec![0], vec![1]],
        );

        let value = collection_information
            .evaluate_projected_state_values(&[0, 0], &[0.0])
            .unwrap();

        assert_eq!(value, 5.0);
    }

    #[test]
    fn canonical_collection_computes_max_additive_subset() {
        let task = canonical_sample_task();
        let patterns = PatternCollection::new(vec![
            Pattern::new(vec![0], vec![]),
            Pattern::new(vec![1], vec![]),
        ]);

        let collection_information =
            CanonicalPdbCollectionInformation::new(&task, patterns, 32).unwrap();

        assert_eq!(collection_information.max_additive_subsets(), &[vec![0, 1]]);
    }
}
