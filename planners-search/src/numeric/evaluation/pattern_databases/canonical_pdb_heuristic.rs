#[cfg(test)]
mod tests;

use std::cell::RefCell;
use std::fmt;

use planners_sas::numeric::numeric_task::AbstractNumericTask;
use planners_sas::numeric::state_registry::{ConcreteState, StateID, StateRegistry};
use serde::{Deserialize, Serialize};

use crate::numeric::evaluation::evaluator::{EvaluationError, EvaluationState};
use crate::numeric::evaluation::heuristic::Heuristic;

use super::max_additive_subsets::{compute_additive_vars, compute_max_additive_subsets};
use super::pattern_collection::PatternCollection;
use super::pattern_database::{PdbHeuristicConfig, PdbInternalHeuristic};
use super::pattern_generator_systematic::{
    SystematicPatternGeneratorConfig, generate_systematic_patterns,
};
use super::pdb_collection::PdbCollection;

#[derive(Default)]
pub(crate) struct PdbValueCache {
    values: Vec<f64>,
    generations: Vec<u32>,
    current_generation: u32,
}

impl PdbValueCache {
    fn begin_evaluation(&mut self, len: usize) {
        if self.values.len() != len {
            self.values.resize(len, 0.0);
            self.generations.resize(len, 0);
            self.current_generation = 1;
            return;
        }

        if self.current_generation == u32::MAX {
            self.generations.fill(0);
            self.current_generation = 1;
        } else {
            self.current_generation += 1;
        }
    }

    fn get(&self, index: usize) -> Option<f64> {
        (self.generations.get(index).copied() == Some(self.current_generation))
            .then(|| self.values[index])
    }

    fn insert(&mut self, index: usize, value: f64) -> Result<(), String> {
        let Some(slot) = self.values.get_mut(index) else {
            return Err(format!("invalid canonical subset pdb cache index {index}"));
        };
        let Some(generation) = self.generations.get_mut(index) else {
            return Err(format!("invalid canonical subset pdb cache generation index {index}"));
        };
        *slot = value;
        *generation = self.current_generation;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
pub struct CanonicalNumericPdbConfig {
    pub max_pdb_states: usize,
    pub max_pattern_size: usize,
    pub only_interesting_patterns: bool,
    pub exploration_heuristic: PdbInternalHeuristic,
    pub frontier_heuristic: PdbInternalHeuristic,
    pub failed_lookup_heuristic: PdbInternalHeuristic,
}

impl Default for CanonicalNumericPdbConfig {
    fn default() -> Self {
        let config = SystematicPatternGeneratorConfig::default();
        Self {
            max_pdb_states: config.max_pdb_states,
            max_pattern_size: config.max_pattern_size,
            only_interesting_patterns: config.only_interesting_patterns,
            exploration_heuristic: Default::default(),
            frontier_heuristic: Default::default(),
            failed_lookup_heuristic: Default::default(),
        }
    }
}

impl fmt::Display for CanonicalNumericPdbConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "max_pdb_states={}, max_pattern_size={}, only_interesting_patterns={}, exploration_heuristic={}, frontier_heuristic={}, failed_lookup_heuristic={}",
            self.max_pdb_states,
            self.max_pattern_size,
            self.only_interesting_patterns,
            self.exploration_heuristic,
            self.frontier_heuristic,
            self.failed_lookup_heuristic,
        )
    }
}

impl From<CanonicalNumericPdbConfig> for SystematicPatternGeneratorConfig {
    fn from(config: CanonicalNumericPdbConfig) -> Self {
        Self {
            max_pdb_states: config.max_pdb_states,
            max_pattern_size: config.max_pattern_size,
            only_interesting_patterns: config.only_interesting_patterns,
        }
    }
}

impl CanonicalNumericPdbConfig {
    pub fn pdb_heuristic_config(&self) -> PdbHeuristicConfig {
        PdbHeuristicConfig {
            exploration_heuristic: self.exploration_heuristic,
            frontier_heuristic: self.frontier_heuristic,
            failed_lookup_heuristic: self.failed_lookup_heuristic,
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
        heuristic_config: PdbHeuristicConfig,
    ) -> Result<Self, String> {
        let pdb_collection =
            PdbCollection::with_heuristic_config(task, patterns, max_pdb_states, heuristic_config)?;
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

    pub fn requires_derived_numeric_values(&self) -> bool {
        self.pdb_collection.requires_derived_numeric_values()
    }

    pub fn supports_direct_concrete_state_projection(&self) -> bool {
        self.pdb_collection.supports_direct_concrete_state_projection()
    }

    fn prepare_pdb_value_cache(&self, pdb_value_cache: &mut PdbValueCache) {
        pdb_value_cache.begin_evaluation(self.pdb_collection.len());
    }

    pub fn expand_numeric_state_values_into(
        &self,
        numeric_values: &[f64],
        expanded_numeric_values: &mut Vec<f64>,
    ) -> Result<(), String> {
        self.pdb_collection
            .expand_numeric_state_values_into(numeric_values, expanded_numeric_values)
    }

    pub(crate) fn evaluate_projected_state_values(
        &self,
        propositional_values: &[usize],
        expanded_numeric_values: &[f64],
        pdb_value_cache: &mut PdbValueCache,
    ) -> Result<f64, String> {
        self.prepare_pdb_value_cache(pdb_value_cache);

        let mut best_value = 0.0_f64;

        for subset in &self.max_additive_subsets {
            let mut subset_value = 0.0;
            for &pdb_id in subset {
                let value = if let Some(cached_value) = pdb_value_cache.get(pdb_id) {
                    cached_value
                } else {
                    let Some(pdb) = self.pdb_collection.pdb(pdb_id) else {
                        return Err(format!("invalid canonical subset pdb index {pdb_id}"));
                    };
                    let value = pdb.lookup_projected_or_fallback_from_expanded_state_values(
                        propositional_values,
                        expanded_numeric_values,
                    )?;
                    pdb_value_cache.insert(pdb_id, value)?;
                    value
                };
                if value.is_infinite() {
                    return Ok(f64::INFINITY);
                }
                subset_value += value;
            }
            best_value = best_value.max(subset_value);
        }

        Ok(best_value)
    }

    pub(crate) fn evaluate_concrete_state(
        &self,
        state: &ConcreteState,
        registry: &StateRegistry<'_>,
        pdb_value_cache: &mut PdbValueCache,
    ) -> Result<f64, String> {
        self.prepare_pdb_value_cache(pdb_value_cache);

        let mut best_value = 0.0_f64;

        for subset in &self.max_additive_subsets {
            let mut subset_value = 0.0;
            for &pdb_id in subset {
                let value = if let Some(cached_value) = pdb_value_cache.get(pdb_id) {
                    cached_value
                } else {
                    let Some(pdb) = self.pdb_collection.pdb(pdb_id) else {
                        return Err(format!("invalid canonical subset pdb index {pdb_id}"));
                    };
                    let value = pdb.lookup_or_fallback_from_concrete_state(state, registry)?;
                    pdb_value_cache.insert(pdb_id, value)?;
                    value
                };
                if value.is_infinite() {
                    return Ok(f64::INFINITY);
                }
                subset_value += value;
            }
            best_value = best_value.max(subset_value);
        }

        Ok(best_value)
    }
}

#[allow(unused)]
pub struct CanonicalNumericPdbHeuristic<'task> {
    name: String,
    task: &'task dyn AbstractNumericTask,
    collection_information: CanonicalPdbCollectionInformation<'task>,
    prop_scratch: RefCell<Vec<usize>>,
    numeric_scratch: RefCell<Vec<f64>>,
    expanded_numeric_scratch: RefCell<Vec<f64>>,
    pdb_value_cache: RefCell<PdbValueCache>,
    state_value_cache: RefCell<Vec<f64>>,
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
            CanonicalPdbCollectionInformation::new(
                task,
                patterns,
                config.max_pdb_states,
                config.pdb_heuristic_config(),
            )?,
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
            expanded_numeric_scratch: RefCell::new(Vec::new()),
            pdb_value_cache: RefCell::new(PdbValueCache::default()),
            state_value_cache: RefCell::new(Vec::new()),
        }
    }

    fn cached_state_value(&self, state_id: StateID) -> Option<f64> {
        self.state_value_cache
            .borrow()
            .get(state_id)
            .copied()
            .filter(|value| !value.is_nan())
    }

    fn cache_state_value(&self, state_id: StateID, value: f64) {
        let mut cache = self.state_value_cache.borrow_mut();
        if cache.len() <= state_id {
            cache.resize(state_id + 1, f64::NAN);
        }
        cache[state_id] = value;
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

    #[allow(unused)]
    fn is_goal_state(&self, propositional_values: &[usize]) -> bool {
        (0..self.task.get_num_goals()).all(|goal_index| {
            let goal = self.task.get_goal_fact(goal_index);
            propositional_values.get(goal.var).copied() == Some(goal.value)
        })
    }
}

impl Heuristic for CanonicalNumericPdbHeuristic<'_> {
    fn compute_heuristic(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError> {
        let state_id = eval_state.state().get_id();
        if let Some(value) = self.cached_state_value(state_id) {
            return Ok(value);
        }

        let (_task, registry) = Self::require_task_and_registry(eval_state)?;
        let mut pdb_value_cache = self.pdb_value_cache.borrow_mut();
        let heuristic_value = if self
            .collection_information
            .supports_direct_concrete_state_projection()
        {
            self.collection_information
                .evaluate_concrete_state(eval_state.state(), registry, &mut pdb_value_cache)
                .map_err(EvaluationError::ComputationFailed)?
        } else {
            let mut propositional_values = self.prop_scratch.borrow_mut();
            let mut numeric_values = self.numeric_scratch.borrow_mut();
            registry
                .fill_state_and_numeric_vars_with_options(
                    eval_state.state(),
                    &mut propositional_values,
                    &mut numeric_values,
                    self.collection_information
                        .requires_derived_numeric_values(),
                )
                .map_err(|err| EvaluationError::InvalidState(format!("{err:?}")))?;

            let mut expanded_numeric_values = self.expanded_numeric_scratch.borrow_mut();
            self.collection_information
                .expand_numeric_state_values_into(&numeric_values, &mut expanded_numeric_values)
                .map_err(EvaluationError::ComputationFailed)?;

            self.collection_information
                .evaluate_projected_state_values(
                    &propositional_values,
                    &expanded_numeric_values,
                    &mut pdb_value_cache,
                )
                .map_err(EvaluationError::ComputationFailed)?
        };

        self.cache_state_value(state_id, heuristic_value);
        Ok(heuristic_value)
    }

    fn heuristic_name(&self) -> String {
        self.name.clone()
    }
}
