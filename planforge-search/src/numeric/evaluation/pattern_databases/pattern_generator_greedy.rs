#[cfg(test)]
mod tests;

use std::fmt;

use planforge_sas::numeric::numeric_task::AbstractNumericTask;
use serde::{Deserialize, Serialize};
use tracing::info;

use super::numeric_size_estimator::NumericSizeEstimator;
use super::numeric_support::NumericSupportContext;
use super::pattern_database::PdbHeuristicConfig;
use super::projected_task::Pattern;
use super::variable_order_finder::{GreedyVariableOrderType, VariableOrderFinder};

pub const DEFAULT_MAX_PDB_STATES: usize = 100_000;
pub const DEFAULT_NUMERIC_FIRST: bool = true;
pub const DEFAULT_RANDOM_SEED: u64 = 0;

#[derive(
    Debug,
    Clone,
    Copy,
    Deserialize,
    Serialize,
    PartialEq,
    Eq,
    planforge_search::config::ApplyOptions,
)]
pub struct GreedyPatternGeneratorConfig {
    pub max_pdb_states: usize,
    pub numeric_first: bool,
    pub random_seed: u64,
    pub variable_order_type: GreedyVariableOrderType,
    pub exploration_heuristic: super::pattern_database::PdbInternalHeuristic,
    pub frontier_heuristic: super::pattern_database::PdbInternalHeuristic,
    pub failed_lookup_heuristic: super::pattern_database::PdbInternalHeuristic,
}

impl Default for GreedyPatternGeneratorConfig {
    fn default() -> Self {
        Self {
            max_pdb_states: DEFAULT_MAX_PDB_STATES,
            numeric_first: DEFAULT_NUMERIC_FIRST,
            random_seed: DEFAULT_RANDOM_SEED,
            variable_order_type: GreedyVariableOrderType::default(),
            exploration_heuristic: Default::default(),
            frontier_heuristic: Default::default(),
            failed_lookup_heuristic: Default::default(),
        }
    }
}

impl fmt::Display for GreedyPatternGeneratorConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "max_pdb_states={}, numeric_first={}, random_seed={}, variable_order_type={}, exploration_heuristic={}, frontier_heuristic={}, failed_lookup_heuristic={}",
            self.max_pdb_states,
            self.numeric_first,
            self.random_seed,
            self.variable_order_type,
            self.exploration_heuristic,
            self.frontier_heuristic,
            self.failed_lookup_heuristic,
        )
    }
}

impl GreedyPatternGeneratorConfig {
    pub fn pdb_heuristic_config(&self) -> PdbHeuristicConfig {
        PdbHeuristicConfig {
            exploration_heuristic: self.exploration_heuristic,
            frontier_heuristic: self.frontier_heuristic,
            failed_lookup_heuristic: self.failed_lookup_heuristic,
        }
    }
}

pub fn generate_greedy_pattern(
    task: &dyn AbstractNumericTask,
    config: GreedyPatternGeneratorConfig,
) -> Pattern {
    let numeric_support = NumericSupportContext::new(task);
    let numeric_size_estimator = NumericSizeEstimator::new(task);
    let mut order = VariableOrderFinder::new(
        task,
        &numeric_support,
        config.variable_order_type,
        config.numeric_first,
        config.random_seed,
    );

    let mut pattern = Pattern::new(Vec::new(), Vec::new());
    let mut size = 1usize;

    while !order.done() {
        let Some((next_var_id, is_numeric)) = order.next() else {
            break;
        };

        let next_var_size = if is_numeric {
            numeric_size_estimator
                .estimate_domain_size(next_var_id)
                .max(1)
        } else {
            task.get_variable_domain_size(next_var_id)
                .unwrap_or(1)
                .max(1)
        };

        if size.saturating_mul(next_var_size) > config.max_pdb_states {
            break;
        }

        if is_numeric {
            pattern.numeric.push(next_var_id);
        } else {
            pattern.regular.push(next_var_id);
        }
        size *= next_var_size;
    }

    validate_and_normalize_pattern(task, &numeric_support, &mut pattern);
    info!(
        "Greedy pattern: propositional {:?}; numeric {:?}",
        pattern.regular, pattern.numeric
    );

    pattern
}

fn validate_and_normalize_pattern(
    task: &dyn AbstractNumericTask,
    numeric_support: &NumericSupportContext,
    pattern: &mut Pattern,
) {
    pattern.normalize_in_place();

    if let Some(&var_id) = pattern.regular.first() {
        assert!(
            var_id < task.variables().len(),
            "regular variable number too low/high in pattern"
        );
    }
    if let Some(&var_id) = pattern.regular.last() {
        assert!(
            var_id < task.variables().len(),
            "regular variable number too high in pattern"
        );
    }

    let helper_space_len = numeric_support.helper_space_len(task);
    if let Some(&var_id) = pattern.numeric.first() {
        assert!(
            var_id < helper_space_len,
            "numeric variable number too low/high in pattern"
        );
    }
    if let Some(&var_id) = pattern.numeric.last() {
        assert!(
            var_id < helper_space_len,
            "numeric variable number too high in pattern"
        );
    }
}
