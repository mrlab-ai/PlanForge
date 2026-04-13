use std::fmt;

use planners_sas::numeric::numeric_task::AbstractNumericTask;
use serde::{Deserialize, Serialize};

use super::numeric_size_estimator::NumericSizeEstimator;
use super::pattern_database::PdbHeuristicConfig;
use super::numeric_support::NumericSupportContext;
use super::projected_task::Pattern;
use super::variable_order_finder::{GreedyVariableOrderType, VariableOrderFinder};

pub const DEFAULT_MAX_PDB_STATES: usize = 100_000;
pub const DEFAULT_NUMERIC_FIRST: bool = true;
pub const DEFAULT_RANDOM_SEED: i32 = 0;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
pub struct GreedyPatternGeneratorConfig {
    pub max_pdb_states: usize,
    pub numeric_first: bool,
    pub random_seed: i32,
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
            numeric_size_estimator.estimate_domain_size(next_var_id).max(1)
        } else {
            task.get_variable_domain_size(next_var_id as i32)
                .unwrap_or(1)
                .max(1) as usize
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
    println!(
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
        assert!(var_id < task.variables().len(), "regular variable number too low/high in pattern");
    }
    if let Some(&var_id) = pattern.regular.last() {
        assert!(var_id < task.variables().len(), "regular variable number too high in pattern");
    }

    let helper_space_len = numeric_support.helper_space_len(task);
    if let Some(&var_id) = pattern.numeric.first() {
        assert!(var_id < helper_space_len, "numeric variable number too low/high in pattern");
    }
    if let Some(&var_id) = pattern.numeric.last() {
        assert!(var_id < helper_space_len, "numeric variable number too high in pattern");
    }
}

#[cfg(test)]
mod tests {
    use planners_sas::numeric::axioms::{AssignmentAxiom, CalOperator, ComparisonAxiom, ComparisonOperator};
    use planners_sas::numeric::numeric_task::{ExplicitVariable, Fact, Metric, NumericRootTask, NumericType, NumericVariable};

    use super::*;

    fn simple_var(name: &str, axiom_layer: i32) -> ExplicitVariable {
        ExplicitVariable::new(
            2,
            name.to_string(),
            vec![format!("{name}=0"), format!("{name}=1")],
            axiom_layer,
            1,
        )
    }

    fn numeric_goal_task() -> NumericRootTask {
        NumericRootTask::new(
            1,
            Metric::new(true, -1),
            vec![simple_var("cmp", 0)],
            vec![
                NumericVariable::new("threshold".to_string(), NumericType::Constant, -1),
                NumericVariable::new("x".to_string(), NumericType::Regular, -1),
            ],
            vec![Fact::new(0, 1)],
            vec![],
            vec![0],
            vec![1.0, 0.0],
            vec![],
            vec![],
            vec![ComparisonAxiom::new(0, 1, 0, ComparisonOperator::GreaterThanOrEqual)],
            vec![],
            (0, 0),
        )
    }

    #[test]
    fn greedy_pattern_config_defaults_match_fd_defaults() {
        let config = GreedyPatternGeneratorConfig::default();
        assert_eq!(config.max_pdb_states, 100_000);
        assert!(config.numeric_first);
        assert_eq!(config.random_seed, 0);
        assert_eq!(config.variable_order_type, GreedyVariableOrderType::GoalCgLevel);
    }

    #[test]
    fn greedy_pattern_uses_fd_goal_ordering() {
        let task = numeric_goal_task();
        let pattern = generate_greedy_pattern(&task, GreedyPatternGeneratorConfig::default());
        assert_eq!(pattern.numeric.first().copied(), Some(1));
    }

    #[test]
    fn greedy_pattern_respects_budget_like_fd() {
        let task = numeric_goal_task();
        let pattern = generate_greedy_pattern(
            &task,
            GreedyPatternGeneratorConfig {
                max_pdb_states: 0,
                ..GreedyPatternGeneratorConfig::default()
            },
        );
        assert!(pattern.regular.is_empty());
        assert!(pattern.numeric.is_empty());
    }
}
