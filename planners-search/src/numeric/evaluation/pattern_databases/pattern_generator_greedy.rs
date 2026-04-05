use std::collections::BTreeSet;
use std::fmt;

use planners_sas::numeric::numeric_task::{AbstractNumericTask, NumericType};
use serde::{Deserialize, Serialize};

use crate::numeric::evaluation::domain_abstractions::comparison_expression::ComparisonTree;

use super::causal_graph::{CausalGraphVariable, MixedCausalGraph};
use super::numeric_size_estimator::NumericSizeEstimator;
use super::projected_task::Pattern;
use super::variable_order_finder::{
    GreedyVariableOrderType, order_causal_graph_variables, order_variable_ids,
};

pub const DEFAULT_MAX_PDB_STATES: usize = 100_000;
pub const DEFAULT_NUMERIC_FIRST: bool = true;
pub const DEFAULT_RANDOM_SEED: i32 = 0;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
pub struct GreedyPatternGeneratorConfig {
    pub max_pdb_states: usize,
    pub numeric_first: bool,
    pub random_seed: i32,
    pub variable_order_type: GreedyVariableOrderType,
}

impl Default for GreedyPatternGeneratorConfig {
    fn default() -> Self {
        Self {
            max_pdb_states: DEFAULT_MAX_PDB_STATES,
            numeric_first: DEFAULT_NUMERIC_FIRST,
            random_seed: DEFAULT_RANDOM_SEED,
            variable_order_type: GreedyVariableOrderType::default(),
        }
    }
}

impl fmt::Display for GreedyPatternGeneratorConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "max_pdb_states={}, numeric_first={}, random_seed={}, variable_order_type={}",
            self.max_pdb_states, self.numeric_first, self.random_seed, self.variable_order_type
        )
    }
}

pub fn generate_greedy_pattern(
    task: &dyn AbstractNumericTask,
    config: GreedyPatternGeneratorConfig,
) -> Pattern {
    let (goal_regular, goal_numeric, true_goal_regular) = collect_goal_variables(task);
    let causal_graph = MixedCausalGraph::new(task);
    let numeric_size_estimator = NumericSizeEstimator::new(task);

    let mut pattern = Pattern {
        regular: Vec::new(),
        numeric: Vec::new(),
    };
    let mut size = 1usize;

    let goal_regular_ids = ordered_regular_ids(
        goal_regular.iter().copied().collect(),
        &causal_graph,
        config,
    );
    let true_goal_regular_ids = ordered_regular_ids(
        true_goal_regular.iter().copied().collect(),
        &causal_graph,
        config,
    );
    let goal_numeric_ids = ordered_numeric_ids(
        goal_numeric.iter().copied().collect(),
        &causal_graph,
        config,
    );

    if config.numeric_first {
        add_numeric_variables(
            task,
            &goal_numeric_ids,
            &numeric_size_estimator,
            &mut pattern,
            &mut size,
            &config,
        );
        add_regular_variables(task, &goal_regular_ids, &mut pattern, &mut size, &config);
    } else {
        add_regular_variables(task, &goal_regular_ids, &mut pattern, &mut size, &config);
        add_numeric_variables(
            task,
            &goal_numeric_ids,
            &numeric_size_estimator,
            &mut pattern,
            &mut size,
            &config,
        );
    }

    if pattern.regular.is_empty() {
        for &var_id in &true_goal_regular_ids {
            let domain_size = task
                .get_variable_domain_size(var_id as i32)
                .unwrap_or(1)
                .max(1) as usize;
            if size.saturating_mul(domain_size) > config.max_pdb_states {
                break;
            }
            pattern.regular.push(var_id);
            size *= domain_size;
            break;
        }
    }

    expand_pattern_with_predecessors(
        task,
        &causal_graph,
        &numeric_size_estimator,
        &mut pattern,
        &mut size,
        &config,
    );

    pattern
}

fn ordered_ids(mut ids: Vec<usize>, config: GreedyPatternGeneratorConfig) -> Vec<usize> {
    order_variable_ids(&mut ids, config.variable_order_type, config.random_seed);
    ids
}

fn ordered_regular_ids(
    ids: Vec<usize>,
    graph: &MixedCausalGraph,
    config: GreedyPatternGeneratorConfig,
) -> Vec<usize> {
    let mut variables: Vec<CausalGraphVariable> =
        ids.into_iter().map(CausalGraphVariable::Regular).collect();
    order_causal_graph_variables(
        &mut variables,
        graph,
        config.variable_order_type,
        config.random_seed,
    );
    variables
        .into_iter()
        .filter_map(|variable| match variable {
            CausalGraphVariable::Regular(var_id) => Some(var_id),
            CausalGraphVariable::Numeric(_) => None,
        })
        .collect()
}

fn ordered_numeric_ids(
    ids: Vec<usize>,
    graph: &MixedCausalGraph,
    config: GreedyPatternGeneratorConfig,
) -> Vec<usize> {
    let mut variables: Vec<CausalGraphVariable> =
        ids.into_iter().map(CausalGraphVariable::Numeric).collect();
    order_causal_graph_variables(
        &mut variables,
        graph,
        config.variable_order_type,
        config.random_seed,
    );
    variables
        .into_iter()
        .filter_map(|variable| match variable {
            CausalGraphVariable::Numeric(var_id) => Some(var_id),
            CausalGraphVariable::Regular(_) => None,
        })
        .collect()
}

fn expand_pattern_with_predecessors(
    task: &dyn AbstractNumericTask,
    graph: &MixedCausalGraph,
    numeric_size_estimator: &NumericSizeEstimator,
    pattern: &mut Pattern,
    size: &mut usize,
    config: &GreedyPatternGeneratorConfig,
) {
    let mut frontier: Vec<CausalGraphVariable> = pattern
        .regular
        .iter()
        .copied()
        .map(CausalGraphVariable::Regular)
        .chain(
            pattern
                .numeric
                .iter()
                .copied()
                .map(CausalGraphVariable::Numeric),
        )
        .collect();

    let mut expanded = BTreeSet::new();
    while !frontier.is_empty() {
        let mut predecessor_candidates: BTreeSet<CausalGraphVariable> = BTreeSet::new();
        for variable in frontier.drain(..) {
            if !expanded.insert(variable) {
                continue;
            }
            for predecessor in graph.predecessors_of(variable) {
                if pattern_contains(pattern, predecessor) {
                    continue;
                }
                predecessor_candidates.insert(predecessor);
            }
        }

        if predecessor_candidates.is_empty() {
            break;
        }

        let mut ordered_candidates: Vec<_> = predecessor_candidates.into_iter().collect();
        order_causal_graph_variables(
            &mut ordered_candidates,
            graph,
            config.variable_order_type,
            config.random_seed,
        );

        let mut next_frontier = Vec::new();
        if config.numeric_first {
            add_ordered_predecessors(
                task,
                &ordered_candidates,
                numeric_size_estimator,
                pattern,
                size,
                config,
                true,
                &mut next_frontier,
            );
            add_ordered_predecessors(
                task,
                &ordered_candidates,
                numeric_size_estimator,
                pattern,
                size,
                config,
                false,
                &mut next_frontier,
            );
        } else {
            add_ordered_predecessors(
                task,
                &ordered_candidates,
                numeric_size_estimator,
                pattern,
                size,
                config,
                false,
                &mut next_frontier,
            );
            add_ordered_predecessors(
                task,
                &ordered_candidates,
                numeric_size_estimator,
                pattern,
                size,
                config,
                true,
                &mut next_frontier,
            );
        }

        if next_frontier.is_empty() {
            break;
        }
        frontier = next_frontier;
    }
}

fn add_ordered_predecessors(
    task: &dyn AbstractNumericTask,
    ordered_candidates: &[CausalGraphVariable],
    numeric_size_estimator: &NumericSizeEstimator,
    pattern: &mut Pattern,
    size: &mut usize,
    config: &GreedyPatternGeneratorConfig,
    numeric: bool,
    next_frontier: &mut Vec<CausalGraphVariable>,
) {
    for &candidate in ordered_candidates {
        match candidate {
            CausalGraphVariable::Regular(var_id) if !numeric => {
                if pattern.regular.contains(&var_id)
                    || task.get_variable_axiom_layer(var_id as i32).unwrap_or(-1) != -1
                {
                    continue;
                }
                let domain_size = task
                    .get_variable_domain_size(var_id as i32)
                    .unwrap_or(1)
                    .max(1) as usize;
                if size.saturating_mul(domain_size) > config.max_pdb_states {
                    continue;
                }
                pattern.regular.push(var_id);
                *size *= domain_size;
                next_frontier.push(candidate);
            }
            CausalGraphVariable::Numeric(var_id) if numeric => {
                if pattern.numeric.contains(&var_id)
                    || task.numeric_variables()[var_id].get_type() != &NumericType::Regular
                {
                    continue;
                }
                let domain_size = numeric_size_estimator.estimate_domain_size(var_id);
                if size.saturating_mul(domain_size) > config.max_pdb_states {
                    continue;
                }
                pattern.numeric.push(var_id);
                *size *= domain_size;
                next_frontier.push(candidate);
            }
            _ => {}
        }
    }
}

fn pattern_contains(pattern: &Pattern, variable: CausalGraphVariable) -> bool {
    match variable {
        CausalGraphVariable::Regular(var_id) => pattern.regular.contains(&var_id),
        CausalGraphVariable::Numeric(var_id) => pattern.numeric.contains(&var_id),
    }
}

fn add_regular_variables(
    task: &dyn AbstractNumericTask,
    variable_ids: &[usize],
    pattern: &mut Pattern,
    size: &mut usize,
    config: &GreedyPatternGeneratorConfig,
) {
    for &var_id in variable_ids {
        if pattern.regular.contains(&var_id) {
            continue;
        }
        let domain_size = task
            .get_variable_domain_size(var_id as i32)
            .unwrap_or(1)
            .max(1) as usize;
        if size.saturating_mul(domain_size) > config.max_pdb_states {
            break;
        }
        pattern.regular.push(var_id);
        *size *= domain_size;
    }
}

fn add_numeric_variables(
    task: &dyn AbstractNumericTask,
    numeric_var_ids: &[usize],
    numeric_size_estimator: &NumericSizeEstimator,
    pattern: &mut Pattern,
    size: &mut usize,
    config: &GreedyPatternGeneratorConfig,
) {
    for &numeric_var_id in numeric_var_ids {
        if pattern.numeric.contains(&numeric_var_id)
            || task.numeric_variables()[numeric_var_id].get_type() != &NumericType::Regular
        {
            continue;
        }
        let domain_size = numeric_size_estimator.estimate_domain_size(numeric_var_id);
        if size.saturating_mul(domain_size) > config.max_pdb_states {
            break;
        }
        pattern.numeric.push(numeric_var_id);
        *size *= domain_size;
    }
}

fn collect_goal_variables(
    task: &dyn AbstractNumericTask,
) -> (BTreeSet<usize>, BTreeSet<usize>, BTreeSet<usize>) {
    let mut regular = BTreeSet::new();
    let mut numeric = BTreeSet::new();
    let goal_related_propositional_vars = collect_goal_related_propositional_closure(task);
    let mut true_goal_regular: BTreeSet<usize> = goal_related_propositional_vars
        .iter()
        .copied()
        .filter(|&var_id| task.get_variable_axiom_layer(var_id as i32).unwrap_or(-1) == -1)
        .collect();

    for goal_index in 0..usize::try_from(task.get_num_goals().max(0)).unwrap_or(0) {
        let goal = task.get_goal_fact(goal_index as i32);
        let goal_var_id = goal.var() as usize;
        if task
            .get_variable_axiom_layer(goal_var_id as i32)
            .unwrap_or(-1)
            == -1
        {
            regular.insert(goal_var_id);
            true_goal_regular.insert(goal_var_id);
        }
    }

    for (comparison_axiom_id, comparison_axiom) in task.comparison_axioms().iter().enumerate() {
        let Some(affected_var_id) = usize::try_from(comparison_axiom.get_affected_var_id()).ok()
        else {
            continue;
        };
        if !goal_related_propositional_vars.contains(&affected_var_id) {
            continue;
        }

        if let Ok(tree) = ComparisonTree::from_task(task, comparison_axiom_id) {
            for numeric_var_id in tree.regular_numeric_var_dependencies(task) {
                let Ok(numeric_var_id) = usize::try_from(numeric_var_id) else {
                    continue;
                };
                if task
                    .numeric_variables()
                    .get(numeric_var_id)
                    .map(|numeric_var| numeric_var.get_type() == &NumericType::Regular)
                    .unwrap_or(false)
                {
                    numeric.insert(numeric_var_id);
                }
            }
        } else {
            for numeric_var_id in [
                comparison_axiom.get_left_var_id(),
                comparison_axiom.get_right_var_id(),
            ] {
                let Ok(numeric_var_id) = usize::try_from(numeric_var_id) else {
                    continue;
                };
                if task
                    .numeric_variables()
                    .get(numeric_var_id)
                    .map(|numeric_var| numeric_var.get_type() == &NumericType::Regular)
                    .unwrap_or(false)
                {
                    numeric.insert(numeric_var_id);
                }
            }
        }
    }

    (regular, numeric, true_goal_regular)
}

fn collect_goal_related_propositional_closure(task: &dyn AbstractNumericTask) -> BTreeSet<usize> {
    let mut goal_related: BTreeSet<usize> = (0..task.get_num_goals())
        .filter_map(|goal_id| usize::try_from(task.get_goal_fact(goal_id).var()).ok())
        .collect();

    loop {
        let mut changed = false;

        for axiom in task.axioms() {
            let affected_var_id = axiom.var_id() as usize;
            if goal_related.contains(&affected_var_id) {
                for condition in axiom.conditions() {
                    if let Ok(condition_var_id) = usize::try_from(condition.var()) {
                        changed |= goal_related.insert(condition_var_id);
                    }
                }
            }
        }

        if !changed {
            break;
        }
    }

    goal_related
}

fn collect_goal_related_propositional_vars(task: &dyn AbstractNumericTask) -> BTreeSet<usize> {
    collect_goal_related_propositional_closure(task)
        .into_iter()
        .filter(|&var_id| task.get_variable_axiom_layer(var_id as i32).unwrap_or(-1) == -1)
        .collect()
}

#[cfg(test)]
mod tests {
    use planners_sas::numeric::axioms::{AssignmentAxiom, CalOperator};
    use planners_sas::numeric::axioms::{ComparisonAxiom, ComparisonOperator, PropositionalAxiom};
    use planners_sas::numeric::numeric_task::{
        AssignmentEffect, AssignmentOperation, ExplicitVariable, Fact, Metric, NumericRootTask,
        NumericType, NumericVariable, Operator,
    };

    use super::*;
    use crate::numeric::evaluation::pattern_databases::variable_order_finder::GreedyVariableOrderType;

    fn simple_var(name: &str, axiom_layer: i32) -> ExplicitVariable {
        ExplicitVariable::new(
            2,
            name.to_string(),
            vec![format!("{name}=0"), format!("{name}=1")],
            axiom_layer,
            1,
        )
    }

    fn sample_task() -> NumericRootTask {
        NumericRootTask::new(
            1,
            Metric::new(true, -1),
            vec![
                simple_var("p", -1),
                ExplicitVariable::new(
                    3,
                    "cmp".to_string(),
                    vec!["t".to_string(), "f".to_string(), "u".to_string()],
                    0,
                    2,
                ),
            ],
            vec![
                NumericVariable::new("c".to_string(), NumericType::Constant, -1),
                NumericVariable::new("x".to_string(), NumericType::Regular, -1),
            ],
            vec![Fact::new(1, 0)],
            vec![],
            vec![0, 2],
            vec![1.0, 0.0],
            vec![Operator::new(
                "inc".to_string(),
                vec![],
                vec![],
                vec![AssignmentEffect::new(
                    1,
                    AssignmentOperation::Plus,
                    0,
                    false,
                    vec![],
                )],
                1,
            )],
            vec![PropositionalAxiom::new(vec![Fact::new(1, 0)], 0, 1, 0)],
            vec![ComparisonAxiom::new(
                1,
                1,
                0,
                ComparisonOperator::GreaterThanOrEqual,
            )],
            vec![],
            (0, 0),
        )
    }

    fn operator_predecessor_task() -> NumericRootTask {
        NumericRootTask::new(
            1,
            Metric::new(true, -1),
            vec![
                simple_var("pre", -1),
                simple_var("goal", -1),
                simple_var("other", -1),
            ],
            vec![],
            vec![Fact::new(1, 1)],
            vec![],
            vec![0, 0, 0],
            vec![],
            vec![Operator::new(
                "achieve-goal".to_string(),
                vec![Fact::new(0, 1)],
                vec![planners_sas::numeric::numeric_task::Effect::new(
                    vec![],
                    1,
                    0,
                    1,
                )],
                vec![],
                1,
            )],
            vec![],
            vec![],
            vec![],
            (0, 0),
        )
    }

    #[test]
    fn greedy_pattern_prefers_goal_variables() {
        let task = sample_task();
        let pattern = generate_greedy_pattern(&task, GreedyPatternGeneratorConfig::default());

        assert!(pattern.numeric.contains(&1));
    }

    #[test]
    fn greedy_pattern_config_defaults_match_expected_port_defaults() {
        let config = GreedyPatternGeneratorConfig::default();

        assert_eq!(config.max_pdb_states, 100_000);
        assert!(config.numeric_first);
        assert_eq!(config.random_seed, 0);
        assert_eq!(
            config.variable_order_type,
            GreedyVariableOrderType::CgGoalLevel
        );
    }

    #[test]
    fn greedy_pattern_includes_true_goal_support_var() {
        let task = NumericRootTask::new(
            1,
            Metric::new(true, -1),
            vec![
                simple_var("support", -1),
                ExplicitVariable::new(
                    2,
                    "goal".to_string(),
                    vec!["off".to_string(), "on".to_string()],
                    1,
                    0,
                ),
            ],
            vec![],
            vec![Fact::new(1, 1)],
            vec![],
            vec![0, 0],
            vec![],
            vec![],
            vec![PropositionalAxiom::new(vec![Fact::new(0, 1)], 1, 0, 1)],
            vec![],
            vec![],
            (0, 0),
        );

        let pattern = generate_greedy_pattern(&task, GreedyPatternGeneratorConfig::default());

        assert!(pattern.regular.contains(&0));
    }

    #[test]
    fn greedy_pattern_expands_via_causal_predecessors_not_all_variables() {
        let task = operator_predecessor_task();
        let pattern = generate_greedy_pattern(
            &task,
            GreedyPatternGeneratorConfig {
                max_pdb_states: 32,
                ..GreedyPatternGeneratorConfig::default()
            },
        );

        assert!(pattern.regular.contains(&1));
        assert!(pattern.regular.contains(&0));
        assert!(!pattern.regular.contains(&2));
    }

    #[test]
    fn greedy_pattern_respects_estimated_numeric_domain_size_budget() {
        let task = sample_task();
        let pattern = generate_greedy_pattern(
            &task,
            GreedyPatternGeneratorConfig {
                max_pdb_states: 2,
                ..GreedyPatternGeneratorConfig::default()
            },
        );

        assert!(!pattern.numeric.contains(&1));
    }

    #[test]
    fn greedy_pattern_collects_regular_numeric_dependencies_from_comparison_trees() {
        let task = NumericRootTask::new(
            1,
            Metric::new(true, -1),
            vec![ExplicitVariable::new(
                2,
                "goal".to_string(),
                vec!["off".to_string(), "on".to_string()],
                0,
                0,
            )],
            vec![
                NumericVariable::new("c5".to_string(), NumericType::Constant, -1),
                NumericVariable::new("x".to_string(), NumericType::Regular, -1),
                NumericVariable::new("y".to_string(), NumericType::Regular, -1),
                NumericVariable::new("sum".to_string(), NumericType::Derived, 0),
            ],
            vec![Fact::new(0, 0)],
            vec![],
            vec![0],
            vec![5.0, 0.0, 0.0, 0.0],
            vec![],
            vec![],
            vec![ComparisonAxiom::new(
                0,
                3,
                0,
                ComparisonOperator::GreaterThanOrEqual,
            )],
            vec![AssignmentAxiom::new(3, CalOperator::Sum, 1, 2)],
            (0, 0),
        );

        let pattern = generate_greedy_pattern(&task, GreedyPatternGeneratorConfig::default());

        assert!(pattern.numeric.contains(&1));
        assert!(pattern.numeric.contains(&2));
    }
}
