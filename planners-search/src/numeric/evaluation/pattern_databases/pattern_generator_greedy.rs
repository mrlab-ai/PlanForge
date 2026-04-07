#[cfg(test)]
mod tests;

use std::collections::BTreeSet;
use std::fmt;

use planners_sas::numeric::numeric_task::{AbstractNumericTask, NumericType};
use serde::{Deserialize, Serialize};

use crate::numeric::evaluation::domain_abstractions::comparison_expression::ComparisonTree;

use super::causal_graph::{CausalGraphVariable, MixedCausalGraph};
use super::numeric_size_estimator::NumericSizeEstimator;
use super::numeric_support::NumericSupportContext;
use super::projected_task::Pattern;
use super::variable_order_finder::{GreedyVariableOrderType, order_causal_graph_variables};

pub const DEFAULT_MAX_PDB_STATES: usize = 100_000;
pub const DEFAULT_NUMERIC_FIRST: bool = true;
pub const DEFAULT_RANDOM_SEED: u64 = 0;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
pub struct GreedyPatternGeneratorConfig {
    pub max_pdb_states: usize,
    pub numeric_first: bool,
    pub random_seed: u64,
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
    let numeric_support = NumericSupportContext::new(task);
    let (goal_regular, goal_numeric, true_goal_regular) =
        collect_goal_variables(task, &numeric_support);
    let causal_graph = MixedCausalGraph::new(task);
    let numeric_size_estimator = NumericSizeEstimator::new(task);

    debug_print_goal_variables(
        task,
        &goal_regular,
        &goal_numeric,
        &true_goal_regular,
        &numeric_size_estimator,
        &numeric_support,
        &config,
    );

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

    println!(
        "  ordered goal regular vars: {:?}",
        describe_regular_vars(task, &goal_regular_ids)
    );
    println!(
        "  ordered true goal regular vars: {:?}",
        describe_regular_vars(task, &true_goal_regular_ids)
    );
    println!(
        "  ordered goal numeric vars: {:?}",
        describe_numeric_vars(
            task,
            &goal_numeric_ids,
            &numeric_size_estimator,
            &numeric_support
        )
    );

    if config.numeric_first {
        add_numeric_variables(
            task,
            &goal_numeric_ids,
            &numeric_size_estimator,
            &mut pattern,
            &mut size,
            &config,
            &numeric_support,
            "goal-numeric",
        );
        add_regular_variables(
            task,
            &goal_regular_ids,
            &mut pattern,
            &mut size,
            &config,
            "goal-regular",
        );
    } else {
        add_regular_variables(
            task,
            &goal_regular_ids,
            &mut pattern,
            &mut size,
            &config,
            "goal-regular",
        );
        add_numeric_variables(
            task,
            &goal_numeric_ids,
            &numeric_size_estimator,
            &mut pattern,
            &mut size,
            &config,
            &numeric_support,
            "goal-numeric",
        );
    }

    if pattern.regular.is_empty() {
        #[allow(clippy::never_loop)]
        for &var_id in &true_goal_regular_ids {
            let domain_size = task.get_variable_domain_size(var_id).unwrap_or(1).max(1);
            if size.saturating_mul(domain_size) > config.max_pdb_states {
                break;
            }
            pattern.regular.push(var_id);
            size *= domain_size;
            println!(
                "  fallback accepted regular {} domain={} size={}",
                describe_regular_var(task, var_id),
                domain_size,
                size
            );
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
        &numeric_support,
    );

    println!(
        "  final pattern regular={:?} numeric={:?} final_size={}",
        describe_regular_vars(task, &pattern.regular),
        describe_numeric_vars(
            task,
            &pattern.numeric,
            &numeric_size_estimator,
            &numeric_support
        ),
        size
    );

    pattern
}

fn debug_print_goal_variables(
    task: &dyn AbstractNumericTask,
    goal_regular: &BTreeSet<usize>,
    goal_numeric: &BTreeSet<usize>,
    true_goal_regular: &BTreeSet<usize>,
    numeric_size_estimator: &NumericSizeEstimator,
    numeric_support: &NumericSupportContext,
    config: &GreedyPatternGeneratorConfig,
) {
    println!("\n=== GREEDY PATTERN GENERATION ===");
    println!("  config: {config}");
    println!(
        "  goal regular vars: {:?}",
        describe_regular_vars(task, &goal_regular.iter().copied().collect::<Vec<_>>())
    );
    println!(
        "  true goal regular vars: {:?}",
        describe_regular_vars(task, &true_goal_regular.iter().copied().collect::<Vec<_>>())
    );
    println!(
        "  goal numeric vars: {:?}",
        describe_numeric_vars(
            task,
            &goal_numeric.iter().copied().collect::<Vec<_>>(),
            numeric_size_estimator,
            numeric_support,
        )
    );
}

fn describe_regular_var(task: &dyn AbstractNumericTask, var_id: usize) -> String {
    let name = task.get_variable_name(var_id).unwrap_or("<unknown>");
    format!("{var_id}:{name}")
}

fn describe_numeric_var(
    task: &dyn AbstractNumericTask,
    numeric_var_id: usize,
    numeric_size_estimator: &NumericSizeEstimator,
    numeric_support: &NumericSupportContext,
) -> String {
    if let Some(numeric_var) = task.numeric_variables().get(numeric_var_id) {
        format!(
            "{}:{}({:?}, est={})",
            numeric_var_id,
            numeric_var.name(),
            numeric_var.get_type(),
            numeric_size_estimator.estimate_domain_size(numeric_var_id)
        )
    } else if let Some(source_numeric_var_id) =
        numeric_support.helper_source_numeric_var_id(numeric_var_id)
    {
        let source_numeric_var = &task.numeric_variables()[source_numeric_var_id];
        format!(
            "{}:helper({}:{}, est={})",
            numeric_var_id,
            source_numeric_var_id,
            source_numeric_var.name(),
            numeric_size_estimator.estimate_domain_size(numeric_var_id)
        )
    } else {
        format!(
            "{}:<unknown>(est={})",
            numeric_var_id,
            numeric_size_estimator.estimate_domain_size(numeric_var_id)
        )
    }
}

fn describe_regular_vars(task: &dyn AbstractNumericTask, var_ids: &[usize]) -> Vec<String> {
    var_ids
        .iter()
        .copied()
        .map(|var_id| describe_regular_var(task, var_id))
        .collect()
}

fn describe_numeric_vars(
    task: &dyn AbstractNumericTask,
    numeric_var_ids: &[usize],
    numeric_size_estimator: &NumericSizeEstimator,
    numeric_support: &NumericSupportContext,
) -> Vec<String> {
    numeric_var_ids
        .iter()
        .copied()
        .map(|numeric_var_id| {
            describe_numeric_var(
                task,
                numeric_var_id,
                numeric_size_estimator,
                numeric_support,
            )
        })
        .collect()
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
    numeric_support: &NumericSupportContext,
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

        println!(
            "  predecessor candidates: {:?}",
            ordered_predecessor_preview(
                task,
                &predecessor_candidates,
                numeric_size_estimator,
                numeric_support,
            )
        );

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
                numeric_support,
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
                numeric_support,
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
                numeric_support,
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
                numeric_support,
                &mut next_frontier,
            );
        }

        if next_frontier.is_empty() {
            break;
        }
        frontier = next_frontier;
    }
}

#[allow(clippy::too_many_arguments)]
fn add_ordered_predecessors(
    task: &dyn AbstractNumericTask,
    ordered_candidates: &[CausalGraphVariable],
    numeric_size_estimator: &NumericSizeEstimator,
    pattern: &mut Pattern,
    size: &mut usize,
    config: &GreedyPatternGeneratorConfig,
    numeric: bool,
    numeric_support: &NumericSupportContext,
    next_frontier: &mut Vec<CausalGraphVariable>,
) {
    for &candidate in ordered_candidates {
        match candidate {
            CausalGraphVariable::Regular(var_id) if !numeric => {
                if pattern.regular.contains(&var_id)
                    || task
                        .get_variable_axiom_layer(var_id)
                        .unwrap_or(None)
                        .is_some()
                {
                    println!(
                        "  skip predecessor regular {} reason={}",
                        describe_regular_var(task, var_id),
                        if pattern.regular.contains(&var_id) {
                            "already-in-pattern"
                        } else {
                            "derived-propositional"
                        }
                    );
                    continue;
                }
                let domain_size = task.get_variable_domain_size(var_id).unwrap_or(1).max(1);
                if size.saturating_mul(domain_size) > config.max_pdb_states {
                    println!(
                        "  reject predecessor regular {} domain={} current_size={} limit={}",
                        describe_regular_var(task, var_id),
                        domain_size,
                        *size,
                        config.max_pdb_states
                    );
                    continue;
                }
                pattern.regular.push(var_id);
                *size *= domain_size;
                println!(
                    "  accept predecessor regular {} domain={} new_size={}",
                    describe_regular_var(task, var_id),
                    domain_size,
                    *size
                );
                next_frontier.push(candidate);
            }
            CausalGraphVariable::Numeric(var_id) if numeric => {
                if pattern.numeric.contains(&var_id)
                    || !is_pattern_numeric_candidate(task, var_id, numeric_support)
                {
                    println!(
                        "  skip predecessor numeric {} reason={}",
                        describe_numeric_var(task, var_id, numeric_size_estimator, numeric_support),
                        if pattern.numeric.contains(&var_id) {
                            "already-in-pattern"
                        } else {
                            "unsupported-type"
                        }
                    );
                    continue;
                }
                let domain_size = numeric_size_estimator.estimate_domain_size(var_id);
                if size.saturating_mul(domain_size) > config.max_pdb_states {
                    println!(
                        "  reject predecessor numeric {} current_size={} limit={}",
                        describe_numeric_var(task, var_id, numeric_size_estimator, numeric_support),
                        *size,
                        config.max_pdb_states
                    );
                    continue;
                }
                pattern.numeric.push(var_id);
                *size *= domain_size;
                println!(
                    "  accept predecessor numeric {} new_size={}",
                    describe_numeric_var(task, var_id, numeric_size_estimator, numeric_support),
                    *size
                );
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
    label: &str,
) {
    for &var_id in variable_ids {
        if pattern.regular.contains(&var_id) {
            println!(
                "  skip {label} regular {} reason=already-in-pattern",
                describe_regular_var(task, var_id)
            );
            continue;
        }
        let domain_size = task.get_variable_domain_size(var_id).unwrap_or(1).max(1);
        if size.saturating_mul(domain_size) > config.max_pdb_states {
            println!(
                "  reject {label} regular {} domain={} current_size={} limit={}",
                describe_regular_var(task, var_id),
                domain_size,
                *size,
                config.max_pdb_states
            );
            break;
        }
        pattern.regular.push(var_id);
        *size *= domain_size;
        println!(
            "  accept {label} regular {} domain={} new_size={}",
            describe_regular_var(task, var_id),
            domain_size,
            *size
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn add_numeric_variables(
    task: &dyn AbstractNumericTask,
    numeric_var_ids: &[usize],
    numeric_size_estimator: &NumericSizeEstimator,
    pattern: &mut Pattern,
    size: &mut usize,
    config: &GreedyPatternGeneratorConfig,
    numeric_support: &NumericSupportContext,
    label: &str,
) {
    for &numeric_var_id in numeric_var_ids {
        if pattern.numeric.contains(&numeric_var_id)
            || !is_pattern_numeric_candidate(task, numeric_var_id, numeric_support)
        {
            println!(
                "  skip {label} numeric {} reason={}",
                describe_numeric_var(
                    task,
                    numeric_var_id,
                    numeric_size_estimator,
                    numeric_support
                ),
                if pattern.numeric.contains(&numeric_var_id) {
                    "already-in-pattern"
                } else {
                    "unsupported-type"
                }
            );
            continue;
        }
        let domain_size = numeric_size_estimator.estimate_domain_size(numeric_var_id);
        if size.saturating_mul(domain_size) > config.max_pdb_states {
            println!(
                "  reject {label} numeric {} current_size={} limit={}",
                describe_numeric_var(
                    task,
                    numeric_var_id,
                    numeric_size_estimator,
                    numeric_support
                ),
                *size,
                config.max_pdb_states
            );
            break;
        }
        pattern.numeric.push(numeric_var_id);
        *size *= domain_size;
        println!(
            "  accept {label} numeric {} new_size={}",
            describe_numeric_var(
                task,
                numeric_var_id,
                numeric_size_estimator,
                numeric_support
            ),
            *size
        );
    }
}

fn collect_goal_variables(
    task: &dyn AbstractNumericTask,
    numeric_support: &NumericSupportContext,
) -> (BTreeSet<usize>, BTreeSet<usize>, BTreeSet<usize>) {
    let mut regular = BTreeSet::new();
    let mut numeric = BTreeSet::new();
    let goal_related_propositional_vars = collect_goal_related_propositional_closure(task);
    println!(
        "  goal-related propositional closure: {:?}",
        describe_regular_vars(
            task,
            &goal_related_propositional_vars
                .iter()
                .copied()
                .collect::<Vec<_>>()
        )
    );
    let mut true_goal_regular: BTreeSet<usize> = goal_related_propositional_vars
        .iter()
        .copied()
        .filter(|&var_id| {
            task.get_variable_axiom_layer(var_id)
                .unwrap_or(None)
                .is_none()
        })
        .collect();

    for goal_index in 0..task.get_num_goals() {
        let goal = task.get_goal_fact(goal_index);
        let goal_var_id = goal.var;
        if task
            .get_variable_axiom_layer(goal_var_id)
            .unwrap_or(None)
            .is_none()
        {
            regular.insert(goal_var_id);
            true_goal_regular.insert(goal_var_id);
        }
    }

    for (comparison_axiom_id, comparison_axiom) in task.comparison_axioms().iter().enumerate() {
        let affected_var_id = comparison_axiom.get_affected_var_id();
        if !goal_related_propositional_vars.contains(&affected_var_id) {
            continue;
        }

        println!(
            "  inspect comparison axiom {} affected_var={} left={} right={}",
            comparison_axiom_id,
            describe_regular_var(task, affected_var_id),
            comparison_axiom.get_left_var_id(),
            comparison_axiom.get_right_var_id()
        );

        if ComparisonTree::from_task(task, comparison_axiom_id).is_ok() {
            let dependencies = numeric_support.comparison_support_ids(task, comparison_axiom_id);
            println!(
                "    comparison dependencies: {:?}",
                describe_numeric_vars(
                    task,
                    &dependencies,
                    &NumericSizeEstimator::new(task),
                    numeric_support,
                )
            );
            for numeric_var_id in dependencies {
                if is_pattern_numeric_candidate(task, numeric_var_id, numeric_support) {
                    numeric.insert(numeric_var_id);
                }
            }
        } else {
            for numeric_var_id in numeric_support.comparison_support_ids(task, comparison_axiom_id)
            {
                if is_pattern_numeric_candidate(task, numeric_var_id, numeric_support) {
                    numeric.insert(numeric_var_id);
                }
            }
        }
    }

    (regular, numeric, true_goal_regular)
}

fn ordered_predecessor_preview(
    task: &dyn AbstractNumericTask,
    candidates: &BTreeSet<CausalGraphVariable>,
    numeric_size_estimator: &NumericSizeEstimator,
    numeric_support: &NumericSupportContext,
) -> Vec<String> {
    candidates
        .iter()
        .map(|candidate| match candidate {
            CausalGraphVariable::Regular(var_id) => {
                format!("regular:{}", describe_regular_var(task, *var_id))
            }
            CausalGraphVariable::Numeric(var_id) => {
                format!(
                    "numeric:{}",
                    describe_numeric_var(task, *var_id, numeric_size_estimator, numeric_support)
                )
            }
        })
        .collect()
}

fn is_pattern_numeric_candidate(
    task: &dyn AbstractNumericTask,
    numeric_var_id: usize,
    numeric_support: &NumericSupportContext,
) -> bool {
    task.numeric_variables()
        .get(numeric_var_id)
        .map(|numeric_var| numeric_var.get_type() == &NumericType::Regular)
        .unwrap_or_else(|| numeric_support.is_helper_var_id(task, numeric_var_id))
}

fn collect_goal_related_propositional_closure(task: &dyn AbstractNumericTask) -> BTreeSet<usize> {
    let mut goal_related: BTreeSet<usize> = (0..task.get_num_goals()).collect();

    loop {
        let mut changed = false;

        for axiom in task.axioms() {
            let affected_var_id = axiom.var_id();
            if goal_related.contains(&affected_var_id) {
                for condition in axiom.conditions() {
                    changed |= goal_related.insert(condition.var);
                }
            }
        }

        if !changed {
            break;
        }
    }

    goal_related
}
