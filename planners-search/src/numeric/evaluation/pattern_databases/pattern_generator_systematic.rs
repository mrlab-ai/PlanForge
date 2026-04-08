use std::collections::BTreeSet;
use std::fmt;

use planners_sas::numeric::numeric_task::{
    AbstractNumericTask, ExplicitVariable, Fact, Metric, NumericRootTask, NumericType,
    NumericVariable, Operator,
};
use serde::{Deserialize, Serialize};

use super::causal_graph::{CausalGraphVariable, MixedCausalGraph};
use super::numeric_size_estimator::NumericSizeEstimator;
use super::numeric_support::NumericSupportContext;
use super::pattern_collection::PatternCollection;
use super::pattern_generator_greedy::DEFAULT_MAX_PDB_STATES;
use super::projected_task::{Pattern, ProjectedTask};
use super::variable_order_finder::{order_causal_graph_variables, GreedyVariableOrderType};

pub const DEFAULT_MAX_PATTERN_SIZE: usize = 2;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
pub struct SystematicPatternGeneratorConfig {
    pub max_pdb_states: usize,
    pub max_pattern_size: usize,
    pub random_seed: i32,
    pub variable_order_type: GreedyVariableOrderType,
}

impl Default for SystematicPatternGeneratorConfig {
    fn default() -> Self {
        Self {
            max_pdb_states: DEFAULT_MAX_PDB_STATES,
            max_pattern_size: DEFAULT_MAX_PATTERN_SIZE,
            random_seed: 0,
            variable_order_type: GreedyVariableOrderType::default(),
        }
    }
}

impl fmt::Display for SystematicPatternGeneratorConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "max_pdb_states={}, max_pattern_size={}, random_seed={}, variable_order_type={}",
            self.max_pdb_states, self.max_pattern_size, self.random_seed, self.variable_order_type,
        )
    }
}

pub fn generate_systematic_patterns(
    task: &dyn AbstractNumericTask,
    config: SystematicPatternGeneratorConfig,
) -> PatternCollection {
    if config.max_pattern_size == 0 || config.max_pdb_states == 0 {
        return PatternCollection::empty();
    }

    let numeric_support = NumericSupportContext::new(task);
    let causal_graph = MixedCausalGraph::new(task);
    let size_estimator = NumericSizeEstimator::new(task);
    let seed_variables = collect_seed_variables(task, &numeric_support);
    let ordered_candidates = ordered_relevant_candidates(&causal_graph, &seed_variables, config);
    let mut collection = PatternCollection::empty();

    for &seed in &ordered_candidates {
        if !seed_variables.contains(&seed) {
            continue;
        }

        let Some((pattern, estimated_size)) =
            singleton_pattern(task, &size_estimator, seed, config)
        else {
            continue;
        };

        collection.push(pattern.clone());
        extend_patterns(
            task,
            &causal_graph,
            &size_estimator,
            &ordered_candidates,
            &mut collection,
            pattern,
            estimated_size,
            config,
            ordered_candidates
                .iter()
                .position(|candidate| *candidate == seed)
                .map(|index| index + 1)
                .unwrap_or(0),
        );
    }

    collection
}

fn extend_patterns(
    task: &dyn AbstractNumericTask,
    causal_graph: &MixedCausalGraph,
    size_estimator: &NumericSizeEstimator,
    ordered_candidates: &[CausalGraphVariable],
    collection: &mut PatternCollection,
    pattern: Pattern,
    estimated_size: usize,
    config: SystematicPatternGeneratorConfig,
    min_candidate_index: usize,
) {
    if pattern.total_len() >= config.max_pattern_size {
        return;
    }

    let expansion_candidates = expansion_candidates(
        causal_graph,
        ordered_candidates,
        &pattern,
        min_candidate_index,
    );

    for (candidate_index, candidate) in expansion_candidates {
        let mut next_pattern = pattern.clone();
        let Some(next_size) = try_extend_pattern(
            task,
            size_estimator,
            &mut next_pattern,
            estimated_size,
            candidate,
            config.max_pdb_states,
        ) else {
            continue;
        };

        if collection.push(next_pattern.clone()) {
            extend_patterns(
                task,
                causal_graph,
                size_estimator,
                ordered_candidates,
                collection,
                next_pattern,
                next_size,
                config,
                candidate_index + 1,
            );
        }
    }
}

fn expansion_candidates(
    causal_graph: &MixedCausalGraph,
    ordered_candidates: &[CausalGraphVariable],
    pattern: &Pattern,
    min_candidate_index: usize,
) -> Vec<(usize, CausalGraphVariable)> {
    let pattern_variables = pattern_variables(pattern);
    let mut expandable = BTreeSet::new();

    for variable in &pattern_variables {
        for predecessor in causal_graph.predecessors_of(*variable) {
            if pattern_variables.contains(&predecessor) {
                continue;
            }
            expandable.insert(predecessor);
        }
    }

    ordered_candidates
        .iter()
        .copied()
        .enumerate()
        .filter(|(index, candidate)| {
            *index >= min_candidate_index && expandable.contains(candidate)
        })
        .collect()
}

fn singleton_pattern(
    task: &dyn AbstractNumericTask,
    size_estimator: &NumericSizeEstimator,
    variable: CausalGraphVariable,
    config: SystematicPatternGeneratorConfig,
) -> Option<(Pattern, usize)> {
    let mut pattern = Pattern::new(Vec::new(), Vec::new());
    let estimated_size = try_extend_pattern(
        task,
        size_estimator,
        &mut pattern,
        1,
        variable,
        config.max_pdb_states,
    )?;
    Some((pattern, estimated_size))
}

fn try_extend_pattern(
    task: &dyn AbstractNumericTask,
    size_estimator: &NumericSizeEstimator,
    pattern: &mut Pattern,
    current_size: usize,
    candidate: CausalGraphVariable,
    max_pdb_states: usize,
) -> Option<usize> {
    let factor = match candidate {
        CausalGraphVariable::Regular(var_id) => task
            .get_variable_domain_size(var_id as i32)
            .ok()
            .and_then(|size| usize::try_from(size.max(1)).ok())
            .unwrap_or(1),
        CausalGraphVariable::Numeric(var_id) => size_estimator.estimate_domain_size(var_id),
    };

    let next_size = current_size.saturating_mul(factor.max(1));
    if next_size > max_pdb_states {
        return None;
    }

    let inserted = match candidate {
        CausalGraphVariable::Regular(var_id) => pattern.add_regular_var(var_id),
        CausalGraphVariable::Numeric(var_id) => pattern.add_numeric_var(var_id),
    };

    inserted.then_some(next_size)
}

fn ordered_relevant_candidates(
    causal_graph: &MixedCausalGraph,
    seed_variables: &BTreeSet<CausalGraphVariable>,
    config: SystematicPatternGeneratorConfig,
) -> Vec<CausalGraphVariable> {
    let mut reachable = seed_variables.clone();
    let mut agenda: Vec<_> = seed_variables.iter().copied().collect();

    while let Some(variable) = agenda.pop() {
        for predecessor in causal_graph.predecessors_of(variable) {
            if reachable.insert(predecessor) {
                agenda.push(predecessor);
            }
        }
    }

    let mut ordered: Vec<_> = reachable.into_iter().collect();
    order_causal_graph_variables(
        &mut ordered,
        causal_graph,
        config.variable_order_type,
        config.random_seed,
    );
    ordered
}

fn collect_seed_variables(
    task: &dyn AbstractNumericTask,
    numeric_support: &NumericSupportContext,
) -> BTreeSet<CausalGraphVariable> {
    let mut seed_variables = BTreeSet::new();
    let goal_related_propositional_vars = collect_goal_related_propositional_closure(task);

    for goal_index in 0..usize::try_from(task.get_num_goals().max(0)).unwrap_or(0) {
        let goal = task.get_goal_fact(goal_index as i32);
        let goal_var_id = goal.var() as usize;
        if task
            .get_variable_axiom_layer(goal_var_id as i32)
            .unwrap_or(-1)
            == -1
        {
            seed_variables.insert(CausalGraphVariable::Regular(goal_var_id));
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

        for numeric_var_id in numeric_support.comparison_support_ids(task, comparison_axiom_id) {
            if is_pattern_numeric_candidate(task, numeric_var_id, numeric_support) {
                seed_variables.insert(CausalGraphVariable::Numeric(numeric_var_id));
            }
        }
    }

    seed_variables
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

fn pattern_variables(pattern: &Pattern) -> BTreeSet<CausalGraphVariable> {
    pattern
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
        .collect()
}

#[cfg(test)]
mod tests {
    use planners_sas::numeric::axioms::{ComparisonAxiom, ComparisonOperator, PropositionalAxiom};

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

    fn propositional_predecessor_task() -> NumericRootTask {
        NumericRootTask::new(
            1,
            Metric::new(true, -1),
            vec![simple_var("q", -1), simple_var("p", -1)],
            vec![],
            vec![Fact::new(1, 1)],
            vec![],
            vec![0, 0],
            vec![],
            vec![Operator::new(
                "set-goal".to_string(),
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

    fn numeric_goal_task() -> NumericRootTask {
        NumericRootTask::new(
            1,
            Metric::new(true, -1),
            vec![simple_var("cmp", 0), simple_var("goal", -1)],
            vec![
                NumericVariable::new("threshold".to_string(), NumericType::Constant, -1),
                NumericVariable::new("x".to_string(), NumericType::Regular, -1),
            ],
            vec![Fact::new(1, 1)],
            vec![],
            vec![0, 0],
            vec![1.0, 0.0],
            vec![],
            vec![PropositionalAxiom::new(vec![Fact::new(0, 0)], 1, 0, 1)],
            vec![ComparisonAxiom::new(
                0,
                1,
                0,
                ComparisonOperator::GreaterThanOrEqual,
            )],
            vec![],
            (0, 0),
        )
    }

    #[test]
    fn systematic_generator_includes_goal_singleton_and_predecessor_pair() {
        let task = propositional_predecessor_task();
        let collection = generate_systematic_patterns(
            &task,
            SystematicPatternGeneratorConfig {
                max_pattern_size: 2,
                ..SystematicPatternGeneratorConfig::default()
            },
        );

        assert!(collection.contains(&Pattern::new(vec![1], vec![])));
        assert!(collection.contains(&Pattern::new(vec![0, 1], vec![])));
    }

    #[test]
    fn systematic_generator_returns_projectable_numeric_patterns() {
        let task = numeric_goal_task();
        let collection =
            generate_systematic_patterns(&task, SystematicPatternGeneratorConfig::default());

        assert!(collection.contains(&Pattern::new(vec![], vec![1])));
        assert!(collection
            .iter()
            .all(|pattern| ProjectedTask::new(&task, pattern).is_ok()));
    }
}
