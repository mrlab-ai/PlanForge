use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use std::collections::{BTreeMap, BTreeSet};
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
use super::variable_order_finder::GreedyVariableOrderType;

pub const DEFAULT_MAX_PATTERN_SIZE: usize = 2;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
pub struct SystematicPatternGeneratorConfig {
    pub max_pdb_states: usize,
    pub max_pattern_size: usize,
    pub only_interesting_patterns: bool,
    pub random_seed: i32,
    pub variable_order_type: GreedyVariableOrderType,
}

impl Default for SystematicPatternGeneratorConfig {
    fn default() -> Self {
        Self {
            max_pdb_states: DEFAULT_MAX_PDB_STATES,
            max_pattern_size: DEFAULT_MAX_PATTERN_SIZE,
            only_interesting_patterns: true,
            random_seed: 0,
            variable_order_type: GreedyVariableOrderType::default(),
        }
    }
}

impl fmt::Display for SystematicPatternGeneratorConfig {
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
    if config.only_interesting_patterns {
        build_interesting_patterns(
            task,
            &causal_graph,
            &seed_variables,
            config.max_pattern_size,
            config,
        )
    } else {
        let ordered_candidates =
            ordered_relevant_candidates(&causal_graph, &seed_variables, config);
        generate_relevant_patterns_naive(
            task,
            &size_estimator,
            &ordered_candidates,
            config,
        )
    }
}

fn build_interesting_patterns(
    task: &dyn AbstractNumericTask,
    causal_graph: &MixedCausalGraph,
    seed_variables: &BTreeSet<CausalGraphVariable>,
    max_pattern_size: usize,
    config: SystematicPatternGeneratorConfig,
) -> PatternCollection {
    let sga_patterns = build_sga_patterns(
        task,
        causal_graph,
        seed_variables,
        max_pattern_size,
        config,
    );
    let mut sga_patterns_by_variable: BTreeMap<CausalGraphVariable, Vec<&Pattern>> =
        BTreeMap::new();
    for pattern in &sga_patterns {
        for variable in pattern_variables(pattern) {
            sga_patterns_by_variable
                .entry(variable)
                .or_default()
                .push(pattern);
        }
    }

    let mut collection = Vec::new();
    let mut seen = BTreeSet::new();
    for pattern in &sga_patterns {
        enqueue_pattern_if_new(pattern.clone(), &mut collection, &mut seen);
    }

    let mut pattern_index = 0;
    while let Some(pattern) = collection.get(pattern_index).cloned() {
        let connection_points = compute_connection_points(causal_graph, &pattern);
        for connection_point in connection_points {
            let Some(candidates) = sga_patterns_by_variable.get(&connection_point) else {
                continue;
            };
            for candidate in candidates {
                if pattern.total_len() + candidate.total_len() > max_pattern_size {
                    break;
                }
                if patterns_are_disjoint(&pattern, candidate) {
                    enqueue_pattern_if_new(
                        compute_union_pattern(&pattern, candidate),
                        &mut collection,
                        &mut seen,
                    );
                }
            }
        }
        pattern_index += 1;
    }

    PatternCollection::new(collection)
}

fn build_sga_patterns(
    task: &dyn AbstractNumericTask,
    causal_graph: &MixedCausalGraph,
    seed_variables: &BTreeSet<CausalGraphVariable>,
    max_pattern_size: usize,
    config: SystematicPatternGeneratorConfig,
) -> Vec<Pattern> {
    let mut patterns = Vec::new();
    let mut seen = BTreeSet::new();

    let mut ordered_seeds: Vec<_> = seed_variables.iter().copied().collect();
    order_causal_graph_variables(
        &mut ordered_seeds,
        causal_graph,
        config.variable_order_type,
        config.random_seed,
    );

    for seed in ordered_seeds {
        let mut pattern = Pattern::new(Vec::new(), Vec::new());
        match seed {
            CausalGraphVariable::Regular(var_id) => {
                let inserted = pattern.add_regular_var(var_id);
                assert!(inserted, "goal singleton regular variable inserted twice");
            }
            CausalGraphVariable::Numeric(var_id) => {
                let inserted = pattern.add_numeric_var(var_id);
                assert!(inserted, "goal singleton numeric variable inserted twice");
            }
        }
        enqueue_pattern_if_new(pattern, &mut patterns, &mut seen);
    }

    let mut pattern_index = 0;
    while let Some(pattern) = patterns.get(pattern_index).cloned() {
        if pattern.total_len() == max_pattern_size {
            break;
        }

        let neighbors = compute_eff_pre_neighbors(causal_graph, &pattern);
        for neighbor in neighbors {
            let mut next_pattern = pattern.clone();
            match neighbor {
                CausalGraphVariable::Regular(var_id) => {
                    let inserted = next_pattern.add_regular_var(var_id);
                    assert!(inserted, "eff-pre neighbor regular variable already in pattern");
                }
                CausalGraphVariable::Numeric(var_id) => {
                    let inserted = next_pattern.add_numeric_var(var_id);
                    assert!(inserted, "eff-pre neighbor numeric variable already in pattern");
                }
            }
            enqueue_pattern_if_new(next_pattern, &mut patterns, &mut seen);
        }

        pattern_index += 1;
    }

    patterns
}

fn generate_relevant_patterns_naive(
    task: &dyn AbstractNumericTask,
    size_estimator: &NumericSizeEstimator,
    ordered_candidates: &[CausalGraphVariable],
    config: SystematicPatternGeneratorConfig,
) -> PatternCollection {
    let mut collection = Vec::new();
    let mut seen = BTreeSet::new();
    extend_patterns_naive(
        task,
        size_estimator,
        ordered_candidates,
        &mut collection,
        &mut seen,
        Pattern::new(Vec::new(), Vec::new()),
        1,
        config,
        0,
    );
    PatternCollection::new(collection)
}

fn extend_patterns_naive(
    task: &dyn AbstractNumericTask,
    size_estimator: &NumericSizeEstimator,
    ordered_candidates: &[CausalGraphVariable],
    collection: &mut Vec<Pattern>,
    seen: &mut BTreeSet<Pattern>,
    pattern: Pattern,
    estimated_size: usize,
    config: SystematicPatternGeneratorConfig,
    min_candidate_index: usize,
) {
    if pattern.total_len() >= config.max_pattern_size {
        return;
    }

    for (candidate_index, candidate) in ordered_candidates.iter().copied().enumerate() {
        if candidate_index < min_candidate_index {
            continue;
        }

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

        if seen.insert(next_pattern.clone()) {
            collection.push(next_pattern.clone());
            extend_patterns_naive(
                task,
                size_estimator,
                ordered_candidates,
                collection,
                seen,
                next_pattern,
                next_size,
                config,
                candidate_index + 1,
            );
        }
    }
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

fn order_causal_graph_variables(
    variables: &mut [CausalGraphVariable],
    causal_graph: &MixedCausalGraph,
    variable_order_type: GreedyVariableOrderType,
    random_seed: i32,
) {
    match variable_order_type {
        GreedyVariableOrderType::CgGoalRandom => {
            let mut rng = SmallRng::seed_from_u64(random_seed as i64 as u64);
            variables.shuffle(&mut rng);
        }
        GreedyVariableOrderType::CgGoalLevel => {
            variables.sort_by_key(|&variable| {
                (
                    std::cmp::Reverse(causal_graph.predecessor_count(variable)),
                    causal_graph.goal_distance(variable).unwrap_or(usize::MAX / 2),
                    causal_graph.causal_level(variable).unwrap_or(usize::MAX / 2),
                    variable,
                )
            });
        }
        GreedyVariableOrderType::GoalCgLevel => {
            variables.sort_by_key(|&variable| {
                (
                    causal_graph.goal_distance(variable).unwrap_or(usize::MAX / 2),
                    std::cmp::Reverse(causal_graph.predecessor_count(variable)),
                    causal_graph.causal_level(variable).unwrap_or(usize::MAX / 2),
                    variable,
                )
            });
        }
    }
}

fn enqueue_pattern_if_new(
    pattern: Pattern,
    collection: &mut Vec<Pattern>,
    seen: &mut BTreeSet<Pattern>,
) {
    if seen.insert(pattern.clone()) {
        collection.push(pattern);
    }
}

fn compute_eff_pre_neighbors(
    causal_graph: &MixedCausalGraph,
    pattern: &Pattern,
) -> Vec<CausalGraphVariable> {
    let pattern_variables = pattern_variables(pattern);
    let mut neighbors = BTreeSet::new();

    for variable in &pattern_variables {
        for neighbor in causal_graph.eff_pre_neighbors_of(*variable) {
            if !pattern_variables.contains(&neighbor) {
                neighbors.insert(neighbor);
            }
        }
    }

    neighbors.into_iter().collect()
}

fn compute_connection_points(
    causal_graph: &MixedCausalGraph,
    pattern: &Pattern,
) -> Vec<CausalGraphVariable> {
    let pattern_variables = pattern_variables(pattern);
    let mut candidates = BTreeSet::new();

    for variable in &pattern_variables {
        for predecessor in causal_graph.predecessors_of(*variable) {
            candidates.insert(predecessor);
        }
    }

    for variable in &pattern_variables {
        candidates.remove(variable);
        for eff_pre_neighbor in causal_graph.eff_pre_neighbors_of(*variable) {
            candidates.remove(&eff_pre_neighbor);
        }
    }

    candidates.into_iter().collect()
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

fn patterns_are_disjoint(lhs: &Pattern, rhs: &Pattern) -> bool {
    let mut lhs_regular = lhs.regular.iter();
    let mut rhs_regular = rhs.regular.iter();
    let mut lhs_regular_next = lhs_regular.next();
    let mut rhs_regular_next = rhs_regular.next();
    while let (Some(lhs_var), Some(rhs_var)) = (lhs_regular_next, rhs_regular_next) {
        if lhs_var == rhs_var {
            return false;
        }
        if lhs_var < rhs_var {
            lhs_regular_next = lhs_regular.next();
        } else {
            rhs_regular_next = rhs_regular.next();
        }
    }

    let mut lhs_numeric = lhs.numeric.iter();
    let mut rhs_numeric = rhs.numeric.iter();
    let mut lhs_numeric_next = lhs_numeric.next();
    let mut rhs_numeric_next = rhs_numeric.next();
    while let (Some(lhs_var), Some(rhs_var)) = (lhs_numeric_next, rhs_numeric_next) {
        if lhs_var == rhs_var {
            return false;
        }
        if lhs_var < rhs_var {
            lhs_numeric_next = lhs_numeric.next();
        } else {
            rhs_numeric_next = rhs_numeric.next();
        }
    }

    true
}

fn compute_union_pattern(lhs: &Pattern, rhs: &Pattern) -> Pattern {
    Pattern::new(
        lhs.regular
            .iter()
            .copied()
            .chain(rhs.regular.iter().copied())
            .collect(),
        lhs.numeric
            .iter()
            .copied()
            .chain(rhs.numeric.iter().copied())
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use planners_sas::numeric::axioms::{
        AssignmentAxiom, CalOperator, ComparisonAxiom, ComparisonOperator, PropositionalAxiom,
    };
    use planners_sas::numeric::numeric_task::{AssignmentEffect, AssignmentOperation};

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

    fn eff_eff_goal_join_task() -> NumericRootTask {
        NumericRootTask::new(
            1,
            Metric::new(true, -1),
            vec![simple_var("g1", -1), simple_var("g2", -1)],
            vec![],
            vec![Fact::new(0, 1), Fact::new(1, 1)],
            vec![],
            vec![0, 0],
            vec![],
            vec![Operator::new(
                "set-both".to_string(),
                vec![],
                vec![
                    planners_sas::numeric::numeric_task::Effect::new(vec![], 0, 0, 1),
                    planners_sas::numeric::numeric_task::Effect::new(vec![], 1, 0, 1),
                ],
                vec![],
                1,
            )],
            vec![],
            vec![],
            vec![],
            (0, 0),
        )
    }

    fn helper_goal_with_unsupported_numeric_effect_task() -> NumericRootTask {
        NumericRootTask::new(
            1,
            Metric::new(true, -1),
            vec![simple_var("cmp", 0), simple_var("goal", -1)],
            vec![
                NumericVariable::new("const2".to_string(), NumericType::Constant, -1),
                NumericVariable::new("x".to_string(), NumericType::Regular, -1),
                NumericVariable::new("y".to_string(), NumericType::Regular, -1),
                NumericVariable::new("sum".to_string(), NumericType::Derived, 0),
            ],
            vec![Fact::new(1, 1)],
            vec![],
            vec![0, 0],
            vec![2.0, 1.0, 1.0, 2.0],
            vec![Operator::new(
                "scale-x".to_string(),
                vec![],
                vec![],
                vec![AssignmentEffect::new(
                    1,
                    AssignmentOperation::Times,
                    0,
                    false,
                    vec![],
                )],
                1,
            )],
            vec![PropositionalAxiom::new(vec![Fact::new(0, 0)], 1, 0, 1)],
            vec![ComparisonAxiom::new(
                0,
                3,
                0,
                ComparisonOperator::GreaterThanOrEqual,
            )],
            vec![AssignmentAxiom::new(3, CalOperator::Sum, 1, 2)],
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
    }

    #[test]
    fn systematic_generator_joins_disjoint_sga_patterns_via_connection_points() {
        let task = eff_eff_goal_join_task();
        let collection = generate_systematic_patterns(
            &task,
            SystematicPatternGeneratorConfig {
                max_pattern_size: 2,
                ..SystematicPatternGeneratorConfig::default()
            },
        );

        assert!(collection.contains(&Pattern::new(vec![0], vec![])));
        assert!(collection.contains(&Pattern::new(vec![1], vec![])));
        assert!(collection.contains(&Pattern::new(vec![0, 1], vec![])));
    }

    #[test]
    fn systematic_generator_keeps_patterns_rejected_by_projected_task() {
        let task = helper_goal_with_unsupported_numeric_effect_task();
        let helper_var_id = task.numeric_variables().len();

        assert!(ProjectedTask::new(&task, &Pattern::new(vec![], vec![helper_var_id])).is_err());

        let collection = generate_systematic_patterns(
            &task,
            SystematicPatternGeneratorConfig {
                max_pattern_size: 2,
                ..SystematicPatternGeneratorConfig::default()
            },
        );

        assert!(collection.contains(&Pattern::new(vec![], vec![helper_var_id])));
    }

    #[test]
    fn systematic_generator_supports_naive_mode() {
        let task = propositional_predecessor_task();
        let collection = generate_systematic_patterns(
            &task,
            SystematicPatternGeneratorConfig {
                only_interesting_patterns: false,
                max_pattern_size: 2,
                ..SystematicPatternGeneratorConfig::default()
            },
        );

        assert!(collection.contains(&Pattern::new(vec![1], vec![])));
        assert!(collection.contains(&Pattern::new(vec![0], vec![])));
        assert!(collection.contains(&Pattern::new(vec![0, 1], vec![])));
    }
}
