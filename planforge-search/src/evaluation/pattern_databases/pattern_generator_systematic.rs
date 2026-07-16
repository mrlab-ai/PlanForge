#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use planforge_sas::numeric_task::{AbstractNumericTask, NumericType};
use serde::{Deserialize, Serialize};

use super::pattern_collection::PatternCollection;
use super::projected_task::Pattern;
use crate::causal_graph::{CausalGraph, CausalGraphVariable, RestrictedCausalGraph};

pub const DEFAULT_SYSTEMATIC_MAX_PDB_STATES: usize = 50_000;
pub const DEFAULT_MAX_PATTERN_SIZE: usize = 2;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
pub struct SystematicPatternGeneratorConfig {
    pub max_pdb_states: usize,
    pub max_pattern_size: usize,
    pub only_interesting_patterns: bool,
}

impl Default for SystematicPatternGeneratorConfig {
    fn default() -> Self {
        Self {
            max_pdb_states: DEFAULT_SYSTEMATIC_MAX_PDB_STATES,
            max_pattern_size: DEFAULT_MAX_PATTERN_SIZE,
            only_interesting_patterns: true,
        }
    }
}

impl fmt::Display for SystematicPatternGeneratorConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "max_pdb_states={}, max_pattern_size={}, only_interesting_patterns={}",
            self.max_pdb_states, self.max_pattern_size, self.only_interesting_patterns,
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

    if !config.only_interesting_patterns {
        panic!("not implemented: numeric systematic naive pattern generation");
    }

    let causal_graph = RestrictedCausalGraph::new(task)
        .expect("systematic PDB patterns require a restricted task");
    let seed_variables = collect_seed_variables(task, &causal_graph);
    build_interesting_patterns(
        &causal_graph,
        &seed_variables,
        config.max_pattern_size,
        config,
    )
}

fn build_interesting_patterns(
    causal_graph: &CausalGraph,
    seed_variables: &[CausalGraphVariable],
    max_pattern_size: usize,
    config: SystematicPatternGeneratorConfig,
) -> PatternCollection {
    let sga_patterns = build_sga_patterns(causal_graph, seed_variables, max_pattern_size, config);
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
                if cpp_systematic_combined_size(&pattern, candidate) > max_pattern_size {
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
    causal_graph: &CausalGraph,
    seed_variables: &[CausalGraphVariable],
    max_pattern_size: usize,
    _config: SystematicPatternGeneratorConfig,
) -> Vec<Pattern> {
    let mut patterns = Vec::new();
    let mut seen = BTreeSet::new();

    for seed in seed_variables {
        let mut pattern = Pattern::new(Vec::new(), Vec::new());
        match *seed {
            CausalGraphVariable::Propositional(var_id) => {
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
                CausalGraphVariable::Propositional(var_id) => {
                    let inserted = next_pattern.add_regular_var(var_id);
                    assert!(
                        inserted,
                        "eff-pre neighbor regular variable already in pattern"
                    );
                }
                CausalGraphVariable::Numeric(var_id) => {
                    let inserted = next_pattern.add_numeric_var(var_id);
                    assert!(
                        inserted,
                        "eff-pre neighbor numeric variable already in pattern"
                    );
                }
            }
            enqueue_pattern_if_new(next_pattern, &mut patterns, &mut seen);
        }

        pattern_index += 1;
    }

    patterns
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
    causal_graph: &CausalGraph,
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
    causal_graph: &CausalGraph,
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
    causal_graph: &CausalGraph,
) -> Vec<CausalGraphVariable> {
    let mut seed_variables = Vec::new();
    let mut seen = BTreeSet::new();
    let goal_related_propositional_vars = collect_goal_related_propositional_closure(task);

    for goal_index in 0..task.get_num_goals() {
        let goal = task.get_goal_fact(goal_index);
        if task
            .get_variable_axiom_layer(goal.var())
            .unwrap_or(None)
            .is_none()
        {
            push_seed_variable(
                &mut seed_variables,
                &mut seen,
                CausalGraphVariable::Propositional(goal.var()),
            );
        }
    }

    for (comparison_axiom_id, comparison_axiom) in task.comparison_axioms().iter().enumerate() {
        let affected_var_id = comparison_axiom.get_affected_var_id();
        if !goal_related_propositional_vars.contains(&affected_var_id) {
            continue;
        }

        if let Some(numeric_var_id) = causal_graph.comparison_numeric_var(comparison_axiom_id)
            && is_pattern_numeric_candidate(task, numeric_var_id)
        {
            push_seed_variable(
                &mut seed_variables,
                &mut seen,
                CausalGraphVariable::Numeric(numeric_var_id),
            );
        }
    }

    seed_variables
}

fn push_seed_variable(
    seed_variables: &mut Vec<CausalGraphVariable>,
    seen: &mut BTreeSet<CausalGraphVariable>,
    variable: CausalGraphVariable,
) {
    if seen.insert(variable) {
        seed_variables.push(variable);
    }
}

fn cpp_systematic_combined_size(pattern: &Pattern, candidate: &Pattern) -> usize {
    pattern.regular.len()
        + pattern.numeric.len()
        + candidate.regular.len()
        + candidate.numeric.len()
}

fn is_pattern_numeric_candidate(task: &dyn AbstractNumericTask, numeric_var_id: usize) -> bool {
    task.numeric_variables()
        .get(numeric_var_id)
        .map(|numeric_var| numeric_var.get_type() == &NumericType::Regular)
        .unwrap_or(false)
}

fn collect_goal_related_propositional_closure(task: &dyn AbstractNumericTask) -> BTreeSet<usize> {
    let mut goal_related: BTreeSet<usize> = (0..task.get_num_goals())
        .map(|goal_id| task.get_goal_fact(goal_id).var())
        .collect();

    loop {
        let mut changed = false;

        for axiom in task.axioms() {
            let affected_var_id = axiom.var_id();
            if goal_related.contains(&affected_var_id) {
                for condition in axiom.conditions() {
                    changed |= goal_related.insert(condition.var());
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
        .map(CausalGraphVariable::Propositional)
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
