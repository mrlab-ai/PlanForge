#[cfg(test)]
mod tests;

use std::collections::BTreeSet;

use planforge_sas::numeric_task::{
    AbstractNumericTask, AssignmentEffect, AssignmentOperation, NumericType, Operator,
};

use super::pattern_collection::PatternCollection;
use super::projected_task::Pattern;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NumericVariableAdditivity {
    pub prop_to_prop: Vec<Vec<bool>>,
    pub prop_to_num: Vec<Vec<bool>>,
    pub num_to_prop: Vec<Vec<bool>>,
    pub num_to_num: Vec<Vec<bool>>,
}

pub fn are_patterns_additive(
    pattern1: &Pattern,
    pattern2: &Pattern,
    are_additive: &NumericVariableAdditivity,
) -> bool {
    for &var1 in &pattern1.regular {
        for &var2 in &pattern2.regular {
            if !are_additive.prop_to_prop[var1][var2] {
                return false;
            }
        }
        for &var2 in &pattern2.numeric {
            if !are_additive.prop_to_num[var1][var2] {
                return false;
            }
        }
    }

    for &var1 in &pattern1.numeric {
        for &var2 in &pattern2.numeric {
            if !are_additive.num_to_num[var1][var2] {
                return false;
            }
        }
        for &var2 in &pattern2.regular {
            if !are_additive.num_to_prop[var1][var2] {
                return false;
            }
        }
    }

    true
}

pub fn compute_additive_vars(task: &dyn AbstractNumericTask) -> NumericVariableAdditivity {
    let num_prop_vars = task.variables().len();
    let num_num_vars = task.numeric_variables().len();

    let mut are_additive = NumericVariableAdditivity {
        prop_to_prop: vec![vec![true; num_prop_vars]; num_prop_vars],
        prop_to_num: vec![vec![true; num_num_vars]; num_prop_vars],
        num_to_prop: vec![vec![true; num_prop_vars]; num_num_vars],
        num_to_num: vec![vec![true; num_num_vars]; num_num_vars],
    };

    for operator in task.get_operators() {
        let propositional_targets: Vec<_> = operator
            .effects()
            .iter()
            .map(|effect| effect.var_id())
            .collect();
        let numeric_targets = affected_numeric_targets(task, operator);

        for &var1 in &propositional_targets {
            for &var2 in &propositional_targets {
                are_additive.prop_to_prop[var1][var2] = false;
            }
            for &var2 in &numeric_targets {
                are_additive.prop_to_num[var1][var2] = false;
                are_additive.num_to_prop[var2][var1] = false;
            }
        }

        for &var1 in &numeric_targets {
            for &var2 in &numeric_targets {
                are_additive.num_to_num[var1][var2] = false;
            }
        }
    }

    are_additive
}

fn affected_numeric_targets(
    task: &dyn AbstractNumericTask,
    operator: &Operator,
) -> BTreeSet<usize> {
    let mut targets = BTreeSet::new();

    for effect in operator.assignment_effects() {
        if !assignment_effect_can_change_numeric_value(task, effect) {
            continue;
        }

        let affected_var_id = effect.affected_var_id();
        if task
            .numeric_variables()
            .get(affected_var_id)
            .is_some_and(|variable| variable.get_type() == &NumericType::Regular)
        {
            targets.insert(affected_var_id);
        }

        match effect.operation() {
            AssignmentOperation::Assign
            | AssignmentOperation::Plus
            | AssignmentOperation::Minus
            | AssignmentOperation::Times
            | AssignmentOperation::Divide => {}
        }
    }

    targets
}

fn assignment_effect_can_change_numeric_value(
    task: &dyn AbstractNumericTask,
    effect: &AssignmentEffect,
) -> bool {
    match effect.operation() {
        AssignmentOperation::Plus | AssignmentOperation::Minus => {
            if task
                .numeric_variables()
                .get(effect.var_id())
                .is_some_and(|numeric_var| numeric_var.get_type() == &NumericType::Constant)
            {
                let initial_numeric_values = task.get_initial_numeric_state_values();
                return initial_numeric_values
                    .get(effect.var_id())
                    .is_none_or(|value| *value != 0.0);
            }
            true
        }
        AssignmentOperation::Assign | AssignmentOperation::Times | AssignmentOperation::Divide => {
            true
        }
    }
}

pub fn compute_max_additive_subsets(
    patterns: &PatternCollection,
    are_additive: &NumericVariableAdditivity,
) -> Vec<Vec<usize>> {
    let mut compatibility_graph = vec![Vec::new(); patterns.len()];

    for left in 0..patterns.len() {
        for right in (left + 1)..patterns.len() {
            if are_patterns_additive(
                &patterns.as_slice()[left],
                &patterns.as_slice()[right],
                are_additive,
            ) {
                compatibility_graph[left].push(right);
                compatibility_graph[right].push(left);
            }
        }
    }

    let mut maximal_cliques = Vec::new();
    bron_kerbosch(
        &compatibility_graph,
        &mut Vec::new(),
        (0..patterns.len()).collect(),
        Vec::new(),
        &mut maximal_cliques,
    );

    let mut nondominated = prune_dominated_subsets(patterns, &maximal_cliques);
    if nondominated.is_empty() && !patterns.is_empty() {
        nondominated = (0..patterns.len()).map(|index| vec![index]).collect();
    }
    nondominated
}

fn bron_kerbosch(
    graph: &[Vec<usize>],
    current: &mut Vec<usize>,
    candidates: Vec<usize>,
    excluded: Vec<usize>,
    maximal_cliques: &mut Vec<Vec<usize>>,
) {
    if candidates.is_empty() && excluded.is_empty() {
        let mut clique = current.clone();
        clique.sort_unstable();
        maximal_cliques.push(clique);
        return;
    }

    let pivot = candidates
        .iter()
        .chain(excluded.iter())
        .copied()
        .max_by_key(|&vertex| graph[vertex].len());
    let pivot_neighbors: BTreeSet<_> = pivot
        .map(|vertex| graph[vertex].iter().copied().collect())
        .unwrap_or_default();

    let mut remaining_candidates = candidates.clone();
    let mut local_excluded = excluded;
    let vertices: Vec<_> = candidates
        .iter()
        .copied()
        .filter(|candidate| !pivot_neighbors.contains(candidate))
        .collect();

    for vertex in vertices {
        current.push(vertex);

        let neighbors: BTreeSet<_> = graph[vertex].iter().copied().collect();
        let next_candidates = remaining_candidates
            .iter()
            .copied()
            .filter(|candidate| neighbors.contains(candidate))
            .collect();
        let next_excluded = local_excluded
            .iter()
            .copied()
            .filter(|candidate| neighbors.contains(candidate))
            .collect();

        bron_kerbosch(
            graph,
            current,
            next_candidates,
            next_excluded,
            maximal_cliques,
        );
        current.pop();

        remaining_candidates.retain(|candidate| *candidate != vertex);
        local_excluded.push(vertex);
    }
}

fn prune_dominated_subsets(
    patterns: &PatternCollection,
    subsets: &[Vec<usize>],
) -> Vec<Vec<usize>> {
    let mut nondominated = Vec::new();
    let mut removed = vec![false; subsets.len()];

    for left_id in 0..subsets.len() {
        let left = &subsets[left_id];
        let mut useful = true;

        for right_id in 0..subsets.len() {
            if left_id == right_id || removed[right_id] {
                continue;
            }

            if collection_dominates(patterns, &subsets[right_id], left) {
                useful = false;
                break;
            }
        }

        if useful {
            let mut subset = left.clone();
            subset.sort_unstable();
            nondominated.push(subset);
        } else {
            removed[left_id] = true;
        }
    }

    nondominated.sort();
    nondominated.dedup();
    nondominated
}

fn collection_dominates(
    patterns: &PatternCollection,
    superset: &[usize],
    subset: &[usize],
) -> bool {
    subset.iter().all(|&subset_id| {
        let subset_pattern = &patterns.as_slice()[subset_id];
        superset
            .iter()
            .any(|&superset_id| subset_pattern.is_subset_of(&patterns.as_slice()[superset_id]))
    })
}
