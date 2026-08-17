#[cfg(test)]
mod tests;

use std::collections::BTreeSet;

use crate::evaluation::maximal_cliques::maximal_cliques;
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
    let maximal_cliques = maximal_cliques(patterns.len(), |left, right| {
        are_patterns_additive(
            &patterns.as_slice()[left],
            &patterns.as_slice()[right],
            are_additive,
        )
    });

    let mut nondominated = prune_dominated_subsets(patterns, &maximal_cliques);
    if nondominated.is_empty() && !patterns.is_empty() {
        nondominated = (0..patterns.len()).map(|index| vec![index]).collect();
    }
    nondominated
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
