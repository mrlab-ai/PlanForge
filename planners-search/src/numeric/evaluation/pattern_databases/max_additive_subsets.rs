use std::collections::BTreeSet;

use planners_sas::numeric::numeric_task::{AbstractNumericTask, AssignmentOperation, Operator};

use super::numeric_support::NumericSupportContext;
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
    let numeric_support = NumericSupportContext::new(task);
    let num_prop_vars = task.variables().len();
    let num_num_vars = numeric_support.helper_space_len(task);
    let helper_dependency_sets: Vec<BTreeSet<usize>> = numeric_support
        .auxiliary_numeric_vars()
        .iter()
        .map(|auxiliary_numeric_var| {
            numeric_support
                .numeric_var_leaf_support_ids(task, auxiliary_numeric_var.source_numeric_var_id)
                .into_iter()
                .collect()
        })
        .collect();

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
            .map(|effect| effect.var_id() as usize)
            .collect();
        let numeric_targets =
            affected_numeric_targets(&numeric_support, &helper_dependency_sets, operator);

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
    numeric_support: &NumericSupportContext,
    helper_dependency_sets: &[BTreeSet<usize>],
    operator: &Operator,
) -> BTreeSet<usize> {
    let mut targets = BTreeSet::new();
    let mut direct_targets = BTreeSet::new();

    for effect in operator.assignment_effects() {
        let affected_var_id = effect.affected_var_id() as usize;
        direct_targets.insert(affected_var_id);
        targets.insert(affected_var_id);

        if let Some(helper_id) = numeric_support.helper_id_for_derived(affected_var_id) {
            targets.insert(helper_id);
        }

        match effect.operation() {
            AssignmentOperation::Assign
            | AssignmentOperation::Plus
            | AssignmentOperation::Minus
            | AssignmentOperation::Times
            | AssignmentOperation::Divide => {}
        }
    }

    for (auxiliary_numeric_var, dependency_set) in numeric_support
        .auxiliary_numeric_vars()
        .iter()
        .zip(helper_dependency_sets.iter())
    {
        if direct_targets
            .iter()
            .any(|target_var_id| dependency_set.contains(target_var_id))
        {
            targets.insert(auxiliary_numeric_var.helper_id);
        }
    }

    targets
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

#[cfg(test)]
mod tests {
    use planners_sas::numeric::numeric_task::{
        AssignmentEffect, Effect, ExplicitVariable, Fact, Metric, NumericRootTask, NumericType,
        NumericVariable, Operator,
    };

    use super::*;

    fn simple_var(name: &str) -> ExplicitVariable {
        ExplicitVariable::new(
            2,
            name.to_string(),
            vec![format!("{name}=0"), format!("{name}=1")],
            -1,
            1,
        )
    }

    fn disjoint_effect_task() -> NumericRootTask {
        NumericRootTask::new(
            1,
            Metric::new(true, -1),
            vec![simple_var("p"), simple_var("q")],
            vec![NumericVariable::new(
                "x".to_string(),
                NumericType::Regular,
                -1,
            )],
            vec![Fact::new(0, 1), Fact::new(1, 1)],
            vec![],
            vec![0, 0],
            vec![0.0],
            vec![
                Operator::new(
                    "set-p".to_string(),
                    vec![],
                    vec![Effect::new(vec![], 0, 0, 1)],
                    vec![],
                    1,
                ),
                Operator::new(
                    "set-q".to_string(),
                    vec![],
                    vec![Effect::new(vec![], 1, 0, 1)],
                    vec![],
                    1,
                ),
            ],
            vec![],
            vec![],
            vec![],
            (0, 0),
        )
    }

    fn shared_effect_task() -> NumericRootTask {
        NumericRootTask::new(
            1,
            Metric::new(true, -1),
            vec![simple_var("p"), simple_var("q")],
            vec![
                NumericVariable::new("c".to_string(), NumericType::Constant, -1),
                NumericVariable::new("x".to_string(), NumericType::Regular, -1),
                NumericVariable::new("y".to_string(), NumericType::Regular, -1),
            ],
            vec![],
            vec![],
            vec![0, 0],
            vec![1.0, 0.0, 0.0],
            vec![Operator::new(
                "touch-both".to_string(),
                vec![],
                vec![Effect::new(vec![], 0, 0, 1)],
                vec![AssignmentEffect::new(
                    1,
                    AssignmentOperation::Plus,
                    0,
                    false,
                    vec![],
                )],
                1,
            )],
            vec![],
            vec![],
            vec![],
            (0, 0),
        )
    }

    #[test]
    fn computes_additive_patterns_for_disjoint_effects() {
        let task = disjoint_effect_task();
        let patterns = PatternCollection::new(vec![
            Pattern::new(vec![0], vec![]),
            Pattern::new(vec![1], vec![]),
        ]);

        let additivity = compute_additive_vars(&task);
        let subsets = compute_max_additive_subsets(&patterns, &additivity);

        assert_eq!(subsets, vec![vec![0, 1]]);
    }

    #[test]
    fn marks_prop_and_numeric_as_non_additive_when_same_operator_touches_both() {
        let task = shared_effect_task();
        let additivity = compute_additive_vars(&task);

        assert!(!additivity.prop_to_num[0][1]);
        assert!(!additivity.num_to_prop[1][0]);
    }
}
