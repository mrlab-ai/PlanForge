use anyhow::{Context, Result, ensure};
use planners_sas::numeric::numeric_task::{AbstractNumericTask, NumericType};

use super::comparison_expression::{ComparisonTree, Interval};
use super::domain_abstraction::NumericPartitions;

fn has_refined_regular_dependency(
    task: &dyn AbstractNumericTask,
    numeric_domain_sizes: &[usize],
    numeric_var_id: usize,
    visiting: &mut [bool],
) -> bool {
    let Some(numeric_var) = task.numeric_variables().get(numeric_var_id) else {
        return false;
    };
    match numeric_var.get_type() {
        NumericType::Regular | NumericType::Cost => {
            numeric_domain_sizes
                .get(numeric_var_id)
                .copied()
                .unwrap_or(1)
                > 1
        }
        NumericType::Constant => false,
        NumericType::Derived => {
            if *visiting.get(numeric_var_id).unwrap_or(&false) {
                return false;
            }
            visiting[numeric_var_id] = true;
            let depends = task
                .assignment_axioms()
                .iter()
                .find(|axiom| axiom.get_affected_var_id() == numeric_var_id)
                .is_some_and(|axiom| {
                    has_refined_regular_dependency(
                        task,
                        numeric_domain_sizes,
                        axiom.get_left_var_id(),
                        visiting,
                    ) || has_refined_regular_dependency(
                        task,
                        numeric_domain_sizes,
                        axiom.get_right_var_id(),
                        visiting,
                    )
                });
            visiting[numeric_var_id] = false;
            depends
        }
    }
}

pub fn should_preserve_refined_derived_root(
    task: &dyn AbstractNumericTask,
    numeric_domain_sizes: &[usize],
    numeric_var_id: usize,
) -> bool {
    if task
        .numeric_variables()
        .get(numeric_var_id)
        .is_none_or(|var| var.get_type() != &NumericType::Derived)
    {
        return false;
    }

    let mut visiting = vec![false; task.numeric_variables().len()];
    !has_refined_regular_dependency(task, numeric_domain_sizes, numeric_var_id, &mut visiting)
}

pub fn seed_numeric_intervals_from_initial_state(task: &dyn AbstractNumericTask) -> Vec<Interval> {
    let initial_numeric_values = task.get_initial_numeric_state_values();
    let mut numeric_intervals: Vec<Interval> =
        vec![Interval::unbounded(); task.numeric_variables().len()];
    for (i, v) in task.numeric_variables().iter().enumerate() {
        if v.get_type() == &NumericType::Constant {
            numeric_intervals[i] = Interval::singleton(initial_numeric_values[i]);
        }
    }
    numeric_intervals
}

pub fn fill_derived_numeric_intervals_from_comparison_trees(
    comparison_trees: &[ComparisonTree],
    numeric_intervals: &mut [Interval],
) {
    for tree in comparison_trees {
        let _ = tree.evaluate_interval_and_fill(numeric_intervals);
    }
}

pub fn prepare_comparison_tree_inputs_from_initial_state(
    task: &dyn AbstractNumericTask,
    comparison_trees: &[ComparisonTree],
) -> Result<Vec<Interval>> {
    let initial_numeric_values = task.get_initial_numeric_state_values();
    let mut numeric_intervals: Vec<Interval> = Vec::with_capacity(initial_numeric_values.len());
    for (numeric_var_id, &value) in initial_numeric_values.iter().enumerate() {
        ensure!(
            value.is_finite() && !value.is_nan(),
            "initial numeric value for var {numeric_var_id} must be finite, got {value}"
        );
        numeric_intervals.push(Interval::singleton(value));
    }
    fill_derived_numeric_intervals_from_comparison_trees(comparison_trees, &mut numeric_intervals);
    Ok(numeric_intervals)
}

pub fn prepare_comparison_tree_inputs_from_abstract_state(
    task: &dyn AbstractNumericTask,
    comparison_trees: &[ComparisonTree],
    partitions: &NumericPartitions,
    state_hash: usize,
    num_props: usize,
    numeric_domain_sizes: &[usize],
    hash_multipliers: &[usize],
) -> Result<Vec<Interval>> {
    let num_numeric_vars = task.numeric_variables().len();
    ensure!(
        numeric_domain_sizes.len() == num_numeric_vars,
        "numeric_domain_sizes length mismatch: {} != {num_numeric_vars}",
        numeric_domain_sizes.len()
    );

    let initial_numeric_values = task.get_initial_numeric_state_values();
    let mut numeric_intervals = vec![Interval::new(0.0, 0.0, false, false); num_numeric_vars];
    let mut refined_derived_intervals: Vec<(usize, Interval)> = Vec::new();
    for (numeric_var_id, numeric_var) in task.numeric_variables().iter().enumerate() {
        match numeric_var.get_type() {
            NumericType::Constant => {
                let value = initial_numeric_values[numeric_var_id];
                ensure!(
                    value.is_finite() && !value.is_nan(),
                    "constant numeric value for var {numeric_var_id} must be finite, got {value}"
                );
                numeric_intervals[numeric_var_id] = Interval::singleton(value);
            }
            NumericType::Derived => {
                if numeric_domain_sizes[numeric_var_id] > 1 {
                    let abs_var = num_props + numeric_var_id;
                    ensure!(
                        abs_var < hash_multipliers.len(),
                        "missing hash multiplier for abstract numeric var {abs_var}"
                    );
                    let mult = hash_multipliers[abs_var] as i64;
                    let dom = numeric_domain_sizes[numeric_var_id] as i64;
                    ensure!(
                        dom > 0,
                        "numeric domain size must be > 0 for var {numeric_var_id}"
                    );
                    let part = (((state_hash as i64) / mult) % dom) as usize;
                    let interval = partitions
                        .partition_interval(numeric_var_id, part)
                        .with_context(|| {
                            format!(
                                "missing partition interval for numeric var {numeric_var_id} part {part}"
                            )
                        })?;
                    numeric_intervals[numeric_var_id] = interval;
                    refined_derived_intervals.push((numeric_var_id, interval));
                }
            }
            _ => {
                let abs_var = num_props + numeric_var_id;
                ensure!(
                    abs_var < hash_multipliers.len(),
                    "missing hash multiplier for abstract numeric var {abs_var}"
                );
                let mult = hash_multipliers[abs_var] as i64;
                let dom = numeric_domain_sizes[numeric_var_id] as i64;
                ensure!(
                    dom > 0,
                    "numeric domain size must be > 0 for var {numeric_var_id}"
                );
                let part = (((state_hash as i64) / mult) % dom) as usize;
                let interval = partitions.partition_interval(numeric_var_id, part).with_context(|| {
                    format!(
                        "missing partition interval for numeric var {numeric_var_id} part {part}"
                    )
                })?;
                numeric_intervals[numeric_var_id] = interval;
            }
        }
    }

    fill_derived_numeric_intervals_from_comparison_trees(comparison_trees, &mut numeric_intervals);
    for (numeric_var_id, interval) in refined_derived_intervals {
        if should_preserve_refined_derived_root(task, numeric_domain_sizes, numeric_var_id) {
            numeric_intervals[numeric_var_id] = interval;
        }
    }
    for interval in &mut numeric_intervals {
        if interval.is_empty() {
            *interval = Interval::unbounded();
        }
    }

    Ok(numeric_intervals)
}

pub fn evaluate_comparison_tree_from_initial_state(
    task: &dyn AbstractNumericTask,
    tree: &ComparisonTree,
) -> Result<Option<bool>> {
    let numeric_intervals =
        prepare_comparison_tree_inputs_from_initial_state(task, std::slice::from_ref(tree))?;
    Ok(tree.evaluate_interval_with_refined_roots(&numeric_intervals, &[]))
}

pub fn evaluate_comparison_tree_from_abstract_state(
    task: &dyn AbstractNumericTask,
    tree: &ComparisonTree,
    partitions: &NumericPartitions,
    state_hash: usize,
    num_props: usize,
    numeric_domain_sizes: &[usize],
    hash_multipliers: &[usize],
) -> Result<Option<bool>> {
    let numeric_intervals = prepare_comparison_tree_inputs_from_abstract_state(
        task,
        std::slice::from_ref(tree),
        partitions,
        state_hash,
        num_props,
        numeric_domain_sizes,
        hash_multipliers,
    )?;
    let refined_numeric_roots: Vec<bool> =
        numeric_domain_sizes.iter().map(|&size| size > 1).collect();
    Ok(tree.evaluate_interval_with_refined_roots(&numeric_intervals, &refined_numeric_roots))
}
