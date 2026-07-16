use anyhow::{Context, Result, ensure};
use planforge_sas::numeric_task::{AbstractNumericTask, NumericType};
use planforge_sas::utils::float_tolerance;

use super::comparison_expression::{ComparisonTree, Interval};
use super::domain_abstraction::NumericPartitions;

pub fn seed_numeric_intervals_from_initial_state(task: &dyn AbstractNumericTask) -> Vec<Interval> {
    let initial_numeric_values = task.get_initial_numeric_state_values();
    let mut numeric_intervals: Vec<Interval> =
        vec![Interval::unbounded(); task.numeric_variables().len()];
    for (i, v) in task.numeric_variables().iter().enumerate() {
        if v.get_type() == &NumericType::Constant {
            numeric_intervals[i] =
                Interval::singleton(float_tolerance::canonicalize(initial_numeric_values[i]));
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
    for (numeric_var_id, &raw_value) in initial_numeric_values.iter().enumerate() {
        if task.numeric_variables()[numeric_var_id].get_type() == &NumericType::Derived {
            numeric_intervals.push(Interval::new(0.0, 0.0, false, false));
            continue;
        }
        let value = float_tolerance::canonicalize(raw_value);
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
    let mut buf = Vec::new();
    prepare_comparison_tree_inputs_from_abstract_state_into(
        task,
        comparison_trees,
        partitions,
        state_hash,
        num_props,
        numeric_domain_sizes,
        hash_multipliers,
        &mut buf,
    )?;
    Ok(buf)
}

/// Resize-and-fill variant for callers that re-evaluate this on many states
/// and want to reuse one `Vec<Interval>` across the loop.
pub fn prepare_comparison_tree_inputs_from_abstract_state_into(
    task: &dyn AbstractNumericTask,
    comparison_trees: &[ComparisonTree],
    partitions: &NumericPartitions,
    state_hash: usize,
    num_props: usize,
    numeric_domain_sizes: &[usize],
    hash_multipliers: &[usize],
    out: &mut Vec<Interval>,
) -> Result<()> {
    let num_numeric_vars = task.numeric_variables().len();
    ensure!(
        numeric_domain_sizes.len() == num_numeric_vars,
        "numeric_domain_sizes length mismatch: {} != {num_numeric_vars}",
        numeric_domain_sizes.len()
    );

    let initial_numeric_values = task.get_initial_numeric_state_values();
    out.clear();
    out.resize(num_numeric_vars, Interval::new(0.0, 0.0, false, false));
    for (numeric_var_id, numeric_var) in task.numeric_variables().iter().enumerate() {
        match numeric_var.get_type() {
            NumericType::Constant => {
                let value = float_tolerance::canonicalize(initial_numeric_values[numeric_var_id]);
                ensure!(
                    value.is_finite() && !value.is_nan(),
                    "constant numeric value for var {numeric_var_id} must be finite, got {value}"
                );
                out[numeric_var_id] = Interval::singleton(value);
            }
            NumericType::Derived if numeric_domain_sizes[numeric_var_id] == 1 => {}
            NumericType::Derived | NumericType::Regular => {
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
                out[numeric_var_id] = interval;
            }
            NumericType::Cost => {
                out[numeric_var_id] = Interval::unbounded();
            }
        }
    }

    fill_derived_numeric_intervals_from_comparison_trees(comparison_trees, out);
    for interval in out.iter_mut() {
        if interval.is_empty() {
            *interval = Interval::unbounded();
        }
    }

    Ok(())
}

pub fn evaluate_comparison_tree_from_initial_state(
    task: &dyn AbstractNumericTask,
    tree: &ComparisonTree,
) -> Result<Option<bool>> {
    let mut numeric_intervals =
        prepare_comparison_tree_inputs_from_initial_state(task, std::slice::from_ref(tree))?;
    Ok(tree.evaluate_interval_and_fill(&mut numeric_intervals))
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
    let mut numeric_intervals = prepare_comparison_tree_inputs_from_abstract_state(
        task,
        std::slice::from_ref(tree),
        partitions,
        state_hash,
        num_props,
        numeric_domain_sizes,
        hash_multipliers,
    )?;
    Ok(tree.evaluate_interval_and_fill(&mut numeric_intervals))
}
