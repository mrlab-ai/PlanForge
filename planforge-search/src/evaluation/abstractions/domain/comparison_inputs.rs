//! The intervals numeric conditions are evaluated over in an abstract state.

use anyhow::{Result, ensure};
use planforge_sas::numeric_task::{AbstractNumericTask, NumericType, VariableIndex, ZERO_VALUE};
use planforge_sas::utils::float_tolerance;

use super::layout::StateLayout;
use super::partitions::NumericPartitions;
use planforge_sas::numeric_conditions::NumericCondition;
use planforge_sas::utils::interval::Interval;

/// Propagate every comparison tree's arithmetic nodes into the intervals of the
/// derived numeric variables they define.
///
/// `evaluate_interval_and_fill` fills the intervals unconditionally and returns
/// the tri-state truth value of the comparison over them; `None` means the
/// comparison is indeterminate on this box, not that the fill failed. Only
/// callers that want the truth value keep it.
pub fn fill_derived_numeric_intervals_from_comparison_trees(
    comparison_trees: &[NumericCondition],
    numeric_intervals: &mut [Interval],
) {
    for tree in comparison_trees {
        let _ = tree.evaluate_interval_and_fill(numeric_intervals);
    }
}

/// Fills `out` with the interval of every numeric variable over abstract
/// state `state`: a constant at its value, a regular or refined derived
/// variable at its partition, the cost unbounded, and an unrefined derived
/// variable through the comparison trees defining it. A derived variable no
/// tree defines, or one its operands leave undefined, is unbounded.
pub(super) fn fill_state_intervals(
    task: &dyn AbstractNumericTask,
    comparison_trees: &[NumericCondition],
    partitions: &NumericPartitions,
    layout: &StateLayout,
    state: usize,
    out: &mut Vec<Interval>,
) -> Result<()> {
    let initial_numeric_values = task.get_initial_numeric_state_values();
    out.clear();
    for (numeric_var_id, numeric_var) in task.numeric_variables().iter().enumerate() {
        let numeric_var_id = VariableIndex::from_usize(numeric_var_id);
        out.push(match numeric_var.get_type() {
            NumericType::Constant => {
                let value =
                    float_tolerance::canonicalize(initial_numeric_values[numeric_var_id.index()]);
                ensure!(
                    value.is_finite(),
                    "constant numeric value for var {numeric_var_id} must be finite, got {value}"
                );
                Interval::singleton(value)
            }
            NumericType::Derived if layout.numeric_domain_sizes()[numeric_var_id.index()] == 1 => {
                // Empty until the comparison trees fill it in.
                Interval::new(ZERO_VALUE, ZERO_VALUE, false, false)
            }
            NumericType::Derived | NumericType::Regular => partitions
                .partition_interval(numeric_var_id, layout.partition(state, numeric_var_id)),
            NumericType::Cost => Interval::unbounded(),
        });
    }

    fill_derived_numeric_intervals_from_comparison_trees(comparison_trees, out);
    for interval in out.iter_mut() {
        if interval.is_empty() {
            *interval = Interval::unbounded();
        }
    }
    Ok(())
}
