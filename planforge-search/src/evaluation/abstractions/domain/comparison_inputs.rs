//! The intervals numeric conditions are evaluated over in an abstract state.

use planforge_sas::numeric_task::{AbstractNumericTask, NumericType, VariableIndex, ZERO_VALUE};

use super::layout::StateLayout;
use super::partitions::NumericPartitions;
use planforge_sas::numeric_conditions::NumericCondition;
use planforge_sas::utils::interval::Interval;

use crate::evaluation::abstractions::common::regions::trivial_interval;

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

/// Fills `out` with the interval of every numeric variable that numeric
/// conditions are evaluated over: a refined coordinate at the interval
/// `refined` gives it, an unrefined derived variable through the comparison
/// trees defining it, and every other variable at its trivial interval. A
/// derived variable no tree defines, or one its operands leave undefined, is
/// unbounded.
pub(super) fn fill_condition_intervals(
    task: &dyn AbstractNumericTask,
    refined: impl Fn(VariableIndex) -> Option<Interval>,
    out: &mut Vec<Interval>,
) {
    let is_unrefined_derived = |var_id: VariableIndex| {
        task.numeric_variables()[var_id.index()].get_type() == &NumericType::Derived
            && refined(var_id).is_none()
    };
    out.clear();
    out.extend(
        (0..task.numeric_variables().len())
            .map(VariableIndex::from_usize)
            .map(|var_id| {
                if is_unrefined_derived(var_id) {
                    // Empty until the comparison trees fill it in.
                    Interval::new(ZERO_VALUE, ZERO_VALUE, false, false)
                } else {
                    refined(var_id).unwrap_or_else(|| trivial_interval(task, var_id))
                }
            }),
    );
    fill_derived_numeric_intervals_from_comparison_trees(task.numeric_conditions().all(), out);
    for (var_id, interval) in out.iter_mut().enumerate() {
        if interval.is_empty() {
            assert!(
                is_unrefined_derived(VariableIndex::from_usize(var_id)),
                "numeric variable {var_id} has an empty interval"
            );
            *interval = Interval::unbounded();
        }
    }
}

/// Fills `out` with the interval of every numeric variable over abstract
/// state `state`; see [`fill_condition_intervals`].
pub(super) fn fill_state_intervals(
    task: &dyn AbstractNumericTask,
    partitions: &NumericPartitions,
    layout: &StateLayout,
    state: usize,
    out: &mut Vec<Interval>,
) {
    fill_condition_intervals(
        task,
        |var_id| {
            (layout.numeric_domain_sizes()[var_id.index()] > 1
                || task.numeric_variables()[var_id.index()].get_type() == &NumericType::Regular)
                .then(|| partitions.partition_interval(var_id, layout.partition(state, var_id)))
        },
        out,
    );
}
