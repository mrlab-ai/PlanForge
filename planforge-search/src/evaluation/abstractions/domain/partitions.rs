#[cfg(test)]
mod tests;

use planforge_sas::numeric_task::{
    AbstractNumericTask, AssignmentOperation, NumericType, NumericValue, VariableIndex,
};
use planforge_sas::utils::float_tolerance;

use crate::evaluation::abstractions::common::cegar::split::numeric_split_intervals;

use planforge_sas::numeric_conditions::ArithOp;
use planforge_sas::utils::interval::Interval;

#[derive(Debug, Clone, PartialEq)]
pub struct NumericPartitions {
    partitions_by_numeric_var: Vec<Vec<Interval>>,
}

impl NumericPartitions {
    pub fn trivial(task: &dyn AbstractNumericTask) -> Self {
        let initial_numeric_values = task.get_initial_numeric_state_values();
        let partitions_by_numeric_var: Vec<Vec<Interval>> = task
            .numeric_variables()
            .iter()
            .enumerate()
            .map(|(i, v)| match v.get_type() {
                NumericType::Constant => {
                    let value = float_tolerance::canonicalize(initial_numeric_values[i]);
                    if value.is_finite() {
                        vec![Interval::singleton(value)]
                    } else {
                        vec![Interval::unbounded()]
                    }
                }
                _ => vec![Interval::unbounded()],
            })
            .collect();
        Self {
            partitions_by_numeric_var,
        }
    }

    pub fn with_partitions(partitions_by_numeric_var: Vec<Vec<Interval>>) -> Self {
        Self {
            partitions_by_numeric_var,
        }
    }

    pub fn partitions(&self, numeric_var_id: VariableIndex) -> Option<&[Interval]> {
        self.partitions_by_numeric_var
            .get(numeric_var_id.index())
            .map(|v| v.as_slice())
    }

    pub fn partition_interval(
        &self,
        numeric_var_id: VariableIndex,
        partition_id: usize,
    ) -> Option<Interval> {
        self.partitions_by_numeric_var
            .get(numeric_var_id.index())
            .and_then(|parts| parts.get(partition_id).copied())
    }

    pub fn reachable_partitions(
        &self,
        numeric_var_id: VariableIndex,
        source_partition: usize,
        operation: &AssignmentOperation,
        rhs: Interval,
    ) -> Vec<usize> {
        let Some(source_interval) = self.partition_interval(numeric_var_id, source_partition)
        else {
            return vec![];
        };

        // Concrete numeric states are canonicalized after every assignment.
        // Keep abstract transition images on the same lattice so a rounding
        // artifact cannot manufacture overlap with an adjacent partition.
        let result_interval = match operation {
            AssignmentOperation::Assign => rhs,
            AssignmentOperation::Plus => ArithOp::Add.apply_interval(source_interval, rhs),
            AssignmentOperation::Minus => ArithOp::Sub.apply_interval(source_interval, rhs),
            AssignmentOperation::Times => ArithOp::Mul.apply_interval(source_interval, rhs),
            AssignmentOperation::Divide => ArithOp::Div.apply_interval(source_interval, rhs),
        }
        .canonicalized();

        let Some(targets) = self.partitions(numeric_var_id) else {
            return vec![];
        };

        let mut out: Vec<usize> = Vec::new();
        for (target_id, &target_interval) in targets.iter().enumerate() {
            if result_interval.intersects(&target_interval) {
                out.push(target_id);
            }
        }
        out
    }

    /// Splits the partition that contains `value` into two partitions.
    ///
    /// Returns `true` if a split was applied.
    /// Split the partition of `numeric_var_id` that contains `value` at
    /// `value`, which belongs to the lower part iff `include_in_lower`. On an
    /// integer lattice the parts are closed at the neighbouring integers.
    /// Returns whether a strict split happened.
    pub fn split_at(
        &mut self,
        numeric_var_id: VariableIndex,
        value: NumericValue,
        include_in_lower: bool,
        integer_lattice: bool,
    ) -> bool {
        // Numeric states use the canonical ABS_EPSILON lattice. Storing an
        // uncanonicalized flaw/regression value as a partition boundary makes
        // forward image and inverse-image overlap tests disagree.
        let value = float_tolerance::canonicalize(value);
        let parts = &mut self.partitions_by_numeric_var[numeric_var_id.index()];
        let Some(part_id) = partition_for_value(parts, value) else {
            return false;
        };
        let iv = parts[part_id];
        if !iv.can_split_at(value, include_in_lower) {
            return false;
        }
        let Ok((lower, upper)) =
            numeric_split_intervals(iv, value, include_in_lower, integer_lattice)
        else {
            return false;
        };
        parts[part_id] = lower;
        parts.insert(part_id + 1, upper);
        true
    }
}

/// The partition of `partitions` that contains `value`.
///
/// This is the one value-to-partition mapping of domain abstractions: CEGAR's
/// flaw search and splits, the initial-state hash of the distance tables, and
/// the heuristic's state lookup all go through it, so a value is assigned to
/// the same abstract value everywhere.
///
/// The mapping is exact. Numeric values are not integral in general (rational
/// constants, metric costs, derived linear views), so instead of comparing
/// with a tolerance the value is first rounded to the canonical
/// `float_tolerance::ABS_EPSILON` lattice that states and partition
/// boundaries already live on (`split_at` and the state registry canonicalize
/// both), and then compared exactly against the boundaries, honoring their
/// open/closed flags. The result is the same whatever the number of
/// partitions.
///
/// `partitions` are the disjoint, ascending intervals of one numeric variable.
/// `None` means no partition contains the value, which happens only for a
/// value outside a bounded layout (a constant that changed) or NaN.
pub(crate) fn partition_for_value(partitions: &[Interval], value: NumericValue) -> Option<usize> {
    let value = float_tolerance::canonicalize(value);
    if value.is_nan() {
        return None;
    }
    // The first partition that does not lie entirely below the value.
    let index = partitions
        .partition_point(|iv| iv.upper < value || (iv.upper == value && !iv.upper_closed));
    let found = partitions
        .get(index)
        .filter(|iv| iv.contains(value))
        .map(|_| index);
    debug_assert_eq!(
        found,
        partitions.iter().position(|iv| iv.contains(value)),
        "partitions {partitions:?} are not disjoint and ascending"
    );
    found
}
