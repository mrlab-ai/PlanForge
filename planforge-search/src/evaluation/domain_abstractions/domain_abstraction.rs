#[cfg(test)]
mod tests;

use planforge_sas::numeric_task::{
    AbstractNumericTask, AssignmentOperation, NAN_VALUE, NumericType, NumericValue, VariableIndex,
};
use planforge_sas::utils::float_tolerance;

use super::utils::partition_for_value;
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
                    let value = float_tolerance::canonicalize(
                        *initial_numeric_values.get(i).unwrap_or(&NAN_VALUE),
                    );
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
    pub fn split_at(
        &mut self,
        numeric_var_id: VariableIndex,
        value: NumericValue,
        include_in_lower: bool,
    ) -> bool {
        // Numeric states use the canonical ABS_EPSILON lattice. Storing an
        // uncanonicalized flaw/regression value as a partition boundary makes
        // forward image and inverse-image overlap tests disagree.
        let value = float_tolerance::canonicalize(value);
        let Some(parts) = self
            .partitions_by_numeric_var
            .get_mut(numeric_var_id.index())
        else {
            return false;
        };
        let Some(part_id) = partition_for_value(parts, value) else {
            return false;
        };

        let iv = parts[part_id];
        if !iv.can_split_at(value, include_in_lower) {
            return false;
        }

        let lower = Interval::new(iv.lower, value, iv.lower_closed, include_in_lower);
        let upper = Interval::new(value, iv.upper, !include_in_lower, iv.upper_closed);
        if lower.is_empty() || upper.is_empty() {
            return false;
        }

        parts[part_id] = lower;
        parts.insert(part_id + 1, upper);
        true
    }
}
