#[cfg(test)]
mod tests;

use planforge_sas::numeric_task::{AbstractNumericTask, AssignmentOperation, NumericType};
use planforge_sas::utils::float_tolerance;

use super::utils::EquispacedPartitioning;
use planforge_sas::numeric_conditions::ArithOp;
use planforge_sas::utils::interval::Interval;

#[derive(Debug, Clone, PartialEq)]
pub struct NumericPartitions {
    partitions_by_numeric_var: Vec<Vec<Interval>>,
    /// Per-var cached descriptor for O(1) `partition_for_value` lookup.
    /// `Some` when the layout is contiguous + uniform-width + at most one
    /// unbounded tail on each side; otherwise `None` and callers fall back to
    /// the tolerant binary search. Rebuilt whenever the partitions mutate
    /// (`split_at`, `merge_at`, …) — keep this in sync with the `Vec<Interval>`.
    equispaced_by_numeric_var: Vec<Option<EquispacedPartitioning>>,
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
                        *initial_numeric_values.get(i).unwrap_or(&f64::NAN),
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
        let equispaced_by_numeric_var = compute_equispaced(&partitions_by_numeric_var);
        Self {
            partitions_by_numeric_var,
            equispaced_by_numeric_var,
        }
    }

    pub fn with_partitions(partitions_by_numeric_var: Vec<Vec<Interval>>) -> Self {
        let equispaced_by_numeric_var = compute_equispaced(&partitions_by_numeric_var);
        Self {
            partitions_by_numeric_var,
            equispaced_by_numeric_var,
        }
    }

    pub fn partitions(&self, numeric_var_id: usize) -> Option<&[Interval]> {
        self.partitions_by_numeric_var
            .get(numeric_var_id)
            .map(|v| v.as_slice())
    }

    /// Equispaced-partition descriptor for `numeric_var_id`, if its current
    /// layout fits the `EquispacedPartitioning` shape.
    ///
    /// Currently unused: the previous fast-path consumer was the heuristic's
    /// `numeric_partition_for_projected_value`, which fell out of sync with
    /// the tolerant `partition_for_value` on values that lie *exactly* on a
    /// partition boundary (the cast-based lookup ignores the per-interval
    /// closed/open flags, so boundary-aligned values — i.e., every CEGAR
    /// split point — could land in the wrong partition). Kept because the
    /// descriptor is still maintained on `split_at`; a future fix that makes
    /// `EquispacedPartitioning::lookup` boundary-aware can re-enable it.
    #[allow(dead_code)]
    pub(crate) fn equispaced(&self, numeric_var_id: usize) -> Option<&EquispacedPartitioning> {
        self.equispaced_by_numeric_var
            .get(numeric_var_id)
            .and_then(Option::as_ref)
    }

    pub fn partition_interval(
        &self,
        numeric_var_id: usize,
        partition_id: usize,
    ) -> Option<Interval> {
        self.partitions_by_numeric_var
            .get(numeric_var_id)
            .and_then(|parts| parts.get(partition_id).copied())
    }

    pub fn reachable_partitions(
        &self,
        numeric_var_id: usize,
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
    pub fn split_at(&mut self, numeric_var_id: usize, value: f64, include_in_lower: bool) -> bool {
        // Numeric states use the canonical ABS_EPSILON lattice. Storing an
        // uncanonicalized flaw/regression value as a partition boundary makes
        // forward image and inverse-image overlap tests disagree.
        let value = float_tolerance::canonicalize(value);
        let Some(parts) = self.partitions_by_numeric_var.get_mut(numeric_var_id) else {
            return false;
        };
        let Some(part_id) = parts.iter().position(|iv| iv.contains(value)) else {
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
        // Cache is per-var; recompute only the affected entry.
        self.equispaced_by_numeric_var[numeric_var_id] =
            EquispacedPartitioning::detect(&self.partitions_by_numeric_var[numeric_var_id]);
        true
    }
}

fn compute_equispaced(
    partitions_by_numeric_var: &[Vec<Interval>],
) -> Vec<Option<EquispacedPartitioning>> {
    partitions_by_numeric_var
        .iter()
        .map(|parts| EquispacedPartitioning::detect(parts))
        .collect()
}
