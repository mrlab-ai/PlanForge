#[cfg(test)]
mod tests;

use std::collections::HashMap;

use planners_sas::numeric::numeric_task::{
    AbstractNumericTask, AssignmentOperation, Fact, NumericType,
};

use super::comparison_expression::{ArithOp, ComparisonTree, Interval};

#[derive(Debug, Clone, PartialEq)]
pub struct NumericPartitions {
    partitions_by_numeric_var: Vec<Vec<Interval>>,
}

impl NumericPartitions {
    pub fn trivial(task: &dyn AbstractNumericTask) -> Self {
        let partitions_by_numeric_var = task
            .numeric_variables()
            .iter()
            .map(|v| match v.get_type() {
                NumericType::Regular => vec![Interval::unbounded()],
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

    pub fn partitions(&self, numeric_var_id: usize) -> Option<&[Interval]> {
        self.partitions_by_numeric_var
            .get(numeric_var_id)
            .map(|v| v.as_slice())
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

        let result_interval = match operation {
            AssignmentOperation::Assign => rhs,
            AssignmentOperation::Plus => ArithOp::Add.apply_interval(source_interval, rhs),
            AssignmentOperation::Minus => ArithOp::Sub.apply_interval(source_interval, rhs),
            AssignmentOperation::Times => ArithOp::Mul.apply_interval(source_interval, rhs),
            AssignmentOperation::Divide => ArithOp::Div.apply_interval(source_interval, rhs),
        };

        let Some(targets) = self.partitions(numeric_var_id) else {
            return vec![];
        };

        let mut out: Vec<usize> = Vec::new();
        for (target_id, &target_interval) in targets.iter().enumerate() {
            if intervals_overlap(result_interval, target_interval) {
                out.push(target_id);
            }
        }
        out
    }

    /// Splits the partition that contains `value` into two partitions.
    ///
    /// Returns `true` if a split was applied.
    pub fn split_at(&mut self, numeric_var_id: usize, value: f64, include_in_lower: bool) -> bool {
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
        true
    }
}

fn intervals_overlap(a: Interval, b: Interval) -> bool {
    if a.is_empty() || b.is_empty() {
        return false;
    }

    // Check a.max < b.min
    if (a.upper < b.lower) || (a.upper == b.lower && (!a.upper_closed || !b.lower_closed)) {
        return false;
    }

    // Check b.max < a.min
    if (b.upper < a.lower) || (b.upper == a.lower && (!b.upper_closed || !a.lower_closed)) {
        return false;
    }

    true
}

#[derive(Debug, Clone)]
pub struct ComparisonAxiomIndex {
    trees: Vec<ComparisonTree>,
    by_affected_var_id: HashMap<i32, usize>,
}

impl ComparisonAxiomIndex {
    pub fn from_task(task: &dyn AbstractNumericTask) -> Result<Self, String> {
        let mut trees: Vec<ComparisonTree> = Vec::new();
        let mut by_affected_var_id: HashMap<i32, usize> = HashMap::new();

        for comparison_axiom_id in 0..task.comparison_axioms().len() {
            let tree = ComparisonTree::from_task(task, comparison_axiom_id)
                .map_err(|e| format!("failed to build comparison tree: {e:?}"))?;
            let idx = trees.len();
            by_affected_var_id.insert(tree.affected_var_id, idx);
            trees.push(tree);
        }

        Ok(Self {
            trees,
            by_affected_var_id,
        })
    }

    pub fn is_comparison_axiom_variable(&self, prop_var_id: i32) -> bool {
        self.by_affected_var_id.contains_key(&prop_var_id)
    }

    pub fn comparison_tree(&self, prop_var_id: i32) -> Option<&ComparisonTree> {
        let tree_idx = *self.by_affected_var_id.get(&prop_var_id)?;
        self.trees.get(tree_idx)
    }

    /// Returns `true` if the given propositional precondition is *definitively* contradicted
    /// by evaluating its comparison axiom over the provided numeric intervals.
    ///
    /// This mirrors numeric-fd's “optimistic filtering”: reject only if definite contradiction;
    /// unknown (`None`) never contradicts.
    pub fn precondition_is_contradicted(&self, pre: &Fact, numeric_intervals: &[Interval]) -> bool {
        let var_id = pre.var() as i32;
        let Some(tree) = self.comparison_tree(var_id) else {
            return false;
        };

        let required_truth = pre.value() == 0;
        match tree.evaluate_interval(numeric_intervals) {
            Some(actual_truth) => actual_truth != required_truth,
            None => false,
        }
    }
}
