#[cfg(test)]
mod tests;

use planforge_sas::numeric_task::{AbstractNumericTask, NumericValue, VariableIndex};
use planforge_sas::utils::value_lattice::ValueLattice;

use crate::evaluation::abstractions::common::cegar::split::numeric_split_intervals;

use planforge_sas::utils::interval::Interval;

use crate::evaluation::abstractions::common::regions::trivial_interval;

/// The partitions of every numeric variable. Boundaries are canonical, so
/// equal partitions hash alike by their canonical bits.
#[derive(Debug, Clone, PartialEq)]
pub struct NumericPartitions {
    partitions_by_numeric_var: Vec<Vec<Interval>>,
}

impl Eq for NumericPartitions {}

impl std::hash::Hash for NumericPartitions {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for partitions in &self.partitions_by_numeric_var {
            partitions.len().hash(state);
            for interval in partitions {
                interval.lower.hash(state);
                interval.upper.hash(state);
                (interval.lower_closed, interval.upper_closed).hash(state);
            }
        }
    }
}

impl NumericPartitions {
    pub fn trivial(task: &dyn AbstractNumericTask) -> Self {
        let partitions_by_numeric_var = (0..task.numeric_variables().len())
            .map(|var_id| vec![trivial_interval(task, VariableIndex::from_usize(var_id))])
            .collect();
        Self {
            partitions_by_numeric_var,
        }
    }

    #[cfg(test)]
    pub fn with_partitions(partitions_by_numeric_var: Vec<Vec<Interval>>) -> Self {
        Self {
            partitions_by_numeric_var,
        }
    }

    /// The partitions of `numeric_var_id`: disjoint, ascending, and covering
    /// every value the variable can take.
    pub fn partitions(&self, numeric_var_id: VariableIndex) -> &[Interval] {
        &self.partitions_by_numeric_var[numeric_var_id.index()]
    }

    /// The number of partitions of every numeric variable.
    pub fn numeric_domain_sizes(&self) -> Vec<usize> {
        self.partitions_by_numeric_var
            .iter()
            .map(Vec::len)
            .collect()
    }

    pub fn partition_interval(
        &self,
        numeric_var_id: VariableIndex,
        partition_id: usize,
    ) -> Interval {
        self.partitions(numeric_var_id)[partition_id]
    }

    /// The partitions of `numeric_var_id` that `interval` intersects,
    /// ascending. Partitions are disjoint and ordered, so they are one run
    /// found by binary search.
    pub fn intersecting(
        &self,
        numeric_var_id: VariableIndex,
        interval: Interval,
    ) -> impl Iterator<Item = usize> + '_ {
        let parts = self.partitions(numeric_var_id);
        let first = parts.partition_point(|part| {
            part.upper < interval.lower
                || (part.upper == interval.lower && !(part.upper_closed && interval.lower_closed))
        });
        (first..parts.len()).take_while(move |&part_id| parts[part_id].intersects(&interval))
    }

    /// Split the partition of `numeric_var_id` that contains `value` at
    /// `value`, which belongs to the lower part iff `include_in_lower`. On an
    /// integer lattice the parts are closed at the neighbouring integers.
    /// Returns whether a strict split happened.
    pub fn split_at(
        &mut self,
        numeric_var_id: VariableIndex,
        value: NumericValue,
        include_in_lower: bool,
        lattice: ValueLattice,
    ) -> bool {
        let parts = &mut self.partitions_by_numeric_var[numeric_var_id.index()];
        let Some(part_id) = partition_for_value(parts, value) else {
            return false;
        };
        let iv = parts[part_id];
        if !iv.can_split_at(value, include_in_lower) {
            return false;
        }
        let Some((lower, upper)) = numeric_split_intervals(iv, value, include_in_lower, lattice)
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
/// flaw search and splits, the initial abstract state of the distance tables, and
/// the heuristic's state lookup all go through it, so a value is assigned to
/// the same abstract value everywhere.
///
/// The mapping is exact. Numeric values are not integral in general (rational
/// constants, metric costs, derived linear views), so instead of comparing
/// with a tolerance the value is first rounded to the canonical
/// `float_tolerance::ABS_EPSILON` lattice that states and partition
/// boundaries already live on (NumericValue canonicalizes
/// both), and then compared exactly against the boundaries, honoring their
/// open/closed flags. The result is the same whatever the number of
/// partitions.
///
/// `partitions` are the disjoint, ascending intervals of one numeric variable.
/// `None` means no partition contains the value, which happens only for a
/// value outside a bounded layout (a constant that changed).
pub(crate) fn partition_for_value(partitions: &[Interval], value: NumericValue) -> Option<usize> {
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
