//! Splits: the refinement steps of CEGAR. A split divides one abstract state on
//! one dimension; a Cartesian abstraction splits exactly that state, a domain
//! abstraction the dimension in every state.

use planforge_sas::utils::value_lattice::ValueLattice;
use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use planforge_sas::numeric_task::{
    ExplicitValueIndex, INF_VALUE, NEG_INF_VALUE, NumericValue, VariableIndex,
};
use planforge_sas::utils::interval::Interval;

use crate::evaluation::abstractions::common::cartesian_set::CartesianSet;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum Split {
    Propositional {
        state_id: usize,
        var_id: VariableIndex,
        wanted: Vec<ExplicitValueIndex>,
        witness_value: ExplicitValueIndex,
    },
    Numeric {
        state_id: usize,
        var_id: VariableIndex,
        boundary: NumericValue,
        lower_includes_boundary: bool,
        witness_value: NumericValue,
        lattice: ValueLattice,
    },
}

impl Split {
    pub(crate) fn state_id(&self) -> usize {
        match self {
            Self::Propositional { state_id, .. } | Self::Numeric { state_id, .. } => *state_id,
        }
    }

    pub(crate) fn dimension(&self) -> SplitDimension {
        match self {
            Self::Propositional { var_id, .. } => SplitDimension::Propositional(*var_id),
            Self::Numeric { var_id, .. } => SplitDimension::Numeric(*var_id),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum SplitDimension {
    Propositional(VariableIndex),
    Numeric(VariableIndex),
}

/// One of the two children of a split, in the order of a refinement node:
/// the wanted values of a propositional split or the lower values of a
/// numeric one first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ChildSide {
    First,
    Second,
}

impl ChildSide {
    /// The first child if `first`, else the second.
    pub(crate) fn first_if(first: bool) -> Self {
        if first { Self::First } else { Self::Second }
    }
}

/// The two children a split divides a region into.
pub(crate) struct SplitChildren {
    /// The child that keeps the split's witness.
    pub(crate) witness: CartesianSet,
    pub(crate) other: CartesianSet,
    pub(crate) witness_side: ChildSide,
}

impl SplitChildren {
    /// The children `first` and `second`, of which `witness_side` keeps the
    /// witness.
    fn new(first: CartesianSet, second: CartesianSet, witness_side: ChildSide) -> Self {
        let (witness, other) = match witness_side {
            ChildSide::First => (first, second),
            ChildSide::Second => (second, first),
        };
        Self {
            witness,
            other,
            witness_side,
        }
    }
}

/// The two children `split` splits `parent` into.
pub(crate) fn split_child_regions(parent: &CartesianSet, split: &Split) -> Result<SplitChildren> {
    match split {
        Split::Propositional {
            var_id,
            wanted,
            witness_value,
            ..
        } => {
            let current = parent
                .propositions()
                .get(var_id.index())
                .with_context(|| format!("split references missing prop var {var_id}"))?;
            ensure!(
                wanted.windows(2).all(|values| values[0] < values[1]),
                "propositional split values must be sorted and unique: {wanted:?}"
            );
            let wanted_values = current
                .iter()
                .copied()
                .filter(|value| wanted.binary_search(value).is_ok())
                .collect::<Vec<_>>();
            let other_values = current
                .iter()
                .copied()
                .filter(|value| wanted.binary_search(value).is_err())
                .collect::<Vec<_>>();
            ensure!(
                !wanted_values.is_empty() && !other_values.is_empty(),
                "non-strict propositional split on var {var_id}: current={current:?}, wanted={wanted:?}"
            );
            let witness_side =
                ChildSide::first_if(wanted_values.binary_search(witness_value).is_ok());
            let mut wanted_region = parent.clone();
            wanted_region.narrow_prop(*var_id, wanted_values);
            let mut other_region = parent.clone();
            other_region.narrow_prop(*var_id, other_values);
            Ok(SplitChildren::new(
                wanted_region,
                other_region,
                witness_side,
            ))
        }
        Split::Numeric {
            var_id,
            boundary,
            lower_includes_boundary,
            witness_value,
            lattice,
            ..
        } => {
            let current = *parent
                .numeric
                .get(var_id.index())
                .with_context(|| format!("split references missing numeric var {var_id}"))?;
            ensure!(
                current.can_split_at(*boundary, *lower_includes_boundary),
                "non-strict numeric split on var {var_id} at {boundary}: parent={current:?}, include_lower={lower_includes_boundary}"
            );
            let (lower, upper) = numeric_split_intervals(
                current,
                *boundary,
                *lower_includes_boundary,
                *lattice,
            )
            .with_context(|| {
                format!(
                    "non-strict numeric split on var {var_id} at {boundary}: parent={current:?}, include_lower={lower_includes_boundary}, lattice={lattice:?}"
                )
            })?;
            let witness_is_lower = lower.contains(*witness_value);
            ensure!(
                witness_is_lower ^ upper.contains(*witness_value),
                "numeric split does not place witness {witness_value} in exactly one child"
            );
            let witness_side = ChildSide::first_if(witness_is_lower);
            let mut lower_region = parent.clone();
            Arc::make_mut(&mut lower_region.numeric)[var_id.index()] = lower;
            let mut upper_region = parent.clone();
            Arc::make_mut(&mut upper_region.numeric)[var_id.index()] = upper;
            Ok(SplitChildren::new(lower_region, upper_region, witness_side))
        }
    }
}

/// Split a coordinate at neighbouring reachable lattice points. Query bounds
/// need not themselves lie on the lattice.
pub(crate) fn numeric_split_intervals(
    parent: Interval,
    boundary: NumericValue,
    lower_includes_boundary: bool,
    lattice: ValueLattice,
) -> Option<(Interval, Interval)> {
    let (lower_bound, upper_bound) = lattice.split_bounds(boundary, lower_includes_boundary)?;
    let lower = parent.intersection(&Interval::new(NEG_INF_VALUE, lower_bound, false, true));
    let upper = parent.intersection(&Interval::new(upper_bound, INF_VALUE, true, false));
    (!lower.is_empty() && !upper.is_empty()).then_some((lower, upper))
}
