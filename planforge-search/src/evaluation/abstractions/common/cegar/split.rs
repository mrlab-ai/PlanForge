//! Splits: the refinement steps of CEGAR. A split divides one abstract state on
//! one dimension; a Cartesian abstraction splits exactly that state, a domain
//! abstraction the dimension in every state.

use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use planforge_sas::numeric_task::{
    ExplicitValueIndex, INF_VALUE, NEG_INF_VALUE, NumericValue, ONE_VALUE, VariableIndex,
};
use planforge_sas::utils::float_tolerance;
use planforge_sas::utils::interval::Interval;

use crate::evaluation::abstractions::common::cartesian_set::CartesianSet;

/// Whether two costs or values agree up to a relative tolerance of
/// [`float_tolerance::AGREEMENT_EPSILON`].
///
/// TODO(canonical values): this is also the integer-lattice test (here and in
/// `semantics.rs`), where a relative tolerance is wrong: from |v| >= 5e6 a
/// half-integer counts as an integer, and an integer-lattice split at it
/// drops the states between `b` and `b + 1`. If every numeric value is
/// canonical, the test should be exact (`v == v.round()`), but
/// [`float_tolerance::canonicalize`] does not yet keep large integers fixed
/// (see there). Decide both together; costs keep this tolerance.
pub(crate) fn approximately_equal(left: f64, right: f64) -> bool {
    (left - right).abs()
        <= float_tolerance::AGREEMENT_EPSILON * left.abs().max(right.abs()).max(1.0)
}

#[derive(Debug, Clone)]
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
        integer_lattice: bool,
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

/// Equal for the same split of the same state; numeric values compare by
/// their canonical bits.
impl PartialEq for Split {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Propositional {
                    state_id,
                    var_id,
                    wanted,
                    witness_value,
                },
                Self::Propositional {
                    state_id: other_state_id,
                    var_id: other_var_id,
                    wanted: other_wanted,
                    witness_value: other_witness_value,
                },
            ) => {
                state_id == other_state_id
                    && var_id == other_var_id
                    && wanted == other_wanted
                    && witness_value == other_witness_value
            }
            (
                Self::Numeric {
                    state_id,
                    var_id,
                    boundary,
                    lower_includes_boundary,
                    witness_value,
                    integer_lattice,
                },
                Self::Numeric {
                    state_id: other_state_id,
                    var_id: other_var_id,
                    boundary: other_boundary,
                    lower_includes_boundary: other_lower_includes_boundary,
                    witness_value: other_witness_value,
                    integer_lattice: other_integer_lattice,
                },
            ) => {
                state_id == other_state_id
                    && var_id == other_var_id
                    && float_tolerance::canonical_bits(*boundary)
                        == float_tolerance::canonical_bits(*other_boundary)
                    && lower_includes_boundary == other_lower_includes_boundary
                    && float_tolerance::canonical_bits(*witness_value)
                        == float_tolerance::canonical_bits(*other_witness_value)
                    && integer_lattice == other_integer_lattice
            }
            _ => false,
        }
    }
}

impl Eq for Split {}

impl std::hash::Hash for Split {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::Propositional {
                state_id,
                var_id,
                wanted,
                witness_value,
            } => (state_id, var_id, wanted, witness_value).hash(state),
            Self::Numeric {
                state_id,
                var_id,
                boundary,
                lower_includes_boundary,
                witness_value,
                integer_lattice,
            } => (
                state_id,
                var_id,
                float_tolerance::canonical_bits(*boundary),
                lower_includes_boundary,
                float_tolerance::canonical_bits(*witness_value),
                integer_lattice,
            )
                .hash(state),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
            integer_lattice,
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
                *integer_lattice,
            )
            .with_context(|| {
                format!(
                    "non-strict numeric split on var {var_id} at {boundary}: parent={current:?}, include_lower={lower_includes_boundary}, integer_lattice={integer_lattice}"
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

/// The lower and upper child of `parent` when it is split at `boundary`, or
/// `None` if `boundary` does not split `parent` into two nonempty children.
/// On an integer lattice the children are closed at the neighbouring
/// integers, and a non-integer boundary splits nothing.
pub(crate) fn numeric_split_intervals(
    parent: Interval,
    boundary: NumericValue,
    lower_includes_boundary: bool,
    integer_lattice: bool,
) -> Option<(Interval, Interval)> {
    let (lower_bound, upper_bound, lower_closed, upper_closed) = if integer_lattice {
        if !boundary.is_finite() || !approximately_equal(boundary.value(), boundary.value().round())
        {
            return None;
        }
        if lower_includes_boundary {
            (boundary, boundary + ONE_VALUE, true, true)
        } else {
            (boundary - ONE_VALUE, boundary, true, true)
        }
    } else {
        (
            boundary,
            boundary,
            lower_includes_boundary,
            !lower_includes_boundary,
        )
    };
    let lower = parent.intersection(&Interval::new(
        NEG_INF_VALUE,
        lower_bound,
        false,
        lower_closed,
    ));
    let upper = parent.intersection(&Interval::new(upper_bound, INF_VALUE, upper_closed, false));
    (!lower.is_empty() && !upper.is_empty()).then_some((lower, upper))
}
