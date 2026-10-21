//! Deviation splits: separate the source of a concrete step from the states
//! of its abstract state that reach the planned abstract successor.

use std::fmt;

use anyhow::{Context, Result, ensure};
use planforge_sas::numeric_task::{ExplicitValueIndex, NumericValue, OperatorIndex, VariableIndex};
use planforge_sas::utils::interval::Interval;

use crate::evaluation::abstractions::common::cegar::semantics::CegarSemantics;
use crate::evaluation::abstractions::common::cegar::split::{Split, numeric_split_intervals};
use crate::evaluation::abstractions::common::cegar::view::AbstractionView;

/// A concrete step that leaves abstract state `source_state_id` by `op_id`
/// for a state outside the planned successor `target_state_id`.
#[derive(Clone, Copy)]
pub(crate) struct DeviationWitness<'a> {
    pub(crate) source_state_id: usize,
    pub(crate) target_state_id: usize,
    pub(crate) op_id: OperatorIndex,
    pub(crate) successor_prop: &'a [ExplicitValueIndex],
    pub(crate) source_numeric: &'a [NumericValue],
    pub(crate) successor_numeric: &'a [NumericValue],
}

/// Why a numeric dimension the successor leaves the target on has no split.
#[derive(Debug, Clone, Copy)]
enum NumericRejection {
    /// The preimage of the target contains the source.
    SourceInPreimage,
    /// Only an infinite boundary separates the source from the preimage.
    InfiniteBoundary { parent: Interval },
    /// The separating boundary does not split the source state strictly.
    NotStrict {
        parent: Interval,
        boundary: NumericValue,
        lower_includes_boundary: bool,
    },
}

/// A numeric dimension on which the successor leaves the target and no
/// split separates the source.
#[derive(Debug, Clone, Copy)]
struct RejectedNumericSplit {
    var_id: VariableIndex,
    source: NumericValue,
    successor: NumericValue,
    target: Interval,
    preimage: Interval,
    reason: NumericRejection,
}

impl fmt::Display for RejectedNumericSplit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self {
            var_id,
            source,
            successor,
            target,
            preimage,
            reason,
        } = self;
        write!(
            f,
            "var {var_id}: source={source}, successor={successor}, target={target:?}, preimage={preimage:?}"
        )?;
        match reason {
            NumericRejection::SourceInPreimage => write!(f, " contains source"),
            NumericRejection::InfiniteBoundary { parent } => {
                write!(
                    f,
                    ", parent={parent:?} has only infinite separating boundary"
                )
            }
            NumericRejection::NotStrict {
                parent,
                boundary,
                lower_includes_boundary,
            } => write!(
                f,
                ", parent={parent:?}, boundary={boundary}, lower_includes_boundary={lower_includes_boundary} is not strict"
            ),
        }
    }
}

/// The splits of the source state that separate the deviating step of
/// `witness` from the steps that reach its planned successor.
pub(crate) fn split_deviation_candidates(
    view: &impl AbstractionView,
    semantics: &CegarSemantics<'_>,
    witness: DeviationWitness<'_>,
) -> Result<Vec<Split>> {
    let DeviationWitness {
        source_state_id,
        target_state_id,
        op_id,
        successor_prop,
        source_numeric,
        successor_numeric,
    } = witness;
    let op = &semantics.task().get_operators()[op_id.index()];
    let target = view.region(target_state_id);
    let mut candidates = Vec::new();
    for (var_id, allowed) in target.propositions().iter().enumerate() {
        let var_id = VariableIndex::from_usize(var_id);
        if !semantics.regions().is_operator_variable(var_id) {
            continue;
        }
        let value = successor_prop[var_id.index()];
        if allowed.binary_search(&value).is_err() {
            ensure!(
                !op.effects().iter().any(|effect| effect.var_id() == var_id),
                "operator {op_id} effect image admitted wrong target prop region for var {var_id}"
            );
            candidates.push(Split::Propositional {
                state_id: source_state_id,
                var_id,
                wanted: allowed.clone(),
                witness_value: value,
            });
        }
    }

    let source_region = view.region(source_state_id);
    let mut rejected = Vec::new();
    for (var_id, &target_interval) in target.numeric.iter().enumerate() {
        let var_id = VariableIndex::from_usize(var_id);
        let successor = successor_numeric[var_id.index()];
        if target_interval.contains(successor) {
            continue;
        }
        let preimage = semantics
            .coordinates()
            .preimage(target_interval, op_id, var_id)
            .with_context(|| {
                format!(
                    "abstract transition for operator {op_id} has no numeric preimage for var {var_id} and target {target_interval:?}"
                )
            })?;
        let source = source_numeric[var_id.index()];
        let mut reject = |reason| {
            rejected.push(RejectedNumericSplit {
                var_id,
                source,
                successor,
                target: target_interval,
                preimage,
                reason,
            })
        };
        if preimage.contains(source) {
            reject(NumericRejection::SourceInPreimage);
            continue;
        }
        let (boundary, lower_includes_boundary) =
            if source < preimage.lower || (source == preimage.lower && !preimage.lower_closed) {
                (preimage.lower, !preimage.lower_closed)
            } else {
                ensure!(
                    source > preimage.upper || (source == preimage.upper && !preimage.upper_closed),
                    "numeric successor mismatch has no separating preimage boundary"
                );
                (preimage.upper, preimage.upper_closed)
            };
        let parent = source_region.numeric[var_id.index()];
        ensure!(
            parent.contains(source),
            "abstract source state {source_state_id} interval {parent:?} does not contain concrete numeric var {var_id}={source}"
        );
        if !boundary.is_finite() {
            reject(NumericRejection::InfiniteBoundary { parent });
            continue;
        }
        let integer_lattice = semantics.numeric_integer_lattice()[var_id.index()];
        if numeric_split_intervals(parent, boundary, lower_includes_boundary, integer_lattice)
            .is_none()
        {
            reject(NumericRejection::NotStrict {
                parent,
                boundary,
                lower_includes_boundary,
            });
            continue;
        }
        candidates.push(Split::Numeric {
            state_id: source_state_id,
            var_id,
            boundary,
            lower_includes_boundary,
            witness_value: source,
            integer_lattice,
        });
    }
    ensure!(
        !candidates.is_empty(),
        "concrete successor maps from abstract state {source_state_id} to a state other than abstract target {target_state_id}, but no sound strict split exists for operator {op_id} ({}); numeric split rejections: [{}]",
        op.name(),
        rejected
            .iter()
            .map(RejectedNumericSplit::to_string)
            .collect::<Vec<_>>()
            .join("; ")
    );
    Ok(candidates)
}
