//! Comparison splits: separate a concrete witness from the states of its
//! abstract state that decide a comparison the other way.

use anyhow::{Context, Result, bail, ensure};
use planforge_sas::numeric_conditions::ConditionValue;
use planforge_sas::numeric_task::{ExplicitValueIndex, NumericValue};

use crate::evaluation::abstractions::common::cegar::selection::{
    COMPARISON_TAG, retain_min_growth_splits, select_split, select_tied_split,
};
use crate::evaluation::abstractions::common::cegar::semantics::CegarSemantics;
use crate::evaluation::abstractions::common::cegar::split::{Split, numeric_split_intervals};
use crate::evaluation::abstractions::common::cegar::view::AbstractionView;
use crate::evaluation::abstractions::common::cegar::{SplitKind, SplitSelection};
use crate::evaluation::abstractions::common::coordinates::coordinate_value;

/// A split of a comparison's operand, and how its children evaluate the
/// comparison.
struct ComparisonCandidate {
    split: Split,
    /// The witness's child decides the comparison as the witness does.
    decides_witness: bool,
    /// The witness's child decides it as the witness does, the other child
    /// the other way.
    separates_truth: bool,
}

/// The split of abstract state `state_id` that refines comparison `tree_id`
/// at a concrete witness with `numeric` values, whose comparison variable
/// has `witness_value`: preferably one whose witness child decides the
/// comparison as the witness does.
pub(super) fn split_comparison(
    view: &impl AbstractionView,
    semantics: &CegarSemantics<'_>,
    state_id: usize,
    tree_id: usize,
    witness_value: ExplicitValueIndex,
    numeric: &[NumericValue],
) -> Result<Split> {
    let tree = semantics
        .task()
        .numeric_conditions()
        .get(tree_id)
        .with_context(|| format!("missing comparison tree {tree_id}"))?;
    let concrete_truth = tree.evaluate_point(numeric);
    ensure!(
        concrete_truth == comparison_truth(witness_value)?,
        "comparison tree {tree_id} evaluates to {concrete_truth} for a witness whose comparison value is {witness_value}"
    );
    let state = view.region(state_id);
    // The intervals of a child, written over the region's in place.
    let mut child_numeric = state.numeric.to_vec();
    let mut boundaries = Vec::new();
    let mut candidates = Vec::new();
    // The coordinates the condition is over, or its regular operands when it
    // is over no coordinate.
    for var_id in semantics.coordinates().refinement_dimensions(tree) {
        let witness_value = coordinate_value(semantics.coordinates().views(), numeric, var_id);
        ensure!(
            witness_value.is_finite(),
            "comparison split witness for numeric var {var_id} is non-finite: {witness_value}"
        );
        let parent = state.numeric[var_id.index()];
        boundaries.clear();
        if semantics.split_kind() == SplitKind::Wanted {
            boundaries.extend_from_slice(semantics.wanted_split_boundaries());
        }
        boundaries.push(witness_value);
        boundaries.sort_by(NumericValue::total_cmp);
        boundaries.dedup_by(|left, right| *left == *right);

        let lattice = semantics.numeric_lattices()[var_id.index()];
        for &boundary in &boundaries {
            for lower_includes_boundary in [true, false] {
                let Some((lower, upper)) =
                    numeric_split_intervals(parent, boundary, lower_includes_boundary, lattice)
                else {
                    continue;
                };
                let (witness_child, other_child) = if lower.contains(witness_value) {
                    (lower, upper)
                } else {
                    ensure!(
                        upper.contains(witness_value),
                        "comparison split at {boundary} loses witness {witness_value} for numeric var {var_id}"
                    );
                    (upper, lower)
                };
                child_numeric[var_id.index()] = witness_child;
                let witness_result = tree.evaluate_interval(&child_numeric);
                ensure!(
                    witness_result != Some(!concrete_truth),
                    "comparison interval for tree {tree_id} excludes its concrete witness after splitting numeric var {var_id}"
                );
                child_numeric[var_id.index()] = other_child;
                let other_result = tree.evaluate_interval(&child_numeric);
                child_numeric[var_id.index()] = parent;
                let decides_witness = witness_result == Some(concrete_truth);
                candidates.push(ComparisonCandidate {
                    split: Split::Numeric {
                        state_id,
                        var_id,
                        boundary,
                        lower_includes_boundary,
                        witness_value,
                        lattice,
                    },
                    decides_witness,
                    separates_truth: decides_witness && other_result == Some(!concrete_truth),
                });
            }
        }
    }
    ensure!(
        !candidates.is_empty(),
        "comparison tree {tree_id} has no strict split in abstract state {state_id}"
    );
    let min_growth = semantics.split_selection() == SplitSelection::MinGrowth;
    if min_growth {
        retain_min_growth_splits(view, &mut candidates, |candidate| &candidate.split)?;
    }
    if semantics.split_kind() == SplitKind::Wanted {
        retain_if_any(&mut candidates, |candidate| candidate.separates_truth);
    }
    retain_if_any(&mut candidates, |candidate| candidate.decides_witness);
    let splits = candidates
        .into_iter()
        .map(|candidate| candidate.split)
        .collect();
    if min_growth {
        // The candidates left are refinable and grow the abstraction least.
        Ok(select_tied_split(semantics, splits, COMPARISON_TAG))
    } else {
        select_split(view, semantics, splits, COMPARISON_TAG)
    }
}

/// Keep the candidates that are `preferred`, if there are any.
fn retain_if_any<T>(candidates: &mut Vec<T>, preferred: impl Fn(&T) -> bool) {
    if candidates.iter().any(&preferred) {
        candidates.retain(preferred);
    }
}

/// The truth a comparison variable's value stands for.
fn comparison_truth(prop_value: ExplicitValueIndex) -> Result<bool> {
    match ConditionValue::from_usize(prop_value.index()) {
        Some(ConditionValue::True) => Ok(true),
        Some(ConditionValue::False) => Ok(false),
        None => bail!("invalid comparison fact value {prop_value}"),
    }
}
