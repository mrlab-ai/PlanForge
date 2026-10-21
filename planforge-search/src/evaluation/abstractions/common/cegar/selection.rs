//! Choosing the split that refines a flaw.

use std::collections::HashSet;

use anyhow::{Result, ensure};

use super::SplitSelection;
use super::semantics::CegarSemantics;
use super::split::Split;
use super::view::AbstractionView;

// The decisions split selection draws a seeded choice for: the splits of a
// failed goal, precondition or deviation, of a comparison fact, and of all
// flaws of the plan. Each tag spells four letters of its decision in ASCII
// ("GOAL", "PREC", "DEVI", "COMP", "ENTI" for the entire plan).
pub(crate) const GOAL_TAG: u64 = 0x474F_414C;
pub(crate) const PRECONDITION_TAG: u64 = 0x5052_4543;
pub(crate) const DEVIATION_TAG: u64 = 0x4445_5649;
pub(crate) const COMPARISON_TAG: u64 = 0x434F_4D50;
pub(crate) const WHOLE_PLAN_TAG: u64 = 0x454E_5449;

/// Keep the candidates whose split the abstraction can take within its
/// limits, if there are any.
pub(crate) fn retain_refinable_splits<V: AbstractionView, T>(
    view: &V,
    candidates: &mut Vec<T>,
    split: impl Fn(&T) -> &Split,
) {
    if candidates
        .iter()
        .any(|candidate| view.refinable(split(candidate)))
    {
        candidates.retain(|candidate| view.refinable(split(candidate)));
    }
}

/// Keep the candidates whose split grows the abstraction least, among the
/// refinable ones if there are any.
pub(crate) fn retain_min_growth_splits<V: AbstractionView, T>(
    view: &V,
    candidates: &mut Vec<T>,
    split: impl Fn(&T) -> &Split,
) -> Result<()> {
    ensure!(
        !candidates.is_empty(),
        "cannot rank an empty split candidate set by growth"
    );
    retain_refinable_splits(view, candidates, &split);
    let growths = candidates
        .iter()
        .map(|candidate| view.growth(split(candidate)))
        .collect::<Result<Vec<_>>>()?;
    let minimum = growths
        .iter()
        .min()
        .expect("a nonempty candidate set has a minimum growth");
    let mut growth = growths.iter();
    candidates.retain(|_| growth.next().expect("one growth per candidate") == minimum);
    Ok(())
}

/// The split that refines a flaw among `candidates`, preferring refinable
/// ones: by the configured split selection, with ties broken for decision
/// `tag`.
pub(crate) fn select_split(
    view: &impl AbstractionView,
    semantics: &CegarSemantics<'_>,
    mut candidates: Vec<Split>,
    tag: u64,
) -> Result<Split> {
    ensure!(
        !candidates.is_empty(),
        "cannot select a refinement from an empty candidate set"
    );
    match semantics.split_selection() {
        SplitSelection::MinGrowth => {
            retain_min_growth_splits(view, &mut candidates, |split| split)?;
            Ok(select_tied_split(semantics, candidates, tag))
        }
        SplitSelection::Random => {
            retain_refinable_splits(view, &mut candidates, |split| split);
            let index = semantics.choose_random_split_index(candidates.len());
            Ok(candidates.swap_remove(index))
        }
    }
}

/// The split among the equally good, nonempty `candidates` that the tie
/// breaking picks for decision `tag`.
pub(crate) fn select_tied_split(
    semantics: &CegarSemantics<'_>,
    mut candidates: Vec<Split>,
    tag: u64,
) -> Split {
    let index = semantics.choose_split_index(&candidates, tag);
    candidates.swap_remove(index)
}

pub(crate) fn push_unique_split(
    candidates: &mut Vec<Split>,
    seen: &mut HashSet<Split>,
    split: Split,
) {
    if seen.insert(split.clone()) {
        candidates.push(split);
    }
}
