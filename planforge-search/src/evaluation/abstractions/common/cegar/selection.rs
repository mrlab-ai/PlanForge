//! Choosing the split that refines a flaw.

use std::collections::HashSet;

use anyhow::{Context, Result, ensure};

use super::SplitSelection;
use super::progression::PlanCheck;
use super::semantics::CegarSemantics;
use super::split::{Split, SplitIdentity};
use super::view::AbstractionView;

/// Keep the candidates whose split grows the abstraction least, among the
/// refinable ones if there are any.
pub(crate) fn retain_min_growth_splits<V: AbstractionView, T>(
    view: &V,
    candidates: &mut Vec<T>,
    split: impl Fn(&T) -> &Split,
) -> Result<()> {
    if candidates
        .iter()
        .any(|candidate| view.refinable(split(candidate)))
    {
        candidates.retain(|candidate| view.refinable(split(candidate)));
    }
    let growths = candidates
        .iter()
        .map(|candidate| view.growth(split(candidate)))
        .collect::<Result<Vec<_>>>()?;
    let minimum = growths
        .iter()
        .min()
        .context("cannot rank an empty split candidate set by growth")?;
    let keep = growths
        .iter()
        .map(|growth| growth == minimum)
        .collect::<Vec<_>>();
    let mut index = 0;
    candidates.retain(|_| {
        let retain = keep[index];
        index += 1;
        retain
    });
    Ok(())
}

fn select_min_growth_split(
    view: &impl AbstractionView,
    semantics: &CegarSemantics<'_>,
    mut candidates: Vec<Split>,
    tag: u64,
) -> Result<Split> {
    ensure!(
        !candidates.is_empty(),
        "cannot select a refinement from an empty candidate set"
    );
    retain_min_growth_splits(view, &mut candidates, |split| split)?;
    let index = semantics.choose_split_index(&candidates, tag);
    Ok(candidates.swap_remove(index))
}

pub(crate) fn select_refinement_split(
    view: &impl AbstractionView,
    semantics: &CegarSemantics<'_>,
    mut candidates: Vec<Split>,
    tag: u64,
) -> Result<Split> {
    if candidates.iter().any(|split| view.refinable(split)) {
        candidates.retain(|split| view.refinable(split));
    }
    match semantics.split_selection() {
        SplitSelection::MinGrowth => select_min_growth_split(view, semantics, candidates, tag),
        SplitSelection::Random => {
            ensure!(
                !candidates.is_empty(),
                "cannot select a refinement from an empty candidate set"
            );
            let index = semantics.choose_random_split_index(candidates.len());
            Ok(candidates.swap_remove(index))
        }
    }
}

pub(crate) fn select_refinement(
    view: &impl AbstractionView,
    semantics: &CegarSemantics<'_>,
    candidates: Vec<Split>,
) -> Result<PlanCheck> {
    Ok(PlanCheck::Refine(select_refinement_split(
        view,
        semantics,
        candidates,
        0x454E_5449,
    )?))
}

pub(crate) fn push_unique_split(
    candidates: &mut Vec<Split>,
    identities: &mut HashSet<SplitIdentity>,
    split: Split,
) {
    if identities.insert(SplitIdentity::from(&split)) {
        candidates.push(split);
    }
}
