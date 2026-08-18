use std::sync::Arc;

use planforge_sas::utils::interval::Interval;

/// Sparse propositional active-set ID, narrowed to `u32` to halve the per-value
/// storage cost of `StateRegion::propositions`. Variable / value IDs come from
/// the SAS preprocessor, which already bounds them well below `u32::MAX`.
pub type PropValueId = u32;

#[derive(Debug, Clone, PartialEq)]
pub struct StateRegion {
    pub propositions: Arc<[Vec<PropValueId>]>,
    pub numeric: Arc<[Interval]>,
}

impl StateRegion {
    pub fn overlaps(&self, other: &Self) -> bool {
        prop_regions_overlap(&self.propositions, &other.propositions)
            && numeric_regions_overlap(&self.numeric, &other.numeric)
    }
}

pub(super) fn state_region_is_nonempty(region: &StateRegion) -> bool {
    region.propositions.iter().all(|values| !values.is_empty())
        && region.numeric.iter().all(|interval| !interval.is_empty())
}

pub(crate) fn state_region_intersection(
    left: &StateRegion,
    right: &StateRegion,
) -> Option<StateRegion> {
    debug_assert_eq!(
        left.propositions.len(),
        right.propositions.len(),
        "state-region propositional dimension mismatch"
    );
    debug_assert_eq!(
        left.numeric.len(),
        right.numeric.len(),
        "state-region numeric dimension mismatch"
    );
    let propositions = left
        .propositions
        .iter()
        .zip(right.propositions.iter())
        .map(|(left, right)| sorted_value_intersection(left, right))
        .collect::<Vec<_>>();
    if propositions.iter().any(Vec::is_empty) {
        return None;
    }
    let numeric = left
        .numeric
        .iter()
        .copied()
        .zip(right.numeric.iter().copied())
        .map(|(left, right)| left.intersection(&right))
        .collect::<Vec<_>>();
    if numeric.iter().any(Interval::is_empty) {
        return None;
    }
    Some(StateRegion {
        propositions: propositions.into(),
        numeric: numeric.into(),
    })
}

fn sorted_value_intersection(left: &[PropValueId], right: &[PropValueId]) -> Vec<PropValueId> {
    let mut intersection = Vec::with_capacity(left.len().min(right.len()));
    let (mut left_index, mut right_index) = (0, 0);
    while left_index < left.len() && right_index < right.len() {
        match left[left_index].cmp(&right[right_index]) {
            std::cmp::Ordering::Less => left_index += 1,
            std::cmp::Ordering::Greater => right_index += 1,
            std::cmp::Ordering::Equal => {
                intersection.push(left[left_index]);
                left_index += 1;
                right_index += 1;
            }
        }
    }
    intersection
}

fn sorted_value_difference(left: &[PropValueId], right: &[PropValueId]) -> Vec<PropValueId> {
    left.iter()
        .copied()
        .filter(|value| right.binary_search(value).is_err())
        .collect()
}

/// Returns a disjoint cover of `region \\ removed`. `removed` must be a
/// nonempty subset of `region`.
pub(super) fn subtract_state_region(
    region: &StateRegion,
    removed: &StateRegion,
) -> Vec<StateRegion> {
    let intersection = state_region_intersection(region, removed)
        .expect("removed state region must intersect its parent region");
    debug_assert_eq!(
        &intersection, removed,
        "removed state region must be a subset of its parent region"
    );

    let mut core = region.clone();
    let mut result = Vec::new();
    for var_id in 0..core.propositions.len() {
        let outside =
            sorted_value_difference(&core.propositions[var_id], &removed.propositions[var_id]);
        if !outside.is_empty() {
            let mut piece = core.clone();
            Arc::make_mut(&mut piece.propositions)[var_id] = outside;
            result.push(piece);
        }
        Arc::make_mut(&mut core.propositions)[var_id] = removed.propositions[var_id].clone();
    }
    for var_id in 0..core.numeric.len() {
        let parent = core.numeric[var_id];
        let cut = removed.numeric[var_id];
        let lower = Interval::new(
            parent.lower,
            cut.lower,
            parent.lower_closed,
            !cut.lower_closed,
        );
        if !lower.is_empty() {
            let mut piece = core.clone();
            Arc::make_mut(&mut piece.numeric)[var_id] = lower;
            result.push(piece);
        }
        let upper = Interval::new(
            cut.upper,
            parent.upper,
            !cut.upper_closed,
            parent.upper_closed,
        );
        if !upper.is_empty() {
            let mut piece = core.clone();
            Arc::make_mut(&mut piece.numeric)[var_id] = upper;
            result.push(piece);
        }
        Arc::make_mut(&mut core.numeric)[var_id] = cut;
    }
    debug_assert!(result.iter().all(state_region_is_nonempty));
    debug_assert!(regional_regions_are_disjoint(&result));
    result
}

fn regional_regions_are_disjoint(regions: &[StateRegion]) -> bool {
    regions.iter().enumerate().all(|(index, region)| {
        regions[index + 1..]
            .iter()
            .all(|other| !region.overlaps(other))
    })
}

fn prop_regions_overlap(left: &[Vec<PropValueId>], right: &[Vec<PropValueId>]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter()
        .zip(right.iter())
        .all(|(l, r)| sorted_value_sets_overlap(l, r))
}

pub(crate) fn sorted_value_sets_overlap(left: &[PropValueId], right: &[PropValueId]) -> bool {
    let mut i = 0;
    let mut j = 0;
    while i < left.len() && j < right.len() {
        match left[i].cmp(&right[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => return true,
        }
    }
    false
}

fn numeric_regions_overlap(left: &[Interval], right: &[Interval]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter().zip(right.iter()).all(|(l, r)| l.intersects(r))
}
