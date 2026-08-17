use std::sync::Arc;

use planforge_sas::utils::float_tolerance;
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

    pub fn merge_hull(&mut self, other: &Self) {
        merge_state_region(self, other);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TransitionRegion {
    pub source: Arc<StateRegion>,
    pub target: Arc<StateRegion>,
}

impl TransitionRegion {
    pub fn overlaps(&self, other: &Self) -> bool {
        self.source.overlaps(&other.source) && self.target.overlaps(&other.target)
    }

    pub fn overlaps_parts(&self, source: &StateRegion, target: &StateRegion) -> bool {
        self.source.overlaps(source) && self.target.overlaps(target)
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

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub(super) struct TransitionRegionKey {
    source: StateRegionKey,
    target: StateRegionKey,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub(super) struct StateRegionKey {
    propositions: Arc<[Vec<PropValueId>]>,
    numeric: Vec<IntervalKey>,
}

#[derive(Copy, Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(super) struct IntervalKey {
    lower_bits: u64,
    upper_bits: u64,
    lower_closed: bool,
    upper_closed: bool,
}

pub(super) fn state_region_key(region: &StateRegion) -> StateRegionKey {
    StateRegionKey {
        propositions: region.propositions.clone(),
        numeric: region
            .numeric
            .iter()
            .map(|interval| IntervalKey {
                lower_bits: float_tolerance::canonical_bits(interval.lower),
                upper_bits: float_tolerance::canonical_bits(interval.upper),
                lower_closed: interval.lower_closed,
                upper_closed: interval.upper_closed,
            })
            .collect(),
    }
}

pub(super) fn transition_region_key(region: &TransitionRegion) -> TransitionRegionKey {
    transition_region_key_parts(&region.source, &region.target)
}

pub(super) fn transition_region_key_parts(
    source: &StateRegion,
    target: &StateRegion,
) -> TransitionRegionKey {
    TransitionRegionKey {
        source: state_region_key(source),
        target: state_region_key(target),
    }
}

pub(super) fn interval_key(interval: &Interval) -> IntervalKey {
    IntervalKey {
        lower_bits: float_tolerance::canonical_bits(interval.lower),
        upper_bits: float_tolerance::canonical_bits(interval.upper),
        lower_closed: interval.lower_closed,
        upper_closed: interval.upper_closed,
    }
}

pub(super) fn merge_transition_region(target: &mut TransitionRegion, source: &TransitionRegion) {
    merge_state_region(Arc::make_mut(&mut target.source), &source.source);
    merge_state_region(Arc::make_mut(&mut target.target), &source.target);
}

fn merge_state_region(target: &mut StateRegion, source: &StateRegion) {
    for (target_values, source_values) in Arc::make_mut(&mut target.propositions)
        .iter_mut()
        .zip(source.propositions.iter())
    {
        target_values.extend(source_values.iter().copied());
        target_values.sort_unstable();
        target_values.dedup();
    }
    for (target_interval, source_interval) in Arc::make_mut(&mut target.numeric)
        .iter_mut()
        .zip(source.numeric.iter())
    {
        *target_interval = interval_hull(*target_interval, *source_interval);
    }
}

fn interval_hull(left: Interval, right: Interval) -> Interval {
    let (lower, lower_closed) = if left.lower < right.lower {
        (left.lower, left.lower_closed)
    } else if left.lower > right.lower {
        (right.lower, right.lower_closed)
    } else {
        (left.lower, left.lower_closed || right.lower_closed)
    };
    let (upper, upper_closed) = if left.upper > right.upper {
        (left.upper, left.upper_closed)
    } else if left.upper < right.upper {
        (right.upper, right.upper_closed)
    } else {
        (left.upper, left.upper_closed || right.upper_closed)
    };
    Interval::new(lower, upper, lower_closed, upper_closed)
}

pub(super) fn state_regions_have_common_intersection<'a, I>(
    query: Option<&'a StateRegion>,
    selected: I,
    candidate: &'a StateRegion,
) -> bool
where
    I: Iterator<Item = &'a StateRegion> + Clone,
{
    let regions = query
        .into_iter()
        .chain(selected)
        .chain(std::iter::once(candidate));
    state_regions_have_common_intersection_from_slice(regions)
}

fn state_regions_have_common_intersection_from_slice<'a, I>(regions: I) -> bool
where
    I: Iterator<Item = &'a StateRegion> + Clone,
{
    let mut regions = regions.peekable();
    let Some(first) = regions.peek().copied() else {
        return true;
    };
    if regions.clone().any(|region| {
        region.propositions.len() != first.propositions.len()
            || region.numeric.len() != first.numeric.len()
    }) {
        return false;
    }

    for prop_id in 0..first.propositions.len() {
        let mut smallest = first.propositions[prop_id].as_slice();
        for region in regions.clone() {
            let values = region.propositions[prop_id].as_slice();
            if values.len() < smallest.len() {
                smallest = values;
            }
        }
        if !smallest.iter().any(|value| {
            regions
                .clone()
                .all(|region| region.propositions[prop_id].binary_search(value).is_ok())
        }) {
            return false;
        }
    }

    for numeric_id in 0..first.numeric.len() {
        if !intervals_have_common_intersection(
            regions.clone().map(|region| region.numeric[numeric_id]),
        ) {
            return false;
        }
    }
    true
}

fn intervals_have_common_intersection(intervals: impl Iterator<Item = Interval>) -> bool {
    let mut lower = f64::NEG_INFINITY;
    let mut lower_closed = false;
    let mut upper = f64::INFINITY;
    let mut upper_closed = false;
    for interval in intervals {
        if interval.lower > lower {
            lower = interval.lower;
            lower_closed = interval.lower_closed;
        } else if interval.lower == lower {
            lower_closed = lower_closed && interval.lower_closed;
        }

        if interval.upper < upper {
            upper = interval.upper;
            upper_closed = interval.upper_closed;
        } else if interval.upper == upper {
            upper_closed = upper_closed && interval.upper_closed;
        }
    }
    !Interval::new(lower, upper, lower_closed, upper_closed).is_empty()
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
