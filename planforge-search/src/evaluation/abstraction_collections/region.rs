use std::sync::Arc;

use planforge_sas::utils::interval::Interval;

/// Sparse propositional active-set ID, narrowed to `u32` to halve the per-value
/// storage cost of `StateRegion::propositions`. Variable / value IDs come from
/// the SAS preprocessor, which already bounds them well below `u32::MAX`.
pub type PropValueId = u32;

#[derive(Debug, Clone, PartialEq)]
pub struct StateRegion {
    /// Per-dimension value sets, indexed by propositional variable. Private so
    /// that no caller can narrow a dimension without recording it in
    /// [`Self::constrained_props`]; a missing entry there would make two
    /// disjoint regions look overlapping, which silently inflates the cost a
    /// region may claim. Narrow through [`Self::narrow_prop`] instead.
    propositions: Arc<[Vec<PropValueId>]>,
    pub numeric: Arc<[Interval]>,
    /// Propositional dimensions that exclude at least one of their values,
    /// ascending. A dimension outside this list admits its whole domain, so it
    /// can never be what makes two regions disjoint: overlap, intersection and
    /// subtraction only have to look at the dimensions constrained on *both*
    /// sides.
    ///
    /// This is what makes regional cost partitioning affordable. Measured over
    /// 935 operator regions on delivery/pfile1, a region constrains a median of
    /// 0 and a mean of 0.64 of its 49 propositional dimensions, so walking all
    /// of them spends essentially all of its time confirming that universal
    /// dimensions are universal.
    ///
    /// A superset is always sound -- it only costs time -- so derived regions
    /// may carry the union of their inputs' lists rather than recomputing which
    /// dimensions ended up constrained.
    constrained_props: Arc<[u32]>,
}

impl StateRegion {
    /// A region over dense per-dimension value sets, with the constrained
    /// dimensions derived from the propositional domain sizes.
    ///
    /// `prop_domain_sizes` must have one entry per propositional dimension; a
    /// dimension whose value set is shorter than its domain is constrained.
    /// Panics on a length mismatch, which would mean the caller built the
    /// region against a different variable ordering than it is describing.
    pub fn new(
        propositions: Vec<Vec<PropValueId>>,
        numeric: Vec<Interval>,
        prop_domain_sizes: &[usize],
    ) -> Self {
        assert_eq!(
            propositions.len(),
            prop_domain_sizes.len(),
            "state region has {} propositional dimensions but {} domain sizes",
            propositions.len(),
            prop_domain_sizes.len()
        );
        let constrained_props = propositions
            .iter()
            .zip(prop_domain_sizes.iter())
            .enumerate()
            .filter(|(_, (values, domain_size))| values.len() != **domain_size)
            .map(|(var_id, _)| {
                u32::try_from(var_id).expect("propositional dimension count exceeds u32")
            })
            .collect::<Vec<_>>();
        Self {
            propositions: propositions.into(),
            numeric: numeric.into(),
            constrained_props: constrained_props.into(),
        }
    }

    /// A region whose constrained dimensions are already known, for regions
    /// derived from others (intersections, subtraction pieces) where the
    /// inputs' lists already bound which dimensions can be constrained.
    pub fn with_constrained_props(
        propositions: Vec<Vec<PropValueId>>,
        numeric: Vec<Interval>,
        constrained_props: Arc<[u32]>,
    ) -> Self {
        debug_assert!(
            constrained_props.windows(2).all(|w| w[0] < w[1]),
            "constrained propositional dimensions must be ascending and deduplicated"
        );
        debug_assert!(
            constrained_props
                .last()
                .is_none_or(|&var_id| (var_id as usize) < propositions.len()),
            "constrained propositional dimension is out of range"
        );
        Self {
            propositions: propositions.into(),
            numeric: numeric.into(),
            constrained_props,
        }
    }

    /// A region whose every propositional dimension counts as constrained.
    ///
    /// Test regions are written without a notion of domain size, so this takes
    /// the sound superset: treating every dimension as possibly constrained is
    /// exactly the behaviour that predates the sparse dimension list, which
    /// keeps the tests measuring what they were written to measure.
    #[cfg(test)]
    pub(crate) fn with_all_props_constrained(
        propositions: Vec<Vec<PropValueId>>,
        numeric: Vec<Interval>,
    ) -> Self {
        let constrained_props = (0..propositions.len())
            .map(|var_id| u32::try_from(var_id).expect("propositional dimension exceeds u32"))
            .collect::<Vec<_>>();
        Self::with_constrained_props(propositions, numeric, constrained_props.into())
    }

    /// Propositional dimensions that may exclude values, ascending.
    pub fn constrained_props(&self) -> &[u32] {
        &self.constrained_props
    }

    /// The per-dimension value sets, indexed by propositional variable.
    pub fn propositions(&self) -> &[Vec<PropValueId>] {
        &self.propositions
    }

    /// The propositional value sets as a shared handle.
    ///
    /// Only for checking that a derived region shares its parent's storage
    /// instead of deep-copying it. Immutable: narrowing still has to go through
    /// [`Self::narrow_prop`].
    #[cfg(test)]
    pub(crate) fn propositions_arc(&self) -> &Arc<[Vec<PropValueId>]> {
        &self.propositions
    }

    /// Restrict one propositional dimension to `values`, recording the
    /// dimension as constrained.
    ///
    /// `values` must be sorted, deduplicated and a subset of the dimension's
    /// current values: this narrows a region, it does not redefine it.
    pub fn narrow_prop(&mut self, var_id: usize, values: Vec<PropValueId>) {
        debug_assert!(
            values.windows(2).all(|w| w[0] < w[1]),
            "propositional values must be ascending and deduplicated"
        );
        debug_assert!(
            values
                .iter()
                .all(|value| self.propositions[var_id].binary_search(value).is_ok()),
            "narrowing dimension {var_id} to values outside the region"
        );
        Arc::make_mut(&mut self.propositions)[var_id] = values;
        let dim = u32::try_from(var_id).expect("propositional var id exceeds u32");
        if let Err(insert_at) = self.constrained_props.binary_search(&dim) {
            let mut dims = self.constrained_props.to_vec();
            dims.insert(insert_at, dim);
            self.constrained_props = dims.into();
        }
    }

    pub fn overlaps(&self, other: &Self) -> bool {
        // `debug_assert!` would still type-check the call in release builds, where
        // the oracle does not exist, so gate the statement itself.
        #[cfg(debug_assertions)]
        assert!(
            self.debug_sparse_overlap_matches_dense(other),
            "sparse propositional overlap disagrees with the dense answer: some dimension is \
             narrowed without being listed as constrained (self: {:?}, other: {:?})",
            self.constrained_props,
            other.constrained_props
        );
        prop_regions_overlap(self, other) && numeric_regions_overlap(&self.numeric, &other.numeric)
    }

    /// Whether the sparse propositional overlap agrees with walking every
    /// dimension.
    ///
    /// A complete oracle for the invariant the sparse form relies on, and it
    /// needs no knowledge of domain sizes: if two value sets at a dimension are
    /// disjoint then both are strict subsets of that domain, so both regions
    /// must list the dimension. Disagreement therefore means some producer
    /// narrowed a dimension without recording it.
    #[cfg(debug_assertions)]
    fn debug_sparse_overlap_matches_dense(&self, other: &Self) -> bool {
        let dense = self.propositions.len() == other.propositions.len()
            && self
                .propositions
                .iter()
                .zip(other.propositions.iter())
                .all(|(left, right)| sorted_value_sets_overlap(left, right));
        prop_regions_overlap(self, other) == dense
    }
}

/// The dimensions constrained in either region, ascending.
///
/// Sound for any region derived from both: a dimension universal in both inputs
/// stays universal under intersection and cannot gain a constraint under
/// subtraction.
fn union_constrained_props(left: &StateRegion, right: &StateRegion) -> Arc<[u32]> {
    let (left_dims, right_dims) = (&left.constrained_props, &right.constrained_props);
    if left_dims.is_empty() {
        return Arc::clone(right_dims);
    }
    if right_dims.is_empty() {
        return Arc::clone(left_dims);
    }
    let mut union = Vec::with_capacity(left_dims.len() + right_dims.len());
    let (mut i, mut j) = (0, 0);
    while i < left_dims.len() && j < right_dims.len() {
        match left_dims[i].cmp(&right_dims[j]) {
            std::cmp::Ordering::Less => {
                union.push(left_dims[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                union.push(right_dims[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                union.push(left_dims[i]);
                i += 1;
                j += 1;
            }
        }
    }
    union.extend_from_slice(&left_dims[i..]);
    union.extend_from_slice(&right_dims[j..]);
    union.into()
}

pub(super) fn state_region_is_nonempty(region: &StateRegion) -> bool {
    region.propositions.iter().all(|values| !values.is_empty())
        && region.numeric.iter().all(|interval| !interval.is_empty())
}

/// Whether `inner` is contained in `outer`.
///
/// Only dimensions `outer` constrains can exclude anything, so unconstrained
/// dimensions are skipped: `inner`'s values there are a subset of the whole
/// domain by definition.
fn region_contains(outer: &StateRegion, inner: &StateRegion) -> bool {
    debug_assert_eq!(outer.propositions.len(), inner.propositions.len());
    debug_assert_eq!(outer.numeric.len(), inner.numeric.len());
    outer.constrained_props.iter().all(|&var_id| {
        let var_id = var_id as usize;
        sorted_value_set_contains(&outer.propositions[var_id], &inner.propositions[var_id])
    }) && outer
        .numeric
        .iter()
        .zip(inner.numeric.iter())
        .all(|(outer_interval, inner_interval)| {
            outer_interval.lower_is_lower_or_equal(inner_interval)
                && outer_interval.upper_is_higher_or_equal(inner_interval)
        })
}

/// Whether every value of `inner` appears in `outer`. Both must be sorted.
fn sorted_value_set_contains(outer: &[PropValueId], inner: &[PropValueId]) -> bool {
    if inner.len() > outer.len() {
        return false;
    }
    let mut outer_index = 0;
    for value in inner {
        while outer_index < outer.len() && outer[outer_index] < *value {
            outer_index += 1;
        }
        if outer_index == outer.len() || outer[outer_index] != *value {
            return false;
        }
        outer_index += 1;
    }
    true
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
    // When one side contains the other, the intersection *is* that side. Taking
    // it is four `Arc` clones, against rebuilding every value set. This is the
    // common case rather than a lucky one: operator regions constrain a median
    // of zero dimensions, so intersecting a transition's source region with one
    // usually returns the source region unchanged.
    if region_contains(right, left) {
        return Some(left.clone());
    }
    if region_contains(left, right) {
        return Some(right.clone());
    }

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
    Some(StateRegion::with_constrained_props(
        propositions,
        numeric,
        union_constrained_props(left, right),
    ))
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
    // Every piece narrows one dimension of `core`, so the dimensions either
    // input constrains bound the dimensions any piece can constrain.
    core.constrained_props = union_constrained_props(region, removed);
    let mut result = Vec::new();
    // A dimension universal in both regions contributes no piece: the
    // difference of two whole domains is empty, and copying `removed`'s values
    // into `core` there would be a no-op. Only the constrained dimensions can
    // split anything off.
    for index in 0..core.constrained_props.len() {
        let var_id = core.constrained_props[index] as usize;
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

/// Whether two regions share a value on every propositional dimension.
///
/// Only dimensions constrained in *both* regions can fail: if either side
/// admits the whole domain, the other side's values are all available. So this
/// intersects the two constrained-dimension lists instead of walking every
/// dimension.
fn prop_regions_overlap(left: &StateRegion, right: &StateRegion) -> bool {
    if left.propositions.len() != right.propositions.len() {
        return false;
    }
    // Walk the shorter list and probe the longer one: a merge of both would cost
    // their combined length, while the dimensions that actually need testing are
    // only those in the intersection, which is typically near-empty.
    let (short, long) = if left.constrained_props.len() <= right.constrained_props.len() {
        (&left.constrained_props, &right.constrained_props)
    } else {
        (&right.constrained_props, &left.constrained_props)
    };
    short.iter().all(|&var_id| {
        if long.binary_search(&var_id).is_err() {
            // Universal on the other side, so every value here is available.
            return true;
        }
        let var_id = var_id as usize;
        sorted_value_sets_overlap(&left.propositions[var_id], &right.propositions[var_id])
    })
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
