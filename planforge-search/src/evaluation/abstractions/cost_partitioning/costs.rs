//! Label and regional saturated cost partitioning for abstraction components.
//!
//! Vocabulary:
//! - A *region* is a set of states represented by a [`CartesianSet`].
//! - An *operator region* is the region on which a concrete operator's cost is
//!   claimed.
//! - *Regional* means per-region cost accounting.
//!
//! Each [`LabelSourceSet::source`] stores the *regressed
//! preimage source* of an abstract operator's effect — the intersection of the
//! abstract source region with the inverse image of the abstract target region
//! under the operator's numeric effect (computed by
//! `RegionSemantics::transition_source_set` for both representations).
//!
//! Unbounded preimages are ordinary regions. Cost is allocated on their exact
//! operator region and remains available on disjoint regions.
//!
//! [`CartesianSet`]: crate::evaluation::abstractions::common::cartesian_set::CartesianSet
//! [`LabelSourceSet::source`]: crate::evaluation::abstractions::common::transition_system::LabelSourceSet::source

mod remaining;
mod saturation;
#[cfg(test)]
mod tests;
mod usage;

pub use remaining::RemainingCosts;
pub use saturation::*;

/// The operation an expired SCP table deadline names.
const SCP_TABLE_CONSTRUCTION: &str = "SCP table construction";

/// How many usage cells or allocation entries pass between two deadline
/// checks. Each is a geometric operation on Cartesian sets, far costlier than
/// reading the clock, so a passed deadline is noticed within a few of them.
const GEOMETRIC_POLL_STRIDE: usize = 64;

/// How many transitions or states a saturation pass handles between two
/// deadline checks. Each is a few arithmetic operations, so the clock is
/// read rarely enough to cost nothing measurable.
const SATURATION_POLL_STRIDE: usize = 1024;
