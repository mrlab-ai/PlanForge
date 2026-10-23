//! How much of one operator's cost is used where: a disjoint cover of the
//! states with nonzero usage, with a lazily built index for overlap queries.

use std::cell::RefCell;
use std::collections::HashSet;
use std::ops::ControlFlow;

use anyhow::Result;
use planforge_sas::numeric_task::{NEG_INF_VALUE, NumericValue};
use planforge_sas::utils::float_tolerance;

use super::{GEOMETRIC_POLL_STRIDE, SCP_TABLE_CONSTRUCTION};
use crate::evaluation::abstractions::common::cartesian_set::{
    CartesianSet, cartesian_set_intersection, cartesian_set_is_nonempty, subtract_cartesian_set,
};
use crate::resource_limits::Deadline;

/// A disjoint partition of the part of the concrete state space on which an
/// operator cost has already been allocated. States outside all cells have
/// usage zero. Keeping usage, rather than residual cost, avoids materializing
/// the universal state region.
#[derive(Debug, Default)]
pub(super) struct SourceSetUsage {
    pub(super) cells: Vec<UsageCell>,
    pub(super) index: RefCell<UsageIndex>,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct UsageCell {
    pub(super) region: CartesianSet,
    pub(super) amount: f64,
}

/// The usage one table allocates of one operator's cost: while every
/// allocation takes the full cost, the regions it takes it on, which may
/// overlap; otherwise their pointwise maximum.
#[derive(Debug)]
pub(super) enum TableUsageEnvelope {
    Full(Vec<CartesianSet>),
    Fractional(SourceSetUsage),
}

impl Default for TableUsageEnvelope {
    fn default() -> Self {
        Self::Full(Vec::new())
    }
}

impl TableUsageEnvelope {
    pub(super) fn maximize(
        &mut self,
        region: &CartesianSet,
        amount: f64,
        base_cost: f64,
        deadline: Deadline,
    ) -> Result<()> {
        debug_assert!(amount.is_finite() && amount > 0.0);
        debug_assert!(base_cost.is_finite() && base_cost > 0.0);
        if amount + float_tolerance::SEARCH_EPSILON >= base_cost {
            match self {
                Self::Full(regions) => regions.push(region.clone()),
                Self::Fractional(usage) => usage.maximize(region, base_cost, deadline)?,
            }
            return Ok(());
        }

        if let Self::Full(regions) = self {
            let regions = std::mem::take(regions);
            let mut usage = SourceSetUsage::default();
            for full_region in regions {
                usage.maximize(&full_region, base_cost, deadline)?;
            }
            *self = Self::Fractional(usage);
        }
        let Self::Fractional(usage) = self else {
            unreachable!("full regional envelope must have been promoted to fractional overlay")
        };
        usage.maximize(region, amount, deadline)
    }
}

/// The usage covers below this many cells are scanned: a scan of a few dozen
/// overlap tests is cheaper than building and querying the sorted index.
const USAGE_INDEX_MIN_CELLS: usize = 32;
/// The cells of one index block, which a query skips when the block's
/// largest upper bound lies below it: blocks this small skip most cells while
/// keeping the per-block bounds few.
const USAGE_INDEX_BLOCK_SIZE: usize = 32;

/// The cells sorted by their lower bound on the numeric dimension with the
/// most distinct bounds, in blocks that know their largest upper bound.
#[derive(Debug)]
pub(super) struct SortedUsageIndex {
    primary_dim: usize,
    sorted_cell_ids: Vec<usize>,
    blocks: Vec<UsageIndexBlock>,
}

#[derive(Debug, Clone, Copy)]
struct UsageIndexBlock {
    start: usize,
    end: usize,
    max_upper: NumericValue,
}

impl SourceSetUsage {
    pub(super) fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    pub(super) fn append_full_cover(&mut self, regions: Vec<CartesianSet>, base_cost: f64) {
        debug_assert!(base_cost.is_finite() && base_cost > 0.0);
        self.cells
            .extend(regions.into_iter().map(|region| UsageCell {
                region,
                amount: base_cost,
            }));
        *self.index.get_mut() = UsageIndex::Stale;
    }

    pub(super) fn overlaps(&self, query: &CartesianSet) -> bool {
        let mut overlaps = false;
        self.visit_overlapping(query, |_| {
            overlaps = true;
            ControlFlow::Break(())
        });
        overlaps
    }

    /// The largest usage on `query`.
    pub(super) fn max_over(&self, query: &CartesianSet) -> f64 {
        let mut maximum = 0.0_f64;
        self.visit_overlapping(query, |cell| {
            maximum = maximum.max(cell.amount);
            ControlFlow::Continue(())
        });
        maximum
    }

    /// Visits the cells that overlap `query` until `visit` breaks, through
    /// the index once there are enough cells to build one.
    fn visit_overlapping(
        &self,
        query: &CartesianSet,
        mut visit: impl FnMut(&UsageCell) -> ControlFlow<()>,
    ) {
        let mut visit_if_overlapping = |cell: &UsageCell| {
            if cell.region.overlaps(query) {
                visit(cell)
            } else {
                ControlFlow::Continue(())
            }
        };
        if self.cells.len() >= USAGE_INDEX_MIN_CELLS {
            UsageIndex::ensure_built(&self.index, &self.cells);
        }
        let index = self.index.borrow();
        let Some(index) = index.ready() else {
            for cell in &self.cells {
                if visit_if_overlapping(cell).is_break() {
                    return;
                }
            }
            return;
        };
        let query_interval = query.numeric[index.primary_dim];
        let candidate_end = index.sorted_cell_ids.partition_point(|&cell_id| {
            self.cells[cell_id].region.numeric[index.primary_dim].lower <= query_interval.upper
        });
        for block in &index.blocks {
            if block.start >= candidate_end {
                break;
            }
            if block.max_upper < query_interval.lower {
                continue;
            }
            for &cell_id in &index.sorted_cell_ids[block.start..block.end.min(candidate_end)] {
                if visit_if_overlapping(&self.cells[cell_id]).is_break() {
                    return;
                }
            }
        }
    }

    /// Pointwise maximum assignment. This is used to form the allocation
    /// envelope of one abstraction table: a concrete transition maps to one
    /// abstract transition in that table, even when conservative operator regions
    /// overlap.
    fn maximize(&mut self, region: &CartesianSet, amount: f64, deadline: Deadline) -> Result<()> {
        debug_assert!(amount.is_finite() && amount >= 0.0);
        if amount <= float_tolerance::SEARCH_EPSILON {
            return Ok(());
        }
        self.overlay(region, |old| old.max(amount), amount, deadline)
    }

    /// Pointwise addition. Table envelopes are independent cost partitions and
    /// therefore add across completed tables.
    pub(super) fn add_with_deadline(
        &mut self,
        region: &CartesianSet,
        amount: f64,
        deadline: Deadline,
    ) -> Result<()> {
        debug_assert!(amount.is_finite() && amount >= 0.0);
        if amount <= float_tolerance::SEARCH_EPSILON {
            return Ok(());
        }
        self.overlay(region, |old| old + amount, amount, deadline)
    }

    fn overlay(
        &mut self,
        region: &CartesianSet,
        update_existing: impl Fn(f64) -> f64,
        uncovered_amount: f64,
        deadline: Deadline,
    ) -> Result<()> {
        debug_assert!(cartesian_set_is_nonempty(region));
        let old_cells = std::mem::take(&mut self.cells);
        let mut new_cells = Vec::with_capacity(old_cells.len() + 1);
        let mut uncovered = vec![region.clone()];

        let mut old_cells = old_cells.into_iter();
        while let Some(cell) = old_cells.next() {
            if new_cells.len().is_multiple_of(GEOMETRIC_POLL_STRIDE)
                && let Err(error) = deadline.ensure(SCP_TABLE_CONSTRUCTION)
            {
                // add_saturation_steps reduces costs before cp.add_h_values, so partial overlays never contribute h-values.
                new_cells.push(cell);
                new_cells.extend(old_cells);
                self.cells = new_cells;
                *self.index.get_mut() = UsageIndex::Stale;
                return Err(error);
            }
            let Some(intersection) = cartesian_set_intersection(&cell.region, region) else {
                new_cells.push(cell);
                continue;
            };
            for remainder in subtract_cartesian_set(&cell.region, &intersection) {
                new_cells.push(UsageCell {
                    region: remainder,
                    amount: cell.amount,
                });
            }
            new_cells.push(UsageCell {
                region: intersection.clone(),
                amount: update_existing(cell.amount),
            });
            uncovered = uncovered
                .into_iter()
                .flat_map(|piece| {
                    let Some(covered_piece) = cartesian_set_intersection(&piece, &intersection)
                    else {
                        return vec![piece];
                    };
                    subtract_cartesian_set(&piece, &covered_piece)
                })
                .collect();
        }

        new_cells.extend(uncovered.into_iter().map(|region| UsageCell {
            region,
            amount: uncovered_amount,
        }));
        debug_assert!(
            new_cells
                .iter()
                .all(|cell| cell.amount > float_tolerance::SEARCH_EPSILON)
        );
        debug_assert!(cells_are_disjoint(&new_cells));
        self.cells = new_cells;
        *self.index.get_mut() = UsageIndex::Stale;
        Ok(())
    }
}

/// Lazily built numeric index over a usage cover.
///
/// `Unusable` is the reason this is not an `Option`: building the index fails
/// whenever the cells share no finite numeric dimension to sort on, and with an
/// `Option` that failure is indistinguishable from "not built yet", so every
/// query retries the build -- which allocates a hash set per numeric dimension
/// and visits every cell -- and throws the result away.
#[derive(Debug, Default)]
pub(super) enum UsageIndex {
    /// The cells changed, or nothing has been built yet.
    #[default]
    Stale,
    /// The cells cannot be indexed; queries scan them linearly.
    Unusable,
    Ready(SortedUsageIndex),
}

impl UsageIndex {
    /// Build on first use, and remember a failed attempt.
    fn ensure_built(index: &RefCell<Self>, cells: &[UsageCell]) {
        if matches!(&*index.borrow(), UsageIndex::Stale) {
            *index.borrow_mut() = match SortedUsageIndex::build(cells) {
                Some(built) => UsageIndex::Ready(built),
                None => UsageIndex::Unusable,
            };
        }
    }

    pub(super) fn ready(&self) -> Option<&SortedUsageIndex> {
        match self {
            UsageIndex::Ready(index) => Some(index),
            UsageIndex::Stale | UsageIndex::Unusable => None,
        }
    }
}

impl SortedUsageIndex {
    fn build(cells: &[UsageCell]) -> Option<Self> {
        let numeric_dimensions = cells.first()?.region.numeric.len();
        if numeric_dimensions == 0
            || cells
                .iter()
                .any(|cell| cell.region.numeric.len() != numeric_dimensions)
        {
            return None;
        }
        let primary_dim = (0..numeric_dimensions)
            .max_by_key(|&dimension| {
                let mut bounds = HashSet::with_capacity(cells.len());
                for cell in cells {
                    let interval = cell.region.numeric[dimension];
                    bounds.insert((interval.lower, interval.upper));
                }
                bounds.len()
            })
            .filter(|&dimension| {
                cells.iter().any(|cell| {
                    let interval = cell.region.numeric[dimension];
                    interval.lower.is_finite() || interval.upper.is_finite()
                })
            })?;
        let mut sorted_cell_ids = (0..cells.len()).collect::<Vec<_>>();
        sorted_cell_ids.sort_unstable_by(|&left, &right| {
            let left = cells[left].region.numeric[primary_dim];
            let right = cells[right].region.numeric[primary_dim];
            left.lower
                .cmp(&right.lower)
                .then_with(|| left.upper.cmp(&right.upper))
        });
        let blocks = sorted_cell_ids
            .chunks(USAGE_INDEX_BLOCK_SIZE)
            .enumerate()
            .map(|(block_id, cell_ids)| UsageIndexBlock {
                start: block_id * USAGE_INDEX_BLOCK_SIZE,
                end: block_id * USAGE_INDEX_BLOCK_SIZE + cell_ids.len(),
                max_upper: cell_ids
                    .iter()
                    .map(|&cell_id| cells[cell_id].region.numeric[primary_dim].upper)
                    .fold(NEG_INF_VALUE, |a, b| a.max(b)),
            })
            .collect();
        Some(Self {
            primary_dim,
            sorted_cell_ids,
            blocks,
        })
    }
}

pub(super) fn cells_are_disjoint(cells: &[UsageCell]) -> bool {
    cells.iter().enumerate().all(|(index, cell)| {
        cells[index + 1..]
            .iter()
            .all(|other| !cell.region.overlaps(&other.region))
    })
}
