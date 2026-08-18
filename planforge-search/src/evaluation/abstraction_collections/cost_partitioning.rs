//! Label and regional saturated cost partitioning for abstraction components.
//!
//! Vocabulary:
//! - A *region* is geometry represented by [`StateRegion`] or [`TransitionRegion`]
//!   (defined in `region.rs`).
//! - An *operator region* is the region on which a concrete operator's cost is
//!   claimed.
//! - *Regional* means per-region cost accounting.
//!
//! Each [`OperatorRegion::source`] stores the *regressed
//! preimage source* of an abstract operator's effect — the intersection of the
//! abstract source region with the inverse image of the abstract target region
//! under the operator's numeric effect (computed in
//! `domain_abstraction_factory::build_operator_region`).
//!
//! Unbounded preimages are ordinary regions. Cost is allocated on their exact
//! operator region and remains available on disjoint regions.

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
#[cfg(test)]
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use planforge_sas::utils::float_tolerance;

#[cfg(test)]
use planforge_sas::utils::interval::Interval;

#[path = "explicit_scp.rs"]
mod explicit_scp;
#[path = "region.rs"]
mod region;

pub use explicit_scp::*;
pub use region::*;

const MAX_ABSTRACT_OPERATOR_REDUCTION_PIECES: usize = 4096;

#[derive(Debug)]
pub struct TransitionResidualCosts {
    operator_residuals: Vec<OperatorResidual>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LmCutResidualOperatorCostPartition {
    pub fallback_cost: f64,
    pub variants: Vec<LmCutResidualCostVariant>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LmCutResidualCostVariant {
    pub cost: f64,
    pub source_region: StateRegion,
}

#[derive(Debug)]
struct OperatorResidual {
    base_cost: f64,
    /// Regions where prior tables consumed the complete operator cost. These
    /// may overlap because pointwise max and sum are both equal to the base
    /// cost there; keeping the cover avoids an unnecessary geometric union.
    full_regional_usage: RegionalUsage,
    /// Exact disjoint overlay for genuinely fractional regional allocations.
    regional_usage: RegionalUsage,
    uniform_cost_cache: Cell<Option<f64>>,
    generation: Cell<u64>,
}

/// A disjoint partition of the part of the concrete state space on which an
/// operator cost has already been allocated. States outside all cells have
/// usage zero. Keeping usage, rather than residual cost, avoids materializing
/// the universal state region.
#[derive(Debug, Clone, Default)]
struct RegionalUsage {
    cells: Vec<RegionalUsageCell>,
    index: RefCell<Option<RegionalUsageIndex>>,
}

#[derive(Debug, Clone, PartialEq)]
struct RegionalUsageCell {
    region: StateRegion,
    amount: f64,
}

#[derive(Debug, Clone)]
enum TableRegionalEnvelope {
    Full(Vec<StateRegion>),
    Fractional(RegionalUsage),
}

impl Default for TableRegionalEnvelope {
    fn default() -> Self {
        Self::Full(Vec::new())
    }
}

impl PartialEq for RegionalUsage {
    fn eq(&self, other: &Self) -> bool {
        self.cells == other.cells
    }
}

impl TableRegionalEnvelope {
    fn maximize(
        &mut self,
        region: &StateRegion,
        amount: f64,
        base_cost: f64,
        deadline: Option<Instant>,
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
            let mut usage = RegionalUsage::default();
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

const REGIONAL_INDEX_MIN_CELLS: usize = 32;
const REGIONAL_INDEX_BLOCK_SIZE: usize = 32;

#[derive(Debug, Clone)]
struct RegionalUsageIndex {
    primary_dim: usize,
    sorted_cell_ids: Vec<usize>,
    blocks: Vec<RegionalUsageIndexBlock>,
}

#[derive(Debug, Clone, Copy)]
struct RegionalUsageIndexBlock {
    start: usize,
    end: usize,
    max_upper: f64,
}

impl RegionalUsage {
    fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    fn append_full_cover(&mut self, regions: Vec<StateRegion>, base_cost: f64) {
        debug_assert!(base_cost.is_finite() && base_cost > 0.0);
        self.cells
            .extend(regions.into_iter().map(|region| RegionalUsageCell {
                region,
                amount: base_cost,
            }));
        *self.index.get_mut() = None;
    }

    fn overlaps(&self, query: &StateRegion) -> bool {
        if self.cells.len() < REGIONAL_INDEX_MIN_CELLS {
            return self.cells.iter().any(|cell| cell.region.overlaps(query));
        }
        if self.index.borrow().is_none() {
            *self.index.borrow_mut() = RegionalUsageIndex::build(&self.cells);
        }
        let index = self.index.borrow();
        let Some(index) = index.as_ref() else {
            return self.cells.iter().any(|cell| cell.region.overlaps(query));
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
            if index.sorted_cell_ids[block.start..block.end.min(candidate_end)]
                .iter()
                .any(|&cell_id| self.cells[cell_id].region.overlaps(query))
            {
                return true;
            }
        }
        false
    }

    fn max_over(&self, query: &StateRegion) -> f64 {
        if self.cells.len() < REGIONAL_INDEX_MIN_CELLS {
            return self.max_over_cell_ids(query, 0..self.cells.len());
        }
        if self.index.borrow().is_none() {
            *self.index.borrow_mut() = RegionalUsageIndex::build(&self.cells);
        }
        let index = self.index.borrow();
        let Some(index) = index.as_ref() else {
            return self.max_over_cell_ids(query, 0..self.cells.len());
        };
        let query_interval = query.numeric[index.primary_dim];
        let candidate_end = index.sorted_cell_ids.partition_point(|&cell_id| {
            self.cells[cell_id].region.numeric[index.primary_dim].lower <= query_interval.upper
        });
        let mut maximum = 0.0_f64;
        for block in &index.blocks {
            if block.start >= candidate_end {
                break;
            }
            if block.max_upper < query_interval.lower {
                continue;
            }
            for &cell_id in &index.sorted_cell_ids[block.start..block.end.min(candidate_end)] {
                let cell = &self.cells[cell_id];
                if cell.region.overlaps(query) {
                    maximum = maximum.max(cell.amount);
                }
            }
        }
        maximum
    }

    fn max_over_cell_ids(&self, query: &StateRegion, cell_ids: impl Iterator<Item = usize>) -> f64 {
        cell_ids
            .filter_map(|cell_id| {
                let cell = &self.cells[cell_id];
                cell.region.overlaps(query).then_some(cell.amount)
            })
            .fold(0.0, f64::max)
    }

    /// Pointwise maximum assignment. This is used to form the allocation
    /// envelope of one abstraction table: a concrete transition maps to one
    /// abstract transition in that table, even when conservative operator regions
    /// overlap.
    fn maximize(
        &mut self,
        region: &StateRegion,
        amount: f64,
        deadline: Option<Instant>,
    ) -> Result<()> {
        debug_assert!(amount.is_finite() && amount >= 0.0);
        if amount <= float_tolerance::SEARCH_EPSILON {
            return Ok(());
        }
        self.overlay(region, |old| old.max(amount), amount, deadline)
    }

    /// Pointwise addition. Table envelopes are independent cost partitions and
    /// therefore add across completed tables.
    #[cfg(test)]
    fn add(&mut self, region: &StateRegion, amount: f64) {
        self.add_with_deadline(region, amount, None)
            .expect("an unbounded regional-usage update cannot exceed a deadline");
    }

    fn add_with_deadline(
        &mut self,
        region: &StateRegion,
        amount: f64,
        deadline: Option<Instant>,
    ) -> Result<()> {
        debug_assert!(amount.is_finite() && amount >= 0.0);
        if amount <= float_tolerance::SEARCH_EPSILON {
            return Ok(());
        }
        self.overlay(region, |old| old + amount, amount, deadline)
    }

    fn overlay(
        &mut self,
        region: &StateRegion,
        update_existing: impl Fn(f64) -> f64,
        uncovered_amount: f64,
        deadline: Option<Instant>,
    ) -> Result<()> {
        debug_assert!(state_region_is_nonempty(region));
        let old_cells = std::mem::take(&mut self.cells);
        let mut new_cells = Vec::with_capacity(old_cells.len() + 1);
        let mut uncovered = vec![region.clone()];

        let mut old_cells = old_cells.into_iter();
        while let Some(cell) = old_cells.next() {
            if new_cells.len().is_multiple_of(64)
                && let Err(error) = ensure_scp_table_deadline(deadline)
            {
                new_cells.push(cell);
                new_cells.extend(old_cells);
                self.cells = new_cells;
                *self.index.get_mut() = None;
                return Err(error);
            }
            let Some(intersection) = state_region_intersection(&cell.region, region) else {
                new_cells.push(cell);
                continue;
            };
            for remainder in subtract_state_region(&cell.region, &intersection) {
                new_cells.push(RegionalUsageCell {
                    region: remainder,
                    amount: cell.amount,
                });
            }
            new_cells.push(RegionalUsageCell {
                region: intersection.clone(),
                amount: update_existing(cell.amount),
            });
            uncovered = uncovered
                .into_iter()
                .flat_map(|piece| {
                    let Some(covered_piece) = state_region_intersection(&piece, &intersection)
                    else {
                        return vec![piece];
                    };
                    subtract_state_region(&piece, &covered_piece)
                })
                .collect();
        }

        new_cells.extend(uncovered.into_iter().map(|region| RegionalUsageCell {
            region,
            amount: uncovered_amount,
        }));
        new_cells.retain(|cell| cell.amount > float_tolerance::SEARCH_EPSILON);
        debug_assert!(regional_usage_cells_are_disjoint(&new_cells));
        self.cells = new_cells;
        *self.index.get_mut() = None;
        Ok(())
    }
}

impl RegionalUsageIndex {
    fn build(cells: &[RegionalUsageCell]) -> Option<Self> {
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
                    bounds.insert((interval.lower.to_bits(), interval.upper.to_bits()));
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
                .total_cmp(&right.lower)
                .then_with(|| left.upper.total_cmp(&right.upper))
        });
        let blocks = sorted_cell_ids
            .chunks(REGIONAL_INDEX_BLOCK_SIZE)
            .enumerate()
            .map(|(block_id, cell_ids)| RegionalUsageIndexBlock {
                start: block_id * REGIONAL_INDEX_BLOCK_SIZE,
                end: block_id * REGIONAL_INDEX_BLOCK_SIZE + cell_ids.len(),
                max_upper: cell_ids
                    .iter()
                    .map(|&cell_id| cells[cell_id].region.numeric[primary_dim].upper)
                    .fold(f64::NEG_INFINITY, f64::max),
            })
            .collect();
        Some(Self {
            primary_dim,
            sorted_cell_ids,
            blocks,
        })
    }
}

fn regional_usage_cells_are_disjoint(cells: &[RegionalUsageCell]) -> bool {
    cells.iter().enumerate().all(|(index, cell)| {
        cells[index + 1..]
            .iter()
            .all(|other| !cell.region.overlaps(&other.region))
    })
}

impl TransitionResidualCosts {
    pub fn from_operator_costs(costs: &[f64]) -> Self {
        let operator_residuals = costs
            .iter()
            .map(|&base_cost| OperatorResidual {
                base_cost,
                full_regional_usage: RegionalUsage::default(),
                regional_usage: RegionalUsage::default(),
                uniform_cost_cache: Cell::new(None),
                generation: Cell::new(0),
            })
            .collect();
        Self { operator_residuals }
    }

    pub fn num_reductions(&self) -> usize {
        self.operator_residuals
            .iter()
            .map(|residual| {
                residual.full_regional_usage.cells.len() + residual.regional_usage.cells.len()
            })
            .sum()
    }

    pub fn has_reductions(&self) -> bool {
        self.operator_residuals.iter().any(|residual| {
            !residual.full_regional_usage.is_empty() || !residual.regional_usage.is_empty()
        })
    }

    pub fn base_cost(&self, concrete_op_id: usize) -> f64 {
        self.operator_residuals
            .get(concrete_op_id)
            .map(|residual| residual.base_cost)
            .unwrap_or(f64::INFINITY)
    }

    pub fn operator_costs_for_label_cp(&self) -> Vec<f64> {
        self.operator_residuals
            .iter()
            .map(|residual| {
                if !residual.base_cost.is_finite() {
                    return f64::INFINITY;
                }
                if let Some(cost) = residual.uniform_cost_cache.get() {
                    return cost;
                }
                let full_regional_reduction = if !residual.full_regional_usage.is_empty() {
                    residual.base_cost
                } else {
                    0.0
                };
                let reduction = full_regional_reduction.max(
                    residual
                        .regional_usage
                        .cells
                        .iter()
                        .map(|cell| cell.amount)
                        .fold(0.0, f64::max),
                );
                let cost = (residual.base_cost - reduction).max(0.0);
                residual.uniform_cost_cache.set(Some(cost));
                cost
            })
            .collect()
    }

    pub fn operator_cost_partitions_for_lmcut(
        &self,
        _max_variants_per_operator: usize,
        _max_guard_conditions_per_variant: usize,
    ) -> Vec<LmCutResidualOperatorCostPartition> {
        // Transition reductions were never populated in production, so
        // fillscp(partitioning=region) gives LM-cut uniform residuals only and
        // the region-conditional variant machinery remains dormant. Activating
        // it requires re-deriving variants from the RegionalUsage cells.
        let uniform_costs = self.operator_costs_for_label_cp();
        self.operator_residuals
            .iter()
            .enumerate()
            .map(|(op_id, residual)| {
                let fallback_cost = residual.base_cost.max(0.0);
                LmCutResidualOperatorCostPartition {
                    fallback_cost: uniform_costs.get(op_id).copied().unwrap_or(fallback_cost),
                    variants: Vec::new(),
                }
            })
            .collect()
    }

    pub fn cost_for_operator_region(
        &self,
        _current_abstraction_id: usize,
        _abstract_op_id: usize,
        operator_region: &OperatorRegion,
    ) -> f64 {
        let Some(residual) = self.operator_residuals.get(operator_region.concrete_op_id) else {
            return f64::INFINITY;
        };
        if !residual.base_cost.is_finite() {
            return f64::INFINITY;
        }
        let regional = if residual
            .full_regional_usage
            .overlaps(&operator_region.source)
        {
            residual.base_cost
        } else {
            residual.regional_usage.max_over(&operator_region.source)
        };
        (residual.base_cost - regional).max(0.0)
    }

    pub fn reduce_by_abstract_operator_regions(
        &mut self,
        producing_abstraction_id: usize,
        operator_regions: &[AbstractOperatorRegions],
        tcf: &AbstractOperatorCostFunction,
    ) -> Result<()> {
        self.reduce_by_abstract_operator_regions_with_deadline(
            producing_abstraction_id,
            operator_regions,
            tcf,
            None,
        )
    }

    pub fn reduce_by_abstract_operator_regions_with_deadline(
        &mut self,
        producing_abstraction_id: usize,
        operator_regions: &[AbstractOperatorRegions],
        tcf: &AbstractOperatorCostFunction,
        deadline: Option<Instant>,
    ) -> Result<()> {
        ensure!(
            operator_regions.len() >= tcf.operator_costs.len(),
            "abstract-operator region/cost function size mismatch: operator_regions={} costs={}",
            operator_regions.len(),
            tcf.operator_costs.len()
        );

        let mut entries = Vec::new();
        for (abstract_op_id, &saturated) in tcf.operator_costs.iter().enumerate() {
            if abstract_op_id.is_multiple_of(64) {
                ensure_scp_table_deadline(deadline)?;
            }
            ensure!(
                !saturated.is_finite() || saturated >= -float_tolerance::SEARCH_EPSILON,
                "negative abstract-operator saturated costs are not supported: abstract op {} has {}",
                abstract_op_id,
                saturated
            );
            if !saturated.is_finite() || saturated <= float_tolerance::SEARCH_EPSILON {
                continue;
            }

            for operator_region in &operator_regions[abstract_op_id].labels {
                let concrete_op_id = operator_region.concrete_op_id;
                let residual = self
                    .operator_residuals
                    .get(concrete_op_id)
                    .with_context(|| {
                        format!(
                            "abstract-operator region reduction references missing concrete operator {concrete_op_id}: operator residual count is {}",
                            self.operator_residuals.len()
                        )
                    })?;
                let current_residual = self.cost_for_operator_region(
                    producing_abstraction_id,
                    abstract_op_id,
                    operator_region,
                );
                ensure!(
                    current_residual.is_finite(),
                    "residual cost for abstract op {abstract_op_id}, concrete op {concrete_op_id} must be finite"
                );
                ensure!(
                    saturated <= current_residual + float_tolerance::SEARCH_EPSILON,
                    "abstract-operator region reduction {saturated} exceeds current residual cost {current_residual} for concrete operator {concrete_op_id}"
                );
                if residual.base_cost <= float_tolerance::SEARCH_EPSILON {
                    continue;
                }
                ensure!(
                    residual.base_cost.is_finite(),
                    "no base residual cost for operator {concrete_op_id}"
                );
                ensure!(
                    saturated <= residual.base_cost + float_tolerance::SEARCH_EPSILON,
                    "residual cost underflow: abstract-operator region reduction {saturated} exceeds base cost {} for operator {concrete_op_id}",
                    residual.base_cost
                );
                entries.push(RegionalCostAllocationEntry {
                    operator_region: operator_region.clone(),
                    amount: saturated,
                });
            }
        }

        self.reduce_by_regional_allocation_with_deadline(
            &RegionalCostAllocation::new(entries),
            deadline,
        )
    }

    pub fn reduce_by_regional_allocation_with_deadline(
        &mut self,
        allocation: &RegionalCostAllocation,
        deadline: Option<Instant>,
    ) -> Result<()> {
        let mut table_envelopes: HashMap<usize, TableRegionalEnvelope> = HashMap::new();
        for (entry_id, entry) in allocation.entries().iter().enumerate() {
            if entry_id.is_multiple_of(64) {
                ensure_scp_table_deadline(deadline)?;
            }
            ensure!(
                entry.amount.is_finite() && entry.amount >= -float_tolerance::SEARCH_EPSILON,
                "regional allocation entry {entry_id} has invalid amount {}",
                entry.amount
            );
            if entry.amount <= float_tolerance::SEARCH_EPSILON {
                continue;
            }
            let concrete_op_id = entry.operator_region.concrete_op_id;
            let residual = self
                .operator_residuals
                .get(concrete_op_id)
                .with_context(|| {
                    format!(
                        "regional allocation references missing concrete operator {concrete_op_id}"
                    )
                })?;
            ensure!(
                residual.base_cost.is_finite()
                    && residual.base_cost > float_tolerance::SEARCH_EPSILON,
                "regional allocation requires a positive finite base cost for operator {concrete_op_id}"
            );
            ensure!(
                entry.amount <= residual.base_cost + float_tolerance::SEARCH_EPSILON,
                "regional allocation {} exceeds base cost {} for operator {concrete_op_id}",
                entry.amount,
                residual.base_cost
            );
            table_envelopes
                .entry(concrete_op_id)
                .or_default()
                .maximize(
                    &entry.operator_region.source,
                    entry.amount,
                    residual.base_cost,
                    deadline,
                )?;
        }

        for (concrete_op_id, envelope) in table_envelopes {
            ensure_scp_table_deadline(deadline)?;
            let residual = self
                .operator_residuals
                .get_mut(concrete_op_id)
                .expect("validated concrete operator region must exist");
            match envelope {
                TableRegionalEnvelope::Full(regions) => {
                    for region in &regions {
                        let already_used = if residual.full_regional_usage.overlaps(region) {
                            residual.base_cost
                        } else {
                            residual.regional_usage.max_over(region)
                        };
                        ensure!(
                            already_used <= float_tolerance::SEARCH_EPSILON,
                            "full regional allocation overlaps prior usage for operator {concrete_op_id}: used={already_used}, base={}",
                            residual.base_cost
                        );
                    }
                    residual
                        .full_regional_usage
                        .append_full_cover(regions, residual.base_cost);
                }
                TableRegionalEnvelope::Fractional(envelope) => {
                    for cell in envelope.cells {
                        let already_used = if residual.full_regional_usage.overlaps(&cell.region) {
                            residual.base_cost
                        } else {
                            residual.regional_usage.max_over(&cell.region)
                        };
                        ensure!(
                            already_used + cell.amount
                                <= residual.base_cost + float_tolerance::SEARCH_EPSILON,
                            "regional residual cost underflow for operator {concrete_op_id}: used={already_used}, allocation={}, base={}",
                            cell.amount,
                            residual.base_cost
                        );
                        residual.regional_usage.add_with_deadline(
                            &cell.region,
                            cell.amount,
                            deadline,
                        )?;
                    }
                }
            }
            residual.invalidate_cache();
        }
        Ok(())
    }

    pub fn reduce_operator_costs_uniform(&mut self, saturated_costs: &[f64]) -> Result<()> {
        ensure!(
            self.operator_residuals.len() == saturated_costs.len(),
            "operator cost vector length mismatch: buckets={}, saturated={}",
            self.operator_residuals.len(),
            saturated_costs.len()
        );
        for (op_id, saturated) in saturated_costs.iter().copied().enumerate() {
            ensure!(
                !saturated.is_finite() || saturated >= -float_tolerance::SEARCH_EPSILON,
                "negative uniform saturated costs are not supported: operator {op_id} has {saturated}"
            );
            if !saturated.is_finite() || saturated <= float_tolerance::SEARCH_EPSILON {
                continue;
            }
            self.operator_residuals[op_id].base_cost =
                subtract_cost(self.operator_residuals[op_id].base_cost, saturated).with_context(
                    || format!("uniform residual reduction underflow for operator {op_id}"),
                )?;
            self.operator_residuals[op_id].invalidate_cache();
        }
        Ok(())
    }
}

impl OperatorResidual {
    fn invalidate_cache(&self) {
        self.generation.set(self.generation.get().wrapping_add(1));
        self.uniform_cost_cache.set(None);
    }
}

fn subtract_cost(cost: f64, saturated: f64) -> Result<f64> {
    ensure!(cost.is_finite(), "residual cost must be finite, got {cost}");
    ensure!(
        saturated.is_finite(),
        "saturated cost must be finite, got {saturated}"
    );
    let reduced = cost - saturated;
    if reduced < 0.0 && reduced > -float_tolerance::SEARCH_EPSILON {
        Ok(0.0)
    } else {
        ensure!(
            reduced >= 0.0,
            "residual cost underflow: {cost} - {saturated} = {reduced}"
        );
        Ok(reduced)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expired_scp_table_deadline_uses_shared_typed_error() {
        let error = ensure_scp_table_deadline(Some(Instant::now())).unwrap_err();

        assert!(crate::resource_limits::is_deadline_exceeded(&error));
    }

    #[test]
    fn regional_overlay_handles_multiple_multidimensional_uncovered_pieces() {
        let region = |x: Interval, y: Interval| StateRegion {
            propositions: Vec::new().into(),
            numeric: vec![x, y].into(),
        };
        let first = region(Interval::closed(0.0, 1.0), Interval::closed(0.0, 1.0));
        let second = region(Interval::closed(0.0, 1.0), Interval::closed(2.0, 3.0));
        let mut usage = RegionalUsage {
            cells: vec![
                RegionalUsageCell {
                    region: first.clone(),
                    amount: 1.0,
                },
                RegionalUsageCell {
                    region: second.clone(),
                    amount: 2.0,
                },
            ],
            index: RefCell::new(None),
        };

        usage.add(
            &region(Interval::closed(0.0, 3.0), Interval::closed(0.0, 3.0)),
            3.0,
        );

        assert_eq!(usage.max_over(&first), 4.0);
        assert_eq!(usage.max_over(&second), 5.0);
        assert_eq!(
            usage.max_over(&region(
                Interval::closed(2.0, 3.0),
                Interval::closed(2.0, 3.0),
            )),
            3.0
        );
        assert!(regional_usage_cells_are_disjoint(&usage.cells));
    }

    #[test]
    fn full_cost_operator_regions_use_overlap_cover_without_geometric_overlay() {
        let region = |lower, upper| StateRegion {
            propositions: Vec::new().into(),
            numeric: vec![Interval::closed(lower, upper)].into(),
        };
        let operator_region = |lower, upper| AbstractOperatorRegions {
            labels: vec![OperatorRegion {
                concrete_op_id: 0,
                source: Arc::new(region(lower, upper)),
            }],
        };
        let operator_regions = vec![operator_region(0.0, 2.0), operator_region(1.0, 3.0)];
        let mut residuals = TransitionResidualCosts::from_operator_costs(&[1.0]);

        residuals
            .reduce_by_abstract_operator_regions(
                0,
                &operator_regions,
                &AbstractOperatorCostFunction {
                    operator_costs: vec![1.0, 1.0],
                },
            )
            .unwrap();

        let residual = &residuals.operator_residuals[0];
        assert_eq!(residual.full_regional_usage.cells.len(), 2);
        assert!(residual.regional_usage.cells.is_empty());
        let overlapping = OperatorRegion {
            concrete_op_id: 0,
            source: Arc::new(region(1.5, 1.5)),
        };
        let disjoint = OperatorRegion {
            concrete_op_id: 0,
            source: Arc::new(region(4.0, 5.0)),
        };
        assert_eq!(residuals.cost_for_operator_region(1, 0, &overlapping), 0.0);
        assert_eq!(residuals.cost_for_operator_region(1, 0, &disjoint), 1.0);
    }

    fn two_state_transition_system() -> AbstractTransitionSystem {
        AbstractTransitionSystem {
            transitions: vec![AbstractTransition {
                transition_id: 0,
                abstract_op_id: 0,
                concrete_op_ids: vec![0],
                source_hash: 0,
                target_hash: 1,
            }],
            duplicate_transition_attempts: 0,
            backward: vec![vec![], vec![0]],
            forward: vec![vec![0], vec![]],
            goal_facts: vec![],
            goal_state_hashes: vec![1],
            initial_state_hash: 0,
            hash_multipliers: vec![],
            numeric_domain_sizes: vec![],
            state_regions: vec![state_region(0).into(), state_region(1).into()],
        }
    }

    #[test]
    fn explicit_label_cost_partitioning_saturates_transition_graph() {
        let system = two_state_transition_system();
        let (distances, saturated) =
            build_explicit_label_cost_partitioning_table(&system, &[5.0], None, None).unwrap();

        assert_eq!(distances, vec![5.0, 0.0]);
        assert_eq!(saturated, vec![5.0]);
    }

    #[test]
    fn explicit_regional_cost_partitioning_uses_operator_regions() {
        let system = two_state_transition_system();
        let operator_regions = vec![AbstractOperatorRegions {
            labels: vec![OperatorRegion {
                concrete_op_id: 0,
                source: state_region(0).into(),
            }],
        }];
        let residual = TransitionResidualCosts::from_operator_costs(&[5.0]);
        let (distances, saturated) = build_explicit_regional_cost_partitioning_table(
            &system,
            &operator_regions,
            &residual,
            0,
            None,
            None,
        )
        .unwrap();

        assert_eq!(distances, vec![5.0, 0.0]);
        assert_eq!(saturated.operator_costs, vec![5.0]);
    }

    fn state_region(value: usize) -> StateRegion {
        StateRegion {
            propositions: vec![vec![value as PropValueId]].into(),
            numeric: Vec::new().into(),
        }
    }

    fn numeric_state_region(lower: f64, upper: f64) -> StateRegion {
        StateRegion {
            propositions: vec![vec![0]].into(),
            numeric: vec![Interval::closed(lower, upper)].into(),
        }
    }

    fn operator_region(lower: f64, upper: f64) -> OperatorRegion {
        operator_region_for_op(0, lower, upper)
    }

    fn operator_region_for_op(concrete_op_id: usize, lower: f64, upper: f64) -> OperatorRegion {
        OperatorRegion {
            concrete_op_id,
            source: numeric_state_region(lower, upper).into(),
        }
    }

    fn operator_region_2d(
        concrete_op_id: usize,
        first: Interval,
        second: Interval,
    ) -> OperatorRegion {
        OperatorRegion {
            concrete_op_id,
            source: StateRegion {
                propositions: vec![vec![0]].into(),
                numeric: vec![first, second].into(),
            }
            .into(),
        }
    }

    fn abstract_regions_for_interval(lower: f64, upper: f64) -> AbstractOperatorRegions {
        AbstractOperatorRegions {
            labels: vec![operator_region(lower, upper)],
        }
    }

    #[test]
    fn operator_region_reductions_apply_to_same_concrete_operator_only() {
        let mut residuals = TransitionResidualCosts::from_operator_costs(&[10.0, 10.0]);
        let reduced = abstract_regions_for_interval(3.0, 7.0);
        residuals
            .reduce_by_abstract_operator_regions(
                0,
                std::slice::from_ref(&reduced),
                &AbstractOperatorCostFunction {
                    operator_costs: vec![3.0],
                },
            )
            .unwrap();

        let query = operator_region(5.0, 8.0);
        assert_eq!(residuals.cost_for_operator_region(1, 0, &query), 7.0);
        let other_op_query = operator_region_for_op(1, 5.0, 8.0);
        assert_eq!(
            residuals.cost_for_operator_region(1, 0, &other_op_query),
            10.0
        );
    }

    #[test]
    fn operator_region_reduction_allows_full_cost() {
        let mut residuals = TransitionResidualCosts::from_operator_costs(&[1.0]);
        let reduced = abstract_regions_for_interval(3.0, 7.0);
        residuals
            .reduce_by_abstract_operator_regions(
                0,
                std::slice::from_ref(&reduced),
                &AbstractOperatorCostFunction {
                    operator_costs: vec![1.0],
                },
            )
            .unwrap();

        assert_eq!(
            residuals.cost_for_operator_region(1, 0, &reduced.labels[0]),
            0.0
        );
    }

    #[test]
    fn same_abstract_operator_alternative_operator_regions_do_not_stack() {
        let mut residuals = TransitionResidualCosts::from_operator_costs(&[1.0]);
        let reduced = AbstractOperatorRegions {
            labels: vec![operator_region(0.0, 10.0), operator_region(5.0, 15.0)],
        };
        residuals
            .reduce_by_abstract_operator_regions(
                0,
                &[reduced],
                &AbstractOperatorCostFunction {
                    operator_costs: vec![0.4],
                },
            )
            .unwrap();

        assert_eq!(
            residuals.cost_for_operator_region(1, 0, &operator_region(7.0, 8.0)),
            0.6
        );
    }

    #[test]
    fn disjoint_operator_region_sources_do_not_reduce_residual_cost() {
        let mut residuals = TransitionResidualCosts::from_operator_costs(&[10.0]);
        residuals
            .reduce_by_abstract_operator_regions(
                0,
                &[abstract_regions_for_interval(0.0, 2.0)],
                &AbstractOperatorCostFunction {
                    operator_costs: vec![4.0],
                },
            )
            .unwrap();

        assert_eq!(
            residuals.cost_for_operator_region(1, 0, &operator_region(3.0, 5.0)),
            10.0
        );
    }

    #[test]
    fn target_hull_overlap_is_ignored_for_abstract_operator_regions() {
        let mut residuals = TransitionResidualCosts::from_operator_costs(&[10.0]);
        residuals
            .reduce_by_abstract_operator_regions(
                0,
                &[abstract_regions_for_interval(1.0, 10.0)],
                &AbstractOperatorCostFunction {
                    operator_costs: vec![4.0],
                },
            )
            .unwrap();

        assert_eq!(
            residuals.cost_for_operator_region(1, 0, &operator_region(10.5, 11.0)),
            10.0
        );
    }

    #[test]
    fn overlapping_operator_region_sources_reduce_residual_cost() {
        let mut residuals = TransitionResidualCosts::from_operator_costs(&[10.0]);
        residuals
            .reduce_by_abstract_operator_regions(
                0,
                &[
                    abstract_regions_for_interval(0.0, 5.0),
                    abstract_regions_for_interval(4.0, 10.0),
                ],
                &AbstractOperatorCostFunction {
                    operator_costs: vec![3.0, 4.0],
                },
            )
            .unwrap();

        assert_eq!(
            residuals.cost_for_operator_region(1, 0, &operator_region(4.5, 4.75)),
            6.0
        );
    }

    #[test]
    fn label_cp_steals_shared_operator_cost() {
        let mut residuals = TransitionResidualCosts::from_operator_costs(&[1.0]);
        residuals
            .reduce_by_abstract_operator_regions(
                0,
                &[abstract_regions_for_interval(0.0, 5.0)],
                &AbstractOperatorCostFunction {
                    operator_costs: vec![1.0],
                },
            )
            .unwrap();

        // Label CP has only one scalar residual for `go_east`: once the first
        // abstraction saturates it, every later abstraction sees zero.
        assert_eq!(residuals.operator_costs_for_label_cp(), vec![0.0]);
    }

    #[test]
    fn region_cp_preserves_residual_for_complementary_abstraction() {
        let mut residuals = TransitionResidualCosts::from_operator_costs(&[1.0]);
        residuals
            .reduce_by_abstract_operator_regions(
                0,
                &[abstract_regions_for_interval(0.0, 5.0)],
                &AbstractOperatorCostFunction {
                    operator_costs: vec![1.0],
                },
            )
            .unwrap();

        // The complementary abstraction starts after the first one's active
        // source region, so region CP preserves the unit residual there.
        let complementary = operator_region(5.0 + 1e-6, 10.0);
        let region_residual = residuals.cost_for_operator_region(1, 0, &complementary);
        assert_eq!(region_residual, 1.0);
        assert!(region_residual > residuals.operator_costs_for_label_cp()[0]);
        assert!(region_residual <= 11.0);
    }

    #[test]
    fn region_cp_overlapping_nested_targets_order_insensitive() {
        fn move_operator_regions(start: usize, end: usize) -> Vec<AbstractOperatorRegions> {
            (start..end)
                .map(|i| AbstractOperatorRegions {
                    labels: vec![OperatorRegion {
                        concrete_op_id: 0,
                        source: StateRegion {
                            propositions: vec![vec![0]].into(),
                            numeric: vec![Interval::new(i as f64, (i + 1) as f64, false, true)]
                                .into(),
                        }
                        .into(),
                    }],
                })
                .collect()
        }

        fn save_operator_region(save_op_id: usize) -> AbstractOperatorRegions {
            AbstractOperatorRegions {
                labels: vec![operator_region_for_op(save_op_id, 0.0, 15.0)],
            }
        }

        fn contribution(
            residuals: &TransitionResidualCosts,
            abstraction_id: usize,
            operator_regions: &[AbstractOperatorRegions],
        ) -> f64 {
            operator_regions
                .iter()
                .enumerate()
                .map(|(abstract_op_id, operator_region)| {
                    operator_region
                        .labels
                        .iter()
                        .map(|label| {
                            residuals.cost_for_operator_region(
                                abstraction_id,
                                abstract_op_id,
                                label,
                            )
                        })
                        .fold(f64::INFINITY, f64::min)
                })
                .sum()
        }

        fn reduce(
            residuals: &mut TransitionResidualCosts,
            abstraction_id: usize,
            operator_regions: &[AbstractOperatorRegions],
        ) {
            residuals
                .reduce_by_abstract_operator_regions(
                    abstraction_id,
                    operator_regions,
                    &AbstractOperatorCostFunction {
                        operator_costs: vec![1.0; operator_regions.len()],
                    },
                )
                .unwrap();
        }

        let mut alpha10 = move_operator_regions(0, 10);
        alpha10.push(save_operator_region(1));
        let mut alpha15 = move_operator_regions(0, 15);
        alpha15.push(save_operator_region(2));

        let label_cp_value = {
            let mut residuals = TransitionResidualCosts::from_operator_costs(&[1.0, 1.0, 1.0]);
            reduce(&mut residuals, 0, &alpha10);
            11.0 + residuals.operator_costs_for_label_cp()[2]
        };
        assert_eq!(label_cp_value, 12.0);

        let alpha10_then_alpha15 = {
            let mut residuals = TransitionResidualCosts::from_operator_costs(&[1.0, 1.0, 1.0]);
            let first = contribution(&residuals, 0, &alpha10);
            reduce(&mut residuals, 0, &alpha10);
            let second = contribution(&residuals, 1, &alpha15);
            first + second
        };
        let alpha15_then_alpha10 = {
            let mut residuals = TransitionResidualCosts::from_operator_costs(&[1.0, 1.0, 1.0]);
            let first = contribution(&residuals, 0, &alpha15);
            reduce(&mut residuals, 0, &alpha15);
            let second = contribution(&residuals, 1, &alpha10);
            first + second
        };

        assert_eq!(alpha10_then_alpha15, 17.0);
        assert_eq!(alpha15_then_alpha10, 17.0);
        assert!(alpha10_then_alpha15 <= 17.0);
        assert!(alpha15_then_alpha10 <= 17.0);
        assert!(alpha10_then_alpha15 >= 16.0);
        assert!(alpha15_then_alpha10 >= 16.0);
        assert!(alpha10_then_alpha15 > label_cp_value);
        assert!(alpha15_then_alpha10 > label_cp_value);
    }

    #[test]
    fn cross_dimension_residual_shared() {
        let mut residuals = TransitionResidualCosts::from_operator_costs(&[1.0]);
        let x_abstraction = AbstractOperatorRegions {
            labels: vec![operator_region_2d(
                0,
                Interval::closed(0.0, 1.0),
                Interval::unbounded(),
            )],
        };
        residuals
            .reduce_by_abstract_operator_regions(
                0,
                &[x_abstraction],
                &AbstractOperatorCostFunction {
                    operator_costs: vec![1.0],
                },
            )
            .unwrap();

        let y_abstraction =
            operator_region_2d(0, Interval::unbounded(), Interval::closed(0.0, 1.0));
        assert_eq!(
            residuals.cost_for_operator_region(1, 0, &y_abstraction),
            0.0
        );
    }

    #[test]
    fn infinite_tail_reduction_preserves_disjoint_tail_cost() {
        let mut residuals = TransitionResidualCosts::from_operator_costs(&[10.0]);
        let tail = abstract_regions_for_interval(f64::NEG_INFINITY, 0.0);
        residuals
            .reduce_by_abstract_operator_regions(
                0,
                &[tail],
                &AbstractOperatorCostFunction {
                    operator_costs: vec![4.0],
                },
            )
            .unwrap();

        assert_eq!(
            residuals.cost_for_operator_region(1, 0, &operator_region(1.0, f64::INFINITY),),
            10.0
        );
        assert_eq!(
            residuals.cost_for_operator_region(
                1,
                0,
                &operator_region(f64::NEG_INFINITY, f64::INFINITY),
            ),
            6.0
        );
    }

    #[test]
    fn open_infinite_tail_does_not_consume_boundary() {
        let mut residuals = TransitionResidualCosts::from_operator_costs(&[1.0]);
        let open_tail = AbstractOperatorRegions {
            labels: vec![OperatorRegion {
                concrete_op_id: 0,
                source: StateRegion {
                    propositions: vec![vec![0]].into(),
                    numeric: vec![Interval::new(f64::NEG_INFINITY, 0.0, false, false)].into(),
                }
                .into(),
            }],
        };
        residuals
            .reduce_by_abstract_operator_regions(
                0,
                &[open_tail],
                &AbstractOperatorCostFunction {
                    operator_costs: vec![1.0],
                },
            )
            .unwrap();

        assert_eq!(
            residuals.cost_for_operator_region(1, 0, &operator_region(0.0, 0.0)),
            1.0
        );
    }

    #[test]
    fn multidimensional_disjoint_regions_preserve_full_cost() {
        let mut residuals = TransitionResidualCosts::from_operator_costs(&[10.0]);
        let lower_y = AbstractOperatorRegions {
            labels: vec![operator_region_2d(
                0,
                Interval::closed(0.0, 10.0),
                Interval::new(f64::NEG_INFINITY, 0.0, false, true),
            )],
        };
        residuals
            .reduce_by_abstract_operator_regions(
                0,
                &[lower_y],
                &AbstractOperatorCostFunction {
                    operator_costs: vec![4.0],
                },
            )
            .unwrap();

        let upper_y = operator_region_2d(
            0,
            Interval::closed(0.0, 10.0),
            Interval::new(0.0, f64::INFINITY, false, false),
        );
        assert_eq!(residuals.cost_for_operator_region(1, 0, &upper_y), 10.0);
    }

    #[test]
    fn perpendicular_tail_allocations_preserve_untouched_corner() {
        let mut residuals = TransitionResidualCosts::from_operator_costs(&[10.0]);
        let left = AbstractOperatorRegions {
            labels: vec![operator_region_2d(
                0,
                Interval::new(f64::NEG_INFINITY, 0.0, false, true),
                Interval::unbounded(),
            )],
        };
        let lower = AbstractOperatorRegions {
            labels: vec![operator_region_2d(
                0,
                Interval::unbounded(),
                Interval::new(f64::NEG_INFINITY, 0.0, false, true),
            )],
        };
        residuals
            .reduce_by_abstract_operator_regions(
                0,
                &[left],
                &AbstractOperatorCostFunction {
                    operator_costs: vec![4.0],
                },
            )
            .unwrap();
        residuals
            .reduce_by_abstract_operator_regions(
                1,
                &[lower],
                &AbstractOperatorCostFunction {
                    operator_costs: vec![3.0],
                },
            )
            .unwrap();

        let upper_right = operator_region_2d(
            0,
            Interval::new(0.0, f64::INFINITY, false, false),
            Interval::new(0.0, f64::INFINITY, false, false),
        );
        let lower_left = operator_region_2d(
            0,
            Interval::new(f64::NEG_INFINITY, 0.0, false, true),
            Interval::new(f64::NEG_INFINITY, 0.0, false, true),
        );
        assert_eq!(residuals.cost_for_operator_region(2, 0, &upper_right), 10.0);
        assert_eq!(residuals.cost_for_operator_region(2, 0, &lower_left), 3.0);
    }

    #[test]
    fn regional_usage_index_matches_exact_overlap_across_blocks() {
        let mut usage = RegionalUsage::default();
        for index in 0..96 {
            let region = numeric_state_region(index as f64, index as f64 + 0.5);
            usage.add(&region, (index % 7 + 1) as f64);
        }
        assert_eq!(usage.cells.len(), 96);

        let query = numeric_state_region(30.25, 66.25);
        let expected = usage
            .cells
            .iter()
            .filter(|cell| cell.region.overlaps(&query))
            .map(|cell| cell.amount)
            .fold(0.0, f64::max);
        assert_eq!(usage.max_over(&query), expected);
        assert!(usage.index.borrow().is_some());
    }
}
