//! Label and regional saturated cost partitioning for abstraction components.
//!
//! Each [`ConcreteOperatorFootprint::source_region`] stores the *regressed
//! preimage source* of an abstract operator's effect — the intersection of the
//! abstract source region with the inverse image of the abstract target region
//! under the operator's numeric effect (computed in
//! `domain_abstraction_factory::build_concrete_operator_footprint`).
//!
//! Unbounded preimages are ordinary regions. Cost is allocated on their exact
//! source footprint and remains available on disjoint regions.

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use planforge_sas::utils::float_tolerance;

use planforge_sas::utils::interval::Interval;

#[path = "explicit_scp.rs"]
mod explicit_scp;
#[path = "region.rs"]
mod region;

pub use explicit_scp::*;
pub use region::*;

const ABSTRACT_OPERATOR_REGION_HASH: usize = usize::MAX;
const MAX_ABSTRACT_OPERATOR_REDUCTION_PIECES: usize = 4096;
const MAX_TOTAL_ABSTRACT_OPERATOR_REDUCTION_PIECES: usize = 50_000;

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
    reductions: Vec<ResidualReduction>,
    reduction_indices: HashMap<TransitionIdentity, usize>,
    generation: Cell<u64>,
    uniform_cost_cache: Cell<Option<f64>>,
    transition_cost_cache: RefCell<HashMap<TransitionQueryKey, CachedCost>>,
    full_reduction_index: RefCell<Option<FullReductionIndex>>,
    /// Lazy sorted index over `reductions` for fast candidate enumeration in
    /// `max_overlap_reduction`. Indexes reductions by the lower bound of their
    /// source-region interval on a chosen primary numeric dimension. Built on
    /// first query after the `generation` advances; invalidated implicitly by
    /// the generation-mismatch check.
    sorted_index: RefCell<Option<SortedReductionIndex>>,
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
    /// abstract transition in that table, even when conservative footprints
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

/// A per-`OperatorResidual` sorted view that lets `max_overlap_reduction`
/// enumerate only the reductions whose primary-dim interval could overlap a
/// query's primary-dim interval, instead of scanning all `reductions` linearly.
///
/// The primary dim is chosen at build time as the numeric dimension with the
/// highest number of distinct lower bounds across the reductions. For
/// operator-residuals where every reduction has the same lower bound on every
/// dim (e.g. only one reduction stored), `primary_dim` is `None` and the
/// fallback is a full scan.
#[derive(Debug)]
struct SortedReductionIndex {
    /// Indices into `reductions`, sorted by the chosen primary dim's lower bound.
    sorted: Vec<usize>,
    primary_dim: Option<usize>,
    generation: u64,
}

impl SortedReductionIndex {
    fn build(reductions: &[ResidualReduction], generation: u64) -> Self {
        let primary_dim = Self::choose_primary_dim(reductions);
        let mut sorted: Vec<usize> = (0..reductions.len()).collect();
        if let Some(dim) = primary_dim {
            sorted.sort_by(|&a, &b| {
                let la = reductions[a].condition.region.source.numeric[dim].lower;
                let lb = reductions[b].condition.region.source.numeric[dim].lower;
                la.partial_cmp(&lb).unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        Self {
            sorted,
            primary_dim,
            generation,
        }
    }

    fn choose_primary_dim(reductions: &[ResidualReduction]) -> Option<usize> {
        if reductions.len() < 2 {
            return None;
        }
        let first = &reductions[0].condition.region.source.numeric;
        let num_dims = first.len();
        let mut best_dim: Option<usize> = None;
        let mut best_distinct = 1usize;
        for dim in 0..num_dims {
            let mut distinct: HashSet<u64> = HashSet::with_capacity(reductions.len().min(64));
            for r in reductions {
                distinct.insert(r.condition.region.source.numeric[dim].lower.to_bits());
            }
            if distinct.len() > best_distinct {
                best_distinct = distinct.len();
                best_dim = Some(dim);
            }
        }
        best_dim
    }

    /// Pre-filter reductions by their primary-dim interval. Returns indices into
    /// `reductions` for entries that could overlap the query on the primary dim.
    /// May return false positives (cleared by the full overlap check downstream);
    /// must not return false negatives.
    fn candidates(
        &self,
        reductions: &[ResidualReduction],
        query: Option<&TransitionCondition>,
    ) -> Vec<usize> {
        let Some(dim) = self.primary_dim else {
            return self.sorted.clone();
        };
        let Some(q) = query else {
            return self.sorted.clone();
        };
        let q_iv = &q.region.source.numeric[dim];
        // Binary search: first `i` where reductions[sorted[i]].lower > q.upper.
        // Everything before is a candidate up to the further upper-bound filter.
        let end = self.sorted.partition_point(|&i| {
            reductions[i].condition.region.source.numeric[dim].lower <= q_iv.upper
        });
        self.sorted[..end]
            .iter()
            .copied()
            .filter(|&i| reductions[i].condition.region.source.numeric[dim].upper >= q_iv.lower)
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq)]
struct ResidualReduction {
    amount: f64,
    condition: TransitionCondition,
}

#[derive(Debug, Clone, PartialEq)]
struct TransitionCondition {
    abstraction_id: usize,
    source_hash: usize,
    abstract_op_id: usize,
    target_hash: usize,
    region: TransitionRegion,
}

impl TransitionCondition {
    fn identity(&self) -> TransitionIdentity {
        TransitionIdentity {
            abstraction_id: self.abstraction_id,
            source_hash: self.source_hash,
            abstract_op_id: self.abstract_op_id,
            target_hash: self.target_hash,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct TransitionIdentity {
    abstraction_id: usize,
    source_hash: usize,
    abstract_op_id: usize,
    target_hash: usize,
}

/// The transition a residual cost is read or reduced for: the concrete operator
/// that pays the cost, the abstraction of the collection the query belongs to,
/// and the abstract operator taking `source_hash` to `target_hash`.
/// `TransitionIdentity` is the same thing without the operator, which is how
/// each operator's own reduction map is keyed.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct OperatorTransition {
    pub concrete_op_id: usize,
    pub abstraction_id: usize,
    pub source_hash: usize,
    pub abstract_op_id: usize,
    pub target_hash: usize,
}

impl OperatorTransition {
    fn identity(&self) -> TransitionIdentity {
        TransitionIdentity {
            abstraction_id: self.abstraction_id,
            source_hash: self.source_hash,
            abstract_op_id: self.abstract_op_id,
            target_hash: self.target_hash,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct TransitionQueryKey {
    abstraction_id: usize,
    source_hash: usize,
    abstract_op_id: usize,
    target_hash: usize,
    region: Option<TransitionRegionKey>,
}

#[derive(Copy, Clone, Debug)]
struct CachedCost {
    generation: u64,
    cost: f64,
}

#[derive(Clone, Debug)]
struct FullReductionIndex {
    generation: u64,
    kind: FullReductionIndexKind,
    all_reductions_full: bool,
}

#[derive(Clone, Debug)]
enum FullReductionIndexKind {
    Prop {
        feature: RegionFeature,
        buckets: HashMap<usize, Vec<usize>>,
    },
    Numeric {
        feature: RegionFeature,
        intervals: Vec<IndexedInterval>,
        /// `prefix_max_upper[i]` is the maximum `interval.upper` across
        /// `intervals[..=i]`. Lets `lookup_full_reduction_overlap` short-circuit
        /// when no candidate's upper bound can reach the query's lower bound.
        prefix_max_upper: Vec<f64>,
    },
}

#[derive(Clone, Debug)]
struct IndexedInterval {
    interval: Interval,
    reduction_id: usize,
}

#[derive(Copy, Clone, Debug)]
enum RegionFeature {
    SourceProp(usize),
    TargetProp(usize),
    SourceNumeric(usize),
    TargetNumeric(usize),
}

impl TransitionResidualCosts {
    pub fn from_operator_costs(costs: &[f64]) -> Self {
        let operator_residuals = costs
            .iter()
            .map(|&base_cost| OperatorResidual {
                base_cost,
                full_regional_usage: RegionalUsage::default(),
                regional_usage: RegionalUsage::default(),
                reductions: Vec::new(),
                reduction_indices: HashMap::new(),
                generation: Cell::new(0),
                uniform_cost_cache: Cell::new(None),
                transition_cost_cache: RefCell::new(HashMap::new()),
                full_reduction_index: RefCell::new(None),
                sorted_index: RefCell::new(None),
            })
            .collect();
        Self { operator_residuals }
    }

    pub fn num_reductions(&self) -> usize {
        self.operator_residuals
            .iter()
            .map(|residual| {
                residual.reductions.len()
                    + residual.full_regional_usage.cells.len()
                    + residual.regional_usage.cells.len()
            })
            .sum()
    }

    pub fn has_reductions(&self) -> bool {
        self.operator_residuals.iter().any(|residual| {
            !residual.reductions.is_empty()
                || !residual.full_regional_usage.is_empty()
                || !residual.regional_usage.is_empty()
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
                let reduction = max_overlap_reduction(None, residual, residual.base_cost)
                    .max(full_regional_reduction)
                    .max(
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
        max_variants_per_operator: usize,
        max_guard_conditions_per_variant: usize,
    ) -> Vec<LmCutResidualOperatorCostPartition> {
        let uniform_costs = self.operator_costs_for_label_cp();
        self.operator_residuals
            .iter()
            .enumerate()
            .map(|(op_id, residual)| {
                let fallback_cost = residual.base_cost.max(0.0);
                if residual.reductions.is_empty()
                    || residual.reductions.len() > max_variants_per_operator
                {
                    return LmCutResidualOperatorCostPartition {
                        fallback_cost: uniform_costs.get(op_id).copied().unwrap_or(fallback_cost),
                        variants: Vec::new(),
                    };
                }

                let mut variants = Vec::with_capacity(residual.reductions.len());
                for reduction in &residual.reductions {
                    if !lmcut_residual_region_is_compact(
                        &reduction.condition.region.source,
                        max_guard_conditions_per_variant,
                    ) {
                        return LmCutResidualOperatorCostPartition {
                            fallback_cost: uniform_costs
                                .get(op_id)
                                .copied()
                                .unwrap_or(fallback_cost),
                            variants: Vec::new(),
                        };
                    }
                    variants.push(LmCutResidualCostVariant {
                        cost: (residual.base_cost - reduction.amount).max(0.0),
                        source_region: reduction.condition.region.source.as_ref().clone(),
                    });
                }

                LmCutResidualOperatorCostPartition {
                    fallback_cost,
                    variants,
                }
            })
            .collect()
    }

    pub fn cost_for_operator_footprint(
        &self,
        current_abstraction_id: usize,
        abstract_op_id: usize,
        footprint: &ConcreteOperatorFootprint,
    ) -> f64 {
        let Some(residual) = self.operator_residuals.get(footprint.concrete_op_id) else {
            return f64::INFINITY;
        };
        if !residual.base_cost.is_finite() {
            return f64::INFINITY;
        }
        let regional = if residual
            .full_regional_usage
            .overlaps(&footprint.source_region)
        {
            residual.base_cost
        } else {
            residual.regional_usage.max_over(&footprint.source_region)
        };
        let legacy = self.cost_for_transition_with_region(
            OperatorTransition {
                concrete_op_id: footprint.concrete_op_id,
                abstraction_id: current_abstraction_id,
                source_hash: ABSTRACT_OPERATOR_REGION_HASH,
                abstract_op_id,
                target_hash: ABSTRACT_OPERATOR_REGION_HASH,
            },
            TransitionRegion {
                source: Arc::clone(&footprint.source_region),
                target: Arc::clone(&footprint.source_region),
            },
            None,
        );
        (legacy - regional).max(0.0)
    }

    #[cfg(test)]
    fn cost_for_transition_with_region_key(
        &self,
        transition: OperatorTransition,
        source_region: &StateRegion,
        target_region: &StateRegion,
        region_key: Option<TransitionRegionKey>,
    ) -> f64 {
        self.cost_for_transition_with_region(
            transition,
            TransitionRegion {
                source: Arc::new(source_region.clone()),
                target: Arc::new(target_region.clone()),
            },
            region_key,
        )
    }

    #[cfg(test)]
    fn reduction_cost_for_transition(
        &self,
        transition: OperatorTransition,
        source_region: &StateRegion,
        target_region: &StateRegion,
    ) -> f64 {
        self.cost_for_transition_with_region_key(
            transition,
            source_region,
            target_region,
            Some(transition_region_key_parts(source_region, target_region)),
        )
    }

    fn cost_for_transition_with_region(
        &self,
        transition: OperatorTransition,
        query_region: TransitionRegion,
        region_key: Option<TransitionRegionKey>,
    ) -> f64 {
        let OperatorTransition {
            concrete_op_id,
            abstraction_id,
            source_hash,
            abstract_op_id,
            target_hash,
        } = transition;
        let Some(residual) = self.operator_residuals.get(concrete_op_id) else {
            return f64::INFINITY;
        };
        if !residual.base_cost.is_finite() {
            return f64::INFINITY;
        }

        let key = TransitionQueryKey {
            abstraction_id,
            source_hash,
            abstract_op_id,
            target_hash,
            region: region_key,
        };
        if let Some(cached) = residual.transition_cost_cache.borrow().get(&key)
            && cached.generation == residual.generation.get()
        {
            return cached.cost;
        }

        let query = TransitionCondition {
            abstraction_id,
            source_hash,
            abstract_op_id,
            target_hash,
            region: query_region,
        };

        if let Some(has_full_overlap) = residual.lookup_full_reduction_overlap(&query) {
            let cost = if has_full_overlap {
                0.0
            } else {
                residual.base_cost
            };
            residual.transition_cost_cache.borrow_mut().insert(
                key,
                CachedCost {
                    generation: residual.generation.get(),
                    cost,
                },
            );
            return cost;
        }
        let reduction = max_overlap_reduction(Some(&query), residual, residual.base_cost);
        let cost = (residual.base_cost - reduction).max(0.0);
        residual.transition_cost_cache.borrow_mut().insert(
            key,
            CachedCost {
                generation: residual.generation.get(),
                cost,
            },
        );
        cost
    }

    pub fn reduce_by_tcf(
        &mut self,
        producing_abstraction_id: usize,
        transition_system: &AbstractTransitionSystem,
        tcf: &AbstractTransitionCostFunction,
    ) -> Result<()> {
        ensure!(
            transition_system.transitions.len() == tcf.transition_costs.len(),
            "transition system/cost function size mismatch: {} vs {}",
            transition_system.transitions.len(),
            tcf.transition_costs.len()
        );
        for transition in &transition_system.transitions {
            let saturated = tcf.transition_costs[transition.transition_id];
            ensure!(
                !saturated.is_finite() || saturated >= -float_tolerance::SEARCH_EPSILON,
                "negative transition saturated costs are not supported: transition {} has {}",
                transition.transition_id,
                saturated
            );
            if !saturated.is_finite() || saturated <= float_tolerance::SEARCH_EPSILON {
                continue;
            }
            for &concrete_op_id in &transition.concrete_op_ids {
                let region = transition_system.transition_region(transition)?;
                self.reduce_exact_transition(
                    OperatorTransition {
                        concrete_op_id,
                        abstraction_id: producing_abstraction_id,
                        source_hash: transition.source_hash,
                        abstract_op_id: transition.abstract_op_id,
                        target_hash: transition.target_hash,
                    },
                    &region,
                    saturated,
                )
                .with_context(|| {
                    format!(
                        "failed to reduce op {concrete_op_id} by transition {}",
                        transition.transition_id
                    )
                })?;
            }
        }
        Ok(())
    }

    pub fn reduce_by_abstract_operator_tcf(
        &mut self,
        producing_abstraction_id: usize,
        transition_system: &AbstractTransitionSystem,
        tcf: &AbstractOperatorCostFunction,
    ) -> Result<bool> {
        let concrete_op_ids = transition_system.concrete_operator_ids_by_abstract_operator();
        ensure!(
            concrete_op_ids.len() == tcf.operator_costs.len(),
            "abstract-operator system/cost function size mismatch: {} vs {}",
            concrete_op_ids.len(),
            tcf.operator_costs.len()
        );
        let transition_counts =
            transition_system.transition_counts_by_abstract_operator(tcf.operator_costs.len());
        let mut total_reduction_pieces = 0usize;
        for (abstract_op_id, &saturated) in tcf.operator_costs.iter().enumerate() {
            if !saturated.is_finite() || saturated <= float_tolerance::SEARCH_EPSILON {
                continue;
            }
            total_reduction_pieces = total_reduction_pieces.saturating_add(
                transition_counts[abstract_op_id]
                    .saturating_mul(concrete_op_ids[abstract_op_id].len()),
            );
            if total_reduction_pieces > MAX_TOTAL_ABSTRACT_OPERATOR_REDUCTION_PIECES {
                return Ok(false);
            }
        }
        if transition_system.state_regions.is_empty() {
            return Ok(false);
        }
        let covers = transition_system.abstract_operator_region_covers();
        for (abstract_op_id, &saturated) in tcf.operator_costs.iter().enumerate() {
            ensure!(
                !saturated.is_finite() || saturated >= -float_tolerance::SEARCH_EPSILON,
                "negative abstract-operator saturated costs are not supported: abstract op {} has {}",
                abstract_op_id,
                saturated
            );
            if !saturated.is_finite() || saturated <= float_tolerance::SEARCH_EPSILON {
                continue;
            }
            let Some(cover) = covers.get(abstract_op_id) else {
                continue;
            };
            for (piece_id, region) in cover.iter().enumerate() {
                for &concrete_op_id in &concrete_op_ids[abstract_op_id] {
                    let Some(residual) = self.operator_residuals.get_mut(concrete_op_id) else {
                        continue;
                    };
                    ensure!(
                        residual.base_cost.is_finite(),
                        "no base residual cost for operator {concrete_op_id}"
                    );
                    ensure!(
                        saturated <= residual.base_cost + float_tolerance::SEARCH_EPSILON,
                        "residual cost underflow: abstract-operator reduction {saturated} exceeds base cost {} for operator {concrete_op_id}",
                        residual.base_cost
                    );
                    let condition = TransitionCondition {
                        abstraction_id: producing_abstraction_id,
                        source_hash: piece_id,
                        abstract_op_id,
                        target_hash: piece_id,
                        region: region.clone(),
                    };
                    let identity = condition.identity();
                    if let Some(&index) = residual.reduction_indices.get(&identity) {
                        let reduction = &mut residual.reductions[index];
                        let new_amount = reduction.amount + saturated;
                        ensure!(
                            new_amount <= residual.base_cost + float_tolerance::SEARCH_EPSILON,
                            "abstract-operator reductions for concrete operator {concrete_op_id} exceed base cost {}",
                            residual.base_cost
                        );
                        reduction.amount = new_amount.min(residual.base_cost);
                    } else {
                        let index = residual.reductions.len();
                        residual.reductions.push(ResidualReduction {
                            amount: saturated.min(residual.base_cost),
                            condition,
                        });
                        let previous = residual.reduction_indices.insert(identity, index);
                        assert!(previous.is_none(), "duplicate residual reduction identity");
                    }
                    residual.invalidate_cache();
                }
            }
        }
        Ok(true)
    }

    pub fn reduce_by_abstract_operator_footprints(
        &mut self,
        producing_abstraction_id: usize,
        footprints: &[AbstractOperatorFootprint],
        tcf: &AbstractOperatorCostFunction,
    ) -> Result<()> {
        self.reduce_by_abstract_operator_footprints_with_deadline(
            producing_abstraction_id,
            footprints,
            tcf,
            None,
        )
    }

    pub fn reduce_by_abstract_operator_footprints_with_deadline(
        &mut self,
        producing_abstraction_id: usize,
        footprints: &[AbstractOperatorFootprint],
        tcf: &AbstractOperatorCostFunction,
        deadline: Option<Instant>,
    ) -> Result<()> {
        ensure!(
            footprints.len() >= tcf.operator_costs.len(),
            "abstract-operator footprint/cost function size mismatch: footprints={} costs={}",
            footprints.len(),
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

            for footprint in &footprints[abstract_op_id].labels {
                let concrete_op_id = footprint.concrete_op_id;
                let current_residual = self.cost_for_operator_footprint(
                    producing_abstraction_id,
                    abstract_op_id,
                    footprint,
                );
                ensure!(
                    current_residual.is_finite(),
                    "residual cost for abstract op {abstract_op_id}, concrete op {concrete_op_id} must be finite"
                );
                ensure!(
                    saturated <= current_residual + float_tolerance::SEARCH_EPSILON,
                    "abstract-operator footprint reduction {saturated} exceeds current residual cost {current_residual} for concrete operator {concrete_op_id}"
                );
                let Some(residual) = self.operator_residuals.get(concrete_op_id) else {
                    continue;
                };
                if residual.base_cost <= float_tolerance::SEARCH_EPSILON {
                    continue;
                }
                ensure!(
                    residual.base_cost.is_finite(),
                    "no base residual cost for operator {concrete_op_id}"
                );
                ensure!(
                    saturated <= residual.base_cost + float_tolerance::SEARCH_EPSILON,
                    "residual cost underflow: abstract-operator footprint reduction {saturated} exceeds base cost {} for operator {concrete_op_id}",
                    residual.base_cost
                );
                entries.push(RegionalCostAllocationEntry {
                    footprint: footprint.clone(),
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
            let concrete_op_id = entry.footprint.concrete_op_id;
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
                    &entry.footprint.source_region,
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
                .expect("validated concrete operator footprint must exist");
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

    fn reduce_exact_transition(
        &mut self,
        transition: OperatorTransition,
        region: &TransitionRegion,
        saturated: f64,
    ) -> Result<()> {
        let concrete_op_id = transition.concrete_op_id;
        ensure!(
            concrete_op_id < self.operator_residuals.len(),
            "concrete operator id out of bounds: {concrete_op_id}"
        );
        let condition = TransitionCondition {
            abstraction_id: transition.abstraction_id,
            source_hash: transition.source_hash,
            abstract_op_id: transition.abstract_op_id,
            target_hash: transition.target_hash,
            region: region.clone(),
        };
        let residual = &mut self.operator_residuals[concrete_op_id];
        let identity = transition.identity();
        if let Some(&index) = residual.reduction_indices.get(&identity) {
            let reduction = &mut residual.reductions[index];
            let new_amount = reduction.amount + saturated;
            ensure!(
                new_amount <= residual.base_cost + float_tolerance::SEARCH_EPSILON,
                "residual cost underflow: transition reductions for operator {concrete_op_id} exceed base cost {}",
                residual.base_cost
            );
            reduction.amount = new_amount.min(residual.base_cost);
            residual.invalidate_cache();
            return Ok(());
        }

        ensure!(
            residual.base_cost.is_finite(),
            "no base residual cost for operator {concrete_op_id}"
        );
        ensure!(
            saturated <= residual.base_cost + float_tolerance::SEARCH_EPSILON,
            "residual cost underflow: transition reduction {saturated} exceeds base cost {} for operator {concrete_op_id}",
            residual.base_cost
        );
        let index = residual.reductions.len();
        residual.reductions.push(ResidualReduction {
            amount: saturated.min(residual.base_cost),
            condition,
        });
        let previous = residual.reduction_indices.insert(identity, index);
        assert!(previous.is_none(), "duplicate residual reduction identity");
        residual.invalidate_cache();
        Ok(())
    }
}

impl OperatorResidual {
    fn invalidate_cache(&self) {
        self.generation.set(self.generation.get().wrapping_add(1));
        self.uniform_cost_cache.set(None);
        self.transition_cost_cache.borrow_mut().clear();
        self.full_reduction_index.borrow_mut().take();
        self.sorted_index.borrow_mut().take();
    }

    /// Ensure the sorted candidate index is built and up to date with the
    /// current generation. Returns `Some(_)` only when the index has at least
    /// `primary_dim` set (i.e. there is some discriminating numeric axis).
    /// Returns `None` when no discriminating dim was found — callers fall back
    /// to a full linear scan, which is correct and is the best we can do for
    /// trivially small reduction sets.
    fn ensure_sorted_index(&self) -> bool {
        let needs_build = {
            let borrow = self.sorted_index.borrow();
            match borrow.as_ref() {
                Some(index) => index.generation != self.generation.get(),
                None => true,
            }
        };
        if needs_build {
            *self.sorted_index.borrow_mut() = Some(SortedReductionIndex::build(
                &self.reductions,
                self.generation.get(),
            ));
        }
        true
    }

    fn lookup_full_reduction_overlap(&self, query: &TransitionCondition) -> Option<bool> {
        if !self.base_cost.is_finite()
            || self.base_cost <= float_tolerance::SEARCH_EPSILON
            || self.reductions.is_empty()
        {
            return None;
        }
        self.ensure_full_reduction_index()?;
        let index_ref = self.full_reduction_index.borrow();
        let index = index_ref.as_ref()?;
        if index.generation != self.generation.get() {
            return None;
        }
        match &index.kind {
            FullReductionIndexKind::Prop { feature, buckets } => {
                let values = query_values_for_feature(&query.region, *feature)?;
                for &value in values {
                    let Some(bucket) = buckets.get(&(value as usize)) else {
                        continue;
                    };
                    if bucket.iter().any(|&reduction_id| {
                        let reduction = &self.reductions[reduction_id];
                        compatible_identities(query, &reduction.condition)
                            && reduction.condition.region.overlaps(&query.region)
                    }) {
                        return Some(true);
                    }
                }
            }
            FullReductionIndexKind::Numeric {
                feature,
                intervals,
                prefix_max_upper,
            } => {
                let query_interval = interval_for_feature(&query.region, *feature)?;
                // Binary-search for the first indexed interval whose lower
                // strictly starts after the query — everything past that point
                // cannot overlap and can be skipped without inspection. Without
                // this, queries whose lower lies above every stored lower would
                // walk the entire intervals vector before hitting `break`.
                let end = intervals.partition_point(|indexed| {
                    !interval_starts_after(&indexed.interval, query_interval)
                });
                // Short-circuit: if the max upper among the candidate prefix is
                // strictly below the query's lower, no candidate can overlap.
                // This is the dominant case when the query sits *above* the
                // stored intervals (e.g. a single-goal abstraction querying
                // high-y cells against a full-goal abstraction that only refined
                // low-y cells). We deliberately use `<` rather than `<=` so a
                // closed boundary on either endpoint still falls through to the
                // exact overlap check below.
                if end > 0 && prefix_max_upper[end - 1] < query_interval.lower {
                    return if index.all_reductions_full {
                        Some(false)
                    } else {
                        None
                    };
                }
                for indexed in &intervals[..end] {
                    if !indexed.interval.intersects(query_interval) {
                        continue;
                    }
                    let reduction = &self.reductions[indexed.reduction_id];
                    if compatible_identities(query, &reduction.condition)
                        && reduction.condition.region.overlaps(&query.region)
                    {
                        return Some(true);
                    }
                }
            }
        }
        if index.all_reductions_full {
            Some(false)
        } else {
            None
        }
    }

    fn ensure_full_reduction_index(&self) -> Option<()> {
        if self
            .full_reduction_index
            .borrow()
            .as_ref()
            .is_some_and(|index| index.generation == self.generation.get())
        {
            return Some(());
        }
        let index =
            build_full_reduction_index(&self.reductions, self.base_cost, self.generation.get())?;
        self.full_reduction_index.borrow_mut().replace(index);
        Some(())
    }
}

fn build_full_reduction_index(
    reductions: &[ResidualReduction],
    cap: f64,
    generation: u64,
) -> Option<FullReductionIndex> {
    if reductions.is_empty() || !cap.is_finite() || cap <= float_tolerance::SEARCH_EPSILON {
        return None;
    }
    let all_reductions_full = reductions
        .iter()
        .all(|reduction| reduction.amount >= cap - float_tolerance::SEARCH_EPSILON);
    let feature = best_full_reduction_feature(reductions, cap)?;
    let kind = match feature {
        RegionFeature::SourceProp(_) | RegionFeature::TargetProp(_) => {
            let mut buckets: HashMap<usize, Vec<usize>> = HashMap::new();
            for (reduction_id, reduction) in reductions.iter().enumerate() {
                if reduction.amount < cap - float_tolerance::SEARCH_EPSILON {
                    continue;
                }
                let value = singleton_value_for_feature(&reduction.condition.region, feature)?;
                buckets.entry(value).or_default().push(reduction_id);
            }
            if buckets.is_empty() {
                return None;
            }
            FullReductionIndexKind::Prop { feature, buckets }
        }
        RegionFeature::SourceNumeric(_) | RegionFeature::TargetNumeric(_) => {
            let mut intervals = Vec::new();
            for (reduction_id, reduction) in reductions.iter().enumerate() {
                if reduction.amount < cap - float_tolerance::SEARCH_EPSILON {
                    continue;
                }
                let interval = *interval_for_feature(&reduction.condition.region, feature)?;
                if interval.is_empty() {
                    return None;
                }
                intervals.push(IndexedInterval {
                    interval,
                    reduction_id,
                });
            }
            if intervals.is_empty() {
                return None;
            }
            intervals.sort_by(|left, right| {
                left.interval
                    .lower
                    .total_cmp(&right.interval.lower)
                    .then_with(|| left.interval.upper.total_cmp(&right.interval.upper))
            });
            let mut prefix_max_upper: Vec<f64> = Vec::with_capacity(intervals.len());
            let mut running = f64::NEG_INFINITY;
            for indexed in &intervals {
                running = running.max(indexed.interval.upper);
                prefix_max_upper.push(running);
            }
            FullReductionIndexKind::Numeric {
                feature,
                intervals,
                prefix_max_upper,
            }
        }
    };
    Some(FullReductionIndex {
        generation,
        kind,
        all_reductions_full,
    })
}

fn best_full_reduction_feature(
    reductions: &[ResidualReduction],
    cap: f64,
) -> Option<RegionFeature> {
    let first_full = reductions
        .iter()
        .find(|reduction| reduction.amount >= cap - float_tolerance::SEARCH_EPSILON)?;
    let source_len = first_full.condition.region.source.propositions.len();
    let target_len = first_full.condition.region.target.propositions.len();
    let mut best = None;
    let mut best_distinct = 0usize;
    for feature in (0..source_len)
        .map(RegionFeature::SourceProp)
        .chain((0..target_len).map(RegionFeature::TargetProp))
    {
        let mut buckets = std::collections::BTreeSet::new();
        let mut usable = false;
        for reduction in reductions
            .iter()
            .filter(|reduction| reduction.amount >= cap - float_tolerance::SEARCH_EPSILON)
        {
            let Some(value) = singleton_value_for_feature(&reduction.condition.region, feature)
            else {
                usable = false;
                break;
            };
            usable = true;
            buckets.insert(value);
        }
        if usable && buckets.len() > best_distinct {
            best = Some(feature);
            best_distinct = buckets.len();
        }
    }
    let source_numeric_len = first_full.condition.region.source.numeric.len();
    let target_numeric_len = first_full.condition.region.target.numeric.len();
    for feature in (0..source_numeric_len)
        .map(RegionFeature::SourceNumeric)
        .chain((0..target_numeric_len).map(RegionFeature::TargetNumeric))
    {
        let mut buckets = std::collections::BTreeSet::new();
        let mut usable = false;
        for reduction in reductions
            .iter()
            .filter(|reduction| reduction.amount >= cap - float_tolerance::SEARCH_EPSILON)
        {
            let Some(interval) = interval_for_feature(&reduction.condition.region, feature) else {
                usable = false;
                break;
            };
            if interval.is_empty() || (!interval.lower.is_finite() && !interval.upper.is_finite()) {
                usable = false;
                break;
            }
            usable = true;
            buckets.insert(interval_key(interval));
        }
        if usable && buckets.len() > best_distinct {
            best = Some(feature);
            best_distinct = buckets.len();
        }
    }
    best
}

fn singleton_value_for_feature(region: &TransitionRegion, feature: RegionFeature) -> Option<usize> {
    let values = match feature {
        RegionFeature::SourceProp(var_id) => region.source.propositions.get(var_id)?,
        RegionFeature::TargetProp(var_id) => region.target.propositions.get(var_id)?,
        RegionFeature::SourceNumeric(_) | RegionFeature::TargetNumeric(_) => return None,
    };
    (values.len() == 1).then_some(values[0] as usize)
}

fn query_values_for_feature(
    region: &TransitionRegion,
    feature: RegionFeature,
) -> Option<&[PropValueId]> {
    match feature {
        RegionFeature::SourceProp(var_id) => region.source.propositions.get(var_id),
        RegionFeature::TargetProp(var_id) => region.target.propositions.get(var_id),
        RegionFeature::SourceNumeric(_) | RegionFeature::TargetNumeric(_) => None,
    }
    .map(Vec::as_slice)
}

fn interval_for_feature(region: &TransitionRegion, feature: RegionFeature) -> Option<&Interval> {
    match feature {
        RegionFeature::SourceNumeric(var_id) => region.source.numeric.get(var_id),
        RegionFeature::TargetNumeric(var_id) => region.target.numeric.get(var_id),
        RegionFeature::SourceProp(_) | RegionFeature::TargetProp(_) => None,
    }
}

fn interval_starts_after(left: &Interval, right: &Interval) -> bool {
    left.lower > right.upper
        || (left.lower == right.upper && !(left.lower_closed && right.upper_closed))
}

fn max_overlap_reduction(
    query: Option<&TransitionCondition>,
    residual: &OperatorResidual,
    cap: f64,
) -> f64 {
    if !cap.is_finite() || cap <= float_tolerance::SEARCH_EPSILON {
        return 0.0;
    }
    let reductions = &residual.reductions;
    if reductions.is_empty() {
        return 0.0;
    }
    residual.ensure_sorted_index();
    let index_ref = residual.sorted_index.borrow();
    let candidates: Vec<usize> = match index_ref.as_ref() {
        Some(index) => index.candidates(reductions, query),
        None => (0..reductions.len()).collect(),
    };
    drop(index_ref);
    if candidates.is_empty() {
        return 0.0;
    }
    let mut has_subcap_reduction = false;
    if candidates.iter().any(|&i| {
        let reduction = &reductions[i];
        has_subcap_reduction |= reduction.amount < cap - float_tolerance::SEARCH_EPSILON;
        reduction.amount >= cap - float_tolerance::SEARCH_EPSILON
            && query.is_none_or(|query| {
                compatible_identities(query, &reduction.condition)
                    && reduction.condition.region.overlaps(&query.region)
            })
    }) {
        return cap;
    }
    if !has_subcap_reduction {
        return 0.0;
    }
    let mut relevant: Vec<&ResidualReduction> = candidates
        .iter()
        .map(|&i| &reductions[i])
        .filter(|reduction| {
            query.is_none_or(|query| {
                compatible_identities(query, &reduction.condition)
                    && reduction.condition.region.overlaps(&query.region)
            })
        })
        .collect();
    // Exact overlap accounting is exponential in the number of overlapping
    // reductions. For very large overlap sets we deliberately over-approximate
    // the already allocated cost. This can only lower residual costs and make
    // the heuristic weaker; it must not increase allocated cost.
    if relevant.len() > 64 {
        return relevant
            .iter()
            .map(|reduction| reduction.amount.max(0.0))
            .sum::<f64>()
            .min(cap);
    }
    relevant.sort_by(|left, right| {
        right
            .amount
            .partial_cmp(&left.amount)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let suffix: Vec<f64> = {
        let mut suffix = vec![0.0; relevant.len() + 1];
        for index in (0..relevant.len()).rev() {
            suffix[index] = suffix[index + 1] + relevant[index].amount.max(0.0);
        }
        suffix
    };

    /// The part of the branch-and-bound that does not change as it recurses:
    /// the reductions being chosen from, the best-case remaining sum per
    /// suffix, the query they must stay compatible with, and the bound above
    /// which no branch can improve.
    struct SearchSpace<'a> {
        relevant: &'a [&'a ResidualReduction],
        suffix: &'a [f64],
        query: Option<&'a TransitionCondition>,
        cap: f64,
    }

    fn search(
        space: &SearchSpace<'_>,
        index: usize,
        selected: &mut Vec<usize>,
        current_sum: f64,
        best: &mut f64,
    ) {
        if index == space.relevant.len() {
            *best = best.max(current_sum);
            return;
        }
        if *best >= space.cap - float_tolerance::SEARCH_EPSILON {
            return;
        }
        if current_sum + space.suffix[index] <= *best + float_tolerance::SEARCH_EPSILON {
            return;
        }

        let reduction = space.relevant[index];
        if can_add_reduction(space.query, selected, &reduction.condition, space.relevant) {
            selected.push(index);
            search(
                space,
                index + 1,
                selected,
                current_sum + reduction.amount.max(0.0),
                best,
            );
            selected.pop();
        }
        search(space, index + 1, selected, current_sum, best);
    }

    let mut best = 0.0;
    let mut selected = Vec::new();
    search(
        &SearchSpace {
            relevant: &relevant,
            suffix: &suffix,
            query,
            cap,
        },
        0,
        &mut selected,
        0.0,
        &mut best,
    );
    best.min(cap)
}

fn compatible_identities(left: &TransitionCondition, right: &TransitionCondition) -> bool {
    if left.abstraction_id != right.abstraction_id {
        return true;
    }
    if left.abstract_op_id != right.abstract_op_id {
        return false;
    }
    let left_is_abstract_operator_query = left.source_hash == ABSTRACT_OPERATOR_REGION_HASH
        || left.target_hash == ABSTRACT_OPERATOR_REGION_HASH;
    let right_is_abstract_operator_query = right.source_hash == ABSTRACT_OPERATOR_REGION_HASH
        || right.target_hash == ABSTRACT_OPERATOR_REGION_HASH;
    if left_is_abstract_operator_query || right_is_abstract_operator_query {
        return true;
    }

    left.source_hash == right.source_hash && left.target_hash == right.target_hash
}

fn can_add_reduction(
    query: Option<&TransitionCondition>,
    selected: &[usize],
    condition: &TransitionCondition,
    relevant: &[&ResidualReduction],
) -> bool {
    if let Some(query) = query
        && !compatible_identities(query, condition)
    {
        return false;
    }
    for &index in selected {
        if same_abstract_operator_reduction_identity(&relevant[index].condition, condition) {
            return false;
        }
        if !compatible_identities(&relevant[index].condition, condition) {
            return false;
        }
    }
    state_regions_have_common_intersection(
        query.map(|condition| condition.region.source.as_ref()),
        selected
            .iter()
            .map(|&index| relevant[index].condition.region.source.as_ref()),
        &condition.region.source,
    ) && state_regions_have_common_intersection(
        query.map(|condition| condition.region.target.as_ref()),
        selected
            .iter()
            .map(|&index| relevant[index].condition.region.target.as_ref()),
        &condition.region.target,
    )
}

fn same_abstract_operator_reduction_identity(
    left: &TransitionCondition,
    right: &TransitionCondition,
) -> bool {
    left.abstraction_id == right.abstraction_id
        && left.abstract_op_id == right.abstract_op_id
        && left.source_hash == ABSTRACT_OPERATOR_REGION_HASH
        && left.target_hash == ABSTRACT_OPERATOR_REGION_HASH
        && right.source_hash == ABSTRACT_OPERATOR_REGION_HASH
        && right.target_hash == ABSTRACT_OPERATOR_REGION_HASH
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

fn lmcut_residual_region_is_compact(region: &StateRegion, max_guard_conditions: usize) -> bool {
    let prop_guards = region
        .propositions
        .iter()
        .filter(|values| values.len() == 1)
        .count();
    let numeric_guards = region
        .numeric
        .iter()
        .map(|interval| {
            usize::from(interval.lower.is_finite()) + usize::from(interval.upper.is_finite())
        })
        .sum::<usize>();
    let guards = prop_guards + numeric_guards;
    guards > 0 && guards <= max_guard_conditions
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
    fn full_cost_footprints_use_overlap_cover_without_geometric_overlay() {
        let region = |lower, upper| StateRegion {
            propositions: Vec::new().into(),
            numeric: vec![Interval::closed(lower, upper)].into(),
        };
        let footprint = |lower, upper| AbstractOperatorFootprint {
            labels: vec![ConcreteOperatorFootprint {
                concrete_op_id: 0,
                source_region: Arc::new(region(lower, upper)),
            }],
        };
        let footprints = vec![footprint(0.0, 2.0), footprint(1.0, 3.0)];
        let mut residuals = TransitionResidualCosts::from_operator_costs(&[1.0]);

        residuals
            .reduce_by_abstract_operator_footprints(
                0,
                &footprints,
                &AbstractOperatorCostFunction {
                    operator_costs: vec![1.0, 1.0],
                },
            )
            .unwrap();

        let residual = &residuals.operator_residuals[0];
        assert_eq!(residual.full_regional_usage.cells.len(), 2);
        assert!(residual.regional_usage.cells.is_empty());
        let overlapping = ConcreteOperatorFootprint {
            concrete_op_id: 0,
            source_region: Arc::new(region(1.5, 1.5)),
        };
        let disjoint = ConcreteOperatorFootprint {
            concrete_op_id: 0,
            source_region: Arc::new(region(4.0, 5.0)),
        };
        assert_eq!(
            residuals.cost_for_operator_footprint(1, 0, &overlapping),
            0.0
        );
        assert_eq!(residuals.cost_for_operator_footprint(1, 0, &disjoint), 1.0);
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
    fn explicit_regional_cost_partitioning_uses_footprints() {
        let system = two_state_transition_system();
        let footprints = vec![AbstractOperatorFootprint {
            labels: vec![ConcreteOperatorFootprint {
                concrete_op_id: 0,
                source_region: state_region(0).into(),
            }],
        }];
        let residual = TransitionResidualCosts::from_operator_costs(&[5.0]);
        let (distances, saturated) = build_explicit_regional_cost_partitioning_table(
            &system,
            &footprints,
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

    fn region(source: usize, target: usize) -> TransitionRegion {
        TransitionRegion {
            source: state_region(source).into(),
            target: state_region(target).into(),
        }
    }

    /// A transition of concrete operator 0, which is the only operator these
    /// residual-cost tests give a base cost to.
    fn transition(
        abstraction_id: usize,
        source_hash: usize,
        abstract_op_id: usize,
        target_hash: usize,
    ) -> OperatorTransition {
        OperatorTransition {
            concrete_op_id: 0,
            abstraction_id,
            source_hash,
            abstract_op_id,
            target_hash,
        }
    }

    fn numeric_state_region(lower: f64, upper: f64) -> StateRegion {
        StateRegion {
            propositions: vec![vec![0]].into(),
            numeric: vec![Interval::closed(lower, upper)].into(),
        }
    }

    fn numeric_region(source_lower: f64, source_upper: f64) -> TransitionRegion {
        TransitionRegion {
            source: numeric_state_region(source_lower, source_upper).into(),
            target: numeric_state_region(source_lower, source_upper).into(),
        }
    }

    fn concrete_footprint(lower: f64, upper: f64) -> ConcreteOperatorFootprint {
        concrete_footprint_for_op(0, lower, upper)
    }

    fn concrete_footprint_for_op(
        concrete_op_id: usize,
        lower: f64,
        upper: f64,
    ) -> ConcreteOperatorFootprint {
        ConcreteOperatorFootprint {
            concrete_op_id,
            source_region: numeric_state_region(lower, upper).into(),
        }
    }

    fn concrete_footprint_2d(
        concrete_op_id: usize,
        first: Interval,
        second: Interval,
    ) -> ConcreteOperatorFootprint {
        ConcreteOperatorFootprint {
            concrete_op_id,
            source_region: StateRegion {
                propositions: vec![vec![0]].into(),
                numeric: vec![first, second].into(),
            }
            .into(),
        }
    }

    fn footprint(lower: f64, upper: f64) -> AbstractOperatorFootprint {
        AbstractOperatorFootprint {
            labels: vec![concrete_footprint(lower, upper)],
        }
    }

    #[test]
    fn exact_transition_reduction_does_not_reduce_other_transitions() {
        let reduced_region = region(0, 1);
        let mut residuals = TransitionResidualCosts::from_operator_costs(&[5.0]);
        let transition_system = AbstractTransitionSystem {
            transitions: vec![AbstractTransition {
                transition_id: 0,
                abstract_op_id: 7,
                concrete_op_ids: vec![0],
                source_hash: 3,
                target_hash: 4,
            }],
            duplicate_transition_attempts: 0,
            backward: vec![vec![], vec![], vec![], vec![], vec![0]],
            forward: vec![vec![], vec![], vec![], vec![0], vec![]],
            goal_facts: vec![],
            goal_state_hashes: vec![],
            initial_state_hash: 0,
            hash_multipliers: vec![],
            numeric_domain_sizes: vec![],
            state_regions: vec![
                state_region(9).into(),
                state_region(9).into(),
                state_region(9).into(),
                reduced_region.source.clone(),
                reduced_region.target.clone(),
            ],
        };
        let tcf = AbstractTransitionCostFunction {
            transition_costs: vec![2.0],
        };

        residuals
            .reduce_by_tcf(0, &transition_system, &tcf)
            .unwrap();

        assert_eq!(
            residuals.reduction_cost_for_transition(
                transition(0, 3, 7, 4),
                &reduced_region.source,
                &reduced_region.target
            ),
            3.0
        );
        let other_target = state_region(2);
        assert_eq!(
            residuals.reduction_cost_for_transition(
                transition(0, 3, 7, 5),
                &reduced_region.source,
                &other_target
            ),
            5.0
        );
        let overlapping = region(0, 1);
        assert_eq!(
            residuals.reduction_cost_for_transition(
                transition(1, 3, 7, 4),
                &overlapping.source,
                &overlapping.target
            ),
            3.0
        );
        let disjoint = region(1, 0);
        assert_eq!(
            residuals.reduction_cost_for_transition(
                transition(1, 3, 7, 4),
                &disjoint.source,
                &disjoint.target
            ),
            5.0
        );
    }

    #[test]
    fn repeated_exact_transition_reduction_clamps_tiny_negative_to_zero() {
        let reduced_region = region(0, 1);
        let mut residuals = TransitionResidualCosts::from_operator_costs(&[1.0]);
        let transition_system = AbstractTransitionSystem {
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
            goal_state_hashes: vec![],
            initial_state_hash: 0,
            hash_multipliers: vec![],
            numeric_domain_sizes: vec![],
            state_regions: vec![reduced_region.source.clone(), reduced_region.target.clone()],
        };
        residuals
            .reduce_by_tcf(
                0,
                &transition_system,
                &AbstractTransitionCostFunction {
                    transition_costs: vec![0.4],
                },
            )
            .unwrap();
        residuals
            .reduce_by_tcf(
                0,
                &transition_system,
                &AbstractTransitionCostFunction {
                    transition_costs: vec![0.6000000001],
                },
            )
            .unwrap();

        assert_eq!(
            residuals.reduction_cost_for_transition(
                transition(0, 0, 0, 1),
                &reduced_region.source,
                &reduced_region.target
            ),
            0.0
        );
    }

    #[test]
    fn foreign_abstraction_uses_region_overlap() {
        let reduced_region = region(0, 1);
        let mut residuals = TransitionResidualCosts::from_operator_costs(&[5.0]);
        let transition_system = AbstractTransitionSystem {
            transitions: vec![AbstractTransition {
                transition_id: 0,
                abstract_op_id: 7,
                concrete_op_ids: vec![0],
                source_hash: 3,
                target_hash: 4,
            }],
            duplicate_transition_attempts: 0,
            backward: vec![vec![], vec![], vec![], vec![], vec![0]],
            forward: vec![vec![], vec![], vec![], vec![0], vec![]],
            goal_facts: vec![],
            goal_state_hashes: vec![],
            initial_state_hash: 0,
            hash_multipliers: vec![],
            numeric_domain_sizes: vec![],
            state_regions: vec![
                state_region(9).into(),
                state_region(9).into(),
                state_region(9).into(),
                reduced_region.source.clone(),
                reduced_region.target.clone(),
            ],
        };
        residuals
            .reduce_by_tcf(
                0,
                &transition_system,
                &AbstractTransitionCostFunction {
                    transition_costs: vec![2.0],
                },
            )
            .unwrap();

        let disjoint = region(1, 0);
        assert_eq!(
            residuals.reduction_cost_for_transition(
                transition(0, 9, 7, 4),
                &disjoint.source,
                &disjoint.target
            ),
            5.0
        );
        assert_eq!(
            residuals.reduction_cost_for_transition(
                transition(1, 9, 7, 4),
                &disjoint.source,
                &disjoint.target
            ),
            5.0
        );
        let overlapping = region(0, 1);
        assert_eq!(
            residuals.reduction_cost_for_transition(
                transition(1, 9, 7, 4),
                &overlapping.source,
                &overlapping.target
            ),
            3.0
        );
    }

    #[test]
    fn same_abstraction_reductions_need_same_transition_identity() {
        let first_region = region(0, 1);
        let second_region = region(0, 1);
        let mut residuals = TransitionResidualCosts::from_operator_costs(&[10.0]);
        let transition_system = AbstractTransitionSystem {
            transitions: vec![
                AbstractTransition {
                    transition_id: 0,
                    abstract_op_id: 0,
                    concrete_op_ids: vec![0],
                    source_hash: 0,
                    target_hash: 1,
                },
                AbstractTransition {
                    transition_id: 1,
                    abstract_op_id: 1,
                    concrete_op_ids: vec![0],
                    source_hash: 2,
                    target_hash: 3,
                },
            ],
            duplicate_transition_attempts: 0,
            backward: vec![vec![], vec![0], vec![], vec![1]],
            forward: vec![vec![0], vec![], vec![1], vec![]],
            goal_facts: vec![],
            goal_state_hashes: vec![],
            initial_state_hash: 0,
            hash_multipliers: vec![],
            numeric_domain_sizes: vec![],
            state_regions: vec![
                first_region.source.clone(),
                first_region.target.clone(),
                second_region.source.clone(),
                second_region.target.clone(),
            ],
        };
        residuals
            .reduce_by_tcf(
                0,
                &transition_system,
                &AbstractTransitionCostFunction {
                    transition_costs: vec![3.0, 4.0],
                },
            )
            .unwrap();

        let overlapping = region(0, 1);
        assert_eq!(
            residuals.reduction_cost_for_transition(
                transition(1, 99, 99, 100),
                &overlapping.source,
                &overlapping.target
            ),
            6.0
        );
        assert_eq!(residuals.operator_costs_for_label_cp(), vec![6.0]);
    }

    #[test]
    fn disjoint_transition_reductions_use_max_overlap_not_sum() {
        let first_region = numeric_region(0.0, 4.0);
        let second_region = numeric_region(6.0, 10.0);
        let query = numeric_region(0.0, 10.0);
        let mut residuals = TransitionResidualCosts::from_operator_costs(&[10.0]);
        let transition_system = AbstractTransitionSystem {
            transitions: vec![
                AbstractTransition {
                    transition_id: 0,
                    abstract_op_id: 0,
                    concrete_op_ids: vec![0],
                    source_hash: 0,
                    target_hash: 1,
                },
                AbstractTransition {
                    transition_id: 1,
                    abstract_op_id: 1,
                    concrete_op_ids: vec![0],
                    source_hash: 2,
                    target_hash: 3,
                },
            ],
            duplicate_transition_attempts: 0,
            backward: vec![vec![], vec![0], vec![], vec![1]],
            forward: vec![vec![0], vec![], vec![1], vec![]],
            goal_facts: vec![],
            goal_state_hashes: vec![],
            initial_state_hash: 0,
            hash_multipliers: vec![],
            numeric_domain_sizes: vec![],
            state_regions: vec![
                first_region.source.clone(),
                first_region.target.clone(),
                second_region.source.clone(),
                second_region.target.clone(),
            ],
        };
        residuals
            .reduce_by_tcf(
                0,
                &transition_system,
                &AbstractTransitionCostFunction {
                    transition_costs: vec![3.0, 4.0],
                },
            )
            .unwrap();

        assert_eq!(
            residuals.reduction_cost_for_transition(
                transition(1, 99, 99, 100),
                &query.source,
                &query.target
            ),
            6.0
        );
        assert_eq!(residuals.operator_costs_for_label_cp(), vec![6.0]);
    }

    #[test]
    fn footprint_reductions_apply_to_same_concrete_operator_only() {
        let mut residuals = TransitionResidualCosts::from_operator_costs(&[10.0, 10.0]);
        let reduced = footprint(3.0, 7.0);
        residuals
            .reduce_by_abstract_operator_footprints(
                0,
                std::slice::from_ref(&reduced),
                &AbstractOperatorCostFunction {
                    operator_costs: vec![3.0],
                },
            )
            .unwrap();

        let query = concrete_footprint(5.0, 8.0);
        assert_eq!(residuals.cost_for_operator_footprint(1, 0, &query), 7.0);
        let other_op_query = concrete_footprint_for_op(1, 5.0, 8.0);
        assert_eq!(
            residuals.cost_for_operator_footprint(1, 0, &other_op_query),
            10.0
        );
    }

    #[test]
    fn footprint_reduction_allows_full_cost() {
        let mut residuals = TransitionResidualCosts::from_operator_costs(&[1.0]);
        let reduced = footprint(3.0, 7.0);
        residuals
            .reduce_by_abstract_operator_footprints(
                0,
                std::slice::from_ref(&reduced),
                &AbstractOperatorCostFunction {
                    operator_costs: vec![1.0],
                },
            )
            .unwrap();

        assert_eq!(
            residuals.cost_for_operator_footprint(1, 0, &reduced.labels[0]),
            0.0
        );
    }

    #[test]
    fn same_abstract_operator_alternative_footprints_do_not_stack() {
        let mut residuals = TransitionResidualCosts::from_operator_costs(&[1.0]);
        let reduced = AbstractOperatorFootprint {
            labels: vec![concrete_footprint(0.0, 10.0), concrete_footprint(5.0, 15.0)],
        };
        residuals
            .reduce_by_abstract_operator_footprints(
                0,
                &[reduced],
                &AbstractOperatorCostFunction {
                    operator_costs: vec![0.4],
                },
            )
            .unwrap();

        assert_eq!(
            residuals.cost_for_operator_footprint(1, 0, &concrete_footprint(7.0, 8.0)),
            0.6
        );
    }

    #[test]
    fn disjoint_footprint_sources_do_not_reduce_residual_cost() {
        let mut residuals = TransitionResidualCosts::from_operator_costs(&[10.0]);
        residuals
            .reduce_by_abstract_operator_footprints(
                0,
                &[footprint(0.0, 2.0)],
                &AbstractOperatorCostFunction {
                    operator_costs: vec![4.0],
                },
            )
            .unwrap();

        assert_eq!(
            residuals.cost_for_operator_footprint(1, 0, &concrete_footprint(3.0, 5.0)),
            10.0
        );
    }

    #[test]
    fn target_hull_overlap_is_ignored_for_abstract_operator_footprints() {
        let mut residuals = TransitionResidualCosts::from_operator_costs(&[10.0]);
        residuals
            .reduce_by_abstract_operator_footprints(
                0,
                &[footprint(1.0, 10.0)],
                &AbstractOperatorCostFunction {
                    operator_costs: vec![4.0],
                },
            )
            .unwrap();

        assert_eq!(
            residuals.cost_for_operator_footprint(1, 0, &concrete_footprint(10.5, 11.0)),
            10.0
        );
    }

    #[test]
    fn overlapping_footprint_sources_reduce_residual_cost() {
        let mut residuals = TransitionResidualCosts::from_operator_costs(&[10.0]);
        residuals
            .reduce_by_abstract_operator_footprints(
                0,
                &[footprint(0.0, 5.0), footprint(4.0, 10.0)],
                &AbstractOperatorCostFunction {
                    operator_costs: vec![3.0, 4.0],
                },
            )
            .unwrap();

        assert_eq!(
            residuals.cost_for_operator_footprint(1, 0, &concrete_footprint(4.5, 4.75)),
            6.0
        );
    }

    #[test]
    fn label_cp_steals_shared_operator_cost() {
        let mut residuals = TransitionResidualCosts::from_operator_costs(&[1.0]);
        residuals
            .reduce_by_abstract_operator_footprints(
                0,
                &[footprint(0.0, 5.0)],
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
            .reduce_by_abstract_operator_footprints(
                0,
                &[footprint(0.0, 5.0)],
                &AbstractOperatorCostFunction {
                    operator_costs: vec![1.0],
                },
            )
            .unwrap();

        // The complementary abstraction starts after the first one's active
        // source region, so region CP preserves the unit residual there.
        let complementary = concrete_footprint(5.0 + 1e-6, 10.0);
        let region_residual = residuals.cost_for_operator_footprint(1, 0, &complementary);
        assert_eq!(region_residual, 1.0);
        assert!(region_residual > residuals.operator_costs_for_label_cp()[0]);
        assert!(region_residual <= 11.0);
    }

    #[test]
    fn region_cp_overlapping_nested_targets_order_insensitive() {
        fn move_footprints(start: usize, end: usize) -> Vec<AbstractOperatorFootprint> {
            (start..end)
                .map(|i| AbstractOperatorFootprint {
                    labels: vec![ConcreteOperatorFootprint {
                        concrete_op_id: 0,
                        source_region: StateRegion {
                            propositions: vec![vec![0]].into(),
                            numeric: vec![Interval::new(i as f64, (i + 1) as f64, false, true)]
                                .into(),
                        }
                        .into(),
                    }],
                })
                .collect()
        }

        fn save_footprint(save_op_id: usize) -> AbstractOperatorFootprint {
            AbstractOperatorFootprint {
                labels: vec![concrete_footprint_for_op(save_op_id, 0.0, 15.0)],
            }
        }

        fn contribution(
            residuals: &TransitionResidualCosts,
            abstraction_id: usize,
            footprints: &[AbstractOperatorFootprint],
        ) -> f64 {
            footprints
                .iter()
                .enumerate()
                .map(|(abstract_op_id, footprint)| {
                    footprint
                        .labels
                        .iter()
                        .map(|label| {
                            residuals.cost_for_operator_footprint(
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
            footprints: &[AbstractOperatorFootprint],
        ) {
            residuals
                .reduce_by_abstract_operator_footprints(
                    abstraction_id,
                    footprints,
                    &AbstractOperatorCostFunction {
                        operator_costs: vec![1.0; footprints.len()],
                    },
                )
                .unwrap();
        }

        let mut alpha10 = move_footprints(0, 10);
        alpha10.push(save_footprint(1));
        let mut alpha15 = move_footprints(0, 15);
        alpha15.push(save_footprint(2));

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
        let x_abstraction = AbstractOperatorFootprint {
            labels: vec![concrete_footprint_2d(
                0,
                Interval::closed(0.0, 1.0),
                Interval::unbounded(),
            )],
        };
        residuals
            .reduce_by_abstract_operator_footprints(
                0,
                &[x_abstraction],
                &AbstractOperatorCostFunction {
                    operator_costs: vec![1.0],
                },
            )
            .unwrap();

        let y_abstraction =
            concrete_footprint_2d(0, Interval::unbounded(), Interval::closed(0.0, 1.0));
        assert_eq!(
            residuals.cost_for_operator_footprint(1, 0, &y_abstraction),
            0.0
        );
    }

    #[test]
    fn infinite_tail_reduction_preserves_disjoint_tail_cost() {
        let mut residuals = TransitionResidualCosts::from_operator_costs(&[10.0]);
        let tail = footprint(f64::NEG_INFINITY, 0.0);
        residuals
            .reduce_by_abstract_operator_footprints(
                0,
                &[tail],
                &AbstractOperatorCostFunction {
                    operator_costs: vec![4.0],
                },
            )
            .unwrap();

        assert_eq!(
            residuals.cost_for_operator_footprint(1, 0, &concrete_footprint(1.0, f64::INFINITY),),
            10.0
        );
        assert_eq!(
            residuals.cost_for_operator_footprint(
                1,
                0,
                &concrete_footprint(f64::NEG_INFINITY, f64::INFINITY),
            ),
            6.0
        );
    }

    #[test]
    fn open_infinite_tail_does_not_consume_boundary() {
        let mut residuals = TransitionResidualCosts::from_operator_costs(&[1.0]);
        let open_tail = AbstractOperatorFootprint {
            labels: vec![ConcreteOperatorFootprint {
                concrete_op_id: 0,
                source_region: StateRegion {
                    propositions: vec![vec![0]].into(),
                    numeric: vec![Interval::new(f64::NEG_INFINITY, 0.0, false, false)].into(),
                }
                .into(),
            }],
        };
        residuals
            .reduce_by_abstract_operator_footprints(
                0,
                &[open_tail],
                &AbstractOperatorCostFunction {
                    operator_costs: vec![1.0],
                },
            )
            .unwrap();

        assert_eq!(
            residuals.cost_for_operator_footprint(1, 0, &concrete_footprint(0.0, 0.0)),
            1.0
        );
    }

    #[test]
    fn multidimensional_disjoint_regions_preserve_full_cost() {
        let mut residuals = TransitionResidualCosts::from_operator_costs(&[10.0]);
        let lower_y = AbstractOperatorFootprint {
            labels: vec![concrete_footprint_2d(
                0,
                Interval::closed(0.0, 10.0),
                Interval::new(f64::NEG_INFINITY, 0.0, false, true),
            )],
        };
        residuals
            .reduce_by_abstract_operator_footprints(
                0,
                &[lower_y],
                &AbstractOperatorCostFunction {
                    operator_costs: vec![4.0],
                },
            )
            .unwrap();

        let upper_y = concrete_footprint_2d(
            0,
            Interval::closed(0.0, 10.0),
            Interval::new(0.0, f64::INFINITY, false, false),
        );
        assert_eq!(residuals.cost_for_operator_footprint(1, 0, &upper_y), 10.0);
    }

    #[test]
    fn perpendicular_tail_allocations_preserve_untouched_corner() {
        let mut residuals = TransitionResidualCosts::from_operator_costs(&[10.0]);
        let left = AbstractOperatorFootprint {
            labels: vec![concrete_footprint_2d(
                0,
                Interval::new(f64::NEG_INFINITY, 0.0, false, true),
                Interval::unbounded(),
            )],
        };
        let lower = AbstractOperatorFootprint {
            labels: vec![concrete_footprint_2d(
                0,
                Interval::unbounded(),
                Interval::new(f64::NEG_INFINITY, 0.0, false, true),
            )],
        };
        residuals
            .reduce_by_abstract_operator_footprints(
                0,
                &[left],
                &AbstractOperatorCostFunction {
                    operator_costs: vec![4.0],
                },
            )
            .unwrap();
        residuals
            .reduce_by_abstract_operator_footprints(
                1,
                &[lower],
                &AbstractOperatorCostFunction {
                    operator_costs: vec![3.0],
                },
            )
            .unwrap();

        let upper_right = concrete_footprint_2d(
            0,
            Interval::new(0.0, f64::INFINITY, false, false),
            Interval::new(0.0, f64::INFINITY, false, false),
        );
        let lower_left = concrete_footprint_2d(
            0,
            Interval::new(f64::NEG_INFINITY, 0.0, false, true),
            Interval::new(f64::NEG_INFINITY, 0.0, false, true),
        );
        assert_eq!(
            residuals.cost_for_operator_footprint(2, 0, &upper_right),
            10.0
        );
        assert_eq!(
            residuals.cost_for_operator_footprint(2, 0, &lower_left),
            3.0
        );
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
