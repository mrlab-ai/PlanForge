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
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use ordered_float::NotNan;
use planforge_sas::numeric_task::ExplicitFact;
use planforge_sas::utils::float_tolerance;

use crate::evaluation::domain_abstractions::comparison_expression::Interval;

const EPSILON: f64 = 1e-9;
const ABSTRACT_OPERATOR_REGION_HASH: usize = usize::MAX;
const MAX_ABSTRACT_OPERATOR_REDUCTION_PIECES: usize = 4096;
const MAX_TOTAL_ABSTRACT_OPERATOR_REDUCTION_PIECES: usize = 50_000;

#[derive(Debug, Clone, PartialEq)]
pub struct AbstractTransition {
    pub transition_id: usize,
    pub abstract_op_id: usize,
    pub concrete_op_ids: Vec<usize>,
    pub source_hash: usize,
    pub target_hash: usize,
}

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
pub struct AbstractOperatorFootprint {
    pub labels: Vec<ConcreteOperatorFootprint>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConcreteOperatorFootprint {
    pub concrete_op_id: usize,
    pub source_region: Arc<StateRegion>,
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

#[derive(Debug, Clone, PartialEq)]
pub struct AbstractTransitionSystem {
    pub transitions: Vec<AbstractTransition>,
    pub duplicate_transition_attempts: usize,
    pub backward: Vec<Vec<usize>>,
    pub forward: Vec<Vec<usize>>,
    pub goal_facts: Vec<ExplicitFact>,
    pub goal_state_hashes: Vec<usize>,
    pub initial_state_hash: usize,
    pub hash_multipliers: Vec<usize>,
    pub numeric_domain_sizes: Vec<usize>,
    pub state_regions: Vec<Arc<StateRegion>>,
}

impl AbstractTransitionSystem {
    pub fn transition_region(&self, transition: &AbstractTransition) -> Result<TransitionRegion> {
        let source = self
            .state_regions
            .get(transition.source_hash)
            .with_context(|| {
                format!(
                    "missing source state region {} for transition {}",
                    transition.source_hash, transition.transition_id
                )
            })?
            .clone();
        let target = self
            .state_regions
            .get(transition.target_hash)
            .with_context(|| {
                format!(
                    "missing target state region {} for transition {}",
                    transition.target_hash, transition.transition_id
                )
            })?
            .clone();
        Ok(TransitionRegion { source, target })
    }

    pub fn abstract_operator_regions(&self) -> Vec<Option<TransitionRegion>> {
        assert!(
            !self.state_regions.is_empty(),
            "abstract transition system has no materialized state regions"
        );
        let num_abstract_ops = self
            .transitions
            .iter()
            .map(|transition| transition.abstract_op_id)
            .max()
            .map_or(0, |max_id| max_id + 1);
        let mut regions: Vec<Option<TransitionRegion>> = vec![None; num_abstract_ops];
        for transition in &self.transitions {
            let source = self.state_regions[transition.source_hash].clone();
            let target = self.state_regions[transition.target_hash].clone();
            let transition_region = TransitionRegion { source, target };
            match &mut regions[transition.abstract_op_id] {
                Some(region) => merge_transition_region(region, &transition_region),
                None => regions[transition.abstract_op_id] = Some(transition_region),
            }
        }
        regions
    }

    pub fn abstract_operator_region_covers(&self) -> Vec<Vec<TransitionRegion>> {
        assert!(
            !self.state_regions.is_empty(),
            "abstract transition system has no materialized state regions"
        );
        let num_abstract_ops = self
            .transitions
            .iter()
            .map(|transition| transition.abstract_op_id)
            .max()
            .map_or(0, |max_id| max_id + 1);
        let mut covers = vec![Vec::new(); num_abstract_ops];
        let mut seen = vec![std::collections::HashSet::new(); num_abstract_ops];
        for transition in &self.transitions {
            let region = TransitionRegion {
                source: self.state_regions[transition.source_hash].clone(),
                target: self.state_regions[transition.target_hash].clone(),
            };
            let key = transition_region_key(&region);
            if seen[transition.abstract_op_id].insert(key) {
                covers[transition.abstract_op_id].push(region);
            }
        }
        for (abstract_op_id, cover) in covers.iter_mut().enumerate() {
            if cover.len() > MAX_ABSTRACT_OPERATOR_REDUCTION_PIECES {
                let mut hull = cover[0].clone();
                for region in cover.iter().skip(1) {
                    merge_transition_region(&mut hull, region);
                }
                tracing::debug!(
                    "abstract operator {abstract_op_id} reduction cover exceeded {} pieces; using hull fallback",
                    MAX_ABSTRACT_OPERATOR_REDUCTION_PIECES
                );
                cover.clear();
                cover.push(hull);
            }
        }
        covers
    }

    pub fn concrete_operator_ids_by_abstract_operator(&self) -> Vec<Vec<usize>> {
        let num_abstract_ops = self
            .transitions
            .iter()
            .map(|transition| transition.abstract_op_id)
            .max()
            .map_or(0, |max_id| max_id + 1);
        let mut concrete_op_ids = vec![Vec::new(); num_abstract_ops];
        for transition in &self.transitions {
            concrete_op_ids[transition.abstract_op_id]
                .extend(transition.concrete_op_ids.iter().copied());
        }
        for ids in &mut concrete_op_ids {
            ids.sort_unstable();
            ids.dedup();
        }
        concrete_op_ids
    }

    fn transition_counts_by_abstract_operator(&self, num_abstract_ops: usize) -> Vec<usize> {
        let mut counts = vec![0usize; num_abstract_ops];
        for transition in &self.transitions {
            if let Some(count) = counts.get_mut(transition.abstract_op_id) {
                *count = count.saturating_add(1);
            }
        }
        counts
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AbstractTransitionCostFunction {
    pub transition_costs: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AbstractOperatorCostFunction {
    pub operator_costs: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct RegionalCostAllocation {
    entries: Vec<RegionalCostAllocationEntry>,
}

#[derive(Debug, Clone)]
pub struct RegionalCostAllocationEntry {
    pub footprint: ConcreteOperatorFootprint,
    pub amount: f64,
}

impl RegionalCostAllocation {
    pub fn new(entries: Vec<RegionalCostAllocationEntry>) -> Self {
        Self { entries }
    }

    pub fn entries(&self) -> &[RegionalCostAllocationEntry] {
        &self.entries
    }
}

pub fn build_explicit_label_cost_partitioning_table(
    transition_system: &AbstractTransitionSystem,
    operator_costs: &[f64],
    cap_state_id: Option<usize>,
    deadline: Option<Instant>,
) -> Result<(Vec<f64>, Vec<f64>)> {
    ensure_scp_table_deadline(deadline)?;
    let transition_costs = transition_system
        .transitions
        .iter()
        .map(|transition| {
            ensure!(
                !transition.concrete_op_ids.is_empty(),
                "abstract transition {} has no concrete operator labels",
                transition.transition_id
            );
            transition
                .concrete_op_ids
                .iter()
                .map(|&operator_id| {
                    operator_costs.get(operator_id).copied().with_context(|| {
                        format!("missing residual cost for concrete operator {operator_id}")
                    })
                })
                .collect::<Result<Vec<_>>>()
                .map(|costs| costs.into_iter().fold(f64::INFINITY, f64::min))
        })
        .collect::<Result<Vec<_>>>()?;
    let distances = build_explicit_goal_distances(transition_system, &transition_costs, deadline)?;
    let saturation_table = capped_saturation_table(&distances, cap_state_id)?;
    let saturated = saturated_label_costs(
        transition_system,
        &transition_costs,
        operator_costs.len(),
        &saturation_table,
    )?;

    if cap_state_id.is_none() {
        return Ok((distances, saturated));
    }
    let saturated_transition_costs = transition_system
        .transitions
        .iter()
        .map(|transition| {
            transition
                .concrete_op_ids
                .iter()
                .map(|&operator_id| saturated[operator_id])
                .fold(f64::INFINITY, f64::min)
        })
        .collect::<Vec<_>>();
    let global_distances =
        build_explicit_goal_distances(transition_system, &saturated_transition_costs, deadline)?;
    Ok((global_distances, saturated))
}

#[allow(clippy::too_many_arguments)]
pub fn build_explicit_regional_cost_partitioning_table(
    transition_system: &AbstractTransitionSystem,
    footprints: &[AbstractOperatorFootprint],
    residual_costs: &TransitionResidualCosts,
    abstraction_id: usize,
    cap_state_id: Option<usize>,
    deadline: Option<Instant>,
) -> Result<(Vec<f64>, AbstractOperatorCostFunction)> {
    let operator_costs = abstract_operator_costs_from_footprints(
        footprints.len(),
        footprints,
        residual_costs,
        abstraction_id,
        deadline,
    )?;
    let transition_costs =
        transition_costs_from_abstract_operator_costs(transition_system, &operator_costs)?;
    let distances = build_explicit_goal_distances(transition_system, &transition_costs, deadline)?;
    let saturation_table = capped_saturation_table(&distances, cap_state_id)?;
    let saturated =
        saturated_abstract_operator_costs(transition_system, &operator_costs, &saturation_table)?;

    if cap_state_id.is_none() {
        return Ok((
            distances,
            AbstractOperatorCostFunction {
                operator_costs: saturated,
            },
        ));
    }
    let saturated_transition_costs =
        transition_costs_from_abstract_operator_costs(transition_system, &saturated)?;
    let global_distances =
        build_explicit_goal_distances(transition_system, &saturated_transition_costs, deadline)?;
    Ok((
        global_distances,
        AbstractOperatorCostFunction {
            operator_costs: saturated,
        },
    ))
}

fn ensure_scp_table_deadline(deadline: Option<Instant>) -> Result<()> {
    crate::resource_limits::ensure_before_deadline(deadline, "SCP table construction")
}

fn build_explicit_goal_distances(
    transition_system: &AbstractTransitionSystem,
    transition_costs: &[f64],
    deadline: Option<Instant>,
) -> Result<Vec<f64>> {
    ensure!(
        transition_system.transitions.len() == transition_costs.len(),
        "transition system/cost vector size mismatch: {} vs {}",
        transition_system.transitions.len(),
        transition_costs.len()
    );
    let num_states = transition_system.backward.len();
    let mut distances = vec![f64::INFINITY; num_states];
    let mut heap = BinaryHeap::new();
    for &goal_state_id in &transition_system.goal_state_hashes {
        ensure!(
            goal_state_id < num_states,
            "goal state id {goal_state_id} out of bounds for {num_states} states"
        );
        distances[goal_state_id] = 0.0;
        heap.push((
            Reverse(NotNan::new(0.0).expect("zero is not NaN")),
            goal_state_id,
        ));
    }
    let mut expansions = 0usize;
    while let Some((Reverse(distance), target_id)) = heap.pop() {
        if expansions.is_multiple_of(1024) {
            ensure_scp_table_deadline(deadline)?;
        }
        expansions += 1;
        let distance = distance.into_inner();
        if distance > distances[target_id] + EPSILON {
            continue;
        }
        let predecessors = transition_system
            .backward
            .get(target_id)
            .with_context(|| format!("missing predecessor list for abstract state {target_id}"))?;
        for &transition_id in predecessors {
            let transition = transition_system
                .transitions
                .get(transition_id)
                .with_context(|| format!("missing abstract transition {transition_id}"))?;
            ensure!(
                transition.target_hash == target_id,
                "backward transition {transition_id} targets {}, expected {target_id}",
                transition.target_hash
            );
            let cost = transition_costs[transition_id];
            ensure!(
                cost.is_finite() && cost >= -EPSILON,
                "abstract transition {transition_id} has invalid cost {cost}"
            );
            let alternative = distance + cost.max(0.0);
            let source_distance = distances.get_mut(transition.source_hash).with_context(|| {
                format!(
                    "transition {transition_id} source {} out of bounds for {num_states} states",
                    transition.source_hash
                )
            })?;
            if alternative + EPSILON < *source_distance {
                *source_distance = alternative;
                heap.push((
                    Reverse(NotNan::new(alternative).context("abstract distance is NaN")?),
                    transition.source_hash,
                ));
            }
        }
    }
    Ok(distances)
}

fn capped_saturation_table(distances: &[f64], cap_state_id: Option<usize>) -> Result<Vec<f64>> {
    let Some(cap_state_id) = cap_state_id else {
        return Ok(distances.to_vec());
    };
    let h_cap = distances.get(cap_state_id).copied().with_context(|| {
        format!(
            "perimeter cap state {cap_state_id} out of bounds for {} states",
            distances.len()
        )
    })?;
    let mut capped = distances.to_vec();
    if h_cap.is_finite() {
        for value in &mut capped {
            if !value.is_finite() || *value > h_cap {
                *value = f64::NEG_INFINITY;
            }
        }
    }
    Ok(capped)
}

fn saturated_label_costs(
    transition_system: &AbstractTransitionSystem,
    transition_costs: &[f64],
    num_operators: usize,
    distances: &[f64],
) -> Result<Vec<f64>> {
    let mut saturated = vec![0.0_f64; num_operators];
    for transition in &transition_system.transitions {
        let source_h = distances[transition.source_hash];
        let target_h = distances[transition.target_hash];
        if !source_h.is_finite() || !target_h.is_finite() {
            continue;
        }
        let needed = (source_h - target_h).max(0.0);
        ensure!(
            needed <= transition_costs[transition.transition_id] + 1e-7,
            "saturated transition cost {needed} exceeds residual transition cost {}",
            transition_costs[transition.transition_id]
        );
        for &operator_id in &transition.concrete_op_ids {
            let slot = saturated.get_mut(operator_id).with_context(|| {
                format!("transition references missing concrete operator {operator_id}")
            })?;
            *slot = slot.max(needed);
        }
    }
    Ok(saturated)
}

fn transition_costs_from_abstract_operator_costs(
    transition_system: &AbstractTransitionSystem,
    operator_costs: &[f64],
) -> Result<Vec<f64>> {
    transition_system
        .transitions
        .iter()
        .map(|transition| {
            operator_costs
                .get(transition.abstract_op_id)
                .copied()
                .with_context(|| {
                    format!(
                        "transition {} references missing abstract operator {}",
                        transition.transition_id, transition.abstract_op_id
                    )
                })
        })
        .collect()
}

fn saturated_abstract_operator_costs(
    transition_system: &AbstractTransitionSystem,
    operator_costs: &[f64],
    distances: &[f64],
) -> Result<Vec<f64>> {
    let mut saturated = vec![0.0_f64; operator_costs.len()];
    for transition in &transition_system.transitions {
        let source_h = distances[transition.source_hash];
        let target_h = distances[transition.target_hash];
        if !source_h.is_finite() || !target_h.is_finite() {
            continue;
        }
        let needed = (source_h - target_h).max(0.0);
        let operator_cost = *operator_costs
            .get(transition.abstract_op_id)
            .with_context(|| {
                format!(
                    "transition {} references missing abstract operator {}",
                    transition.transition_id, transition.abstract_op_id
                )
            })?;
        ensure!(
            needed <= operator_cost + 1e-7,
            "saturated abstract-operator cost {needed} exceeds residual cost {operator_cost}"
        );
        saturated[transition.abstract_op_id] = saturated[transition.abstract_op_id].max(needed);
    }
    Ok(saturated)
}

#[allow(clippy::too_many_arguments)]
fn abstract_operator_costs_from_footprints(
    num_operators: usize,
    footprints: &[AbstractOperatorFootprint],
    residual_costs: &TransitionResidualCosts,
    abstraction_id: usize,
    deadline: Option<Instant>,
) -> Result<Vec<f64>> {
    ensure!(
        footprints.len() >= num_operators,
        "abstract-operator footprint/operator size mismatch: footprints={} operators={num_operators}",
        footprints.len()
    );
    let has_reductions = residual_costs.has_reductions();
    let mut operator_costs = vec![f64::INFINITY; num_operators];
    for abstract_op_id in 0..num_operators {
        if abstract_op_id.is_multiple_of(64) {
            ensure_scp_table_deadline(deadline)?;
        }
        let footprint = &footprints[abstract_op_id];
        ensure!(
            !footprint.labels.is_empty(),
            "abstract operator {abstract_op_id} has no concrete footprint labels"
        );
        operator_costs[abstract_op_id] = footprint
            .labels
            .iter()
            .map(|label| {
                let residual = if has_reductions {
                    residual_costs.cost_for_operator_footprint(
                        abstraction_id,
                        abstract_op_id,
                        label,
                    )
                } else {
                    residual_costs.base_cost(label.concrete_op_id)
                };
                residual.min(residual_costs.base_cost(label.concrete_op_id))
            })
            .fold(f64::INFINITY, f64::min);
        ensure!(
            operator_costs[abstract_op_id].is_finite(),
            "residual cost for abstract operator {abstract_op_id} is not finite"
        );
    }
    Ok(operator_costs)
}

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
        if amount + EPSILON >= base_cost {
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
        if amount <= EPSILON {
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
        if amount <= EPSILON {
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
        new_cells.retain(|cell| cell.amount > EPSILON);
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

fn state_region_is_nonempty(region: &StateRegion) -> bool {
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
        .map(|(left, right)| intersect_intervals(left, right))
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

fn intersect_intervals(left: Interval, right: Interval) -> Interval {
    let lower = left.lower.max(right.lower);
    let upper = left.upper.min(right.upper);
    Interval::new(
        lower,
        upper,
        if left.lower > right.lower {
            left.lower_closed
        } else if left.lower < right.lower {
            right.lower_closed
        } else {
            left.lower_closed && right.lower_closed
        },
        if left.upper < right.upper {
            left.upper_closed
        } else if left.upper > right.upper {
            right.upper_closed
        } else {
            left.upper_closed && right.upper_closed
        },
    )
}

/// Returns a disjoint cover of `region \\ removed`. `removed` must be a
/// nonempty subset of `region`.
fn subtract_state_region(region: &StateRegion, removed: &StateRegion) -> Vec<StateRegion> {
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

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct TransitionQueryKey {
    abstraction_id: usize,
    source_hash: usize,
    abstract_op_id: usize,
    target_hash: usize,
    region: Option<TransitionRegionKey>,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct TransitionRegionKey {
    source: StateRegionKey,
    target: StateRegionKey,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct StateRegionKey {
    propositions: Arc<[Vec<PropValueId>]>,
    numeric: Vec<IntervalKey>,
}

#[derive(Copy, Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct IntervalKey {
    lower_bits: u64,
    upper_bits: u64,
    lower_closed: bool,
    upper_closed: bool,
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
                let full_regional_reduction = (!residual.full_regional_usage.is_empty())
                    .then_some(residual.base_cost)
                    .unwrap_or(0.0);
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

    #[allow(clippy::too_many_arguments)]
    pub fn cost_for_transition(
        &self,
        concrete_op_id: usize,
        current_abstraction_id: usize,
        source_hash: usize,
        abstract_op_id: usize,
        target_hash: usize,
        source_region: &StateRegion,
        target_region: &StateRegion,
    ) -> f64 {
        self.cost_for_transition_with_region_key(
            concrete_op_id,
            current_abstraction_id,
            source_hash,
            abstract_op_id,
            target_hash,
            source_region,
            target_region,
            Some(TransitionRegionKey {
                source: state_region_key(source_region),
                target: state_region_key(target_region),
            }),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn cost_for_indexed_transition(
        &self,
        concrete_op_id: usize,
        current_abstraction_id: usize,
        source_hash: usize,
        abstract_op_id: usize,
        target_hash: usize,
        source_region: &StateRegion,
        target_region: &StateRegion,
    ) -> f64 {
        self.cost_for_transition_with_region_key(
            concrete_op_id,
            current_abstraction_id,
            source_hash,
            abstract_op_id,
            target_hash,
            source_region,
            target_region,
            None,
        )
    }

    pub fn cost_for_abstract_operator(
        &self,
        concrete_op_id: usize,
        current_abstraction_id: usize,
        abstract_op_id: usize,
        region: &TransitionRegion,
    ) -> f64 {
        self.cost_for_transition_with_region(
            concrete_op_id,
            current_abstraction_id,
            ABSTRACT_OPERATOR_REGION_HASH,
            abstract_op_id,
            ABSTRACT_OPERATOR_REGION_HASH,
            region.clone(),
            None,
        )
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
            footprint.concrete_op_id,
            current_abstraction_id,
            ABSTRACT_OPERATOR_REGION_HASH,
            abstract_op_id,
            ABSTRACT_OPERATOR_REGION_HASH,
            TransitionRegion {
                source: Arc::clone(&footprint.source_region),
                target: Arc::clone(&footprint.source_region),
            },
            None,
        );
        (legacy - regional).max(0.0)
    }

    #[allow(clippy::too_many_arguments)]
    fn cost_for_transition_with_region_key(
        &self,
        concrete_op_id: usize,
        current_abstraction_id: usize,
        source_hash: usize,
        abstract_op_id: usize,
        target_hash: usize,
        source_region: &StateRegion,
        target_region: &StateRegion,
        region_key: Option<TransitionRegionKey>,
    ) -> f64 {
        self.cost_for_transition_with_region(
            concrete_op_id,
            current_abstraction_id,
            source_hash,
            abstract_op_id,
            target_hash,
            TransitionRegion {
                source: Arc::new(source_region.clone()),
                target: Arc::new(target_region.clone()),
            },
            region_key,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn cost_for_transition_with_region(
        &self,
        concrete_op_id: usize,
        current_abstraction_id: usize,
        source_hash: usize,
        abstract_op_id: usize,
        target_hash: usize,
        query_region: TransitionRegion,
        region_key: Option<TransitionRegionKey>,
    ) -> f64 {
        let Some(residual) = self.operator_residuals.get(concrete_op_id) else {
            return f64::INFINITY;
        };
        if !residual.base_cost.is_finite() {
            return f64::INFINITY;
        }

        let key = TransitionQueryKey {
            abstraction_id: current_abstraction_id,
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
            abstraction_id: current_abstraction_id,
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
                !saturated.is_finite() || saturated >= -EPSILON,
                "negative transition saturated costs are not supported: transition {} has {}",
                transition.transition_id,
                saturated
            );
            if !saturated.is_finite() || saturated <= EPSILON {
                continue;
            }
            for &concrete_op_id in &transition.concrete_op_ids {
                let region = transition_system.transition_region(transition)?;
                self.reduce_exact_transition(
                    concrete_op_id,
                    producing_abstraction_id,
                    transition.source_hash,
                    transition.abstract_op_id,
                    transition.target_hash,
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
            if !saturated.is_finite() || saturated <= EPSILON {
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
                !saturated.is_finite() || saturated >= -EPSILON,
                "negative abstract-operator saturated costs are not supported: abstract op {} has {}",
                abstract_op_id,
                saturated
            );
            if !saturated.is_finite() || saturated <= EPSILON {
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
                        saturated <= residual.base_cost + EPSILON,
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
                            new_amount <= residual.base_cost + EPSILON,
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
                !saturated.is_finite() || saturated >= -EPSILON,
                "negative abstract-operator saturated costs are not supported: abstract op {} has {}",
                abstract_op_id,
                saturated
            );
            if !saturated.is_finite() || saturated <= EPSILON {
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
                    saturated <= current_residual + EPSILON,
                    "abstract-operator footprint reduction {saturated} exceeds current residual cost {current_residual} for concrete operator {concrete_op_id}"
                );
                let Some(residual) = self.operator_residuals.get(concrete_op_id) else {
                    continue;
                };
                if residual.base_cost <= EPSILON {
                    continue;
                }
                ensure!(
                    residual.base_cost.is_finite(),
                    "no base residual cost for operator {concrete_op_id}"
                );
                ensure!(
                    saturated <= residual.base_cost + EPSILON,
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
        for (entry_id, entry) in allocation.entries.iter().enumerate() {
            if entry_id.is_multiple_of(64) {
                ensure_scp_table_deadline(deadline)?;
            }
            ensure!(
                entry.amount.is_finite() && entry.amount >= -EPSILON,
                "regional allocation entry {entry_id} has invalid amount {}",
                entry.amount
            );
            if entry.amount <= EPSILON {
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
                residual.base_cost.is_finite() && residual.base_cost > EPSILON,
                "regional allocation requires a positive finite base cost for operator {concrete_op_id}"
            );
            ensure!(
                entry.amount <= residual.base_cost + EPSILON,
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
                            already_used <= EPSILON,
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
                            already_used + cell.amount <= residual.base_cost + EPSILON,
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
                !saturated.is_finite() || saturated >= -EPSILON,
                "negative uniform saturated costs are not supported: operator {op_id} has {saturated}"
            );
            if !saturated.is_finite() || saturated <= EPSILON {
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

    #[allow(clippy::too_many_arguments)]
    fn reduce_exact_transition(
        &mut self,
        concrete_op_id: usize,
        producing_abstraction_id: usize,
        source_hash: usize,
        abstract_op_id: usize,
        target_hash: usize,
        region: &TransitionRegion,
        saturated: f64,
    ) -> Result<()> {
        ensure!(
            concrete_op_id < self.operator_residuals.len(),
            "concrete operator id out of bounds: {concrete_op_id}"
        );
        let condition = TransitionCondition {
            abstraction_id: producing_abstraction_id,
            source_hash,
            abstract_op_id,
            target_hash,
            region: region.clone(),
        };
        let residual = &mut self.operator_residuals[concrete_op_id];
        let identity = condition.identity();
        if let Some(&index) = residual.reduction_indices.get(&identity) {
            let reduction = &mut residual.reductions[index];
            let new_amount = reduction.amount + saturated;
            ensure!(
                new_amount <= residual.base_cost + EPSILON,
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
            saturated <= residual.base_cost + EPSILON,
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

fn state_region_key(region: &StateRegion) -> StateRegionKey {
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

fn transition_region_key(region: &TransitionRegion) -> TransitionRegionKey {
    TransitionRegionKey {
        source: state_region_key(&region.source),
        target: state_region_key(&region.target),
    }
}

fn merge_transition_region(target: &mut TransitionRegion, source: &TransitionRegion) {
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
        if !self.base_cost.is_finite() || self.base_cost <= EPSILON || self.reductions.is_empty() {
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
                    if !intervals_overlap(indexed.interval, *query_interval) {
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
    if reductions.is_empty() || !cap.is_finite() || cap <= EPSILON {
        return None;
    }
    let all_reductions_full = reductions
        .iter()
        .all(|reduction| reduction.amount >= cap - EPSILON);
    let feature = best_full_reduction_feature(reductions, cap)?;
    let kind = match feature {
        RegionFeature::SourceProp(_) | RegionFeature::TargetProp(_) => {
            let mut buckets: HashMap<usize, Vec<usize>> = HashMap::new();
            for (reduction_id, reduction) in reductions.iter().enumerate() {
                if reduction.amount < cap - EPSILON {
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
                if reduction.amount < cap - EPSILON {
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
        .find(|reduction| reduction.amount >= cap - EPSILON)?;
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
            .filter(|reduction| reduction.amount >= cap - EPSILON)
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
            .filter(|reduction| reduction.amount >= cap - EPSILON)
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

fn interval_key(interval: &Interval) -> IntervalKey {
    IntervalKey {
        lower_bits: float_tolerance::canonical_bits(interval.lower),
        upper_bits: float_tolerance::canonical_bits(interval.upper),
        lower_closed: interval.lower_closed,
        upper_closed: interval.upper_closed,
    }
}

fn interval_starts_after(left: &Interval, right: &Interval) -> bool {
    left.lower > right.upper
        || (left.lower == right.upper && !(left.lower_closed && right.upper_closed))
}

fn intervals_overlap(left: Interval, right: Interval) -> bool {
    !Interval::new(
        left.lower.max(right.lower),
        left.upper.min(right.upper),
        if left.lower > right.lower {
            left.lower_closed
        } else if left.lower < right.lower {
            right.lower_closed
        } else {
            left.lower_closed && right.lower_closed
        },
        if left.upper < right.upper {
            left.upper_closed
        } else if left.upper > right.upper {
            right.upper_closed
        } else {
            left.upper_closed && right.upper_closed
        },
    )
    .is_empty()
}

fn max_overlap_reduction(
    query: Option<&TransitionCondition>,
    residual: &OperatorResidual,
    cap: f64,
) -> f64 {
    if !cap.is_finite() || cap <= EPSILON {
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
        has_subcap_reduction |= reduction.amount < cap - EPSILON;
        reduction.amount >= cap - EPSILON
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

    fn search(
        index: usize,
        selected: &mut Vec<usize>,
        current_sum: f64,
        best: &mut f64,
        cap: f64,
        query: Option<&TransitionCondition>,
        relevant: &[&ResidualReduction],
        suffix: &[f64],
    ) {
        if index == relevant.len() {
            *best = best.max(current_sum);
            return;
        }
        if *best >= cap - EPSILON {
            return;
        }
        if current_sum + suffix[index] <= *best + EPSILON {
            return;
        }

        let reduction = relevant[index];
        if can_add_reduction(query, selected, &reduction.condition, relevant) {
            selected.push(index);
            search(
                index + 1,
                selected,
                current_sum + reduction.amount.max(0.0),
                best,
                cap,
                query,
                relevant,
                suffix,
            );
            selected.pop();
        }
        search(
            index + 1,
            selected,
            current_sum,
            best,
            cap,
            query,
            relevant,
            suffix,
        );
    }

    let mut best = 0.0;
    let mut selected = Vec::new();
    search(
        0,
        &mut selected,
        0.0,
        &mut best,
        cap,
        query,
        &relevant,
        &suffix,
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

fn state_regions_have_common_intersection<'a, I>(
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

fn subtract_cost(cost: f64, saturated: f64) -> Result<f64> {
    ensure!(cost.is_finite(), "residual cost must be finite, got {cost}");
    ensure!(
        saturated.is_finite(),
        "saturated cost must be finite, got {saturated}"
    );
    let reduced = cost - saturated;
    if reduced < 0.0 && reduced > -EPSILON {
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

fn prop_regions_overlap(left: &[Vec<PropValueId>], right: &[Vec<PropValueId>]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter()
        .zip(right.iter())
        .all(|(l, r)| sorted_value_sets_overlap(l, r))
}

fn sorted_value_sets_overlap(left: &[PropValueId], right: &[PropValueId]) -> bool {
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
            residuals.cost_for_transition(
                0,
                0,
                3,
                7,
                4,
                &reduced_region.source,
                &reduced_region.target
            ),
            3.0
        );
        let other_target = state_region(2);
        assert_eq!(
            residuals.cost_for_transition(0, 0, 3, 7, 5, &reduced_region.source, &other_target),
            5.0
        );
        let overlapping = region(0, 1);
        assert_eq!(
            residuals.cost_for_transition(0, 1, 3, 7, 4, &overlapping.source, &overlapping.target),
            3.0
        );
        let disjoint = region(1, 0);
        assert_eq!(
            residuals.cost_for_transition(0, 1, 3, 7, 4, &disjoint.source, &disjoint.target),
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
            residuals.cost_for_transition(
                0,
                0,
                0,
                0,
                1,
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
            residuals.cost_for_transition(0, 0, 9, 7, 4, &disjoint.source, &disjoint.target),
            5.0
        );
        assert_eq!(
            residuals.cost_for_transition(0, 1, 9, 7, 4, &disjoint.source, &disjoint.target),
            5.0
        );
        let overlapping = region(0, 1);
        assert_eq!(
            residuals.cost_for_transition(0, 1, 9, 7, 4, &overlapping.source, &overlapping.target),
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
            residuals.cost_for_transition(
                0,
                1,
                99,
                99,
                100,
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
            residuals.cost_for_transition(0, 1, 99, 99, 100, &query.source, &query.target),
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
