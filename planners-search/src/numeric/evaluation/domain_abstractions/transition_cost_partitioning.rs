//! Transition-cost partitioning for numeric-planning domain abstractions.
//!
//! Each [`ConcreteOperatorFootprint::source_region`] stores the *regressed
//! preimage source* of an abstract operator's effect — the intersection of the
//! abstract source region with the inverse image of the abstract target region
//! under the operator's numeric effect (computed in
//! `domain_abstraction_factory::build_concrete_operator_footprint`).
//!
//! [`ConcreteOperatorFootprint::allocable`] encodes finite-support
//! stealability: it is `true` iff the preimage's relevant numeric dimensions
//! are bounded and narrow enough to allow this transition to steal cost from
//! the residual pool. The "narrow enough" threshold is configurable via
//! [`FiniteSupportConfig::max_stealable_width`]; the default (`INFINITY`)
//! reproduces the prior finite-vs-infinite gate exactly.
//!
//! Non-allocable footprints must carry zero saturated cost; this invariant is
//! enforced by [`TransitionResidualCosts::reduce_by_abstract_operator_footprints`].

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result, ensure};
use planners_sas::numeric::numeric_task::ExplicitFact;
use planners_sas::numeric::utils::float_tolerance;
use serde::{Deserialize, Serialize};

use super::comparison_expression::Interval;

const EPSILON: f64 = 1e-9;
const ABSTRACT_OPERATOR_REGION_HASH: usize = usize::MAX;
const MAX_ABSTRACT_OPERATOR_REDUCTION_PIECES: usize = 4096;
const MAX_TOTAL_ABSTRACT_OPERATOR_REDUCTION_PIECES: usize = 50_000;

/// Configuration for the finite-support gate on transition stealability.
///
/// An abstract transition is allowed to "steal" cost (have a positive
/// saturated cost in the cost-partitioning) only when every relevant numeric
/// dimension of its regressed preimage source is either a singleton or has
/// width at most [`max_stealable_width`](Self::max_stealable_width).
///
/// The default `max_stealable_width = f64::INFINITY` reproduces the legacy
/// finite-vs-infinite behavior: every finite preimage passes, every infinite
/// preimage fails.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
pub struct FiniteSupportConfig {
    pub max_stealable_width: f64,
}

impl Default for FiniteSupportConfig {
    fn default() -> Self {
        Self {
            max_stealable_width: f64::INFINITY,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AbstractTransition {
    pub transition_id: usize,
    pub abstract_op_id: usize,
    pub concrete_op_ids: Vec<usize>,
    pub source_hash: usize,
    pub target_hash: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StateRegion {
    pub propositions: Vec<Vec<usize>>,
    pub numeric: Vec<Interval>,
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
    pub source_region: StateRegion,
    pub allocable: bool,
    pub max_allocation_fraction: f64,
    pub non_allocable_reason: Option<NonAllocableFootprintReason>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NonAllocableFootprintReason {
    InfiniteActiveSource,
    UninformativeSource,
    UnsupportedEffectImage,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TransitionRegion {
    pub source: StateRegion,
    pub target: StateRegion,
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
    pub state_regions: Vec<StateRegion>,
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

#[derive(Debug, Clone, PartialEq)]
pub struct AbstractOperatorCostBudget {
    pub label_fractions: Vec<f64>,
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
    reductions: Vec<ResidualReduction>,
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
    propositions: Vec<Vec<usize>>,
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

#[derive(Clone, Debug)]
struct IndexedLaterFootprint {
    interval: Interval,
    later_footprint_id: usize,
}

#[derive(Clone, Debug)]
struct LaterFootprint {
    abstraction_id: usize,
    source_region: StateRegion,
}

#[derive(Clone, Debug, Default)]
struct OperatorFootprintOverlapIndex {
    footprints: Vec<LaterFootprint>,
    numeric: Vec<Vec<IndexedLaterFootprint>>,
    unbounded_by_numeric: Vec<Vec<usize>>,
    unbounded_abstractions_by_numeric: Vec<HashSet<usize>>,
}

#[derive(Clone, Debug, Default)]
struct FootprintOverlapIndex {
    by_operator: HashMap<usize, OperatorFootprintOverlapIndex>,
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
                reductions: Vec::new(),
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
            .map(|residual| residual.reductions.len())
            .sum()
    }

    pub fn has_reductions(&self) -> bool {
        self.operator_residuals
            .iter()
            .any(|residual| !residual.reductions.is_empty())
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
                let reduction = max_overlap_reduction(None, residual, residual.base_cost);
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
                            fallback_cost: uniform_costs.get(op_id).copied().unwrap_or(fallback_cost),
                            variants: Vec::new(),
                        };
                    }
                    variants.push(LmCutResidualCostVariant {
                        cost: (residual.base_cost - reduction.amount).max(0.0),
                        source_region: reduction.condition.region.source.clone(),
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
        self.cost_for_transition_with_region_key(
            concrete_op_id,
            current_abstraction_id,
            ABSTRACT_OPERATOR_REGION_HASH,
            abstract_op_id,
            ABSTRACT_OPERATOR_REGION_HASH,
            &region.source,
            &region.target,
            None,
        )
    }

    pub fn cost_for_operator_footprint(
        &self,
        current_abstraction_id: usize,
        abstract_op_id: usize,
        footprint: &ConcreteOperatorFootprint,
    ) -> f64 {
        if !footprint.allocable {
            return 0.0;
        }
        let region_key = state_region_key(&footprint.source_region);
        self.cost_for_transition_with_region_key(
            footprint.concrete_op_id,
            current_abstraction_id,
            ABSTRACT_OPERATOR_REGION_HASH,
            abstract_op_id,
            ABSTRACT_OPERATOR_REGION_HASH,
            &footprint.source_region,
            &footprint.source_region,
            Some(TransitionRegionKey {
                source: region_key.clone(),
                target: region_key,
            }),
        )
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
        let Some(residual) = self.operator_residuals.get(concrete_op_id) else {
            return f64::INFINITY;
        };
        if !residual.base_cost.is_finite() {
            return f64::INFINITY;
        }
        let query_region = TransitionRegion {
            source: source_region.clone(),
            target: target_region.clone(),
        };

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
                    residual.reductions.push(ResidualReduction {
                        amount: saturated.min(residual.base_cost),
                        condition: TransitionCondition {
                            abstraction_id: producing_abstraction_id,
                            source_hash: piece_id,
                            abstract_op_id,
                            target_hash: piece_id,
                            region: region.clone(),
                        },
                    });
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
        label_rescue_operator_ids: Option<&HashSet<usize>>,
        tcf: &AbstractOperatorCostFunction,
    ) -> Result<()> {
        ensure!(
            footprints.len() >= tcf.operator_costs.len(),
            "abstract-operator footprint/cost function size mismatch: footprints={} costs={}",
            footprints.len(),
            tcf.operator_costs.len()
        );

        let mut pending: Vec<(usize, ResidualReduction)> = Vec::new();
        let mut pending_uniform: Vec<(usize, f64)> = Vec::new();
        let uniform_label_residuals =
            label_rescue_operator_ids.map(|_| self.operator_costs_for_label_cp());
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

            for footprint in &footprints[abstract_op_id].labels {
                if !footprint.allocable {
                    if matches!(
                        footprint.non_allocable_reason,
                        Some(
                            NonAllocableFootprintReason::InfiniteActiveSource
                                | NonAllocableFootprintReason::UninformativeSource
                        )
                    ) && label_rescue_operator_ids
                        .is_some_and(|ids| ids.contains(&footprint.concrete_op_id))
                    {
                        let concrete_op_id = footprint.concrete_op_id;
                        let current_residual = uniform_label_residuals
                            .as_ref()
                            .and_then(|costs| costs.get(concrete_op_id))
                            .copied()
                            .unwrap_or(f64::INFINITY);
                        ensure!(
                            current_residual.is_finite(),
                            "uniform residual cost for rescued abstract op {abstract_op_id}, concrete op {concrete_op_id} must be finite"
                        );
                        ensure!(
                            saturated <= current_residual + EPSILON,
                            "rescued abstract-operator reduction {saturated} exceeds current uniform residual cost {current_residual} for concrete operator {concrete_op_id}"
                        );
                        if let Some((_, existing)) = pending_uniform
                            .iter_mut()
                            .find(|(pending_op_id, _)| *pending_op_id == concrete_op_id)
                        {
                            *existing = existing.max(saturated);
                        } else {
                            pending_uniform.push((concrete_op_id, saturated));
                        }
                        continue;
                    }
                    ensure!(
                        saturated <= EPSILON,
                        "positive abstract-operator saturated cost {saturated} for non-allocable footprint of abstract op {abstract_op_id}, concrete op {}",
                        footprint.concrete_op_id
                    );
                    continue;
                }
                let region = TransitionRegion {
                    source: footprint.source_region.clone(),
                    target: footprint.source_region.clone(),
                };
                let concrete_op_id = footprint.concrete_op_id;
                ensure!(
                    footprint.max_allocation_fraction.is_finite()
                        && footprint.max_allocation_fraction >= -EPSILON
                        && footprint.max_allocation_fraction <= 1.0 + EPSILON,
                    "invalid abstract-operator footprint allocation fraction {} for operator {concrete_op_id}",
                    footprint.max_allocation_fraction
                );
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
                let condition = TransitionCondition {
                    abstraction_id: producing_abstraction_id,
                    source_hash: ABSTRACT_OPERATOR_REGION_HASH,
                    abstract_op_id,
                    target_hash: ABSTRACT_OPERATOR_REGION_HASH,
                    region: region.clone(),
                };
                let amount = saturated.min(residual.base_cost);
                if let Some((_, existing)) =
                    pending.iter_mut().find(|(pending_op_id, reduction)| {
                        *pending_op_id == concrete_op_id && reduction.condition == condition
                    })
                {
                    existing.amount = existing.amount.max(amount);
                } else {
                    pending.push((concrete_op_id, ResidualReduction { amount, condition }));
                }
            }
        }

        for (concrete_op_id, amount) in pending_uniform {
            let Some(residual) = self.operator_residuals.get_mut(concrete_op_id) else {
                continue;
            };
            residual.base_cost = subtract_cost(residual.base_cost, amount).with_context(|| {
                format!(
                    "rescued uniform residual reduction underflow for operator {concrete_op_id}"
                )
            })?;
            residual.invalidate_cache();
        }

        for (concrete_op_id, reduction) in pending {
            let Some(residual) = self.operator_residuals.get_mut(concrete_op_id) else {
                continue;
            };
            if let Some(existing) = residual
                .reductions
                .iter_mut()
                .find(|existing| existing.condition == reduction.condition)
            {
                existing.amount = (existing.amount + reduction.amount).min(residual.base_cost);
            } else {
                residual.reductions.push(reduction);
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
        if let Some(reduction) = residual
            .reductions
            .iter_mut()
            .find(|reduction| reduction.condition == condition)
        {
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
        residual.reductions.push(ResidualReduction {
            amount: saturated.min(residual.base_cost),
            condition,
        });
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
    merge_state_region(&mut target.source, &source.source);
    merge_state_region(&mut target.target, &source.target);
}

fn merge_state_region(target: &mut StateRegion, source: &StateRegion) {
    for (target_values, source_values) in target
        .propositions
        .iter_mut()
        .zip(source.propositions.iter())
    {
        target_values.extend(source_values.iter().copied());
        target_values.sort_unstable();
        target_values.dedup();
    }
    for (target_interval, source_interval) in target.numeric.iter_mut().zip(source.numeric.iter()) {
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
                    let Some(bucket) = buckets.get(&value) else {
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

pub fn compute_lookahead_abstract_operator_cost_budgets(
    footprints_by_abstraction: &[&[AbstractOperatorFootprint]],
    order: &[usize],
    active_abstractions: &[bool],
) -> Result<Vec<Vec<AbstractOperatorCostBudget>>> {
    let mut budgets: Vec<Vec<AbstractOperatorCostBudget>> = footprints_by_abstraction
        .iter()
        .map(|footprints| {
            footprints
                .iter()
                .map(|footprint| AbstractOperatorCostBudget {
                    label_fractions: vec![0.0; footprint.labels.len()],
                })
                .collect()
        })
        .collect();
    let mut suffix_index = FootprintOverlapIndex::default();

    for &abstraction_id in order.iter().rev() {
        if abstraction_id >= footprints_by_abstraction.len() {
            continue;
        }
        let active = active_abstractions
            .get(abstraction_id)
            .copied()
            .unwrap_or(false);
        if active {
            let footprints = footprints_by_abstraction[abstraction_id];
            for (abstract_op_id, footprint) in footprints.iter().enumerate() {
                ensure!(
                    abstract_op_id < budgets[abstraction_id].len(),
                    "missing abstract-operator budget for abstraction {abstraction_id}, abstract op {abstract_op_id}"
                );
                ensure!(
                    budgets[abstraction_id][abstract_op_id]
                        .label_fractions
                        .len()
                        == footprint.labels.len(),
                    "abstract-operator budget label count mismatch for abstraction {abstraction_id}, abstract op {abstract_op_id}"
                );
                for (label_id, label) in footprint.labels.iter().enumerate() {
                    let competitors = suffix_index.count_overlapping_abstractions(label);
                    let lookahead_share = 1.0 / (competitors as f64 + 1.0);
                    budgets[abstraction_id][abstract_op_id].label_fractions[label_id] =
                        label.max_allocation_fraction * lookahead_share;
                }
            }
            suffix_index.add_abstraction(abstraction_id, footprints)?;
        }
    }

    Ok(budgets)
}

impl FootprintOverlapIndex {
    fn add_abstraction(
        &mut self,
        abstraction_id: usize,
        footprints: &[AbstractOperatorFootprint],
    ) -> Result<()> {
        for footprint in footprints {
            for label in &footprint.labels {
                if !label.allocable {
                    continue;
                }
                let operator_index = self.by_operator.entry(label.concrete_op_id).or_default();
                let footprint_id = operator_index.footprints.len();
                operator_index.footprints.push(LaterFootprint {
                    abstraction_id,
                    source_region: label.source_region.clone(),
                });
                if operator_index.numeric.len() < label.source_region.numeric.len() {
                    operator_index
                        .numeric
                        .resize_with(label.source_region.numeric.len(), Vec::new);
                    operator_index
                        .unbounded_by_numeric
                        .resize_with(label.source_region.numeric.len(), Vec::new);
                    operator_index
                        .unbounded_abstractions_by_numeric
                        .resize_with(label.source_region.numeric.len(), HashSet::new);
                }
                for (var_id, interval) in label.source_region.numeric.iter().copied().enumerate() {
                    if interval.is_empty() {
                        continue;
                    }
                    if !interval.lower.is_finite() && !interval.upper.is_finite() {
                        operator_index.unbounded_by_numeric[var_id].push(footprint_id);
                        operator_index.unbounded_abstractions_by_numeric[var_id]
                            .insert(abstraction_id);
                        continue;
                    }
                    operator_index.numeric[var_id].push(IndexedLaterFootprint {
                        interval,
                        later_footprint_id: footprint_id,
                    });
                }
            }
        }
        for operator_index in self.by_operator.values_mut() {
            for intervals in &mut operator_index.numeric {
                intervals.sort_by(|left, right| {
                    left.interval
                        .lower
                        .total_cmp(&right.interval.lower)
                        .then_with(|| left.interval.upper.total_cmp(&right.interval.upper))
                });
            }
        }
        Ok(())
    }

    fn count_overlapping_abstractions(&self, label: &ConcreteOperatorFootprint) -> usize {
        if !label.allocable {
            return 0;
        }
        let Some(operator_index) = self.by_operator.get(&label.concrete_op_id) else {
            return 0;
        };
        let mut best_var = None;
        let mut best_count = usize::MAX;
        for (var_id, query_interval) in label.source_region.numeric.iter().copied().enumerate() {
            if query_interval.is_empty()
                || (!query_interval.lower.is_finite() && !query_interval.upper.is_finite())
            {
                continue;
            }
            let Some(intervals) = operator_index.numeric.get(var_id) else {
                continue;
            };
            let unbounded_count = operator_index
                .unbounded_abstractions_by_numeric
                .get(var_id)
                .map_or(0, HashSet::len);
            let count = count_interval_candidates(intervals, query_interval)
                .saturating_add(unbounded_count);
            if count < best_count {
                best_count = count;
                best_var = Some(var_id);
            }
        }

        let mut abstractions = HashSet::new();
        if let Some(var_id) = best_var {
            if let Some(intervals) = operator_index.numeric.get(var_id) {
                for indexed in intervals {
                    if interval_starts_after(
                        &indexed.interval,
                        &label.source_region.numeric[var_id],
                    ) {
                        break;
                    }
                    if !intervals_overlap(indexed.interval, label.source_region.numeric[var_id]) {
                        continue;
                    }
                    let later = &operator_index.footprints[indexed.later_footprint_id];
                    if later.source_region.overlaps(&label.source_region) {
                        abstractions.insert(later.abstraction_id);
                    }
                }
            }
            if let Some(unbounded_abstractions) =
                operator_index.unbounded_abstractions_by_numeric.get(var_id)
            {
                abstractions.extend(unbounded_abstractions.iter().copied());
            }
        } else {
            for later in &operator_index.footprints {
                if later.source_region.overlaps(&label.source_region) {
                    abstractions.insert(later.abstraction_id);
                }
            }
        }
        abstractions.len()
    }
}

fn count_interval_candidates(intervals: &[IndexedLaterFootprint], query: Interval) -> usize {
    let mut count = 0usize;
    for indexed in intervals {
        if interval_starts_after(&indexed.interval, &query) {
            break;
        }
        if intervals_overlap(indexed.interval, query) {
            count = count.saturating_add(1);
        }
    }
    count
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
    (values.len() == 1).then_some(values[0])
}

fn query_values_for_feature(region: &TransitionRegion, feature: RegionFeature) -> Option<&[usize]> {
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
        query.map(|condition| &condition.region.source),
        selected
            .iter()
            .map(|&index| &relevant[index].condition.region.source),
        &condition.region.source,
    ) && state_regions_have_common_intersection(
        query.map(|condition| &condition.region.target),
        selected
            .iter()
            .map(|&index| &relevant[index].condition.region.target),
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

fn prop_regions_overlap(left: &[Vec<usize>], right: &[Vec<usize>]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter()
        .zip(right.iter())
        .all(|(l, r)| sorted_value_sets_overlap(l, r))
}

fn sorted_value_sets_overlap(left: &[usize], right: &[usize]) -> bool {
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

    fn state_region(value: usize) -> StateRegion {
        StateRegion {
            propositions: vec![vec![value]],
            numeric: vec![],
        }
    }

    fn region(source: usize, target: usize) -> TransitionRegion {
        TransitionRegion {
            source: state_region(source),
            target: state_region(target),
        }
    }

    fn numeric_state_region(lower: f64, upper: f64) -> StateRegion {
        StateRegion {
            propositions: vec![vec![0]],
            numeric: vec![Interval::closed(lower, upper)],
        }
    }

    fn numeric_region(source_lower: f64, source_upper: f64) -> TransitionRegion {
        TransitionRegion {
            source: numeric_state_region(source_lower, source_upper),
            target: numeric_state_region(source_lower, source_upper),
        }
    }

    fn concrete_footprint(lower: f64, upper: f64) -> ConcreteOperatorFootprint {
        concrete_footprint_for_op(0, lower, upper, true)
    }

    fn concrete_footprint_for_op(
        concrete_op_id: usize,
        lower: f64,
        upper: f64,
        allocable: bool,
    ) -> ConcreteOperatorFootprint {
        concrete_footprint_for_op_with_fraction(concrete_op_id, lower, upper, allocable, 1.0)
    }

    fn concrete_footprint_for_op_with_fraction(
        concrete_op_id: usize,
        lower: f64,
        upper: f64,
        allocable: bool,
        max_allocation_fraction: f64,
    ) -> ConcreteOperatorFootprint {
        ConcreteOperatorFootprint {
            concrete_op_id,
            source_region: numeric_state_region(lower, upper),
            allocable,
            max_allocation_fraction: if allocable {
                max_allocation_fraction
            } else {
                0.0
            },
            non_allocable_reason: None,
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
                propositions: vec![vec![0]],
                numeric: vec![first, second],
            },
            allocable: true,
            max_allocation_fraction: 1.0,
            non_allocable_reason: None,
        }
    }

    fn footprint(lower: f64, upper: f64) -> AbstractOperatorFootprint {
        AbstractOperatorFootprint {
            labels: vec![concrete_footprint(lower, upper)],
        }
    }

    fn footprint_with_fraction(
        lower: f64,
        upper: f64,
        max_allocation_fraction: f64,
    ) -> AbstractOperatorFootprint {
        AbstractOperatorFootprint {
            labels: vec![concrete_footprint_for_op_with_fraction(
                0,
                lower,
                upper,
                true,
                max_allocation_fraction,
            )],
        }
    }

    fn budget_fraction(
        budgets: &[Vec<AbstractOperatorCostBudget>],
        abstraction_id: usize,
        abstract_op_id: usize,
        label_id: usize,
    ) -> f64 {
        budgets[abstraction_id][abstract_op_id].label_fractions[label_id]
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
                state_region(9),
                state_region(9),
                state_region(9),
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
                state_region(9),
                state_region(9),
                state_region(9),
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
                None,
                &AbstractOperatorCostFunction {
                    operator_costs: vec![3.0],
                },
            )
            .unwrap();

        let query = concrete_footprint(5.0, 8.0);
        assert_eq!(residuals.cost_for_operator_footprint(1, 0, &query), 7.0);
        let other_op_query = concrete_footprint_for_op(1, 5.0, 8.0, true);
        assert_eq!(
            residuals.cost_for_operator_footprint(1, 0, &other_op_query),
            10.0
        );
    }

    #[test]
    fn footprint_reduction_allows_full_cost_for_allocable_fractional_footprint() {
        let mut residuals = TransitionResidualCosts::from_operator_costs(&[1.0]);
        let reduced = footprint_with_fraction(3.0, 7.0, 0.5);
        residuals
            .reduce_by_abstract_operator_footprints(
                0,
                std::slice::from_ref(&reduced),
                None,
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
                None,
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
                None,
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
                None,
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
                None,
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
    fn lookahead_budget_splits_overlapping_same_label_abstractions() {
        let first = vec![footprint(0.0, 5.0)];
        let second = vec![footprint(3.0, 8.0)];
        let footprints = vec![first.as_slice(), second.as_slice()];
        let budgets =
            compute_lookahead_abstract_operator_cost_budgets(&footprints, &[0, 1], &[true, true])
                .unwrap();

        assert_eq!(budget_fraction(&budgets, 0, 0, 0), 0.5);
        assert_eq!(budget_fraction(&budgets, 1, 0, 0), 1.0);
    }

    #[test]
    fn lookahead_budget_keeps_disjoint_same_label_footprints_full() {
        let first = vec![footprint(0.0, 2.0)];
        let second = vec![footprint(3.0, 8.0)];
        let footprints = vec![first.as_slice(), second.as_slice()];
        let budgets =
            compute_lookahead_abstract_operator_cost_budgets(&footprints, &[0, 1], &[true, true])
                .unwrap();

        assert_eq!(budget_fraction(&budgets, 0, 0, 0), 1.0);
        assert_eq!(budget_fraction(&budgets, 1, 0, 0), 1.0);
    }

    #[test]
    fn lookahead_budget_splits_three_overlapping_abstractions() {
        let first = vec![footprint(0.0, 5.0)];
        let second = vec![footprint(1.0, 6.0)];
        let third = vec![footprint(2.0, 7.0)];
        let footprints = vec![first.as_slice(), second.as_slice(), third.as_slice()];
        let budgets = compute_lookahead_abstract_operator_cost_budgets(
            &footprints,
            &[0, 1, 2],
            &[true, true, true],
        )
        .unwrap();

        assert!((budget_fraction(&budgets, 0, 0, 0) - 1.0 / 3.0).abs() < 1e-9);
        assert_eq!(budget_fraction(&budgets, 1, 0, 0), 0.5);
        assert_eq!(budget_fraction(&budgets, 2, 0, 0), 1.0);
    }

    #[test]
    fn lookahead_budget_counts_unbounded_later_dimension_as_overlap() {
        let first = vec![AbstractOperatorFootprint {
            labels: vec![concrete_footprint_2d(
                0,
                Interval::closed(0.0, 5.0),
                Interval::closed(f64::NEG_INFINITY, f64::INFINITY),
            )],
        }];
        let second = vec![AbstractOperatorFootprint {
            labels: vec![concrete_footprint_2d(
                0,
                Interval::closed(f64::NEG_INFINITY, f64::INFINITY),
                Interval::closed(3.0, 8.0),
            )],
        }];
        let footprints = vec![first.as_slice(), second.as_slice()];
        let budgets =
            compute_lookahead_abstract_operator_cost_budgets(&footprints, &[0, 1], &[true, true])
                .unwrap();

        assert_eq!(budget_fraction(&budgets, 0, 0, 0), 0.5);
        assert_eq!(budget_fraction(&budgets, 1, 0, 0), 1.0);
    }

    #[test]
    fn lookahead_budget_ignores_inactive_later_abstraction() {
        let first = vec![footprint(0.0, 5.0)];
        let second = vec![footprint(3.0, 8.0)];
        let footprints = vec![first.as_slice(), second.as_slice()];
        let budgets =
            compute_lookahead_abstract_operator_cost_budgets(&footprints, &[0, 1], &[true, false])
                .unwrap();

        assert_eq!(budget_fraction(&budgets, 0, 0, 0), 1.0);
        assert_eq!(budget_fraction(&budgets, 1, 0, 0), 0.0);
    }

    #[test]
    fn lookahead_budget_never_competes_across_concrete_operators() {
        let first = vec![footprint(0.0, 5.0)];
        let second = vec![AbstractOperatorFootprint {
            labels: vec![concrete_footprint_for_op(1, 3.0, 8.0, true)],
        }];
        let footprints = vec![first.as_slice(), second.as_slice()];
        let budgets =
            compute_lookahead_abstract_operator_cost_budgets(&footprints, &[0, 1], &[true, true])
                .unwrap();

        assert_eq!(budget_fraction(&budgets, 0, 0, 0), 1.0);
        assert_eq!(budget_fraction(&budgets, 1, 0, 0), 1.0);
    }
}
