use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use anyhow::{Context, Result, ensure};
use planners_sas::numeric::numeric_task::ExplicitFact;
use planners_sas::numeric::utils::float_tolerance;

use super::comparison_expression::Interval;

const EPSILON: f64 = 1e-9;
const ABSTRACT_OPERATOR_REGION_HASH: usize = usize::MAX;

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
}

#[derive(Debug, Clone, PartialEq)]
pub struct AbstractTransitionCostFunction {
    pub transition_costs: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AbstractOperatorCostFunction {
    pub operator_costs: Vec<f64>,
}

#[derive(Debug)]
pub struct TransitionResidualCosts {
    operator_residuals: Vec<OperatorResidual>,
}

#[derive(Debug)]
struct OperatorResidual {
    base_cost: f64,
    reductions: Vec<ResidualReduction>,
    generation: Cell<u64>,
    uniform_cost_cache: Cell<Option<f64>>,
    transition_cost_cache: RefCell<HashMap<TransitionQueryKey, CachedCost>>,
    full_reduction_index: RefCell<Option<FullReductionIndex>>,
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
                reductions: Vec::new(),
                generation: Cell::new(0),
                uniform_cost_cache: Cell::new(None),
                transition_cost_cache: RefCell::new(HashMap::new()),
                full_reduction_index: RefCell::new(None),
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
                let reduction = max_overlap_reduction(None, &residual.reductions, residual.base_cost);
                let cost = (residual.base_cost - reduction).max(0.0);
                residual.uniform_cost_cache.set(Some(cost));
                cost
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

        if let Some(has_full_overlap) = residual.lookup_full_reduction_overlap(&query_region) {
            let cost = if has_full_overlap { 0.0 } else { residual.base_cost };
            residual.transition_cost_cache.borrow_mut().insert(
                key,
                CachedCost {
                    generation: residual.generation.get(),
                    cost,
                },
            );
            return cost;
        }

        let query = TransitionCondition {
            abstraction_id: current_abstraction_id,
            source_hash,
            abstract_op_id,
            target_hash,
            region: query_region,
        };
        let reduction =
            max_overlap_reduction(Some(&query), &residual.reductions, residual.base_cost);
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
    ) -> Result<()> {
        let regions = transition_system.abstract_operator_regions();
        ensure!(
            regions.len() == tcf.operator_costs.len(),
            "abstract-operator system/cost function size mismatch: {} vs {}",
            regions.len(),
            tcf.operator_costs.len()
        );
        let concrete_op_ids = transition_system.concrete_operator_ids_by_abstract_operator();
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
            let Some(region) = regions.get(abstract_op_id).and_then(|region| region.clone())
            else {
                continue;
            };
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
                        source_hash: ABSTRACT_OPERATOR_REGION_HASH,
                        abstract_op_id,
                        target_hash: ABSTRACT_OPERATOR_REGION_HASH,
                        region: region.clone(),
                    },
                });
                residual.invalidate_cache();
            }
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
    }

    fn lookup_full_reduction_overlap(&self, query: &TransitionRegion) -> Option<bool> {
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
                let values = query_values_for_feature(query, *feature)?;
                for &value in values {
                    let Some(bucket) = buckets.get(&value) else {
                        continue;
                    };
                    if bucket.iter().any(|&reduction_id| {
                        self.reductions[reduction_id]
                            .condition
                            .region
                            .overlaps(query)
                    }) {
                        return Some(true);
                    }
                }
            }
            FullReductionIndexKind::Numeric { feature, intervals } => {
                let query_interval = interval_for_feature(query, *feature)?;
                for indexed in intervals {
                    if interval_starts_after(&indexed.interval, query_interval) {
                        break;
                    }
                    if !intervals_overlap(indexed.interval, *query_interval) {
                        continue;
                    }
                    if self.reductions[indexed.reduction_id]
                        .condition
                        .region
                        .overlaps(query)
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
        let index = build_full_reduction_index(
            &self.reductions,
            self.base_cost,
            self.generation.get(),
        )?;
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
            FullReductionIndexKind::Numeric { feature, intervals }
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
            if interval.is_empty()
                || (!interval.lower.is_finite() && !interval.upper.is_finite())
            {
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

fn query_values_for_feature(
    region: &TransitionRegion,
    feature: RegionFeature,
) -> Option<&[usize]> {
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
    reductions: &[ResidualReduction],
    cap: f64,
) -> f64 {
    if !cap.is_finite() || cap <= EPSILON {
        return 0.0;
    }
    let mut has_subcap_reduction = false;
    if reductions.iter().any(|reduction| {
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
    let mut relevant: Vec<&ResidualReduction> = reductions
        .iter()
        .filter(|reduction| {
            query.is_none_or(|query| {
                compatible_identities(query, &reduction.condition)
                    && reduction.condition.region.overlaps(&query.region)
            })
        })
        .collect();
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
    if left.source_hash == ABSTRACT_OPERATOR_REGION_HASH
        || right.source_hash == ABSTRACT_OPERATOR_REGION_HASH
        || left.target_hash == ABSTRACT_OPERATOR_REGION_HASH
        || right.target_hash == ABSTRACT_OPERATOR_REGION_HASH
    {
        return left.abstract_op_id == right.abstract_op_id;
    }
    left.source_hash == right.source_hash
        && left.abstract_op_id == right.abstract_op_id
        && left.target_hash == right.target_hash
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
}
