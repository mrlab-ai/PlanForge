use std::cell::{Cell, RefCell};

use anyhow::{Context, Result, ensure};
use planners_sas::numeric::numeric_task::ExplicitFact;

use super::comparison_expression::Interval;

const EPSILON: f64 = 1e-9;

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
}

#[derive(Debug, Clone, PartialEq)]
pub struct AbstractTransitionCostFunction {
    pub transition_costs: Vec<f64>,
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
    transition_cost_cache: RefCell<std::collections::HashMap<TransitionQueryKey, CachedCost>>,
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
    source_region: StateRegionKey,
    target_region: StateRegionKey,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct StateRegionKey {
    propositions: Vec<Vec<usize>>,
    numeric: Vec<IntervalKey>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
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

impl TransitionResidualCosts {
    pub fn from_operator_costs(costs: &[f64]) -> Self {
        let operator_residuals = costs
            .iter()
            .map(|&base_cost| OperatorResidual {
                base_cost,
                reductions: Vec::new(),
                generation: Cell::new(0),
                uniform_cost_cache: Cell::new(None),
                transition_cost_cache: RefCell::new(std::collections::HashMap::new()),
            })
            .collect();
        Self { operator_residuals }
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
                let reduction = max_overlap_reduction(None, &residual.reductions);
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
            source_region: state_region_key(source_region),
            target_region: state_region_key(target_region),
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
            region: TransitionRegion {
                source: source_region.clone(),
                target: target_region.clone(),
            },
        };
        let reduction = max_overlap_reduction(Some(&query), &residual.reductions);
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
                lower_bits: interval.lower.to_bits(),
                upper_bits: interval.upper.to_bits(),
                lower_closed: interval.lower_closed,
                upper_closed: interval.upper_closed,
            })
            .collect(),
    }
}

impl OperatorResidual {
    fn invalidate_cache(&self) {
        self.generation.set(self.generation.get().wrapping_add(1));
        self.uniform_cost_cache.set(None);
        self.transition_cost_cache.borrow_mut().clear();
    }
}

#[derive(Clone, Debug)]
struct OverlapConstraint {
    region: TransitionRegion,
    identities: Vec<TransitionIdentity>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TransitionIdentity {
    abstraction_id: usize,
    source_hash: usize,
    abstract_op_id: usize,
    target_hash: usize,
}

fn max_overlap_reduction(
    query: Option<&TransitionCondition>,
    reductions: &[ResidualReduction],
) -> f64 {
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
        current: Option<OverlapConstraint>,
        current_sum: f64,
        best: &mut f64,
        relevant: &[&ResidualReduction],
        suffix: &[f64],
    ) {
        if index == relevant.len() {
            *best = best.max(current_sum);
            return;
        }
        if current_sum + suffix[index] <= *best + EPSILON {
            return;
        }

        let reduction = relevant[index];
        let include = match current.as_ref() {
            Some(current) => intersect_constraint_with_condition(current, &reduction.condition),
            None => Some(constraint_from_condition(&reduction.condition)),
        };
        if let Some(include) = include {
            search(
                index + 1,
                Some(include),
                current_sum + reduction.amount.max(0.0),
                best,
                relevant,
                suffix,
            );
        }
        search(index + 1, current, current_sum, best, relevant, suffix);
    }

    let initial = query.map(constraint_from_condition);
    let mut best = 0.0;
    search(0, initial, 0.0, &mut best, &relevant, &suffix);
    best
}

fn condition_identity(condition: &TransitionCondition) -> TransitionIdentity {
    TransitionIdentity {
        abstraction_id: condition.abstraction_id,
        source_hash: condition.source_hash,
        abstract_op_id: condition.abstract_op_id,
        target_hash: condition.target_hash,
    }
}

fn constraint_from_condition(condition: &TransitionCondition) -> OverlapConstraint {
    OverlapConstraint {
        region: condition.region.clone(),
        identities: vec![condition_identity(condition)],
    }
}

fn compatible_identities(left: &TransitionCondition, right: &TransitionCondition) -> bool {
    left.abstraction_id != right.abstraction_id
        || (left.source_hash == right.source_hash
            && left.abstract_op_id == right.abstract_op_id
            && left.target_hash == right.target_hash)
}

fn intersect_constraint_with_condition(
    current: &OverlapConstraint,
    condition: &TransitionCondition,
) -> Option<OverlapConstraint> {
    let identity = condition_identity(condition);
    if current
        .identities
        .iter()
        .any(|existing| existing.abstraction_id == identity.abstraction_id && existing != &identity)
    {
        return None;
    }
    let region = intersect_transition_regions(&current.region, &condition.region)?;
    let mut identities = current.identities.clone();
    if !identities.iter().any(|existing| existing == &identity) {
        identities.push(identity);
    }
    Some(OverlapConstraint { region, identities })
}

fn intersect_transition_regions(
    left: &TransitionRegion,
    right: &TransitionRegion,
) -> Option<TransitionRegion> {
    Some(TransitionRegion {
        source: intersect_state_regions(&left.source, &right.source)?,
        target: intersect_state_regions(&left.target, &right.target)?,
    })
}

fn intersect_state_regions(left: &StateRegion, right: &StateRegion) -> Option<StateRegion> {
    if left.propositions.len() != right.propositions.len()
        || left.numeric.len() != right.numeric.len()
    {
        return None;
    }

    let propositions = left
        .propositions
        .iter()
        .zip(right.propositions.iter())
        .map(|(left, right)| intersect_sorted_value_sets(left, right))
        .collect::<Option<Vec<_>>>()?;
    let numeric = left
        .numeric
        .iter()
        .zip(right.numeric.iter())
        .map(|(left, right)| intersect_intervals(*left, *right))
        .collect::<Option<Vec<_>>>()?;
    Some(StateRegion {
        propositions,
        numeric,
    })
}

fn intersect_sorted_value_sets(left: &[usize], right: &[usize]) -> Option<Vec<usize>> {
    let mut out = Vec::new();
    let mut i = 0;
    let mut j = 0;
    while i < left.len() && j < right.len() {
        match left[i].cmp(&right[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                out.push(left[i]);
                i += 1;
                j += 1;
            }
        }
    }
    (!out.is_empty()).then_some(out)
}

fn intersect_intervals(left: Interval, right: Interval) -> Option<Interval> {
    if !left.intersects(&right) {
        return None;
    }
    let (lower, lower_closed) = if left.lower > right.lower {
        (left.lower, left.lower_closed)
    } else if right.lower > left.lower {
        (right.lower, right.lower_closed)
    } else {
        (left.lower, left.lower_closed && right.lower_closed)
    };
    let (upper, upper_closed) = if left.upper < right.upper {
        (left.upper, left.upper_closed)
    } else if right.upper < left.upper {
        (right.upper, right.upper_closed)
    } else {
        (left.upper, left.upper_closed && right.upper_closed)
    };
    let interval = Interval::new(lower, upper, lower_closed, upper_closed);
    (!interval.is_empty()).then_some(interval)
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
