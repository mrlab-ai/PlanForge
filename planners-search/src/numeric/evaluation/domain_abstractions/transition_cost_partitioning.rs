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

#[derive(Debug, Clone, PartialEq)]
pub struct TransitionResidualCosts {
    operator_buckets: Vec<Vec<ResidualBucket>>,
}

#[derive(Debug, Clone, PartialEq)]
struct ResidualBucket {
    cost: f64,
    condition: TransitionCondition,
}

#[derive(Debug, Clone, PartialEq)]
enum TransitionCondition {
    Any,
    AbstractTransition {
        abstraction_id: usize,
        source_hash: usize,
        abstract_op_id: usize,
        target_hash: usize,
        region: TransitionRegion,
    },
}

impl TransitionResidualCosts {
    pub fn from_operator_costs(costs: &[f64]) -> Self {
        let operator_buckets = costs
            .iter()
            .map(|&cost| {
                vec![ResidualBucket {
                    cost,
                    condition: TransitionCondition::Any,
                }]
            })
            .collect();
        Self { operator_buckets }
    }

    pub fn operator_costs_for_label_cp(&self) -> Vec<f64> {
        self.operator_buckets
            .iter()
            .map(|buckets| {
                let base_cost = buckets
                    .iter()
                    .filter(|bucket| bucket.condition == TransitionCondition::Any)
                    .map(|bucket| bucket.cost)
                    .fold(f64::INFINITY, f64::min);
                if !base_cost.is_finite() {
                    return f64::INFINITY;
                }
                let reduction_sum = buckets
                    .iter()
                    .filter(|bucket| bucket.condition != TransitionCondition::Any)
                    .map(|bucket| (base_cost - bucket.cost).max(0.0))
                    .sum::<f64>();
                (base_cost - reduction_sum).max(0.0)
            })
            .collect()
    }

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
        let Some(buckets) = self.operator_buckets.get(concrete_op_id) else {
            return f64::INFINITY;
        };
        let base_cost = buckets
            .iter()
            .filter(|bucket| bucket.condition == TransitionCondition::Any)
            .map(|bucket| bucket.cost)
            .fold(f64::INFINITY, f64::min);
        if !base_cost.is_finite() {
            return f64::INFINITY;
        }

        let mut reduction_sum = 0.0;
        for bucket in buckets {
            match &bucket.condition {
                TransitionCondition::Any => {}
                TransitionCondition::AbstractTransition {
                    abstraction_id,
                    source_hash: bucket_source_hash,
                    abstract_op_id: bucket_abstract_op_id,
                    target_hash: bucket_target_hash,
                    region,
                    ..
                } => {
                    let same_transition = *abstraction_id == current_abstraction_id
                        && *bucket_source_hash == source_hash
                        && *bucket_abstract_op_id == abstract_op_id
                        && *bucket_target_hash == target_hash;
                    if same_transition || region.overlaps_parts(source_region, target_region) {
                        reduction_sum += (base_cost - bucket.cost).max(0.0);
                    }
                }
            }
        }
        (base_cost - reduction_sum).max(0.0)
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
            self.operator_buckets.len() == saturated_costs.len(),
            "operator cost vector length mismatch: buckets={}, saturated={}",
            self.operator_buckets.len(),
            saturated_costs.len()
        );
        for (op_id, saturated) in saturated_costs.iter().copied().enumerate() {
            if !saturated.is_finite() || saturated <= EPSILON {
                continue;
            }
            for bucket in &mut self.operator_buckets[op_id] {
                bucket.cost = subtract_cost(bucket.cost, saturated).with_context(|| {
                    format!("uniform residual reduction underflow for operator {op_id}")
                })?;
            }
        }
        Ok(())
    }

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
            concrete_op_id < self.operator_buckets.len(),
            "concrete operator id out of bounds: {concrete_op_id}"
        );
        let exact = TransitionCondition::AbstractTransition {
            abstraction_id: producing_abstraction_id,
            source_hash,
            abstract_op_id,
            target_hash,
            region: region.clone(),
        };
        let buckets = &mut self.operator_buckets[concrete_op_id];
        if let Some(bucket) = buckets.iter_mut().find(|bucket| bucket.condition == exact) {
            bucket.cost = subtract_cost(bucket.cost, saturated)?;
            return Ok(());
        }

        let base_cost = buckets
            .iter()
            .filter(|bucket| bucket.condition == TransitionCondition::Any)
            .map(|bucket| bucket.cost)
            .fold(f64::INFINITY, f64::min);
        ensure!(
            base_cost.is_finite(),
            "no base residual cost for operator {concrete_op_id}"
        );
        buckets.push(ResidualBucket {
            cost: subtract_cost(base_cost, saturated)?,
            condition: exact,
        });
        Ok(())
    }
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
    fn overlapping_transition_reductions_accumulate_conservatively() {
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
            3.0
        );
        assert_eq!(residuals.operator_costs_for_label_cp(), vec![3.0]);
    }
}
