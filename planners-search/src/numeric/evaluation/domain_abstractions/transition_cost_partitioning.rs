use anyhow::{Context, Result, ensure};
use planners_sas::numeric::numeric_task::ExplicitFact;

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
pub struct AbstractTransitionSystem {
    pub transitions: Vec<AbstractTransition>,
    pub backward: Vec<Vec<usize>>,
    pub forward: Vec<Vec<usize>>,
    pub goal_facts: Vec<ExplicitFact>,
    pub goal_state_hashes: Vec<usize>,
    pub initial_state_hash: usize,
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum TransitionCondition {
    Any,
    AbstractTransition {
        abstraction_id: usize,
        source_hash: usize,
        abstract_op_id: usize,
        target_hash: usize,
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
                buckets
                    .iter()
                    .map(|bucket| bucket.cost)
                    .fold(f64::INFINITY, f64::min)
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
    ) -> f64 {
        let Some(buckets) = self.operator_buckets.get(concrete_op_id) else {
            return f64::INFINITY;
        };
        let exact = TransitionCondition::AbstractTransition {
            abstraction_id: current_abstraction_id,
            source_hash,
            abstract_op_id,
            target_hash,
        };
        let mut any_cost = f64::INFINITY;
        let mut exact_cost = f64::INFINITY;
        let mut global_min = f64::INFINITY;
        for bucket in buckets {
            global_min = global_min.min(bucket.cost);
            match &bucket.condition {
                TransitionCondition::Any => {
                    any_cost = any_cost.min(bucket.cost);
                }
                condition if *condition == exact => {
                    exact_cost = exact_cost.min(bucket.cost);
                }
                _ => {}
            }
        }
        let has_foreign_specific_bucket = buckets.iter().any(|bucket| {
            matches!(
                bucket.condition,
                TransitionCondition::AbstractTransition { abstraction_id, .. }
                    if abstraction_id != current_abstraction_id
            )
        });
        if has_foreign_specific_bucket {
            return exact_cost.min(global_min);
        }
        exact_cost.min(any_cost)
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
                self.reduce_exact_transition(
                    concrete_op_id,
                    producing_abstraction_id,
                    transition.source_hash,
                    transition.abstract_op_id,
                    transition.target_hash,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_transition_reduction_does_not_reduce_other_transitions() {
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
        };
        let tcf = AbstractTransitionCostFunction {
            transition_costs: vec![2.0],
        };

        residuals
            .reduce_by_tcf(0, &transition_system, &tcf)
            .unwrap();

        assert_eq!(residuals.cost_for_transition(0, 0, 3, 7, 4), 3.0);
        assert_eq!(residuals.cost_for_transition(0, 0, 3, 7, 5), 5.0);
        assert_eq!(residuals.cost_for_transition(0, 1, 3, 7, 4), 3.0);
    }

    #[test]
    fn repeated_exact_transition_reduction_clamps_tiny_negative_to_zero() {
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

        assert_eq!(residuals.cost_for_transition(0, 0, 0, 0, 1), 0.0);
    }

    #[test]
    fn foreign_abstraction_uses_global_minimum_when_disjointness_is_unknown() {
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

        assert_eq!(residuals.cost_for_transition(0, 0, 9, 7, 4), 5.0);
        assert_eq!(residuals.cost_for_transition(0, 1, 9, 7, 4), 3.0);
    }
}
