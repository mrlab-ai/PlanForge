use planforge_sas::state_registry::{ConcreteState, StateRegistry};

use super::PotentialTask;

#[derive(Debug, Clone)]
pub struct NumericPotentialFunction {
    fact_potentials: Vec<Vec<f64>>,
    numeric_potentials: Vec<(usize, f64)>,
    constant_potential: f64,
    conditioned_goal: Option<(usize, usize)>,
}

impl NumericPotentialFunction {
    pub(crate) fn new(
        fact_potentials: Vec<Vec<f64>>,
        numeric_potentials: Vec<f64>,
        constant_potential: f64,
        conditioned_goal: Option<(usize, usize)>,
    ) -> Self {
        Self {
            fact_potentials,
            numeric_potentials: numeric_potentials
                .into_iter()
                .enumerate()
                .filter(|(_, weight)| *weight != 0.0)
                .collect(),
            constant_potential,
            conditioned_goal,
        }
    }

    pub fn value(
        &self,
        state: &ConcreteState,
        registry: &StateRegistry<'_>,
        task: &PotentialTask,
        prop_scratch: &mut Vec<usize>,
        numeric_scratch: &mut Vec<f64>,
    ) -> Result<f64, String> {
        state.fill_state(registry, prop_scratch);
        if let Some((var, value)) = self.conditioned_goal
            && prop_scratch.get(var).copied() == Some(value)
        {
            return Ok(0.0);
        }
        let mut result = self.constant_potential;
        for (var_id, potentials) in self.fact_potentials.iter().enumerate() {
            let value = prop_scratch[var_id];
            result += potentials[value];
        }
        if !self.numeric_potentials.is_empty() {
            let feature_values = task.feature_values(state, registry, numeric_scratch)?;
            for &(feature_id, weight) in &self.numeric_potentials {
                result += weight * feature_values[feature_id];
            }
        }
        Ok(result)
    }

    pub fn numeric_delta_for_operator(&self, task: &PotentialTask, operator_id: usize) -> f64 {
        let operator = &task.operators[operator_id];
        self.numeric_potentials
            .iter()
            .map(|&(feature_id, weight)| weight * operator.numeric_delta(feature_id))
            .sum()
    }

    pub fn conditioned_goal(&self) -> Option<(usize, usize)> {
        self.conditioned_goal
    }

    pub(crate) fn evaluation_signature(&self, numeric_feature_count: usize) -> Vec<f64> {
        let mut signature = vec![self.constant_potential];
        for weights in &self.fact_potentials {
            signature.extend(weights);
        }
        let mut numeric = vec![0.0; numeric_feature_count];
        for &(feature_id, weight) in &self.numeric_potentials {
            numeric[feature_id] = weight;
        }
        signature.extend(numeric);
        signature
    }
}
