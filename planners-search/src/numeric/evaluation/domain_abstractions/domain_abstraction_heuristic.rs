use std::cell::RefCell;

use crate::numeric::evaluation::evaluator::{EvaluationError, EvaluationState};
use crate::numeric::evaluation::heuristic::Heuristic;

use planners_sas::numeric::numeric_task::Operator;
use planners_sas::numeric::state_registry::{ConcreteState, StateRegistry};

use super::comparison_expression::Interval;
use super::domain_abstraction_generator::DomainAbstraction;

/// Heuristic that evaluates a concrete state by mapping it to an abstract state
/// and looking up its precomputed goal distance.
#[derive(Debug, Clone)]
pub struct DomainAbstractionHeuristic {
    name: String,
    abstraction: DomainAbstraction,
    prop_scratch: RefCell<Vec<i32>>,
    numeric_scratch: RefCell<Vec<f64>>,
}

impl DomainAbstractionHeuristic {
    pub fn new(name: Option<String>, abstraction: DomainAbstraction) -> Self {
        Self {
            name: name.unwrap_or_else(|| "domain_abstraction".to_string()),
            abstraction,
            prop_scratch: RefCell::new(Vec::new()),
            numeric_scratch: RefCell::new(Vec::new()),
        }
    }

    pub fn abstraction(&self) -> &DomainAbstraction {
        &self.abstraction
    }

    fn require_task_and_registry<'s, 't>(
        eval_state: &'s EvaluationState<'s, 't>,
    ) -> Result<
        (
            &'t dyn planners_sas::numeric::numeric_task::AbstractNumericTask,
            &'s StateRegistry<'t>,
        ),
        EvaluationError,
    > {
        let task = eval_state.task().ok_or_else(|| {
            EvaluationError::InvalidState(
                "DomainAbstractionHeuristic requires task in EvaluationState".to_string(),
            )
        })?;
        let registry = eval_state.state_registry().ok_or_else(|| {
            EvaluationError::InvalidState(
                "DomainAbstractionHeuristic requires StateRegistry in EvaluationState".to_string(),
            )
        })?;
        Ok((task, registry))
    }

    fn compute_abstract_hash<'s, 't>(
        &self,
        state: &ConcreteState,
        registry: &'s StateRegistry<'t>,
    ) -> Result<i32, EvaluationError> {
        let num_props = self.abstraction.factory.domain_sizes().len();
        let num_numeric = self.abstraction.factory.numeric_domain_sizes().len();

        let mut prop = self.prop_scratch.borrow_mut();
        state.fill_state(registry, &mut prop);
        if prop.len() < num_props {
            return Err(EvaluationError::InvalidState(format!(
                "propositional state too short: {} < {num_props}",
                prop.len()
            )));
        }

        let mut numeric = self.numeric_scratch.borrow_mut();
        registry
            .fill_numeric_vars(state, &mut numeric)
            .map_err(|e| {
                EvaluationError::ComputationFailed(format!("failed to read numeric vars: {e:?}"))
            })?;
        if numeric.len() < num_numeric {
            return Err(EvaluationError::InvalidState(format!(
                "numeric state too short: {} < {num_numeric}",
                numeric.len()
            )));
        }

        let mapping = self.abstraction.factory.domain_mapping();
        let partitions = self.abstraction.factory.partitions();
        let multipliers = &self.abstraction.hash_multipliers;

        if multipliers.len() != num_props + num_numeric {
            return Err(EvaluationError::InvalidState(
                "hash multipliers length mismatch".to_string(),
            ));
        }

        let mut index: i64 = 0;

        for var in 0..num_props {
            let concrete_val = prop[var];
            let cidx = usize::try_from(concrete_val).map_err(|_| {
                EvaluationError::InvalidState(format!(
                    "invalid propositional value {concrete_val} for var {var}"
                ))
            })?;
            let abs_val = *mapping.get(var).and_then(|m| m.get(cidx)).ok_or_else(|| {
                EvaluationError::InvalidState(format!(
                    "missing domain mapping for var {var} value index {cidx}"
                ))
            })?;
            index += i64::from(multipliers[var]) * i64::from(abs_val);
        }

        for num_var_id in 0..num_numeric {
            let val = numeric[num_var_id];
            if !val.is_finite() || val.is_nan() {
                return Err(EvaluationError::InvalidState(format!(
                    "numeric value for var {num_var_id} must be finite, got {val}"
                )));
            }
            let parts = partitions.partitions(num_var_id).ok_or_else(|| {
                EvaluationError::InvalidState(format!(
                    "missing partitions for numeric var {num_var_id}"
                ))
            })?;
            let part = partition_for_value(parts, val).ok_or_else(|| {
                EvaluationError::InvalidState(format!(
                    "numeric value {val} not contained in any partition for var {num_var_id}"
                ))
            })?;
            let abs_var = num_props + num_var_id;
            index += i64::from(multipliers[abs_var]) * i64::from(part);
        }

        i32::try_from(index).map_err(|_| {
            EvaluationError::InvalidState("abstract hash does not fit i32".to_string())
        })
    }
}

impl Heuristic for DomainAbstractionHeuristic {
    fn compute_heuristic(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError> {
        if eval_state.is_goal() {
            return Ok(0.0);
        }

        let (_task, registry) = Self::require_task_and_registry(eval_state)?;
        let state = eval_state.state();

        let hash = self.compute_abstract_hash(state, registry)?;
        let idx = usize::try_from(hash).map_err(|_| {
            EvaluationError::InvalidState(format!("abstract hash negative: {hash}"))
        })?;
        let dist = self
            .abstraction
            .distance_table
            .distances
            .get(idx)
            .copied()
            .ok_or_else(|| {
                EvaluationError::InvalidState(format!(
                    "abstract hash out of bounds: {hash} (len={})",
                    self.abstraction.distance_table.distances.len()
                ))
            })?;
        Ok(dist)
    }

    fn heuristic_name(&self) -> String {
        self.name.clone()
    }

    fn reach_state(
        &mut self,
        _parent_state: &ConcreteState,
        _operator: &Operator,
        _state: &ConcreteState,
    ) -> bool {
        true
    }
}

fn partition_for_value(partitions: &[Interval], value: f64) -> Option<i32> {
    partitions
        .iter()
        .position(|iv| iv.contains(value))
        .and_then(|i| i32::try_from(i).ok())
}
