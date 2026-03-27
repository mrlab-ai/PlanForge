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
        //if eval_state.is_goal() {
        //    return Ok(0.0);
        //}

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

        // Debugging
        if std::env::var("DA_TRACE_STATE_EVAL").unwrap_or_else(|_| "0".to_string()) == "1" {
            let num_props = self.abstraction.factory.domain_sizes().len();
            let mut prop = vec![];
            state.fill_state(registry, &mut prop);
            let mut num = vec![];
            registry.fill_numeric_vars(state, &mut num).unwrap();

            let abstract_prop_sizes = self.abstraction.factory.domain_sizes();
            let abstract_num_sizes = self.abstraction.factory.numeric_domain_sizes();
            let multipliers = &self.abstraction.hash_multipliers;
            let partitions = self.abstraction.factory.partitions();

            let mut prop_str_vec = vec![];
            for (var, val) in prop.iter().enumerate() {
                prop_str_vec.push(format!("v{}={}", var, val));
            }
            let prop_str = prop_str_vec.join(" ");

            let mut abs_prop_str = vec![];
            for var in 0..num_props {
                let dom = abstract_prop_sizes[var];
                let mult = multipliers[var] as i64;
                let val = (((hash as i64) / mult) % (dom as i64)) as i32;
                abs_prop_str.push(format!("v{}={}", var, val));
            }

            let mut num_str_vec = vec![];
            let mut abs_num_str = vec![];

            for (num_id, &dom) in abstract_num_sizes.iter().enumerate() {
                let nv = &_task.numeric_variables()[num_id];
                if matches!(
                    nv.get_type(),
                    planners_sas::numeric::numeric_task::NumericType::Constant
                        | planners_sas::numeric::numeric_task::NumericType::Derived
                ) {
                    continue;
                }

                let abs_var = num_props + num_id;
                let mult = multipliers[abs_var] as i64;
                let part = (((hash as i64) / mult) % (dom as i64)) as usize;

                let is_inf = partitions
                    .partition_interval(num_id, part)
                    .map_or(false, |iv| {
                        iv.lower == f64::NEG_INFINITY && iv.upper == f64::INFINITY
                    });

                if is_inf {
                    continue;
                }

                let iv_str = partitions
                    .partition_interval(num_id, part)
                    .map(|iv| {
                        let left = if iv.lower_closed { '[' } else { '(' };
                        let right = if iv.upper_closed { ']' } else { ')' };
                        let l_str = if iv.lower == f64::NEG_INFINITY {
                            "-inf".to_string()
                        } else {
                            iv.lower.to_string()
                        };
                        let r_str = if iv.upper == f64::INFINITY {
                            "inf".to_string()
                        } else {
                            iv.upper.to_string()
                        };
                        format!("{}{}, {}{}", left, l_str, r_str, right)
                    })
                    .unwrap_or_else(|| "<invalid>".to_string());

                num_str_vec.push(format!("n{}={}", num_id, num[num_id]));
                abs_num_str.push(format!("n{}={}", num_id, iv_str));
            }

            println!("[Evaluate State]");
            println!("  concrete props: {}", prop_str);
            println!("  concrete nums:  {}", num_str_vec.join(" "));
            println!("  abstract props: {}", abs_prop_str.join(" "));
            println!("  abstract nums:  {}", abs_num_str.join(" "));
            println!("  distance:       {}", dist);
        }
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
