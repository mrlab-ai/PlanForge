#[cfg(test)]
mod tests;

use std::cell::RefCell;

use crate::numeric::evaluation::evaluator::{EvaluationError, EvaluationState};
use crate::numeric::evaluation::heuristic::Heuristic;

use planners_sas::numeric::numeric_task::{AbstractNumericTask, NumericType, Operator};
use planners_sas::numeric::state_registry::{ConcreteState, StateRegistry};

use super::comparison_expression::Interval;
use super::domain_abstraction_generator::DomainAbstraction;
use super::numeric_context::{
    fill_derived_numeric_intervals_from_comparison_trees, seed_numeric_intervals_from_initial_state,
};
use super::utils;

const COMPARISON_TRUE_VAL: usize = 0;
const COMPARISON_FALSE_VAL: usize = 1;
const COMPARISON_UNKNOWN_VAL: usize = 2;

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
        task: &'t dyn AbstractNumericTask,
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
            let part = utils::partition_for_value(parts, val).ok_or_else(|| {
                EvaluationError::InvalidState(format!(
                    "numeric value {val} not contained in any partition for var {num_var_id}"
                ))
            })?;
            let abs_var = num_props + num_var_id;
            index += i64::from(multipliers[abs_var]) * i64::from(part);
        }

        let mut prop_index: i64 = 0;
        for var in 0..num_props {
            let abs_val = abstract_propositional_value(var, prop[var], mapping)?;
            prop_index += i64::from(multipliers[var]) * i64::from(abs_val);
        }

        i32::try_from(index + prop_index).map_err(|_| {
            EvaluationError::InvalidState("abstract hash does not fit i32".to_string())
        })
    }
}

fn plant_watering_real_distance_check(
    task: &dyn AbstractNumericTask,
    numeric_values: &[f64],
    heuristic_value: f64,
) -> Option<(f64, bool, String)> {
    const X_AGENT_IDX: usize = 15;
    const Y_AGENT_IDX: usize = 16;
    const CARRYING_IDX: usize = 29;
    const POURED_IDX: usize = 30;
    const TAP_X: f64 = 3.0;
    const TAP_Y: f64 = 3.0;
    const PLANT_X: f64 = 5.0;
    const PLANT_Y: f64 = 5.0;
    const GOAL_POURED: f64 = 10.0;
    const EPS: f64 = 1e-9;

    let vars = task.numeric_variables();
    if vars.len() <= POURED_IDX {
        return None;
    }
    if vars[X_AGENT_IDX].name() != "PNE x(agent1)"
        || vars[Y_AGENT_IDX].name() != "PNE y(agent1)"
        || vars[CARRYING_IDX].name() != "PNE carrying()"
        || vars[POURED_IDX].name() != "PNE poured(plant1)"
    {
        return None;
    }
    if numeric_values.len() <= POURED_IDX {
        return None;
    }

    let x = numeric_values[X_AGENT_IDX];
    let y = numeric_values[Y_AGENT_IDX];
    let carrying = numeric_values[CARRYING_IDX];
    let poured = numeric_values[POURED_IDX];

    if !x.is_finite() || !y.is_finite() || !carrying.is_finite() || !poured.is_finite() {
        return None;
    }

    let remaining = (GOAL_POURED - poured).max(0.0);
    let dist_to_tap = (x - TAP_X).abs().max((y - TAP_Y).abs());
    let dist_tap_to_plant = (TAP_X - PLANT_X).abs().max((TAP_Y - PLANT_Y).abs());
    let dist_to_plant = (x - PLANT_X).abs().max((y - PLANT_Y).abs());

    let real_distance = if carrying + EPS >= remaining {
        dist_to_plant + remaining
    } else {
        dist_to_tap + (remaining - carrying) + dist_tap_to_plant + remaining
    };
    let admissible = heuristic_value <= real_distance + EPS;
    let details = format!(
        "x={} y={} carrying={} poured={} remaining={} h={} h*={}",
        x, y, carrying, poured, remaining, heuristic_value, real_distance
    );
    Some((real_distance, admissible, details))
}

fn abstract_propositional_value(
    var: usize,
    concrete_val: i32,
    mapping: &[Vec<i32>],
) -> Result<i32, EvaluationError> {
    let cidx = usize::try_from(concrete_val).map_err(|_| {
        EvaluationError::InvalidState(format!(
            "invalid propositional value {concrete_val} for var {var}"
        ))
    })?;
    mapping
        .get(var)
        .and_then(|m| m.get(cidx))
        .copied()
        .ok_or_else(|| {
            EvaluationError::InvalidState(format!(
                "missing domain mapping for var {var} value index {cidx}"
            ))
        })
}

impl Heuristic for DomainAbstractionHeuristic {
    fn compute_heuristic(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError> {
        //if eval_state.is_goal() {
        //    return Ok(0.0);
        //}

        let (task, registry) = Self::require_task_and_registry(eval_state)?;
        let state = eval_state.state();

        let hash = self.compute_abstract_hash(task, state, registry)?;
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
            let domain_mapping = self.abstraction.factory.domain_mapping();
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
                if abstract_prop_sizes.get(var).copied().unwrap_or(0) <= 1
                    || domain_mapping
                        .get(var)
                        .is_some_and(|entry| entry.is_empty())
                {
                    continue;
                }
                prop_str_vec.push(format!("v{}={}", var, val));
            }
            let prop_str = prop_str_vec.join(" ");

            let mut abs_prop_str = vec![];
            for var in 0..num_props {
                let dom = abstract_prop_sizes[var];
                if dom <= 1
                    || domain_mapping
                        .get(var)
                        .is_some_and(|entry| entry.is_empty())
                {
                    continue;
                }
                let mult = multipliers[var] as i64;
                let val = (((hash as i64) / mult) % (dom as i64)) as i32;
                abs_prop_str.push(format!("v{}={}", var, val));
            }

            let mut num_str_vec = vec![];
            let mut abs_num_str = vec![];

            for (num_id, &dom) in abstract_num_sizes.iter().enumerate() {
                let nv = &task.numeric_variables()[num_id];
                if dom <= 1 {
                    continue;
                }
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

            utils::debug_print_evaluate_state(
                &prop_str,
                &num_str_vec,
                &abs_prop_str,
                &abs_num_str,
                dist,
            );

            if std::env::var("DA_TRACE_REAL_DISTANCE_CHECK").unwrap_or_else(|_| "0".to_string())
                == "1"
            {
                if let Some((real_distance, admissible, details)) =
                    plant_watering_real_distance_check(task, &num, dist)
                {
                    println!(
                        "  real-check:     admissible={} h*={} {}",
                        admissible, real_distance, details
                    );
                    if !admissible {
                        eprintln!("[DA_INADMISSIBLE] {}", details);
                    }
                }
            }

            if std::env::var("DA_TRACE_COMPARISON_MISMATCH").unwrap_or_else(|_| "0".to_string())
                == "1"
            {
                let mut abstract_intervals = seed_numeric_intervals_from_initial_state(task);
                for num_var_id in 0..task.numeric_variables().len() {
                    if task.numeric_variables()[num_var_id].get_type() != &NumericType::Regular {
                        continue;
                    }
                    let val = num[num_var_id];
                    let Some(parts) = partitions.partitions(num_var_id) else {
                        continue;
                    };
                    let Some(part) = utils::partition_for_value(parts, val) else {
                        continue;
                    };
                    if let Some(iv) = partitions.partition_interval(num_var_id, part as usize) {
                        abstract_intervals[num_var_id] = iv;
                    }
                }
                fill_derived_numeric_intervals_from_comparison_trees(
                    self.abstraction.factory.comparison_trees(),
                    &mut abstract_intervals,
                );

                for tree in self.abstraction.factory.comparison_trees() {
                    let Ok(var_id) = usize::try_from(tree.affected_var_id) else {
                        continue;
                    };
                    if var_id >= prop.len() {
                        continue;
                    }
                    let point_value = if tree.evaluate_point(&num) { 0 } else { 1 };
                    let interval_value = match tree.evaluate_interval(&abstract_intervals) {
                        Some(true) => 0,
                        Some(false) => 1,
                        None => 2,
                    };
                    let hash_value =
                        abs_prop_str_value(var_id, abstract_prop_sizes, multipliers, hash);
                    if point_value != prop[var_id]
                        || interval_value != hash_value
                        || point_value != interval_value
                    {
                        println!(
                            "  cmp mismatch: v{var_id} concrete={} point={} interval={} hash={}",
                            prop[var_id], point_value, interval_value, hash_value
                        );
                    }
                }
            }
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

fn abs_prop_str_value(
    var: usize,
    abstract_prop_sizes: &[i32],
    multipliers: &[i32],
    hash: i32,
) -> i32 {
    let dom = abstract_prop_sizes[var];
    let mult = multipliers[var] as i64;
    (((hash as i64) / mult) % (dom as i64)) as i32
}
