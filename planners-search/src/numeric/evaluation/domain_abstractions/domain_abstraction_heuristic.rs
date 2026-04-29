#[cfg(test)]
mod tests;

use std::cell::RefCell;

use crate::numeric::evaluation::evaluator::{EvaluationError, EvaluationState};
use crate::numeric::evaluation::heuristic::Heuristic;

use planners_sas::numeric::numeric_task::Operator;
use planners_sas::numeric::state_registry::{ConcreteState, StateRegistry};

use super::comparison_expression::{ComparisonTree, ComparisonTreeNode, Interval};
use super::domain_abstraction_generator::DomainAbstraction;
use super::utils;

pub(crate) const COMPARISON_TRUE_VAL: usize = 0;
pub(crate) const COMPARISON_FALSE_VAL: usize = 1;
pub(crate) const COMPARISON_UNKNOWN_VAL: usize = 2;

/// Heuristic that evaluates a concrete state by mapping it to an abstract state
/// and looking up its precomputed goal distance.
#[derive(Debug, Clone)]
pub struct DomainAbstractionHeuristic {
    name: String,
    abstraction: DomainAbstraction,
    prop_scratch: RefCell<Vec<usize>>,
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

    pub fn abstract_state_hash(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<usize, EvaluationError> {
        let (_, registry) = Self::require_task_and_registry(eval_state)?;
        self.compute_abstract_hash(eval_state.state(), registry)
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

    fn compute_abstract_hash<'t>(
        &self,
        state: &ConcreteState,
        registry: &StateRegistry<'t>,
    ) -> Result<usize, EvaluationError> {
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

        let mut index: usize = 0;

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
            index += multipliers[abs_var] * part;
        }

        let mut prop_index: usize = 0;
        for var in 0..num_props {
            let concrete_val = resolved_propositional_value(
                var,
                prop[var],
                &numeric,
                self.abstraction.factory.comparison_trees(),
            )?;
            let abs_val = abstract_propositional_value(var, concrete_val, mapping)?;
            prop_index += multipliers[var] * abs_val;
        }

        Ok(index + prop_index)
    }
}

fn resolved_propositional_value(
    var: usize,
    stored_val: usize,
    numeric: &[f64],
    comparison_trees: &[ComparisonTree],
) -> Result<usize, EvaluationError> {
    let Some(tree) = comparison_trees
        .iter()
        .find(|tree| tree.affected_var_id == var)
    else {
        return Ok(stored_val);
    };

    let eval = evaluate_comparison_tree_on_concrete_numeric_state(tree, numeric)?;
    Ok(match eval {
        Some(true) => COMPARISON_TRUE_VAL,
        Some(false) => COMPARISON_FALSE_VAL,
        None => stored_val.min(COMPARISON_UNKNOWN_VAL),
    })
}

fn evaluate_comparison_tree_on_concrete_numeric_state(
    tree: &ComparisonTree,
    numeric: &[f64],
) -> Result<Option<bool>, EvaluationError> {
    let required_len = comparison_tree_numeric_len(tree);
    if numeric.len() < required_len {
        return Err(EvaluationError::InvalidState(format!(
            "numeric state too short for comparison tree on var {}: {} < {}",
            tree.affected_var_id,
            numeric.len(),
            required_len
        )));
    }

    let mut intervals: Vec<Interval> = numeric
        .iter()
        .map(|&value| Interval::singleton(value))
        .collect();
    Ok(tree.evaluate_interval_and_fill(&mut intervals))
}

fn comparison_tree_numeric_len(tree: &ComparisonTree) -> usize {
    let mut max_numeric_var_id = tree.left_numeric_var_id.max(tree.right_numeric_var_id);
    for node in &tree.nodes {
        match node {
            ComparisonTreeNode::Leaf { numeric_var_id } => {
                max_numeric_var_id = max_numeric_var_id.max(*numeric_var_id);
            }
            ComparisonTreeNode::Arith {
                result_numeric_var_id,
                left_numeric_var_id,
                right_numeric_var_id,
                ..
            } => {
                max_numeric_var_id = max_numeric_var_id
                    .max(*result_numeric_var_id)
                    .max(*left_numeric_var_id)
                    .max(*right_numeric_var_id);
            }
        }
    }
    max_numeric_var_id + 1
}

fn abstract_propositional_value(
    var: usize,
    concrete_val: usize,
    mapping: &[Vec<usize>],
) -> Result<usize, EvaluationError> {
    mapping
        .get(var)
        .and_then(|m| m.get(concrete_val))
        .copied()
        .ok_or_else(|| {
            EvaluationError::InvalidState(format!(
                "missing domain mapping for var {var} value index {concrete_val}"
            ))
        })
}

impl Heuristic for DomainAbstractionHeuristic {
    fn compute_heuristic(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError> {
        // NOTE: I have no idea why I commented that out... Is there a reason?
        //if eval_state.is_goal() {
        //    return Ok(0.0);
        //}

        let (_task, registry) = Self::require_task_and_registry(eval_state)?;
        let state = eval_state.state();

        let hash = self.compute_abstract_hash(state, registry)?;
        let dist = self
            .abstraction
            .distance_table
            .distances
            .get(hash)
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
