#[cfg(test)]
mod tests;

use std::cell::RefCell;

use crate::numeric::evaluation::evaluator::{EvaluationError, EvaluationState};
use crate::numeric::evaluation::heuristic::Heuristic;

use planners_sas::numeric::numeric_task::Operator;
use planners_sas::numeric::state_registry::{ConcreteState, StateRegistry};

use super::abstracted_task::DomainAbstractionTaskProjection;
use super::comparison_expression::{ComparisonTree, ComparisonTreeNode};
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
    projected_numeric_scratch: RefCell<Vec<f64>>,
    active_prop_vars: Vec<usize>,
    active_numeric_vars: Vec<usize>,
    comparison_tree_by_var: Vec<Option<usize>>,
    comparison_tree_required_lens: Vec<usize>,
}

enum NumericValues<'a> {
    Borrowed(&'a [f64]),
    Projected(std::cell::Ref<'a, [f64]>),
}

impl<'a> NumericValues<'a> {
    fn as_slice(&self) -> &[f64] {
        match self {
            Self::Borrowed(values) => values,
            Self::Projected(values) => values,
        }
    }
}

impl DomainAbstractionHeuristic {
    pub fn new(name: Option<String>, abstraction: DomainAbstraction) -> Self {
        let active_prop_vars: Vec<usize> = abstraction
            .factory
            .domain_sizes()
            .iter()
            .enumerate()
            .filter_map(|(var_id, &size)| (size > 1).then_some(var_id))
            .collect();
        let active_numeric_vars: Vec<usize> = abstraction
            .factory
            .numeric_domain_sizes()
            .iter()
            .enumerate()
            .filter_map(|(var_id, &size)| (size > 1).then_some(var_id))
            .collect();
        let mut comparison_tree_by_var = vec![None; abstraction.factory.domain_sizes().len()];
        for (tree_id, tree) in abstraction.factory.comparison_trees().iter().enumerate() {
            if tree.affected_var_id < comparison_tree_by_var.len() {
                comparison_tree_by_var[tree.affected_var_id] = Some(tree_id);
            }
        }
        let comparison_tree_required_lens: Vec<usize> = abstraction
            .factory
            .comparison_trees()
            .iter()
            .map(comparison_tree_numeric_len)
            .collect();
        Self {
            name: name.unwrap_or_else(|| "domain_abstraction".to_string()),
            abstraction,
            prop_scratch: RefCell::new(Vec::new()),
            numeric_scratch: RefCell::new(Vec::new()),
            projected_numeric_scratch: RefCell::new(Vec::new()),
            active_prop_vars,
            active_numeric_vars,
            comparison_tree_by_var,
            comparison_tree_required_lens,
        }
    }

    pub fn abstraction(&self) -> &DomainAbstraction {
        &self.abstraction
    }

    pub fn task_projection(&self) -> Option<&DomainAbstractionTaskProjection> {
        self.abstraction.task_projection.as_ref()
    }

    pub fn abstract_state_hash(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<usize, EvaluationError> {
        let (_, registry) = Self::require_task_and_registry(eval_state)?;
        self.compute_abstract_hash(eval_state.state(), registry)
    }

    pub fn abstract_state_hash_from_state_values(
        &self,
        prop_values: &[usize],
        numeric_values: &[f64],
    ) -> Result<usize, EvaluationError> {
        self.compute_abstract_hash_from_state_values(prop_values, numeric_values, None)
    }

    pub fn abstract_state_hash_from_state_values_with_comparisons(
        &self,
        prop_values: &[usize],
        numeric_values: &[f64],
        comparison_values: &[Option<usize>],
    ) -> Result<usize, EvaluationError> {
        self.compute_abstract_hash_from_state_values(
            prop_values,
            numeric_values,
            Some(comparison_values),
        )
    }

    pub fn abstract_state_hash_from_projected_state_values_with_comparisons(
        &self,
        prop_values: &[usize],
        projected_numeric_values: &[f64],
        comparison_values: &[Option<usize>],
    ) -> Result<usize, EvaluationError> {
        self.compute_abstract_hash_from_projected_state_values(
            prop_values,
            projected_numeric_values,
            Some(comparison_values),
        )
    }

    pub fn fill_comparison_values_from_state_values(
        &self,
        numeric: &[f64],
        out: &mut Vec<Option<usize>>,
    ) -> Result<(), EvaluationError> {
        let numeric_values = self.project_numeric_values(numeric)?;
        let numeric_values = numeric_values.as_slice();
        self.fill_comparison_values_from_projected_state_values(numeric_values, out)
    }

    pub fn fill_comparison_values_from_projected_state_values(
        &self,
        numeric_values: &[f64],
        out: &mut Vec<Option<usize>>,
    ) -> Result<(), EvaluationError> {
        if out.len() < self.comparison_tree_by_var.len() {
            out.resize(self.comparison_tree_by_var.len(), None);
        }
        for (tree_id, tree) in self
            .abstraction
            .factory
            .comparison_trees()
            .iter()
            .enumerate()
        {
            let value = if evaluate_comparison_tree_on_concrete_numeric_state(
                tree,
                numeric_values,
                self.comparison_tree_required_lens[tree_id],
            )? {
                COMPARISON_TRUE_VAL
            } else {
                COMPARISON_FALSE_VAL
            };
            if tree.affected_var_id >= out.len() {
                out.resize(tree.affected_var_id + 1, None);
            }
            out[tree.affected_var_id] = Some(value);
        }
        Ok(())
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
        self.compute_abstract_hash_from_state_values(&prop, &numeric, None)
    }

    fn compute_abstract_hash_from_state_values(
        &self,
        prop_values: &[usize],
        numeric: &[f64],
        comparison_values: Option<&[Option<usize>]>,
    ) -> Result<usize, EvaluationError> {
        let num_props = self.abstraction.factory.domain_sizes().len();

        if prop_values.len() < num_props {
            return Err(EvaluationError::InvalidState(format!(
                "propositional state too short: {} < {num_props}",
                prop_values.len()
            )));
        }

        let numeric_values = self.project_numeric_values(numeric)?;
        let numeric_values = numeric_values.as_slice();
        self.compute_abstract_hash_from_projected_state_values(
            prop_values,
            numeric_values,
            comparison_values,
        )
    }

    fn compute_abstract_hash_from_projected_state_values(
        &self,
        prop_values: &[usize],
        numeric_values: &[f64],
        comparison_values: Option<&[Option<usize>]>,
    ) -> Result<usize, EvaluationError> {
        let num_props = self.abstraction.factory.domain_sizes().len();
        let num_numeric = self.abstraction.factory.numeric_domain_sizes().len();

        if prop_values.len() < num_props {
            return Err(EvaluationError::InvalidState(format!(
                "propositional state too short: {} < {num_props}",
                prop_values.len()
            )));
        }
        if numeric_values.len() < num_numeric {
            return Err(EvaluationError::InvalidState(format!(
                "numeric state too short: {} < {num_numeric}",
                numeric_values.len()
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

        for &num_var_id in &self.active_numeric_vars {
            let val = numeric_values[num_var_id];
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
        for &var in &self.active_prop_vars {
            let concrete_val = resolved_propositional_value(
                var,
                prop_values[var],
                numeric_values,
                self.abstraction.factory.comparison_trees(),
                &self.comparison_tree_by_var,
                &self.comparison_tree_required_lens,
                comparison_values,
            )?;
            let abs_val = abstract_propositional_value(var, concrete_val, mapping)?;
            prop_index += multipliers[var] * abs_val;
        }

        Ok(index + prop_index)
    }

    fn project_numeric_values<'a>(
        &'a self,
        numeric: &'a [f64],
    ) -> Result<NumericValues<'a>, EvaluationError> {
        if let Some(projection) = self.abstraction.task_projection.as_ref() {
            {
                let mut projected_numeric = self.projected_numeric_scratch.borrow_mut();
                projection
                    .project_numeric_values_into(numeric, &mut projected_numeric)
                    .map_err(|e| {
                        EvaluationError::ComputationFailed(format!(
                            "failed to project state into abstracted domain task: {e:#}"
                        ))
                    })?;
            }
            Ok(NumericValues::Projected(std::cell::Ref::map(
                self.projected_numeric_scratch.borrow(),
                |values| values.as_slice(),
            )))
        } else {
            Ok(NumericValues::Borrowed(numeric))
        }
    }
}

fn resolved_propositional_value(
    var: usize,
    stored_val: usize,
    numeric: &[f64],
    comparison_trees: &[ComparisonTree],
    comparison_tree_by_var: &[Option<usize>],
    comparison_tree_required_lens: &[usize],
    comparison_values: Option<&[Option<usize>]>,
) -> Result<usize, EvaluationError> {
    if let Some(value) = comparison_values
        .and_then(|values| values.get(var))
        .copied()
        .flatten()
    {
        return Ok(value);
    }
    let Some(tree_id) = comparison_tree_by_var.get(var).copied().flatten() else {
        return Ok(stored_val);
    };
    let tree = &comparison_trees[tree_id];

    let eval = evaluate_comparison_tree_on_concrete_numeric_state(
        tree,
        numeric,
        comparison_tree_required_lens[tree_id],
    )?;
    Ok(if eval {
        COMPARISON_TRUE_VAL
    } else {
        COMPARISON_FALSE_VAL
    })
}

fn evaluate_comparison_tree_on_concrete_numeric_state(
    tree: &ComparisonTree,
    numeric: &[f64],
    required_len: usize,
) -> Result<bool, EvaluationError> {
    if numeric.len() < required_len {
        return Err(EvaluationError::InvalidState(format!(
            "numeric state too short for comparison tree on var {}: {} < {}",
            tree.affected_var_id,
            numeric.len(),
            required_len
        )));
    }

    Ok(tree.evaluate_point(numeric))
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
