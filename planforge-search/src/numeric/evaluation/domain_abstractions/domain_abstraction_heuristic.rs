#[cfg(test)]
mod tests;

use std::cell::RefCell;
use std::sync::OnceLock;

use crate::numeric::evaluation::evaluator::{EvaluationError, EvaluationState};
use crate::numeric::evaluation::heuristic::Heuristic;

use planforge_sas::numeric::numeric_task::Operator;
use planforge_sas::numeric::state_registry::{ConcreteState, StateRegistry};

use super::comparison_expression::{ComparisonTree, ComparisonTreeNode};
use super::domain_abstraction_generator::DomainAbstraction;
use super::utils;

pub(crate) const COMPARISON_TRUE_VAL: usize = 0;
pub(crate) const COMPARISON_FALSE_VAL: usize = 1;
pub(crate) const COMPARISON_UNKNOWN_VAL: usize = 2;

fn fast_hash_enabled() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| std::env::var_os("DA_NO_FAST_HASH").is_none())
}

#[derive(Debug, Clone)]
pub(crate) struct DomainAbstractionLookupScratch {
    pub(crate) prop: Vec<usize>,
    pub(crate) numeric: Vec<f64>,
    pub(crate) comparisons: Vec<Option<usize>>,
    pub(crate) required_domain_ids: Vec<usize>,
    pub(crate) abstract_state_ids: Vec<Option<usize>>,
    pub(crate) abstraction_value_cache: Vec<Option<f64>>,
}

impl DomainAbstractionLookupScratch {
    pub(crate) fn new() -> Self {
        Self {
            prop: Vec::new(),
            numeric: Vec::new(),
            comparisons: Vec::new(),
            required_domain_ids: Vec::new(),
            abstract_state_ids: Vec::new(),
            abstraction_value_cache: Vec::new(),
        }
    }
}

pub(crate) fn compute_collection_abstract_state_ids(
    heuristics: &[DomainAbstractionHeuristic],
    eval_state: &EvaluationState<'_, '_>,
    required_ids: Option<&[usize]>,
    scratch: &mut DomainAbstractionLookupScratch,
) -> Result<(), EvaluationError> {
    let state = eval_state.state();
    let registry = eval_state.state_registry().ok_or_else(|| {
        EvaluationError::InvalidState(
            "domain abstraction lookup requires state registry".to_string(),
        )
    })?;
    state.fill_state(registry, &mut scratch.prop);
    registry
        .fill_numeric_vars(state, &mut scratch.numeric)
        .map_err(|err| {
            EvaluationError::ComputationFailed(format!("failed to read numeric state: {err:?}"))
        })?;

    let num_domain = heuristics.len();
    let needs_all = required_ids.is_none();
    scratch.required_domain_ids.clear();
    if needs_all {
        scratch.required_domain_ids.extend(0..num_domain);
    } else {
        scratch.required_domain_ids.extend(
            required_ids
                .unwrap_or(&[])
                .iter()
                .copied()
                .filter(|&id| id < num_domain),
        );
    }

    let has_required_domains = !scratch.required_domain_ids.is_empty();
    // The registry's axiom evaluator has already materialized every comparison
    // axiom's truth value into `scratch.prop[affected_var_id]`. In that case
    // we can read those bits directly and skip the per-state `ComparisonTree`
    // walks entirely. `DA_NO_FAST_HASH=1` disables this for A/B benchmarking.
    let prop_has_resolved_comparisons = has_required_domains && fast_hash_enabled();

    scratch.comparisons.clear();
    if !prop_has_resolved_comparisons && scratch.required_domain_ids.len() > 1 {
        if let Some(&first_id) = scratch.required_domain_ids.first() {
            heuristics[first_id].fill_comparison_values_from_projected_state_values(
                &scratch.numeric,
                &mut scratch.comparisons,
            )?;
        }
    }

    scratch.abstract_state_ids.clear();
    scratch.abstract_state_ids.resize(num_domain, None);
    if needs_all {
        for (id, heuristic) in heuristics.iter().enumerate() {
            scratch.abstract_state_ids[id] = Some(hash_with_shared_values(
                heuristic,
                &scratch.prop,
                &scratch.numeric,
                &scratch.comparisons,
                prop_has_resolved_comparisons,
            )?);
        }
    } else if let Some(required_ids) = required_ids {
        for &id in required_ids {
            let Some(heuristic) = heuristics.get(id) else {
                continue;
            };
            scratch.abstract_state_ids[id] = Some(hash_with_shared_values(
                heuristic,
                &scratch.prop,
                &scratch.numeric,
                &scratch.comparisons,
                prop_has_resolved_comparisons,
            )?);
        }
    }

    Ok(())
}

fn hash_with_shared_values(
    heuristic: &DomainAbstractionHeuristic,
    prop_values: &[usize],
    numeric_values: &[f64],
    comparison_values: &[Option<usize>],
    prop_has_resolved_comparisons: bool,
) -> Result<usize, EvaluationError> {
    heuristic.compute_abstract_hash_from_projected_state_values_inner(
        prop_values,
        numeric_values,
        Some(comparison_values),
        prop_has_resolved_comparisons,
    )
}

/// Heuristic that evaluates a concrete state by mapping it to an abstract state
/// and looking up its precomputed goal distance.
#[derive(Debug, Clone)]
pub struct DomainAbstractionHeuristic {
    name: String,
    abstraction: DomainAbstraction,
    prop_scratch: RefCell<Vec<usize>>,
    numeric_scratch: RefCell<Vec<f64>>,
    active_prop_vars: Vec<usize>,
    active_numeric_vars: Vec<usize>,
    comparison_tree_by_var: Vec<Option<usize>>,
    comparison_tree_required_lens: Vec<usize>,
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
            active_prop_vars,
            active_numeric_vars,
            comparison_tree_by_var,
            comparison_tree_required_lens,
        }
    }

    pub fn abstraction(&self) -> &DomainAbstraction {
        &self.abstraction
    }

    fn numeric_partition_for_projected_value(
        &self,
        num_var_id: usize,
        value: f64,
    ) -> Result<usize, EvaluationError> {
        if !value.is_finite() || value.is_nan() {
            return Err(EvaluationError::InvalidState(format!(
                "numeric value for var {num_var_id} must be finite, got {value}"
            )));
        }
        let partitions = self.abstraction.factory.partitions();
        // NOTE: the equispaced fast path is intentionally disabled because
        // its `(value - base) / step` cast does not respect partition
        // boundary closed/open flags. For values exactly on a partition
        // boundary (which happens for every CEGAR-induced split since
        // splits land at concrete state values), the equispaced lookup can
        // return a different partition than the tolerant `partition_for_value`
        // used by `compute_initial_state_hash_determined`. That mismatch
        // makes the heuristic's α(init_concrete) differ from CEGAR's
        // init_state_hash, so `distances[α(init)]` returns a value that
        // CEGAR never tightened — producing h(init) < plan_cost even when
        // CEGAR converged with no flaws.
        let parts = partitions.partitions(num_var_id).ok_or_else(|| {
            EvaluationError::InvalidState(format!(
                "missing partitions for numeric var {num_var_id}"
            ))
        })?;
        utils::partition_for_value(parts, value).ok_or_else(|| {
            EvaluationError::InvalidState(format!(
                "numeric value {value} not contained in any partition for var {num_var_id}"
            ))
        })
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
        self.fill_comparison_values_from_projected_state_values(numeric, out)
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
            &'t dyn planforge_sas::numeric::numeric_task::AbstractNumericTask,
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
        // The registry's buffer already holds correct comparison-axiom-derived
        // bits in `prop`; they were materialized when the state was
        // registered. We can skip the per-evaluation re-evaluation of
        // `ComparisonTree`s entirely.
        // Set DA_NO_FAST_HASH=1 to disable for A/B benchmarking.
        let prop_has_resolved_comparisons = fast_hash_enabled();
        self.compute_abstract_hash_inner(&prop, &numeric, None, prop_has_resolved_comparisons)
    }

    fn compute_abstract_hash_from_state_values(
        &self,
        prop_values: &[usize],
        numeric: &[f64],
        comparison_values: Option<&[Option<usize>]>,
    ) -> Result<usize, EvaluationError> {
        // Conservative path used by external callers: assume `prop_values`
        // does not yet have comparison-axiom-derived bits resolved, so we
        // still consult the comparison trees on the numeric values.
        self.compute_abstract_hash_inner(prop_values, numeric, comparison_values, false)
    }

    fn compute_abstract_hash_inner(
        &self,
        prop_values: &[usize],
        numeric: &[f64],
        comparison_values: Option<&[Option<usize>]>,
        prop_has_resolved_comparisons: bool,
    ) -> Result<usize, EvaluationError> {
        let num_props = self.abstraction.factory.domain_sizes().len();

        if prop_values.len() < num_props {
            return Err(EvaluationError::InvalidState(format!(
                "propositional state too short: {} < {num_props}",
                prop_values.len()
            )));
        }

        self.compute_abstract_hash_from_projected_state_values_inner(
            prop_values,
            numeric,
            comparison_values,
            prop_has_resolved_comparisons,
        )
    }

    fn compute_abstract_hash_from_projected_state_values(
        &self,
        prop_values: &[usize],
        numeric_values: &[f64],
        comparison_values: Option<&[Option<usize>]>,
    ) -> Result<usize, EvaluationError> {
        self.compute_abstract_hash_from_projected_state_values_inner(
            prop_values,
            numeric_values,
            comparison_values,
            false,
        )
    }

    fn compute_abstract_hash_from_projected_state_values_inner(
        &self,
        prop_values: &[usize],
        numeric_values: &[f64],
        comparison_values: Option<&[Option<usize>]>,
        prop_has_resolved_comparisons: bool,
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
        let multipliers = &self.abstraction.hash_multipliers;

        if multipliers.len() != num_props + num_numeric {
            return Err(EvaluationError::InvalidState(
                "hash multipliers length mismatch".to_string(),
            ));
        }

        let mut index: usize = 0;

        for &num_var_id in &self.active_numeric_vars {
            let part =
                self.numeric_partition_for_projected_value(num_var_id, numeric_values[num_var_id])?;
            let abs_var = num_props + num_var_id;
            index += multipliers[abs_var] * part;
        }

        let mut prop_index: usize = 0;
        let _ = prop_has_resolved_comparisons;
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

    // Concrete evaluation on the state's numeric values. This is the
    // deterministic α-image of the concrete state's comparison bit.
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
