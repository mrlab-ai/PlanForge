//! Posthoc-optimization heuristic for a collection of domain abstractions.
//!
//! Given abstractions α_1, …, α_n built with the full operator cost function
//! c, the posthoc-optimization heuristic of a state s is the optimum of
//!
//! ```text
//! min  Σ_o c(o) · Y_o
//! s.t. Σ_{o relevant to α_i} c(o) · Y_o ≥ h_{α_i}(s)   for each i
//!      Y_o ≥ 0
//! ```
//!
//! which by LP duality equals
//!
//! ```text
//! max  Σ_i h_{α_i}(s) · X_i
//! s.t. Σ_{i : o relevant to α_i} X_i ≤ 1   for each o with c(o) > 0
//!      X_i ≥ 0
//! ```
//!
//! The dual has one variable per abstraction (small) and one constraint per
//! positive-cost operator (the large axis), so we solve the dual.
//!
//! The LP is solved by native CPLEX. The constraint matrix and bounds are
//! independent of state, so the model remains loaded and each state changes
//! only the objective, retaining CPLEX's basis and warm-start information.
//!
//! Reference: Pommerening, Röger, Helmert (AAAI 2013), "Getting the most out
//! of pattern databases for classical planning".

#[cfg(test)]
mod tests;

use std::cell::RefCell;
use std::collections::BTreeSet;

use planforge_cplex::{
    Constraint as LpConstraint, Model as LpModel, ObjectiveSense, SolveStatus, Variable,
    assert_unrestricted_license,
};
use planforge_sas::numeric_task::{AbstractNumericTask, metric_operator_cost_from_initial_values};
use rustc_hash::FxHashMap;
use tracing::info;

use crate::evaluation::evaluator::{EvaluationError, EvaluationState};
use crate::evaluation::heuristic::Heuristic;
use crate::evaluation::state_value_cache::StateValueCache;

use super::domain_abstraction_generator::DomainAbstraction;
use super::domain_abstraction_heuristic::{
    DomainAbstractionHeuristic, DomainAbstractionLookupScratch,
    compute_collection_abstract_state_ids,
};

#[derive(Debug, Clone, Default)]
pub struct PostHocOptimizationConfig {
    /// Optional caller-supplied label appended to the heuristic name.
    pub label: Option<String>,
}

#[derive(Debug)]
pub struct PostHocOptimizationHeuristic {
    name: String,
    heuristics: Vec<DomainAbstractionHeuristic>,
    /// For each LP constraint, the (sorted, distinct) abstraction ids that
    /// appear with coefficient 1 in that constraint. Constraint right-hand
    /// side is uniformly 1.
    constraints: Vec<Vec<usize>>,
    lp_model: RefCell<Option<LpModel>>,
    state_value_cache: RefCell<StateValueCache>,
    lookup_scratch: RefCell<DomainAbstractionLookupScratch>,
    diagnostics_logged: RefCell<bool>,
}

impl PostHocOptimizationHeuristic {
    pub fn new(
        name: Option<String>,
        task: &dyn AbstractNumericTask,
        abstractions: Vec<DomainAbstraction>,
    ) -> Result<Self, String> {
        let heuristics: Vec<_> = abstractions
            .into_iter()
            .enumerate()
            .map(|(index, abstraction)| {
                DomainAbstractionHeuristic::new(
                    Some(format!("posthoc_optimization_{index}")),
                    abstraction,
                )
            })
            .collect();

        let mut relevant_operators: Vec<BTreeSet<usize>> = Vec::with_capacity(heuristics.len());
        for heuristic in &heuristics {
            relevant_operators.push(compute_relevant_operator_ids(
                task,
                heuristic.abstraction(),
            )?);
        }

        let original_costs: Vec<f64> = task
            .get_operators()
            .iter()
            .map(|op| metric_operator_cost_from_initial_values(task, op))
            .collect();

        let constraints = build_constraints(&relevant_operators, &original_costs, heuristics.len());

        assert_unrestricted_license().map_err(|error| {
            format!("posthoc_optimization requires unrestricted CPLEX: {error}")
        })?;
        let lp_model = build_lp_model(heuristics.len(), &constraints)?;

        info!(
            "posthoc_optimization: abstractions={}, lp_constraints={} (CPLEX)",
            heuristics.len(),
            constraints.len(),
        );

        Ok(Self {
            name: name.unwrap_or_else(|| "posthoc_optimization".to_string()),
            heuristics,
            constraints,
            lp_model: RefCell::new(lp_model),
            state_value_cache: RefCell::new(StateValueCache::default()),
            lookup_scratch: RefCell::new(DomainAbstractionLookupScratch::new()),
            diagnostics_logged: RefCell::new(false),
        })
    }

    pub fn heuristics(&self) -> &[DomainAbstractionHeuristic] {
        &self.heuristics
    }

    fn evaluate_lp(&self, eval_state: &EvaluationState<'_, '_>) -> Result<f64, EvaluationError> {
        if self.heuristics.is_empty() {
            return Ok(0.0);
        }

        let mut scratch = self.lookup_scratch.borrow_mut();
        compute_collection_abstract_state_ids(&self.heuristics, eval_state, None, &mut scratch)?;

        let mut h_values = vec![0.0_f64; self.heuristics.len()];
        for (abstraction_id, state_id) in scratch.abstract_state_ids.iter().enumerate() {
            let Some(state_id) = *state_id else {
                continue;
            };
            let heuristic = self.heuristics.get(abstraction_id).ok_or_else(|| {
                EvaluationError::InvalidState(format!(
                    "invalid posthoc abstraction index {abstraction_id}"
                ))
            })?;
            let value = heuristic
                .abstraction()
                .distance_table
                .distances
                .get(state_id)
                .copied()
                .ok_or_else(|| {
                    EvaluationError::InvalidState(format!(
                        "abstract hash out of bounds: {state_id} (len={})",
                        heuristic.abstraction().distance_table.distances.len()
                    ))
                })?;
            if value.is_infinite() && value.is_sign_positive() {
                self.log_diagnostics_once(&h_values, &scratch.abstract_state_ids);
                return Ok(f64::INFINITY);
            }
            h_values[abstraction_id] = value.max(0.0);
        }

        self.log_diagnostics_once(&h_values, &scratch.abstract_state_ids);

        self.solve_dual(&h_values)
    }

    fn solve_dual(&self, h_values: &[f64]) -> Result<f64, EvaluationError> {
        // Drop abstractions with h_i = 0: their dual variable is fixed at 0
        // in any optimum (positive value can only tighten a packing
        // constraint shared with a strictly-helpful abstraction).
        let active: Vec<usize> = h_values
            .iter()
            .enumerate()
            .filter_map(|(id, h)| (*h > 0.0).then_some(id))
            .collect();
        if active.is_empty() {
            return Ok(0.0);
        }
        if active.len() == 1 {
            // X_i is capped at 1 by any constraint containing it, and there's
            // always at least one such constraint when h_i > 0 means the goal
            // is reachable in α_i via at least one positive-cost operator.
            // The optimum is h_i.
            return Ok(h_values[active[0]]);
        }

        // Invariant: with at least two active abstractions, each has a
        // positive-cost relevant operator and therefore occurs in the
        // persistent model.
        assert!(
            !self.constraints.is_empty(),
            "posthoc_optimization: active set {active:?} produced no LP rows; \
             this indicates a bug in `build_constraints`",
        );
        assert_eq!(
            h_values.len(),
            self.lp_model
                .borrow()
                .as_ref()
                .expect("nonempty active set requires an LP model")
                .num_columns(),
            "posthoc objective dimension differs from its persistent model",
        );

        let mut model = self.lp_model.borrow_mut();
        let model = model
            .as_mut()
            .expect("nonempty active set requires an LP model");
        model.set_objective(h_values).map_err(lp_evaluation_error)?;
        match model.solve().map_err(lp_evaluation_error)? {
            SolveStatus::Optimal => model
                .objective_value()
                .map(|value| value.max(0.0))
                .map_err(lp_evaluation_error),
            SolveStatus::Infeasible => Ok(f64::INFINITY),
            other => Err(EvaluationError::ComputationFailed(format!(
                "posthoc_optimization: CPLEX returned `{other:?}` for a well-formed LP \
                 (active={} abstractions, {} constraints); not falling back",
                active.len(),
                self.constraints.len(),
            ))),
        }
    }

    fn log_diagnostics_once(&self, h_values: &[f64], abstract_state_ids: &[Option<usize>]) {
        {
            let mut logged = self.diagnostics_logged.borrow_mut();
            if *logged {
                return;
            }
            *logged = true;
        }

        let positive: Vec<_> = h_values
            .iter()
            .enumerate()
            .filter(|(_, h)| **h > 0.0)
            .collect();
        info!(
            "posthoc_optimization first-state diagnostics: abstractions={}, with_positive_h={}, lp_constraints={}",
            self.heuristics.len(),
            positive.len(),
            self.constraints.len(),
        );
        for (id, h) in positive.iter().take(16) {
            info!(
                "posthoc abstraction {id}: h={h}, lookup_state={:?}",
                abstract_state_ids.get(*id).copied().flatten()
            );
        }
    }
}

fn build_lp_model(
    num_abstractions: usize,
    constraints: &[Vec<usize>],
) -> Result<Option<LpModel>, String> {
    if num_abstractions == 0 {
        assert!(
            constraints.is_empty(),
            "zero-abstraction posthoc model cannot contain constraints"
        );
        return Ok(None);
    }
    let variables = vec![Variable::new(0.0, LpModel::infinity(), 0.0); num_abstractions];
    let constraints: Vec<LpConstraint> = constraints
        .iter()
        .map(|constraint| {
            LpConstraint::new(
                -LpModel::infinity(),
                1.0,
                constraint
                    .iter()
                    .map(|&abstraction_id| (abstraction_id, 1.0))
                    .collect(),
            )
        })
        .collect();
    let mut model = LpModel::new("posthoc-optimization").map_err(|error| error.to_string())?;
    model
        .load(ObjectiveSense::Maximize, &variables, &constraints)
        .map_err(|error| error.to_string())?;
    Ok(Some(model))
}

fn lp_evaluation_error(error: planforge_cplex::Error) -> EvaluationError {
    EvaluationError::ComputationFailed(format!("posthoc_optimization CPLEX failure: {error}"))
}

impl Heuristic for PostHocOptimizationHeuristic {
    fn compute_heuristic(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError> {
        let state_id = eval_state.state().get_id();
        if let Some(value) = self.state_value_cache.borrow().get(state_id) {
            return Ok(value);
        }

        let value = self.evaluate_lp(eval_state)?;
        self.state_value_cache.borrow_mut().insert(state_id, value);
        Ok(value)
    }

    fn heuristic_name(&self) -> &str {
        &self.name
    }
}

fn compute_relevant_operator_ids(
    task: &dyn AbstractNumericTask,
    abstraction: &DomainAbstraction,
) -> Result<BTreeSet<usize>, String> {
    if !abstraction.relevant_operator_ids.is_empty() {
        return Ok(abstraction.relevant_operator_ids.iter().copied().collect());
    }

    let task = abstraction.task_for_factory(task);
    let mut generator = abstraction
        .factory
        .make_operator_generator(task, abstraction.combine_labels)
        .map_err(|error| {
            format!("failed to build operator generator for posthoc optimization: {error:#}")
        })?;
    let operators = generator.build_abstract_operators(task).map_err(|error| {
        format!("failed to build abstract operators for posthoc optimization: {error:#}")
    })?;
    let relevant = abstraction
        .factory
        .relevant_operator_ids_from_operators_with_deadline(
            task,
            abstraction.combine_labels,
            &operators,
            None,
        )
        .map_err(|error| {
            format!("failed to compute relevant operator ids for posthoc optimization: {error:#}")
        })?;
    Ok(relevant.into_iter().collect())
}

fn build_constraints(
    relevant_operators: &[BTreeSet<usize>],
    original_costs: &[f64],
    n_abstractions: usize,
) -> Vec<Vec<usize>> {
    let mut operator_to_abstractions: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
    for (abstraction_id, ops) in relevant_operators.iter().enumerate() {
        if abstraction_id >= n_abstractions {
            continue;
        }
        for &operator_id in ops {
            operator_to_abstractions
                .entry(operator_id)
                .or_default()
                .push(abstraction_id);
        }
    }

    let mut seen: FxHashMap<Vec<usize>, ()> = FxHashMap::default();
    let mut constraints = Vec::new();
    for (operator_id, mut abstraction_ids) in operator_to_abstractions {
        let cost = original_costs.get(operator_id).copied().unwrap_or(0.0);
        if !matches!(cost.partial_cmp(&0.0), Some(std::cmp::Ordering::Greater)) {
            // Free operators impose no constraint in the dual LP.
            continue;
        }
        abstraction_ids.sort_unstable();
        abstraction_ids.dedup();
        if abstraction_ids.is_empty() {
            continue;
        }
        if seen.contains_key(&abstraction_ids) {
            continue;
        }
        seen.insert(abstraction_ids.clone(), ());
        constraints.push(abstraction_ids);
    }

    constraints
}
