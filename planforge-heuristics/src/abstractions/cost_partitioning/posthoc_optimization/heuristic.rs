//! Posthoc-optimization heuristic for a collection of abstractions.
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

use planforge_sas::numeric_task::Task;
use std::collections::BTreeSet;

use planforge_cplex::{
    Constraint as LpConstraint, Model as LpModel, ObjectiveSense, SolveStatus, Variable,
    assert_unrestricted_license,
};
use planforge_sas::numeric_task::NumericValue;
use planforge_sas::numeric_task::OperatorIndex;
use planforge_sas::utils::float_tolerance;
use rustc_hash::{FxHashMap, FxHashSet};
use tracing::info;

use planforge_search::evaluation::Heuristic;
use planforge_search::evaluation::state_value_cache::StateValueCache;
use planforge_search::evaluation::{Estimate, EvaluationError, EvaluationState};

use super::super::standalone_estimate;
use crate::abstractions::component::AbstractionComponent;

pub struct PostHocOptimizationHeuristic {
    name: String,
    components: Vec<AbstractionComponent>,
    /// For each LP constraint, the (sorted, distinct) abstraction ids that
    /// appear with coefficient 1 in that constraint. Constraint right-hand
    /// side is uniformly 1.
    constraints: Vec<Vec<usize>>,
    lp_model: LpModel,
    /// Estimates by the state ids of the search's registry.
    state_value_cache: StateValueCache,
    numeric_values: Vec<NumericValue>,
    h_values: Vec<f64>,
}

impl PostHocOptimizationHeuristic {
    /// Post-hoc optimization over `components`, which it reads only the
    /// goal distances and relevant operators of: components built for
    /// standalone use keep no transition data.
    pub fn new(
        name: Option<String>,
        task: &Task,
        components: Vec<AbstractionComponent>,
    ) -> Result<Self, String> {
        if components.is_empty() {
            return Err(
                "posthoc optimization heuristic requires at least one component".to_string(),
            );
        }
        let relevant_operators = components
            .iter()
            .map(|component| component.relevant_operator_ids(task))
            .collect::<Result<Vec<_>, _>>()?;

        let original_costs: Vec<f64> = task.effective_operator_costs().to_vec();

        let constraints = build_constraints(&relevant_operators, &original_costs);

        assert_unrestricted_license().map_err(|error| {
            format!("posthoc_optimization requires unrestricted CPLEX: {error}")
        })?;
        let lp_model = build_lp_model(components.len(), &constraints)?;

        info!(
            "posthoc_optimization: abstractions={}, lp_constraints={} (CPLEX)",
            components.len(),
            constraints.len(),
        );

        Ok(Self {
            name: name.unwrap_or_else(|| "posthoc_optimization".to_string()),
            components,
            constraints,
            lp_model,
            state_value_cache: StateValueCache::default(),
            numeric_values: Vec::new(),
            h_values: Vec::new(),
        })
    }

    fn evaluate_lp(
        &mut self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError> {
        let state = eval_state.values(&mut self.numeric_values)?;
        let mut h_values = std::mem::take(&mut self.h_values);
        h_values.clear();
        for (component_id, component) in self.components.iter().enumerate() {
            let value = standalone_estimate(component, component_id, state);
            if value == f64::INFINITY {
                self.h_values = h_values;
                return Ok(f64::INFINITY);
            }
            h_values.push(value);
        }
        let value = self.solve_dual(&h_values);
        self.h_values = h_values;
        value
    }

    fn solve_dual(&mut self, h_values: &[f64]) -> Result<f64, EvaluationError> {
        // Drop abstractions with h_i = 0: their dual variable is fixed at 0
        // in any optimum (positive value can only tighten a packing
        // constraint shared with a strictly-helpful abstraction).
        let active: Vec<usize> = h_values
            .iter()
            .enumerate()
            .filter_map(|(id, h)| (*h > float_tolerance::SEARCH_EPSILON).then_some(id))
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
        let model = &mut self.lp_model;
        assert_eq!(
            h_values.len(),
            model.num_columns(),
            "posthoc objective dimension differs from its persistent model",
        );
        model.set_objective(h_values).map_err(lp_evaluation_error)?;
        match model.solve().map_err(lp_evaluation_error)? {
            SolveStatus::Optimal => {
                let value = model.objective_value().map_err(lp_evaluation_error)?;
                // Every objective coefficient and variable is non-negative,
                // so only the solver's tolerance can make the optimum
                // negative.
                assert!(
                    value >= -float_tolerance::LP_EPSILON,
                    "posthoc_optimization: the dual optimum {value} is negative"
                );
                Ok(value.max(0.0))
            }
            SolveStatus::Infeasible => {
                unreachable!("posthoc_optimization: the dual LP is feasible at X = 0")
            }
            other => Err(EvaluationError::ComputationFailed(format!(
                "posthoc_optimization: CPLEX returned `{other:?}` for a well-formed LP \
                 (active={} abstractions, {} constraints); not falling back",
                active.len(),
                self.constraints.len(),
            ))),
        }
    }
}

fn build_lp_model(num_abstractions: usize, constraints: &[Vec<usize>]) -> Result<LpModel, String> {
    assert!(num_abstractions > 0, "a posthoc model has a column");
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
    Ok(model)
}

fn lp_evaluation_error(error: planforge_cplex::Error) -> EvaluationError {
    EvaluationError::ComputationFailed(format!("posthoc_optimization CPLEX failure: {error}"))
}

impl Heuristic for PostHocOptimizationHeuristic {
    fn evaluate(
        &mut self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<Estimate, EvaluationError> {
        let state_id = eval_state.state().get_id();
        if let Some(value) = self.state_value_cache.get(state_id) {
            return Ok(Estimate::new(value));
        }

        let value = self.evaluate_lp(eval_state)?;
        self.state_value_cache.insert(state_id, value);
        Ok(Estimate::new(value))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

fn build_constraints(
    relevant_operators: &[BTreeSet<OperatorIndex>],
    original_costs: &[f64],
) -> Vec<Vec<usize>> {
    let mut operator_to_abstractions: FxHashMap<OperatorIndex, Vec<usize>> = FxHashMap::default();
    for (abstraction_id, ops) in relevant_operators.iter().enumerate() {
        for &operator_id in ops {
            operator_to_abstractions
                .entry(operator_id)
                .or_default()
                .push(abstraction_id);
        }
    }

    let mut seen = FxHashSet::default();
    let mut constraints = Vec::new();
    for (operator_id, abstraction_ids) in operator_to_abstractions {
        // Free operators impose no constraint in the dual LP.
        if original_costs[operator_id.index()] <= 0.0 {
            continue;
        }
        // Each abstraction appears once per operator, in increasing order.
        if seen.insert(abstraction_ids.clone()) {
            constraints.push(abstraction_ids);
        }
    }

    constraints
}
