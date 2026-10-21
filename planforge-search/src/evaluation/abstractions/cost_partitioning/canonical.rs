#[cfg(test)]
mod tests;

use std::cell::{Cell, RefCell};
use std::collections::BTreeSet;

use planforge_sas::numeric_task::{AbstractNumericTask, OperatorIndex};
use tracing::{debug, info};

use crate::evaluation::evaluator::{EvaluationError, EvaluationState};
use crate::evaluation::heuristic::Heuristic;
use crate::evaluation::maximal_cliques::maximal_cliques;
use crate::evaluation::state_value_cache::StateValueCache;

use planforge_sas::explicit_state::ExplicitState;

use crate::evaluation::abstractions::component::AbstractionComponent;

pub struct CanonicalAbstractionHeuristic<'task> {
    name: String,
    components: Vec<AbstractionComponent<'task>>,
    max_additive_subsets: Vec<Vec<usize>>,
    relevant_operator_ids: Vec<BTreeSet<OperatorIndex>>,
    component_value_cache: RefCell<Vec<Option<f64>>>,
    state_value_cache: RefCell<StateValueCache>,
    component_state_values: RefCell<ExplicitState>,
    diagnostics_logged: Cell<bool>,
}

impl<'task> CanonicalAbstractionHeuristic<'task> {
    /// The canonical heuristic over the maximal additive subsets of
    /// `components`, which it reads only the goal distances and relevant
    /// operators of: components built for standalone use keep no
    /// transition data.
    pub fn new(
        name: Option<String>,
        task: &dyn AbstractNumericTask,
        components: Vec<AbstractionComponent<'task>>,
    ) -> Result<Self, String> {
        if components.is_empty() {
            return Err(
                "canonical abstraction heuristic requires at least one component".to_string(),
            );
        }
        let relevant_operator_ids: Vec<BTreeSet<OperatorIndex>> = components
            .iter()
            .map(|component| component.relevant_operator_ids(task))
            .collect::<Result<Vec<_>, _>>()?;
        let subsets = compute_max_additive_subsets_from_relevant_operators(&relevant_operator_ids);
        Ok(Self::from_validated_parts(
            name,
            components,
            subsets,
            relevant_operator_ids,
        ))
    }

    fn from_validated_parts(
        name: Option<String>,
        components: Vec<AbstractionComponent<'task>>,
        max_additive_subsets: Vec<Vec<usize>>,
        relevant_operator_ids: Vec<BTreeSet<OperatorIndex>>,
    ) -> Self {
        assert!(
            !max_additive_subsets.is_empty(),
            "a nonempty collection has an additive subset"
        );
        Self {
            name: name.unwrap_or_else(|| "canonical_abstractions".to_string()),
            components,
            max_additive_subsets,
            relevant_operator_ids,
            component_value_cache: RefCell::new(Vec::new()),
            state_value_cache: RefCell::new(StateValueCache::default()),
            component_state_values: RefCell::default(),
            diagnostics_logged: Cell::new(false),
        }
    }

    fn component_value(
        &self,
        component_id: usize,
        state_values: &ExplicitState,
        cache: &mut [Option<f64>],
    ) -> Result<f64, EvaluationError> {
        let slot = cache.get_mut(component_id).ok_or_else(|| {
            EvaluationError::InvalidState(format!(
                "invalid canonical abstraction component index {component_id}"
            ))
        })?;
        if let Some(value) = *slot {
            return Ok(value);
        }
        let component = self.components.get(component_id).ok_or_else(|| {
            EvaluationError::InvalidState(format!(
                "missing canonical abstraction component {component_id}"
            ))
        })?;
        let value = component.standalone_value(state_values).map_err(|error| {
            EvaluationError::ComputationFailed(format!(
                "failed to evaluate {} component {component_id}: {error}",
                component.kind()
            ))
        })?;
        if value.is_nan() || value < 0.0 {
            return Err(EvaluationError::ComputationFailed(format!(
                "{} component {component_id} returned invalid heuristic value {value}",
                component.kind()
            )));
        }
        *slot = Some(value);
        Ok(value)
    }

    fn evaluate_subsets(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError> {
        let mut state_values = self.component_state_values.borrow_mut();
        eval_state.fill_values(&mut state_values)?;
        let mut cache = self.component_value_cache.borrow_mut();
        cache.clear();
        cache.resize(self.components.len(), None);
        let mut best = 0.0_f64;

        for subset in &self.max_additive_subsets {
            let mut sum = 0.0_f64;
            for &component_id in subset {
                let value = self.component_value(component_id, &state_values, &mut cache)?;
                if value.is_infinite() {
                    self.log_diagnostics_once(&cache);
                    return Ok(f64::INFINITY);
                }
                sum += value;
            }
            best = best.max(sum);
        }
        self.log_diagnostics_once(&cache);
        Ok(best)
    }

    fn log_diagnostics_once(&self, values: &[Option<f64>]) {
        if self.diagnostics_logged.replace(true) {
            return;
        }
        info!(
            "canonical abstraction diagnostics: components={}, max_additive_subsets={}",
            self.components.len(),
            self.max_additive_subsets.len()
        );
        for (component_id, component) in self.components.iter().enumerate() {
            debug!(
                "canonical component {component_id}: kind={}, states={}, h={}, relevant_ops={}",
                component.kind(),
                component.num_states(),
                values
                    .get(component_id)
                    .copied()
                    .flatten()
                    .unwrap_or(f64::NAN),
                self.relevant_operator_ids[component_id].len(),
            );
        }
    }
}

impl Heuristic for CanonicalAbstractionHeuristic<'_> {
    fn compute_heuristic(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError> {
        let state_id = eval_state.state().get_id();
        if let Some(value) = self.state_value_cache.borrow().get(state_id) {
            return Ok(value);
        }
        let value = self.evaluate_subsets(eval_state)?;
        self.state_value_cache.borrow_mut().insert(state_id, value);
        Ok(value)
    }

    fn heuristic_name(&self) -> &str {
        &self.name
    }
}

fn are_operator_sets_additive(
    left: &BTreeSet<OperatorIndex>,
    right: &BTreeSet<OperatorIndex>,
) -> bool {
    !left.iter().any(|operator_id| right.contains(operator_id))
}

fn compute_max_additive_subsets_from_relevant_operators(
    relevant_operators: &[BTreeSet<OperatorIndex>],
) -> Vec<Vec<usize>> {
    let mut maximal_cliques = maximal_cliques(relevant_operators.len(), |left, right| {
        are_operator_sets_additive(&relevant_operators[left], &relevant_operators[right])
    });
    maximal_cliques.sort();
    maximal_cliques.dedup();
    maximal_cliques
}
