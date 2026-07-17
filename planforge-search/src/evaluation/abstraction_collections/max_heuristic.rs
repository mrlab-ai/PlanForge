use crate::evaluation::domain_abstractions::domain_abstraction_generator::DomainAbstraction;
use crate::evaluation::evaluator::{EvaluationError, EvaluationState};
use crate::evaluation::heuristic::Heuristic;

use super::component::AbstractionComponent;

pub struct MaxAbstractionHeuristic<'task> {
    name: String,
    components: Vec<AbstractionComponent<'task>>,
}

impl<'task> MaxAbstractionHeuristic<'task> {
    pub fn new(
        name: Option<String>,
        mut components: Vec<AbstractionComponent<'task>>,
    ) -> Result<Self, String> {
        if components.is_empty() {
            return Err("max abstraction heuristic requires at least one component".to_string());
        }
        for component in &mut components {
            component.discard_transition_data();
        }
        Ok(Self {
            name: name.unwrap_or_else(|| "max_abstractions".to_string()),
            components,
        })
    }

    pub fn components(&self) -> &[AbstractionComponent<'task>] {
        &self.components
    }
}

impl Heuristic for MaxAbstractionHeuristic<'_> {
    fn compute_heuristic(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError> {
        let mut best = 0.0_f64;
        for (component_id, component) in self.components.iter().enumerate() {
            let value = component.standalone_value(eval_state).map_err(|error| {
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
            best = best.max(value);
        }
        Ok(best)
    }

    fn proves_initial_state_optimal(&self) -> bool {
        false
    }

    fn heuristic_name(&self) -> String {
        self.name.clone()
    }
}

/// Compatibility wrapper for callers that construct a max heuristic from
/// domain abstractions only.
pub struct MaxDomainAbstractionHeuristic {
    inner: MaxAbstractionHeuristic<'static>,
}

impl MaxDomainAbstractionHeuristic {
    pub fn new(name: Option<String>, abstractions: Vec<DomainAbstraction>) -> Self {
        let components = abstractions
            .into_iter()
            .enumerate()
            .map(|(index, abstraction)| {
                AbstractionComponent::domain(
                    Some(format!("multi_domain_abstraction_{index}")),
                    abstraction,
                )
            })
            .collect();
        Self {
            inner: MaxAbstractionHeuristic::new(
                name.or_else(|| Some("multi_domain_abstractions".to_string())),
                components,
            )
            .expect("domain abstraction collection for max heuristic must not be empty"),
        }
    }
}

impl Heuristic for MaxDomainAbstractionHeuristic {
    fn compute_heuristic(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError> {
        self.inner.compute_heuristic(eval_state)
    }

    fn proves_initial_state_optimal(&self) -> bool {
        self.inner.proves_initial_state_optimal()
    }

    fn heuristic_name(&self) -> String {
        self.inner.heuristic_name()
    }
}
