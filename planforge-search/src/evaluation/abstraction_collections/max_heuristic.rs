use std::cell::RefCell;

use crate::evaluation::evaluator::{EvaluationError, EvaluationState};
use crate::evaluation::heuristic::Heuristic;

use super::component::{AbstractionComponent, ComponentStateValues};

pub struct MaxAbstractionHeuristic<'task> {
    name: String,
    components: Vec<AbstractionComponent<'task>>,
    state_values: RefCell<ComponentStateValues>,
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
            state_values: RefCell::new(ComponentStateValues::default()),
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
        let mut state_values = self.state_values.borrow_mut();
        state_values.fill(eval_state)?;
        let mut best = 0.0_f64;
        for (component_id, component) in self.components.iter().enumerate() {
            let value = component
                .standalone_value_from_state_values(
                    &state_values.propositional,
                    &state_values.numeric,
                )
                .map_err(|error| {
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

    fn heuristic_name(&self) -> &str {
        &self.name
    }
}
