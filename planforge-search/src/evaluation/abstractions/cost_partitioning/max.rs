use std::cell::RefCell;

use crate::evaluation::evaluator::{EvaluationError, EvaluationState};
use crate::evaluation::heuristic::Heuristic;

use planforge_sas::explicit_state::ExplicitState;

use super::standalone_estimate;
use crate::evaluation::abstractions::component::AbstractionComponent;

pub struct MaxAbstractionHeuristic<'task> {
    name: String,
    components: Vec<AbstractionComponent<'task>>,
    state_values: RefCell<ExplicitState>,
}

impl<'task> MaxAbstractionHeuristic<'task> {
    /// The maximum over `components`, which it reads only the goal distances
    /// of: components built for standalone use keep no transition data.
    pub fn new(
        name: Option<String>,
        components: Vec<AbstractionComponent<'task>>,
    ) -> Result<Self, String> {
        if components.is_empty() {
            return Err("max abstraction heuristic requires at least one component".to_string());
        }
        Ok(Self {
            name: name.unwrap_or_else(|| "max_abstractions".to_string()),
            components,
            state_values: RefCell::default(),
        })
    }
}

impl Heuristic for MaxAbstractionHeuristic<'_> {
    fn compute_heuristic(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError> {
        let mut state_values = self.state_values.borrow_mut();
        eval_state.fill_values(&mut state_values)?;
        let mut best = 0.0_f64;
        for (component_id, component) in self.components.iter().enumerate() {
            best = best.max(standalone_estimate(component, component_id, &state_values)?);
        }
        Ok(best)
    }

    fn heuristic_name(&self) -> &str {
        &self.name
    }
}
