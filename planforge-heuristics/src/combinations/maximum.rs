//! Pointwise maximum of arbitrary heuristics. Abstraction children are grouped
//! by the feature builder before reaching this evaluator.

#[cfg(test)]
mod tests;

use planforge_sas::numeric_task::OperatorIndex;
use planforge_search::evaluation::{EvaluationError, EvaluationState, Heuristic};

pub struct MaximumHeuristic<'a> {
    name: String,
    children: Vec<Box<dyn Heuristic + 'a>>,
}

impl<'a> MaximumHeuristic<'a> {
    pub fn new(name: String, children: Vec<Box<dyn Heuristic + 'a>>) -> Self {
        assert!(
            !children.is_empty(),
            "maximum requires at least one heuristic"
        );
        Self { name, children }
    }

    /// The largest estimate of the children; with `preferred`, appending the
    /// union of their preferred operators.
    fn maximum(
        &mut self,
        state: &EvaluationState<'_, '_>,
        mut preferred: Option<&mut Vec<OperatorIndex>>,
    ) -> Result<f64, EvaluationError> {
        let first_preferred = preferred.as_ref().map_or(0, |preferred| preferred.len());
        let mut best = f64::NEG_INFINITY;
        for child in &mut self.children {
            let estimate = match preferred.as_deref_mut() {
                Some(preferred) => child.evaluate_with_preferred(state, preferred)?,
                None => child.evaluate(state)?,
            };
            if estimate.is_nan() {
                return Err(EvaluationError::ComputationFailed(format!(
                    "heuristic `{}` returned NaN",
                    child.name()
                )));
            }
            best = best.max(estimate);
        }
        if let Some(preferred) = preferred {
            preferred[first_preferred..].sort_unstable();
            preferred.dedup();
        }
        Ok(best)
    }
}

impl Heuristic for MaximumHeuristic<'_> {
    fn evaluate(&mut self, state: &EvaluationState<'_, '_>) -> Result<f64, EvaluationError> {
        self.maximum(state, None)
    }

    fn evaluate_with_preferred(
        &mut self,
        state: &EvaluationState<'_, '_>,
        preferred: &mut Vec<OperatorIndex>,
    ) -> Result<f64, EvaluationError> {
        self.maximum(state, Some(preferred))
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn proves_initial_state_optimal(&self) -> bool {
        self.children
            .iter()
            .all(|child| child.proves_initial_state_optimal())
    }

    /// The children's revisions summed: monotone, and growing exactly when
    /// one of theirs grows.
    fn revision(&self) -> u64 {
        self.children.iter().map(|child| child.revision()).sum()
    }

    fn reevaluate_on_every_pop(&self) -> bool {
        self.children
            .iter()
            .any(|child| child.reevaluate_on_every_pop())
    }
}
