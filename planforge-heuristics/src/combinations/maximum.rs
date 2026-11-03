//! Pointwise maximum of arbitrary heuristics. Abstraction children are grouped
//! by the feature builder before reaching this evaluator.

#[cfg(test)]
mod tests;

use planforge_sas::numeric_task::OperatorIndex;
use planforge_search::evaluation::{Estimate, EvaluationError, EvaluationState, Heuristic};

pub struct MaximumHeuristic {
    name: String,
    children: Vec<Box<dyn Heuristic>>,
}

impl std::fmt::Debug for MaximumHeuristic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let children: Vec<&str> = self.children.iter().map(|child| child.name()).collect();
        f.debug_struct("MaximumHeuristic")
            .field("name", &self.name)
            .field("children", &children)
            .finish()
    }
}

impl MaximumHeuristic {
    pub fn new(name: String, children: Vec<Box<dyn Heuristic>>) -> Self {
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
    ) -> Result<Estimate, EvaluationError> {
        let first_preferred = preferred.as_ref().map_or(0, |preferred| preferred.len());
        let mut best = 0.0_f64;
        for child in &mut self.children {
            let estimate = match preferred.as_deref_mut() {
                Some(preferred) => child.evaluate_with_preferred(state, preferred)?,
                None => child.evaluate(state)?,
            };
            best = best.max(estimate.value());
        }
        if let Some(preferred) = preferred {
            preferred[first_preferred..].sort_unstable();
            preferred.dedup();
        }
        Ok(Estimate::new(best))
    }
}

impl Heuristic for MaximumHeuristic {
    fn evaluate(&mut self, state: &EvaluationState<'_, '_>) -> Result<Estimate, EvaluationError> {
        self.maximum(state, None)
    }

    fn evaluate_with_preferred(
        &mut self,
        state: &EvaluationState<'_, '_>,
        preferred: &mut Vec<OperatorIndex>,
    ) -> Result<Estimate, EvaluationError> {
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
