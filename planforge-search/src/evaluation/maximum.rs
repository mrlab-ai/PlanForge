//! Pointwise maximum of arbitrary heuristics. Abstraction children are grouped
//! by the feature builder before reaching this evaluator.
use super::{EvaluationError, EvaluationState, Heuristic};
use planforge_sas::numeric_task::OperatorIndex;
use std::cell::{Cell, RefCell};

pub struct MaximumHeuristic<'a> {
    name: String,
    children: Vec<Box<dyn Heuristic + 'a>>,
    child_revisions: RefCell<Vec<u64>>,
    revision: Cell<u64>,
    preferred: RefCell<Vec<OperatorIndex>>,
    preferred_scratch: RefCell<Vec<OperatorIndex>>,
}
impl<'a> MaximumHeuristic<'a> {
    pub fn new(name: String, children: Vec<Box<dyn Heuristic + 'a>>) -> Self {
        assert!(
            !children.is_empty(),
            "maximum requires at least one heuristic"
        );
        let revisions = children.iter().map(|child| child.revision()).collect();
        Self {
            name,
            children,
            child_revisions: RefCell::new(revisions),
            revision: Cell::new(0),
            preferred: RefCell::default(),
            preferred_scratch: RefCell::default(),
        }
    }
}
impl Heuristic for MaximumHeuristic<'_> {
    fn compute_heuristic(&self, state: &EvaluationState<'_, '_>) -> Result<f64, EvaluationError> {
        let mut best = f64::NEG_INFINITY;
        let mut preferred = self.preferred.borrow_mut();
        let mut scratch = self.preferred_scratch.borrow_mut();
        preferred.clear();
        for child in &self.children {
            let estimate = child.compute_heuristic(state)?;
            if estimate.is_nan() {
                return Err(EvaluationError::ComputationFailed(format!(
                    "heuristic `{}` returned NaN",
                    child.heuristic_name()
                )));
            }
            best = best.max(estimate);
            scratch.clear();
            child.copy_preferred_operator_ids(&mut scratch);
            preferred.extend_from_slice(&scratch);
        }
        preferred.sort_unstable();
        preferred.dedup();
        Ok(best)
    }
    fn heuristic_name(&self) -> &str {
        &self.name
    }
    fn proves_initial_state_optimal(&self) -> bool {
        self.children
            .iter()
            .all(|child| child.proves_initial_state_optimal())
    }
    fn copy_preferred_operator_ids(&self, out: &mut Vec<OperatorIndex>) {
        out.clear();
        out.extend_from_slice(&self.preferred.borrow());
    }
    fn revision(&self) -> u64 {
        let mut revisions = self.child_revisions.borrow_mut();
        let mut changed = false;
        for (child, previous) in self.children.iter().zip(revisions.iter_mut()) {
            let current = child.revision();
            assert!(current >= *previous, "heuristic revisions must be monotone");
            changed |= current != *previous;
            *previous = current;
        }
        if changed {
            self.revision.set(
                self.revision
                    .get()
                    .checked_add(1)
                    .expect("heuristic revision overflow"),
            );
        }
        self.revision.get()
    }
    fn reevaluate_on_every_pop(&self) -> bool {
        self.children
            .iter()
            .any(|child| child.reevaluate_on_every_pop())
    }
}
