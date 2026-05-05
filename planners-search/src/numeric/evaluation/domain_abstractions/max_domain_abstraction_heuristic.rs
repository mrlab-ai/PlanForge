use std::cell::RefCell;

use crate::numeric::evaluation::evaluator::{EvaluationError, EvaluationState};
use crate::numeric::evaluation::heuristic::Heuristic;

use super::domain_abstraction_generator::DomainAbstraction;
use super::domain_abstraction_heuristic::{
    compute_collection_abstract_state_ids, DomainAbstractionHeuristic,
    DomainAbstractionLookupScratch,
};

#[derive(Debug, Clone)]
pub struct MaxDomainAbstractionHeuristic {
    name: String,
    heuristics: Vec<DomainAbstractionHeuristic>,
    lookup_scratch: RefCell<DomainAbstractionLookupScratch>,
}

impl MaxDomainAbstractionHeuristic {
    pub fn new(name: Option<String>, abstractions: Vec<DomainAbstraction>) -> Self {
        let heuristics = abstractions
            .into_iter()
            .enumerate()
            .map(|(index, abstraction)| {
                DomainAbstractionHeuristic::new(
                    Some(format!("multi_domain_abstraction_{index}")),
                    abstraction,
                )
            })
            .collect();

        Self {
            name: name.unwrap_or_else(|| "multi_domain_abstractions".to_string()),
            heuristics,
            lookup_scratch: RefCell::new(DomainAbstractionLookupScratch::new()),
        }
    }

    pub fn heuristics(&self) -> &[DomainAbstractionHeuristic] {
        &self.heuristics
    }
}

impl Heuristic for MaxDomainAbstractionHeuristic {
    fn compute_heuristic(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError> {
        let mut best = f64::NEG_INFINITY;
        let mut scratch = self.lookup_scratch.borrow_mut();
        compute_collection_abstract_state_ids(&self.heuristics, eval_state, None, &mut scratch)?;
        for (heuristic, state_id) in self.heuristics.iter().zip(&scratch.abstract_state_ids) {
            let Some(state_id) = *state_id else {
                continue;
            };
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
            if value > best {
                best = value;
            }
        }

        if best == f64::NEG_INFINITY {
            Ok(0.0)
        } else {
            Ok(best)
        }
    }

    fn heuristic_name(&self) -> String {
        self.name.clone()
    }
}
