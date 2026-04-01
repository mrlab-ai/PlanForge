use crate::numeric::evaluation::evaluator::{EvaluationError, EvaluationState};
use crate::numeric::evaluation::heuristic::Heuristic;

use super::domain_abstraction_generator::DomainAbstraction;
use super::domain_abstraction_heuristic::DomainAbstractionHeuristic;

#[derive(Debug, Clone)]
pub struct MaxDomainAbstractionHeuristic {
    name: String,
    heuristics: Vec<DomainAbstractionHeuristic>,
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
        for heuristic in &self.heuristics {
            let value = heuristic.compute_heuristic(eval_state)?;
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
