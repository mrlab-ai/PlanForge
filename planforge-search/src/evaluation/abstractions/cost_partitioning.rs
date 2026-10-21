//! Admissible combinations of abstraction heuristics: maximization,
//! canonical heuristics, saturated cost partitioning and post-hoc
//! optimization.

pub mod canonical;
pub mod costs;
pub mod max;
// Only compiled with the `cplex` feature. Requesting the heuristic in another
// build produces an explicit configuration error in planforge-searcher.
#[cfg(feature = "cplex")]
pub mod posthoc_optimization;
pub mod saturated;

use planforge_sas::explicit_state::ExplicitState;

use crate::evaluation::abstractions::component::AbstractionComponent;
use crate::evaluation::evaluator::EvaluationError;

/// The goal distance that `component`, the `component_id`-th of a
/// combination, estimates on its own for the state `values`. An abstraction's
/// goal distances are non-negative, so another estimate is a broken
/// invariant.
fn standalone_estimate(
    component: &AbstractionComponent<'_>,
    component_id: usize,
    values: &ExplicitState,
) -> Result<f64, EvaluationError> {
    let value = component.standalone_value(values).map_err(|error| {
        EvaluationError::ComputationFailed(format!(
            "failed to evaluate {} component {component_id}: {error}",
            component.kind()
        ))
    })?;
    assert!(
        value >= 0.0,
        "{} component {component_id} estimates the invalid goal distance {value}",
        component.kind()
    );
    Ok(value)
}
