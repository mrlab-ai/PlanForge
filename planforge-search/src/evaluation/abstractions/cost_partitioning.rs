//! Admissible combinations of abstraction heuristics: maximization,
//! canonical heuristics, saturated cost partitioning and post-hoc
//! optimization.

pub mod canonical;
pub mod costs;
pub mod max;
pub mod posthoc_optimization;
pub mod saturated;

use planforge_sas::explicit_state::StateValues;

use crate::evaluation::abstractions::component::AbstractionComponent;
/// The goal distance that `component`, the `component_id`-th of a
/// combination, estimates on its own for the state `values`. An abstraction's
/// goal distances are non-negative, so another estimate is a broken
/// invariant.
fn standalone_estimate(
    component: &AbstractionComponent<'_>,
    component_id: usize,
    values: StateValues<'_>,
) -> f64 {
    let value = component.standalone_value(values);
    assert!(
        value >= 0.0,
        "{} component {component_id} estimates the invalid goal distance {value}",
        component.kind()
    );
    value
}
