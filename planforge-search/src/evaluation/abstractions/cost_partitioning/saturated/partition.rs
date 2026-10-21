use std::ops::ControlFlow;

use planforge_sas::utils::float_tolerance;
use tracing::{debug, info};

use super::SaturatedCostPartitioningHeuristic;
use super::config::CostPartitioningMethod;
use super::diagnostics::{log_abstract_source_set_summary, log_transition_residual_summary};
use super::lookup::{CostPartitioningHeuristic, current_h_for_distances};
use crate::evaluation::abstractions::component::AbstractionComponent;
use crate::evaluation::abstractions::cost_partitioning::costs::{
    RemainingCosts, SourceSetCostAllocation,
};
use crate::evaluation::evaluator::EvaluationError;
use crate::resource_limits::Deadline;

/// Everything a cost partitioning over this collection needs except the order
/// it is built in: the abstraction components, where one state lands
/// in every component and what each scores there standalone, and the original
/// operator costs the partition divides up.
///
/// Diversification evaluates the same collection at a sample rather than at the
/// state being expanded, which is `PartitionedCollection { abstract_state_ids,
/// abstraction_h, ..collection }`.
#[derive(Clone, Copy)]
pub(super) struct PartitionedCollection<'a, 'task> {
    pub(super) components: &'a [AbstractionComponent<'task>],
    pub(super) abstract_state_ids: &'a [Option<usize>],
    pub(super) abstraction_h: &'a [f64],
    pub(super) original_costs: &'a [f64],
}

/// The costs one component saturates.
pub(super) enum SaturatedCosts {
    /// One cost per operator (label SCP, and every pattern database).
    Label(Vec<f64>),
    /// One cost per source set of a transition (regional SCP).
    Region(SourceSetCostAllocation),
}

impl SaturatedCosts {
    fn amounts(&self) -> Vec<f64> {
        match self {
            Self::Label(costs) => costs.clone(),
            Self::Region(allocation) => allocation
                .entries()
                .iter()
                .map(|entry| entry.amount)
                .collect(),
        }
    }
}

/// The goal distances of `component` under `remaining` and the costs it
/// saturates, by `method`; see [`CegarAbstraction::saturate_labels`] and
/// [`CegarAbstraction::saturate_regions`]. A pattern database has no source
/// sets, so it saturates labels under either method; SCP logs that once when
/// it is built.
fn saturate_component(
    component: &AbstractionComponent<'_>,
    remaining: &RemainingCosts,
    method: CostPartitioningMethod,
    perimeter_state: Option<usize>,
    deadline: Deadline,
) -> anyhow::Result<(Vec<f64>, SaturatedCosts)> {
    match (method, component) {
        (CostPartitioningMethod::Region, AbstractionComponent::Cegar(abstraction)) => abstraction
            .saturate_regions(remaining, perimeter_state, deadline)
            .map(|(distances, allocation)| (distances, SaturatedCosts::Region(allocation))),
        (CostPartitioningMethod::Label, _)
        | (CostPartitioningMethod::Region, AbstractionComponent::PatternDatabase(_)) => {
            saturate_component_labels(
                component,
                &remaining.operator_costs_for_label_cp(),
                perimeter_state,
                deadline,
            )
            .map(|(distances, saturated)| (distances, SaturatedCosts::Label(saturated)))
        }
    }
}

/// Label saturation of `component` under `operator_costs`.
pub(super) fn saturate_component_labels(
    component: &AbstractionComponent<'_>,
    operator_costs: &[f64],
    perimeter_state: Option<usize>,
    deadline: Deadline,
) -> anyhow::Result<(Vec<f64>, Vec<f64>)> {
    match component {
        AbstractionComponent::Cegar(abstraction) => {
            abstraction.saturate_labels(operator_costs, perimeter_state, deadline)
        }
        AbstractionComponent::PatternDatabase(pdb) => match perimeter_state {
            Some(state) => pdb.build_cost_partitioned_distance_table_capped(operator_costs, state),
            None => pdb.build_cost_partitioned_distance_table(operator_costs),
        }
        .map_err(anyhow::Error::msg),
    }
}

/// Whether the construction budget let a cost partitioning saturate every
/// component of its order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Completion {
    Complete,
    /// The budget ended first; the cost partitioning has the tables of the
    /// components saturated before.
    Truncated,
}

impl<'task> SaturatedCostPartitioningHeuristic<'task> {
    /// The cost partitioning of `order`: every component in turn saturates
    /// what the ones before it left, until the order or `deadline` ends.
    pub(super) fn build_cp(
        &self,
        collection: PartitionedCollection<'_, '_>,
        order: &[usize],
        deadline: Deadline,
    ) -> Result<(CostPartitioningHeuristic, Completion), EvaluationError> {
        let mut cp = CostPartitioningHeuristic::default();
        let mut remaining = RemainingCosts::from_operator_costs(collection.original_costs);
        for (position, &component_id) in order.iter().enumerate() {
            let step = if deadline.passed() {
                ControlFlow::Break(())
            } else {
                self.add_saturation_steps(
                    &mut cp,
                    &mut remaining,
                    collection,
                    component_id,
                    deadline,
                )?
            };
            if step.is_break() {
                info!(
                    "scp: the construction budget ended after {position} of the {} components of an order; its cost partitioning keeps their tables",
                    order.len()
                );
                return Ok((cp, Completion::Truncated));
            }
        }
        Ok((cp, Completion::Complete))
    }

    /// Saturates component `component_id` in the passes of the configured
    /// saturator, subtracts what it saturates from `remaining`, and adds its
    /// tables to `cp`. Breaks when `deadline` passes during a pass, which
    /// then adds nothing.
    fn add_saturation_steps(
        &self,
        cp: &mut CostPartitioningHeuristic,
        remaining: &mut RemainingCosts,
        collection: PartitionedCollection<'_, '_>,
        component_id: usize,
        deadline: Deadline,
    ) -> Result<ControlFlow<()>, EvaluationError> {
        let PartitionedCollection {
            components,
            abstract_state_ids,
            abstraction_h,
            ..
        } = collection;
        let component = &components[component_id];
        let method = self.config.partitioning;
        debug!(
            "scp: {method} step component {component_id} ({}), states={}, h={}",
            component.kind(),
            component.num_states(),
            abstraction_h[component_id],
        );
        if method.uses_regions()
            && let Some(source_sets) = component.abstract_source_sets()
        {
            log_abstract_source_set_summary(component_id, source_sets);
        }
        let state_id = abstract_state_ids[component_id];
        for &(phase, limited) in self.config.saturator.passes() {
            let perimeter_state = match (limited, state_id) {
                (false, _) => None,
                (true, Some(state_id)) => Some(state_id),
                // A component without an abstract state for the evaluated
                // state estimates nothing there, so its perimeter holds no
                // state and the pass saturates nothing.
                (true, None) => continue,
            };
            let (distances, saturated) =
                match saturate_component(component, remaining, method, perimeter_state, deadline) {
                    Ok(saturation) => saturation,
                    Err(error) if crate::resource_limits::is_deadline_exceeded(&error) => {
                        return Ok(ControlFlow::Break(()));
                    }
                    Err(error) => {
                        return Err(EvaluationError::ComputationFailed(format!(
                            "failed to saturate component {component_id}: {error:#}"
                        )));
                    }
                };
            let step = format!("{method} {phase}");
            log_table_summary(
                &step,
                component_id,
                &distances,
                &saturated,
                abstract_state_ids,
            );
            if should_skip_zero_current_table(
                self.config.diversify,
                &step,
                component_id,
                &distances,
                abstract_state_ids,
            ) {
                return Ok(ControlFlow::Continue(()));
            }
            let reduced = match &saturated {
                SaturatedCosts::Label(costs) => remaining.reduce_operator_costs_uniform(costs),
                SaturatedCosts::Region(allocation) => {
                    remaining.reduce_by_source_set_allocation_with_deadline(allocation, deadline)
                }
            };
            match reduced {
                Ok(()) => {}
                Err(error) if crate::resource_limits::is_deadline_exceeded(&error) => {
                    return Ok(ControlFlow::Break(()));
                }
                Err(error) => {
                    return Err(EvaluationError::ComputationFailed(format!(
                        "failed to reduce the remaining costs by component {component_id}: {error:#}"
                    )));
                }
            }
            if method.uses_regions() {
                log_transition_residual_summary(remaining);
            }
            cp.add_h_values(component_id, distances);
        }
        Ok(ControlFlow::Continue(()))
    }
}

pub(super) fn should_skip_zero_current_table(
    diversify: bool,
    step: &str,
    abstraction_id: usize,
    distances: &[f64],
    abstract_state_ids: &[Option<usize>],
) -> bool {
    if diversify {
        return false;
    }
    let current_h = current_h_for_distances(abstraction_id, distances, abstract_state_ids);
    if current_h > float_tolerance::SEARCH_EPSILON {
        return false;
    }
    debug!("scp: skipping {step} abstraction {abstraction_id}: current_h=0");
    true
}

fn log_table_summary(
    step: &str,
    abstraction_id: usize,
    distances: &[f64],
    saturated: &SaturatedCosts,
    abstract_state_ids: &[Option<usize>],
) {
    if !tracing::enabled!(tracing::Level::DEBUG) {
        return;
    }
    let (positive_count, total_positive) = saturated
        .amounts()
        .into_iter()
        .filter(|cost| cost.is_finite() && *cost > 0.0)
        .fold((0, 0.0), |(count, total), cost| (count + 1, total + cost));
    let current_h = current_h_for_distances(abstraction_id, distances, abstract_state_ids);
    debug!(
        "scp: {step} abstraction {abstraction_id}: current_h={current_h}, positive_saturated_costs={positive_count}, total_positive_saturated={total_positive:.6}"
    );
}
