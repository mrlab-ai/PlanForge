use super::*;

/// Everything a cost partitioning over this collection needs except the order
/// it is built in: the task, its abstraction components, where one state lands
/// in every component and what each scores there standalone, and the original
/// operator costs the partition divides up.
///
/// Diversification evaluates the same collection at a sample rather than at the
/// state being expanded, which is `PartitionedCollection { abstract_state_ids,
/// abstraction_h, ..collection }`.
#[derive(Clone, Copy)]
pub(super) struct PartitionedCollection<'a, 'task> {
    pub(super) task: &'a dyn AbstractNumericTask,
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
/// saturates, by `method`; see [`saturate_labels`] and [`saturate_regions`].
/// A pattern database has no source sets and saturates labels either way.
fn saturate_component(
    component: &AbstractionComponent<'_>,
    task: &dyn AbstractNumericTask,
    remaining: &RemainingCosts,
    method: CostPartitioningMethod,
    perimeter_state: Option<usize>,
    deadline: Deadline,
) -> anyhow::Result<(Vec<f64>, SaturatedCosts)> {
    let regional = |(distances, allocation)| (distances, SaturatedCosts::Region(allocation));
    match (method, component) {
        (CostPartitioningMethod::Region, AbstractionComponent::Domain(abstraction)) => abstraction
            .saturate_regions(task, remaining, perimeter_state, deadline)
            .map(regional),
        (CostPartitioningMethod::Region, AbstractionComponent::Cartesian(abstraction)) => {
            abstraction
                .saturate_regions(remaining, perimeter_state, deadline)
                .map(regional)
        }
        (CostPartitioningMethod::Label, _)
        | (CostPartitioningMethod::Region, AbstractionComponent::PatternDatabase(_)) => {
            saturate_component_labels(
                component,
                task,
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
    task: &dyn AbstractNumericTask,
    operator_costs: &[f64],
    perimeter_state: Option<usize>,
    deadline: Deadline,
) -> anyhow::Result<(Vec<f64>, Vec<f64>)> {
    match component {
        AbstractionComponent::Domain(abstraction) => {
            abstraction.saturate_labels(task, operator_costs, perimeter_state, deadline)
        }
        AbstractionComponent::Cartesian(abstraction) => {
            abstraction.saturate_labels(operator_costs, perimeter_state, deadline)
        }
        AbstractionComponent::PatternDatabase(pdb) => match perimeter_state {
            Some(state) => pdb.build_cost_partitioned_distance_table_capped(operator_costs, state),
            None => pdb.build_cost_partitioned_distance_table(operator_costs),
        }
        .map_err(anyhow::Error::msg),
    }
}

impl<'task> SaturatedCostPartitioningHeuristic<'task> {
    /// The cost partitioning of `order`: every component in turn saturates
    /// what the ones before it left, until the order or `deadline` ends.
    pub(super) fn build_cp(
        &self,
        collection: PartitionedCollection<'_, '_>,
        order: &[usize],
        deadline: Deadline,
    ) -> Result<CostPartitioningHeuristic, EvaluationError> {
        let mut cp = CostPartitioningHeuristic::default();
        let mut remaining = RemainingCosts::from_operator_costs(collection.original_costs);
        for &component_id in order {
            if deadline.passed() {
                break;
            }
            match self.add_saturation_steps(
                &mut cp,
                &mut remaining,
                collection,
                component_id,
                deadline,
            ) {
                Ok(true) => {}
                Ok(false) => {
                    info!(
                        "scp: component {component_id} stopped while reducing remaining costs (deadline)"
                    );
                    break;
                }
                Err(error) if Self::is_deadline_exceeded(&error) => {
                    info!(
                        "scp: component {component_id} stopped while computing its table (deadline)"
                    );
                    break;
                }
                Err(error) => return Err(error),
            }
        }
        Ok(cp)
    }

    /// Saturates component `component_id` in the passes of the configured
    /// saturator, subtracts what it saturates from `remaining`, and adds its
    /// tables to `cp`. `false` if the deadline passed while reducing.
    fn add_saturation_steps(
        &self,
        cp: &mut CostPartitioningHeuristic,
        remaining: &mut RemainingCosts,
        collection: PartitionedCollection<'_, '_>,
        component_id: usize,
        deadline: Deadline,
    ) -> Result<bool, EvaluationError> {
        let PartitionedCollection {
            task,
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
        let saturator = self.config.saturator;
        if saturator != Saturator::All && state_id.is_none() {
            return Err(EvaluationError::InvalidState(format!(
                "missing abstract state id for PERIM component {component_id}"
            )));
        }
        for &(phase, limited) in saturator.passes() {
            let perimeter_state = if limited { state_id } else { None };
            let (distances, saturated) = saturate_component(
                component,
                task,
                remaining,
                method,
                perimeter_state,
                deadline,
            )
            .map_err(|error| {
                Self::construction_error(
                    &format!("failed to saturate component {component_id}"),
                    error,
                )
            })?;
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
                return Ok(true);
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
                    return Ok(false);
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
        Ok(true)
    }

    pub(super) fn is_deadline_exceeded(error: &EvaluationError) -> bool {
        matches!(error, EvaluationError::ConstructionDeadlineExceeded)
    }

    pub(super) fn construction_error(context: &str, error: anyhow::Error) -> EvaluationError {
        if crate::resource_limits::is_deadline_exceeded(&error) {
            EvaluationError::ConstructionDeadlineExceeded
        } else {
            EvaluationError::ComputationFailed(format!("{context}: {error:#}"))
        }
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
