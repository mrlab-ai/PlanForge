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

/// The costs one abstraction saturates, in the granularity of the remaining
/// cost function they are subtracted from.
pub(super) enum SaturatedCosts {
    /// One cost per operator (label SCP).
    Label(Vec<f64>),
    /// One cost per abstract operator, valid on its source sets.
    AbstractOperator(AbstractOperatorCostFunction),
    /// One cost per source set of an abstract transition (interval SCP).
    Interval(SourceSetCostAllocation),
}

/// Where remaining costs live while an order is saturated.
#[derive(Clone, Copy)]
pub(super) enum SaturationCostSpace {
    /// One remaining cost per operator.
    Label,
    /// Remaining costs per Cartesian set of states (interval SCP).
    Interval,
}

pub(super) struct ComponentSaturation<'a, 'task> {
    pub(super) component: &'a AbstractionComponent<'task>,
    pub(super) task: &'a dyn AbstractNumericTask,
    pub(super) combine_labels: bool,
    pub(super) cost_space: SaturationCostSpace,
}

pub(super) struct SaturationStepContext<'a> {
    pub(super) component_id: usize,
    pub(super) current_state_id: Option<usize>,
    pub(super) abstract_state_ids: &'a [Option<usize>],
    pub(super) saturator: Saturator,
    pub(super) step_prefix: &'a str,
}

pub(super) struct ComponentStepContext<'a, 'task> {
    pub(super) component_id: usize,
    pub(super) component: &'a AbstractionComponent<'task>,
    pub(super) task: &'a dyn AbstractNumericTask,
    pub(super) abstract_state_ids: &'a [Option<usize>],
    pub(super) deadline: Option<Instant>,
    pub(super) cost_space: SaturationCostSpace,
    pub(super) saturator: Saturator,
    pub(super) step_prefix: &'a str,
}

impl ComponentSaturation<'_, '_> {
    pub(super) fn saturate(
        &self,
        residual: &RemainingCosts,
        component_id: usize,
        cap_state_id: Option<usize>,
        deadline: Option<Instant>,
    ) -> Result<(Vec<f64>, SaturatedCosts), EvaluationError> {
        match (self.component, self.cost_space) {
            (AbstractionComponent::Domain(heuristic), SaturationCostSpace::Label) => {
                let abstraction = heuristic.abstraction();
                let abstraction_task = abstraction.task_for_factory(self.task);
                let operator_costs = residual.operator_costs_for_label_cp();
                let (distances, saturated) = if let Some(state_id) = cap_state_id {
                    let table = abstraction
                        .factory
                        .build_goal_distances_for_goals(
                            abstraction_task,
                            self.combine_labels,
                            &operator_costs,
                            &abstraction.distance_table.goal_facts,
                        )
                        .map_err(|error| {
                            EvaluationError::ComputationFailed(format!(
                                "failed to compute domain PERIM cap table: {error:#}"
                            ))
                        })?;
                    let h_cap = table.distances.get(state_id).copied().ok_or_else(|| {
                        EvaluationError::InvalidState(format!(
                            "domain component {component_id} state id {state_id} out of bounds for {} states",
                            table.distances.len()
                        ))
                    })?;
                    SaturatedCostPartitioningHeuristic::saturate_domain_perim(
                        abstraction,
                        abstraction_task,
                        self.combine_labels,
                        &operator_costs,
                        h_cap,
                    )?
                } else {
                    SaturatedCostPartitioningHeuristic::saturate_domain(
                        abstraction,
                        abstraction_task,
                        self.combine_labels,
                        &operator_costs,
                        deadline,
                    )?
                };
                Ok((distances, SaturatedCosts::Label(saturated)))
            }
            (AbstractionComponent::Domain(heuristic), SaturationCostSpace::Interval) => {
                let abstraction = heuristic.abstraction();
                let abstraction_task = abstraction.task_for_factory(self.task);
                let transition_system = abstraction
                    .interval_transition_system(abstraction_task, deadline)
                    .map_err(|error| {
                        SaturatedCostPartitioningHeuristic::construction_error(
                            "failed to build domain regional transition system",
                            error,
                        )
                    })?;
                let (table, saturated) = abstraction
                    .factory
                    .build_interval_cost_partitioned_distance_table(
                        &transition_system,
                        &abstraction.abstract_source_sets,
                        residual,
                        component_id,
                        DistanceTableOptions::default()
                            .with_cap_state(cap_state_id)
                            .with_deadline(deadline),
                    )
                    .map_err(|error| {
                        SaturatedCostPartitioningHeuristic::construction_error(
                            "failed to compute domain regional saturation",
                            error,
                        )
                    })?;
                Ok((table.distances, SaturatedCosts::Interval(saturated)))
            }
            (AbstractionComponent::Cartesian(heuristic), SaturationCostSpace::Label) => {
                let (distances, saturated) = build_explicit_label_cost_partitioning_table(
                    &heuristic.abstraction().transition_system,
                    &residual.operator_costs_for_label_cp(),
                    cap_state_id,
                    deadline,
                )
                .map_err(|error| {
                    SaturatedCostPartitioningHeuristic::construction_error(
                        &format!(
                            "failed to compute Cartesian label saturation for component {component_id}"
                        ),
                        error,
                    )
                })?;
                Ok((distances, SaturatedCosts::Label(saturated)))
            }
            (AbstractionComponent::Cartesian(heuristic), SaturationCostSpace::Interval) => {
                // Phase 5 should give Cartesian components native regional allocations;
                // the Regional arm currently degrades to abstract-operator granularity.
                let abstraction = heuristic.abstraction();
                let (distances, saturated) = build_explicit_regional_cost_partitioning_table(
                    &abstraction.transition_system,
                    &abstraction.abstract_source_sets,
                    residual,
                    component_id,
                    cap_state_id,
                    deadline,
                )
                .map_err(|error| {
                    SaturatedCostPartitioningHeuristic::construction_error(
                        &format!(
                            "failed to compute Cartesian regional saturation for component {component_id}"
                        ),
                        error,
                    )
                })?;
                Ok((distances, SaturatedCosts::AbstractOperator(saturated)))
            }
            (AbstractionComponent::PatternDatabase(pdb), _) => {
                // Phase 5 will add regional PDB support; until then every PDB
                // saturation deliberately degrades to label granularity.
                let operator_costs = residual.operator_costs_for_label_cp();
                let (distances, saturated) = if let Some(state_id) = cap_state_id {
                    let cap_distances = pdb.build_goal_distances(&operator_costs).map_err(|error| {
                        EvaluationError::ComputationFailed(format!(
                            "failed to compute PDB PERIM cap table {component_id}: {error}"
                        ))
                    })?;
                    let h_cap = cap_distances.get(state_id).copied().ok_or_else(|| {
                        EvaluationError::InvalidState(format!(
                            "PDB component {component_id} state id {state_id} out of bounds for {} states",
                            cap_distances.len()
                        ))
                    })?;
                    pdb.build_cost_partitioned_distance_table_capped(&operator_costs, h_cap)
                } else {
                    pdb.build_cost_partitioned_distance_table(&operator_costs)
                }
                .map_err(|error| {
                    EvaluationError::ComputationFailed(format!(
                        "failed to compute PDB saturation {component_id}: {error}"
                    ))
                })?;
                Ok((distances, SaturatedCosts::Label(saturated)))
            }
        }
    }
}

impl<'task> SaturatedCostPartitioningHeuristic<'task> {
    pub(super) fn add_saturation_steps<R, Build, Reduce, Log>(
        &self,
        cp: &mut CostPartitioningHeuristic,
        residual: &mut R,
        context: SaturationStepContext<'_>,
        mut build: Build,
        mut reduce: Reduce,
        mut log: Log,
    ) -> Result<bool, EvaluationError>
    where
        R: ?Sized,
        Build: FnMut(&R, Option<usize>) -> Result<(Vec<f64>, SaturatedCosts), EvaluationError>,
        Reduce: FnMut(&mut R, &SaturatedCosts) -> Result<bool, EvaluationError>,
        Log: FnMut(&str, &[f64], &SaturatedCosts),
    {
        let SaturationStepContext {
            component_id,
            current_state_id,
            abstract_state_ids,
            saturator,
            step_prefix,
        } = context;
        if saturator != Saturator::All && current_state_id.is_none() {
            return Err(EvaluationError::InvalidState(format!(
                "missing abstract state id for PERIM component {component_id}"
            )));
        }
        for &(phase, capped) in saturator.passes() {
            let cap_state_id = if capped { current_state_id } else { None };
            let step = format!("{step_prefix} {phase}");
            let (distances, saturated) = build(residual, cap_state_id)?;
            log(&step, &distances, &saturated);
            if should_skip_zero_current_table(
                self.config.diversify,
                &step,
                component_id,
                &distances,
                abstract_state_ids,
            ) {
                return Ok(true);
            }
            if !reduce(residual, &saturated)? {
                return Ok(false);
            }
            cp.add_h_values(component_id, distances);
        }
        Ok(true)
    }

    pub(super) fn add_label_saturation_step(
        &self,
        cp: &mut CostPartitioningHeuristic,
        remaining_costs: &mut [f64],
        context: ComponentStepContext<'_, '_>,
    ) -> Result<bool, EvaluationError> {
        let ComponentStepContext {
            component_id,
            component,
            task,
            abstract_state_ids,
            deadline,
            step_prefix,
            ..
        } = context;
        let current_state_id = abstract_state_ids.get(component_id).copied().flatten();
        let protocol = ComponentSaturation {
            component,
            task,
            combine_labels: self.config.combine_labels,
            cost_space: SaturationCostSpace::Label,
        };
        self.add_saturation_steps(
            cp,
            remaining_costs,
            SaturationStepContext {
                component_id,
                current_state_id,
                abstract_state_ids,
                saturator: self.config.saturator,
                step_prefix,
            },
            |costs, cap_state_id| {
                let residual = RemainingCosts::from_operator_costs(costs);
                protocol.saturate(&residual, component_id, cap_state_id, deadline)
            },
            |costs, saturated| {
                let SaturatedCosts::Label(saturated) = saturated else {
                    unreachable!("label saturation must return uniform operator costs")
                };
                reduce_costs(costs, saturated)?;
                Ok(true)
            },
            |step, distances, saturated| {
                let SaturatedCosts::Label(saturated) = saturated else {
                    unreachable!("label saturation must return uniform operator costs")
                };
                log_label_table_summary(
                    step,
                    component_id,
                    distances,
                    saturated,
                    abstract_state_ids,
                );
            },
        )
    }

    pub(super) fn add_saturation_step(
        &self,
        cp: &mut CostPartitioningHeuristic,
        remaining_costs: &mut RemainingCosts,
        context: ComponentStepContext<'_, '_>,
    ) -> Result<bool, EvaluationError> {
        let ComponentStepContext {
            component_id,
            component,
            task,
            abstract_state_ids,
            deadline,
            cost_space,
            saturator,
            step_prefix,
        } = context;
        let current_state_id = abstract_state_ids.get(component_id).copied().flatten();
        let protocol = ComponentSaturation {
            component,
            task,
            combine_labels: self.config.combine_labels,
            cost_space,
        };
        let source_sets = match component {
            AbstractionComponent::Domain(heuristic) => {
                heuristic.abstraction().abstract_source_sets.as_slice()
            }
            AbstractionComponent::Cartesian(heuristic) => {
                heuristic.abstraction().abstract_source_sets.as_slice()
            }
            AbstractionComponent::PatternDatabase(_) => &[],
        };
        self.add_saturation_steps(
            cp,
            remaining_costs,
            SaturationStepContext {
                component_id,
                current_state_id,
                abstract_state_ids,
                saturator,
                step_prefix,
            },
            |residual, cap_state_id| {
                protocol.saturate(residual, component_id, cap_state_id, deadline)
            },
            |residual, saturated| match saturated {
                SaturatedCosts::Label(costs) => {
                    residual
                        .reduce_operator_costs_uniform(costs)
                        .map_err(|error| {
                            EvaluationError::ComputationFailed(format!(
                                "failed to reduce uniform residual costs for component {component_id}: {error:#}"
                            ))
                        })?;
                    Ok(true)
                }
                SaturatedCosts::AbstractOperator(costs) => Self::reduce_abstract_operator_costs(
                    residual,
                    component_id,
                    source_sets,
                    costs,
                    deadline,
                    &format!(
                        "failed to reduce abstract-operator residual costs for component {component_id}"
                    ),
                ),
                SaturatedCosts::Interval(allocation) => Self::reduce_source_set_allocation(
                    residual,
                    allocation,
                    deadline,
                    &format!(
                        "failed to reduce regional residual costs for component {component_id}"
                    ),
                ),
            },
            |step, distances, saturated| match saturated {
                SaturatedCosts::Label(costs) => log_transition_table_summary(
                    step,
                    component_id,
                    distances,
                    costs,
                    abstract_state_ids,
                ),
                SaturatedCosts::AbstractOperator(costs) => log_transition_table_summary(
                    step,
                    component_id,
                    distances,
                    &costs.operator_costs,
                    abstract_state_ids,
                ),
                SaturatedCosts::Interval(allocation) => log_transition_table_summary(
                    step,
                    component_id,
                    distances,
                    &allocation_amounts(allocation),
                    abstract_state_ids,
                ),
            },
        )
    }

    pub(super) fn is_deadline_error(error: &anyhow::Error) -> bool {
        crate::resource_limits::is_deadline_exceeded(error)
    }

    pub(super) fn is_deadline_exceeded(error: &EvaluationError) -> bool {
        matches!(error, EvaluationError::ConstructionDeadlineExceeded)
    }

    pub(super) fn construction_error(context: &str, error: anyhow::Error) -> EvaluationError {
        if Self::is_deadline_error(&error) {
            EvaluationError::ConstructionDeadlineExceeded
        } else {
            EvaluationError::ComputationFailed(format!("{context}: {error:#}"))
        }
    }

    pub(super) fn reduce_abstract_operator_costs(
        remaining_costs: &mut RemainingCosts,
        abstraction_id: usize,
        source_sets: &[AbstractOperatorSourceSets],
        tcf: &AbstractOperatorCostFunction,
        deadline: Option<Instant>,
        context: &str,
    ) -> Result<bool, EvaluationError> {
        match remaining_costs.reduce_by_abstract_source_sets_with_deadline(
            abstraction_id,
            source_sets,
            tcf,
            deadline,
        ) {
            Ok(()) => Ok(true),
            Err(error) if Self::is_deadline_error(&error) => Ok(false),
            Err(error) => Err(EvaluationError::ComputationFailed(format!(
                "{context}: {error:#}"
            ))),
        }
    }

    pub(super) fn reduce_source_set_allocation(
        remaining_costs: &mut RemainingCosts,
        allocation: &SourceSetCostAllocation,
        deadline: Option<Instant>,
        context: &str,
    ) -> Result<bool, EvaluationError> {
        match remaining_costs.reduce_by_source_set_allocation_with_deadline(allocation, deadline) {
            Ok(()) => Ok(true),
            Err(error) if Self::is_deadline_error(&error) => Ok(false),
            Err(error) => Err(EvaluationError::ComputationFailed(format!(
                "{context}: {error:#}"
            ))),
        }
    }

    /// The cost partitioning of `order`, label SCP or interval SCP as
    /// configured.
    pub(super) fn build_cp(
        &self,
        collection: PartitionedCollection<'_, '_>,
        order: &[usize],
        deadline: Option<Instant>,
    ) -> Result<CostPartitioningHeuristic, EvaluationError> {
        match self.config.partitioning {
            CostPartitioningMethod::Label => self.build_label_cp(collection, order, deadline),
            CostPartitioningMethod::Region => {
                self.build_interval_cp(collection, order, deadline, self.config.saturator)
            }
        }
    }

    pub(super) fn build_interval_cp(
        &self,
        collection: PartitionedCollection<'_, '_>,
        order: &[usize],
        deadline: Option<Instant>,
        saturator: Saturator,
    ) -> Result<CostPartitioningHeuristic, EvaluationError> {
        let PartitionedCollection {
            task,
            components,
            abstract_state_ids,
            abstraction_h,
            original_costs,
        } = collection;
        let mut cp = CostPartitioningHeuristic::default();
        let mut remaining_costs = RemainingCosts::from_operator_costs(original_costs);

        for sweep in 0..=self.config.residual_sweeps {
            if sweep > 0 {
                info!(
                    "scp: starting regional residual abstraction sweep {sweep}/{}",
                    self.config.residual_sweeps
                );
            }
            for &component_id in order {
                if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                    break;
                }
                let component = components.get(component_id).ok_or_else(|| {
                    EvaluationError::InvalidState(format!(
                        "regional SCP order references missing component {component_id}"
                    ))
                })?;
                if let AbstractionComponent::Domain(heuristic) = component {
                    let abstraction = heuristic.abstraction();
                    debug!(
                        "scp: abstract-operator CP step abstraction {component_id}, abstract_states={}, abstraction_h={}, metadata={}",
                        abstraction_state_count(abstraction),
                        abstraction_h[component_id],
                        abstraction_metadata_summary(abstraction),
                    );
                    log_abstract_source_set_summary(
                        component_id,
                        &abstraction.abstract_source_sets,
                    );
                    self.log_abstract_operator_label_diagnostic(
                        abstraction,
                        abstraction.task_for_factory(task),
                        component_id,
                        abstract_state_ids,
                        &remaining_costs,
                    )?;
                }

                let result = self.add_saturation_step(
                    &mut cp,
                    &mut remaining_costs,
                    ComponentStepContext {
                        component_id,
                        component,
                        task,
                        abstract_state_ids,
                        deadline,
                        cost_space: SaturationCostSpace::Interval,
                        saturator,
                        step_prefix: "abstract-operator",
                    },
                );
                match result {
                    Ok(true) => log_transition_residual_summary(&remaining_costs),
                    Ok(false) => {
                        info!(
                            "scp: abstract-operator component {component_id} stopped while reducing residual costs (deadline)"
                        );
                        break;
                    }
                    Err(error) if Self::is_deadline_exceeded(&error) => {
                        info!(
                            "scp: abstract-operator component {component_id} stopped while computing table (deadline)"
                        );
                        break;
                    }
                    Err(error) => return Err(error),
                }
            }
        }

        Ok(cp)
    }

    pub(super) fn build_label_cp(
        &self,
        collection: PartitionedCollection<'_, '_>,
        order: &[usize],
        deadline: Option<Instant>,
    ) -> Result<CostPartitioningHeuristic, EvaluationError> {
        let PartitionedCollection {
            task,
            components,
            abstract_state_ids,
            original_costs,
            ..
        } = collection;
        let mut cp = CostPartitioningHeuristic::default();
        let mut remaining_costs = original_costs.to_vec();

        for &component_id in order {
            if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                break;
            }
            let component = components.get(component_id).ok_or_else(|| {
                EvaluationError::InvalidState(format!(
                    "label SCP order references missing component {component_id}"
                ))
            })?;
            match self.add_label_saturation_step(
                &mut cp,
                &mut remaining_costs,
                ComponentStepContext {
                    component_id,
                    component,
                    task,
                    abstract_state_ids,
                    deadline,
                    cost_space: SaturationCostSpace::Label,
                    saturator: self.config.saturator,
                    step_prefix: "label",
                },
            ) {
                Ok(true) => {}
                Ok(false) => unreachable!("label residual reduction has no deadline"),
                Err(error) if Self::is_deadline_exceeded(&error) => {
                    info!(
                        "scp: label component {component_id} stopped while computing table (deadline)"
                    );
                    break;
                }
                Err(error) => return Err(error),
            }
        }

        Ok(cp)
    }

    pub(super) fn saturate_domain(
        abstraction: &DomainAbstraction,
        task: &dyn AbstractNumericTask,
        combine_labels: bool,
        remaining_costs: &[f64],
        deadline: Option<Instant>,
    ) -> Result<(Vec<f64>, Vec<f64>), EvaluationError> {
        let start = Instant::now();
        let (table, saturated) = abstraction
            .factory
            .build_cost_partitioned_distance_table(
                task,
                combine_labels,
                remaining_costs,
                DistanceTableOptions::default()
                    .with_goal_facts(&abstraction.distance_table.goal_facts)
                    .with_deadline(deadline),
            )
            .map_err(|error| Self::construction_error("failed to compute SCP table", error))?;
        debug!(
            "scp: label distance-table/CP construction finished in {:.3}s, states={}, saturated_costs={}",
            start.elapsed().as_secs_f64(),
            table.distances.len(),
            saturated.len()
        );
        Ok((table.distances, saturated))
    }

    /// Build a PERIM domain CP entry.
    ///
    /// Saturated costs are computed from a table where states outside the
    /// perimeter are inactive. The returned lookup table is then recomputed
    /// globally from those saturated costs, so stored online CPs remain valid
    /// for later states outside the original perimeter.
    pub(super) fn saturate_domain_perim(
        abstraction: &DomainAbstraction,
        task: &dyn AbstractNumericTask,
        combine_labels: bool,
        remaining_costs: &[f64],
        h_cap: f64,
    ) -> Result<(Vec<f64>, Vec<f64>), EvaluationError> {
        let (table, _) = abstraction
            .factory
            .build_cost_partitioned_distance_table(
                task,
                combine_labels,
                remaining_costs,
                DistanceTableOptions::default()
                    .with_goal_facts(&abstraction.distance_table.goal_facts),
            )
            .map_err(|error| {
                EvaluationError::ComputationFailed(format!(
                    "failed to compute SCP table for PERIM: {error:#}"
                ))
            })?;
        let mut perim_distances = table.distances;
        if h_cap.is_finite() {
            for h in &mut perim_distances {
                if !h.is_finite() || *h > h_cap {
                    *h = f64::NEG_INFINITY;
                }
            }
        }
        let mut generator = abstraction
            .factory
            .make_operator_generator(task, combine_labels)
            .map_err(|error| {
                EvaluationError::ComputationFailed(format!(
                    "failed to create operator generator for PERIM: {error:#}"
                ))
            })?;
        let mut operators = generator.build_abstract_operators(task).map_err(|error| {
            EvaluationError::ComputationFailed(format!(
                "failed to build abstract operators for PERIM: {error:#}"
            ))
        })?;
        apply_operator_costs_from_slice(&mut operators, remaining_costs)?;
        let perim_table = AbstractDistanceTable {
            distances: perim_distances,
            generating_op_ids: table.generating_op_ids,
            initial_state_hash: table.initial_state_hash,
            goal_facts: table.goal_facts,
            hash_multipliers: table.hash_multipliers,
            numeric_domain_sizes: table.numeric_domain_sizes,
        };
        let saturated = abstraction
            .factory
            .saturated_costs_for_table(task, combine_labels, &operators, &perim_table)
            .map_err(|error| {
                EvaluationError::ComputationFailed(format!(
                    "failed to compute PERIM saturated costs: {error:#}"
                ))
            })?;
        let global_table = abstraction
            .factory
            .build_goal_distances_for_goals(
                task,
                combine_labels,
                &saturated,
                &abstraction.distance_table.goal_facts,
            )
            .map_err(|error| {
                EvaluationError::ComputationFailed(format!(
                    "failed to compute global PERIM lookup table: {error:#}"
                ))
            })?;
        Ok((global_table.distances, saturated))
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
    if current_h > 1e-9 {
        return false;
    }
    debug!("scp: skipping {step} abstraction {abstraction_id}: current_h=0");
    true
}

pub(super) fn positive_cost_stats(costs: &[f64]) -> (usize, f64) {
    costs
        .iter()
        .copied()
        .filter(|cost| cost.is_finite() && *cost > 0.0)
        .fold((0, 0.0), |(count, total), cost| (count + 1, total + cost))
}

pub(super) fn allocation_amounts(allocation: &SourceSetCostAllocation) -> Vec<f64> {
    allocation
        .entries()
        .iter()
        .map(|entry| entry.amount)
        .collect()
}

pub(super) fn apply_operator_costs_from_slice(
    operators: &mut [AbstractOperator],
    operator_costs: &[f64],
) -> Result<(), EvaluationError> {
    for op in operators {
        if op.concrete_op_ids.is_empty() {
            return Err(EvaluationError::ComputationFailed(
                "abstract operator without concrete labels".to_string(),
            ));
        }
        let mut cost = f64::INFINITY;
        for &concrete_op_id in &op.concrete_op_ids {
            let concrete_cost = operator_costs
                .get(concrete_op_id.index())
                .copied()
                .ok_or_else(|| {
                    EvaluationError::ComputationFailed(format!(
                        "missing residual cost for concrete operator {concrete_op_id}"
                    ))
                })?;
            if !concrete_cost.is_finite() {
                return Err(EvaluationError::ComputationFailed(format!(
                    "residual cost for concrete operator {concrete_op_id} must be finite"
                )));
            }
            cost = cost.min(concrete_cost);
        }
        op.cost = NumericValue::new(cost);
    }
    Ok(())
}

pub(super) fn reduce_costs(
    remaining_costs: &mut [f64],
    saturated_costs: &[f64],
) -> Result<(), EvaluationError> {
    if remaining_costs.len() != saturated_costs.len() {
        return Err(EvaluationError::ComputationFailed(format!(
            "cost vector length mismatch: remaining={}, saturated={}",
            remaining_costs.len(),
            saturated_costs.len()
        )));
    }
    for (op_id, (r, s)) in remaining_costs
        .iter_mut()
        .zip(saturated_costs.iter())
        .enumerate()
    {
        if !s.is_finite() {
            return Err(EvaluationError::ComputationFailed(format!(
                "saturated cost for operator {op_id} must be finite"
            )));
        }
        if *s < -1e-9 {
            return Err(EvaluationError::ComputationFailed(format!(
                "negative saturated cost for operator {op_id}: {s}"
            )));
        }
        if *s <= 1e-9 {
            continue;
        }
        let new_remaining = *r - *s;
        if new_remaining < -1e-9 {
            return Err(EvaluationError::ComputationFailed(format!(
                "label residual cost underflow for operator {op_id}: remaining={r}, saturated={s}, result={new_remaining}"
            )));
        }
        *r = new_remaining.max(0.0);
    }
    Ok(())
}

/// The goal distances of `component` under the original operator costs, the
/// costs it saturates there, and its initial abstract state. Order generation
/// scores abstractions by these distances and the costs they steal.
pub(super) fn saturate_original_costs(
    component: &AbstractionComponent<'_>,
    component_id: usize,
    task: &dyn AbstractNumericTask,
    combine_labels: bool,
    original_costs: &[f64],
    debug_diagnostics: bool,
) -> Result<(Vec<f64>, Vec<f64>, usize), EvaluationError> {
    Ok(match component {
        AbstractionComponent::Domain(heuristic) => {
            let abstraction = heuristic.abstraction();
            let abstraction_task = abstraction.task_for_factory(task);
            let goal_facts = &abstraction.distance_table.goal_facts;
            let table = abstraction
                .factory
                .build_goal_distances_for_goals(
                    abstraction_task,
                    combine_labels,
                    original_costs,
                    goal_facts,
                )
                .map_err(|error| {
                    EvaluationError::ComputationFailed(format!(
                        "failed to compute goal distances for order generator: {error:#}"
                    ))
                })?;
            let (_, saturated) = abstraction
                .factory
                .build_cost_partitioned_distance_table(
                    abstraction_task,
                    combine_labels,
                    original_costs,
                    DistanceTableOptions::default().with_goal_facts(goal_facts),
                )
                .map_err(|error| {
                    EvaluationError::ComputationFailed(format!(
                        "failed to compute saturated costs for order generator: {error:#}"
                    ))
                })?;
            if debug_diagnostics {
                info!(
                    "scp debug: collection abstraction {component_id}: states={}, goal_facts={}",
                    abstraction_state_count(abstraction),
                    goal_facts.len()
                );
            }
            (table.distances, saturated, table.initial_state_hash)
        }
        AbstractionComponent::Cartesian(heuristic) => {
            let abstraction = heuristic.abstraction();
            let (distances, saturated) = build_explicit_label_cost_partitioning_table(
                &abstraction.transition_system,
                original_costs,
                None,
                None,
            )
            .map_err(|error| {
                EvaluationError::ComputationFailed(format!(
                    "failed to compute Cartesian order-generator table {component_id}: {error:#}"
                ))
            })?;
            (
                distances,
                saturated,
                abstraction.transition_system.initial_state_hash,
            )
        }
        AbstractionComponent::PatternDatabase(pdb) => {
            let distances = pdb.build_goal_distances(original_costs).map_err(|error| {
                EvaluationError::ComputationFailed(format!(
                    "failed to compute PDB goal distances for order generator: {error}"
                ))
            })?;
            let (_, saturated) = pdb
                .build_cost_partitioned_distance_table(original_costs)
                .map_err(|error| {
                    EvaluationError::ComputationFailed(format!(
                        "failed to compute PDB saturated costs for order generator: {error}"
                    ))
                })?;
            (distances, saturated, 0)
        }
    })
}
