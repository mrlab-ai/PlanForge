use planforge_sas::numeric_task::NumericValue;

use super::*;

pub(super) fn apply_operator_costs(
    operators: &mut [AbstractOperator],
    operator_costs: &[f64],
) -> Result<()> {
    for op in operators {
        ensure!(
            !op.concrete_op_ids.is_empty(),
            "abstract operator without concrete labels"
        );
        let mut cost = f64::INFINITY;
        for &concrete_op_id in &op.concrete_op_ids {
            let concrete_cost = *operator_costs
                .get(concrete_op_id.index())
                .with_context(|| {
                    format!("missing residual cost for concrete operator {concrete_op_id}")
                })?;
            ensure!(
                concrete_cost.is_finite(),
                "residual cost for concrete operator {concrete_op_id} must be finite"
            );
            cost = cost.min(concrete_cost);
        }
        op.cost = NumericValue::new(cost);
    }
    Ok(())
}

/// Coarse wall-clock split of one regional table build.
///
/// Phase granularity on purpose: the two transition loops call into region
/// geometry once per (transition, operator) pair, so timing individual calls
/// would cost more than the phases being measured.
#[derive(Default)]
struct RegionalTableTimings {
    transition_costs: std::time::Duration,
    distance_table: std::time::Duration,
    saturation: std::time::Duration,
    lookup_table: std::time::Duration,
    allocation_entries: std::time::Duration,
}

impl RegionalTableTimings {
    fn log(
        &self,
        abstraction_id: usize,
        transitions: usize,
        operator_pairs: usize,
        entries: usize,
    ) {
        tracing::debug!(
            "regional table {abstraction_id}: transitions={transitions} operator_pairs={operator_pairs} \
             entries={entries} | transition_costs={:.3}s distance_table={:.3}s saturation={:.3}s \
             lookup_table={:.3}s allocation_entries={:.3}s",
            self.transition_costs.as_secs_f64(),
            self.distance_table.as_secs_f64(),
            self.saturation.as_secs_f64(),
            self.lookup_table.as_secs_f64(),
            self.allocation_entries.as_secs_f64(),
        );
    }
}

impl DomainAbstractionFactory {
    /// Builds an abstract distance table using the supplied per-concrete-operator costs and
    /// returns the saturated costs induced by the resulting distances.
    pub fn build_cost_partitioned_distance_table(
        &self,
        task: &dyn AbstractNumericTask,
        combine_labels: bool,
        operator_costs: &[f64],
        options: DistanceTableOptions<'_>,
    ) -> Result<(AbstractDistanceTable, Vec<f64>)> {
        let deadline = options.deadline;
        deadline.ensure(ONLINE_SCP)?;
        let mut generator = self.make_operator_generator(task, combine_labels)?;
        let mut operators = generator.build_abstract_operators(task)?;
        deadline.ensure(ONLINE_SCP)?;
        apply_operator_costs(&mut operators, operator_costs)?;
        let match_tree = MatchTree::new(&generator, &operators);
        let table = self.build_distance_table_with_match_tree(
            task,
            &generator,
            &operators,
            &match_tree,
            options,
        )?;
        deadline.ensure(ONLINE_SCP)?;
        let saturated_costs = saturated_operator_costs(task, &operators, &match_tree, &table)?;
        Ok((table, saturated_costs))
    }

    /// Computes saturated costs for the *already-built* distance table and
    /// abstract operators.  This is public so the online SCP heuristic can
    /// cap h-values for PERIM saturation before computing saturated costs.
    pub fn saturated_costs_for_table(
        &self,
        task: &dyn AbstractNumericTask,
        combine_labels: bool,
        operators: &[AbstractOperator],
        table: &AbstractDistanceTable,
    ) -> Result<Vec<f64>> {
        let generator = self.make_operator_generator(task, combine_labels)?;
        let match_tree = MatchTree::new(&generator, operators);
        saturated_operator_costs(task, operators, &match_tree, table)
    }

    /// The state region for `state_hash`, materializing it on first use.
    fn cached_cartesian_set<'cache>(
        &self,
        cache: &'cache mut [Option<CartesianSet>],
        state_hash: usize,
        transition_system: &AbstractTransitionSystem,
    ) -> Result<&'cache CartesianSet> {
        let slot = cache
            .get_mut(state_hash)
            .with_context(|| format!("state hash {state_hash} out of range for region cache"))?;
        if slot.is_none() {
            *slot = Some(self.cartesian_set_from_hash(
                state_hash,
                &transition_system.numeric_domain_sizes,
                &transition_system.hash_multipliers,
            )?);
        }
        Ok(slot.as_ref().expect("region was just materialized"))
    }

    pub fn build_interval_cost_partitioned_distance_table(
        &self,
        transition_system: &AbstractTransitionSystem,
        abstract_source_sets: &[AbstractOperatorSourceSets],
        residual_costs: &RemainingCosts,
        abstraction_id: usize,
        options: DistanceTableOptions<'_>,
    ) -> Result<(AbstractDistanceTable, SourceSetCostAllocation)> {
        let deadline = options.deadline;
        let cap_state_id = options.cap_state_id;
        deadline.ensure(ONLINE_SCP)?;
        ensure!(
            transition_system.cartesian_sets.is_empty(),
            "precise domain regional SCP expects lazy state regions"
        );

        let mut timings = RegionalTableTimings::default();
        let mut operator_pairs = 0usize;
        let phase_start = std::time::Instant::now();

        // Several transitions leave the same abstract state, and both loops below
        // ask for that state's region, so materializing it per transition rebuilds
        // one value set per propositional variable many times over for the same
        // answer. Keep the regions for the duration of the build; a region is four
        // `Arc`s, so reuse is a clone.
        let num_states = transition_system
            .transitions
            .iter()
            .map(|transition| transition.source_hash)
            .max()
            .map_or(0, |max_hash| max_hash + 1);
        let mut source_regions: Vec<Option<CartesianSet>> = vec![None; num_states];

        let mut transition_costs = Vec::with_capacity(transition_system.transitions.len());
        for (transition_id, transition) in transition_system.transitions.iter().enumerate() {
            if transition_id.is_multiple_of(1024) {
                deadline.ensure(ONLINE_SCP)?;
            }
            let source_region = self.cached_cartesian_set(
                &mut source_regions,
                transition.source_hash,
                transition_system,
            )?;
            operator_pairs += transition.concrete_op_ids.len();
            let mut cost = f64::INFINITY;
            for &concrete_op_id in &transition.concrete_op_ids {
                let source_set = precise_source_set_for_transition(
                    transition,
                    concrete_op_id,
                    source_region,
                    abstract_source_sets,
                )?;
                cost = cost.min(residual_costs.cost_for_source_set(
                    abstraction_id,
                    transition_id,
                    &source_set,
                ));
            }
            transition_costs.push(cost);
        }
        timings.transition_costs = phase_start.elapsed();

        let phase_start = std::time::Instant::now();
        let table = self.build_distance_table_with_transition_costs(
            transition_system,
            &transition_costs,
            &transition_system.hash_multipliers,
            &transition_system.numeric_domain_sizes,
        )?;
        timings.distance_table = phase_start.elapsed();

        let capped_table = if let Some(state_id) = cap_state_id {
            let h_cap = table.distances.get(state_id).copied().with_context(|| {
                format!(
                    "regional perimeter state {state_id} out of bounds for {} states",
                    table.distances.len()
                )
            })?;
            let mut capped = table.clone();
            if h_cap.is_finite() {
                for h in &mut capped.distances {
                    if !h.is_finite() || *h > h_cap {
                        *h = f64::NEG_INFINITY;
                    }
                }
            }
            Some(capped)
        } else {
            None
        };
        let saturation_table = capped_table.as_ref().unwrap_or(&table);
        let phase_start = std::time::Instant::now();
        let tcf = self.compute_saturated_transition_costs(
            transition_system,
            &transition_costs,
            saturation_table,
        )?;
        timings.saturation = phase_start.elapsed();

        let phase_start = std::time::Instant::now();
        let lookup_table = if cap_state_id.is_some() {
            self.build_distance_table_with_transition_costs(
                transition_system,
                &tcf.transition_costs,
                &transition_system.hash_multipliers,
                &transition_system.numeric_domain_sizes,
            )?
        } else {
            table
        };
        timings.lookup_table = phase_start.elapsed();

        let phase_start = std::time::Instant::now();
        let mut entries = Vec::new();
        for (transition_id, transition) in transition_system.transitions.iter().enumerate() {
            if transition_id.is_multiple_of(1024) {
                deadline.ensure(ONLINE_SCP)?;
            }
            let saturated = tcf.transition_costs[transition_id];
            if !saturated.is_finite() || saturated <= 1e-9 {
                continue;
            }
            let source_region = self.cached_cartesian_set(
                &mut source_regions,
                transition.source_hash,
                transition_system,
            )?;
            for &concrete_op_id in &transition.concrete_op_ids {
                let source_set = precise_source_set_for_transition(
                    transition,
                    concrete_op_id,
                    source_region,
                    abstract_source_sets,
                )?;
                let current_residual =
                    residual_costs.cost_for_source_set(abstraction_id, transition_id, &source_set);
                ensure!(
                    saturated <= current_residual + 1e-7,
                    "regional transition allocation {saturated} exceeds residual {current_residual} for transition {transition_id}, operator {concrete_op_id}"
                );
                entries.push(SourceSetCost {
                    source_set,
                    amount: saturated,
                });
            }
        }

        timings.allocation_entries = phase_start.elapsed();
        timings.log(
            abstraction_id,
            transition_system.transitions.len(),
            operator_pairs,
            entries.len(),
        );

        Ok((lookup_table, SourceSetCostAllocation::new(entries)))
    }
    pub(super) fn compute_saturated_transition_costs(
        &self,
        transition_system: &AbstractTransitionSystem,
        transition_costs: &[f64],
        table: &AbstractDistanceTable,
    ) -> Result<AbstractTransitionCostFunction> {
        ensure!(
            transition_system.transitions.len() == transition_costs.len(),
            "transition system/cost vector size mismatch: {} vs {}",
            transition_system.transitions.len(),
            transition_costs.len()
        );
        let mut saturated = vec![0.0; transition_system.transitions.len()];
        for transition in &transition_system.transitions {
            let source_h = table.distances[transition.source_hash];
            let target_h = table.distances[transition.target_hash];
            let Some(needed) = saturation_need(
                source_h,
                target_h,
                transition_costs[transition.transition_id],
                "saturated transition cost",
            )?
            else {
                continue;
            };
            saturated[transition.transition_id] = needed;
        }
        Ok(AbstractTransitionCostFunction {
            transition_costs: saturated,
        })
    }
}

/// The saturated cost of every concrete operator for `table`: the most any
/// transition it labels needs, over every transition into a state of finite
/// distance and not only the shortest-path ones, as numeric-fd's
/// `DomainAbstraction::compute_saturated_costs` does.
fn saturated_operator_costs(
    task: &dyn AbstractNumericTask,
    operators: &[AbstractOperator],
    match_tree: &MatchTree,
    table: &AbstractDistanceTable,
) -> Result<Vec<f64>> {
    let mut saturated_costs = vec![0.0_f64; task.get_operators().len()];
    let mut graph = MatchTreeGraph {
        operators,
        match_tree,
        num_states: table.distances.len(),
        applicable: Vec::new(),
    };
    let mut incoming = Vec::new();
    for (target, &target_h) in table.distances.iter().enumerate() {
        if !target_h.is_finite() {
            continue;
        }
        graph.incoming(target, &mut incoming)?;
        for &(source, abstract_op_id, cost) in &incoming {
            let Some(needed) = saturation_need(
                table.distances[source],
                target_h,
                cost,
                "saturated operator cost",
            )?
            else {
                continue;
            };
            for &op_id in &operators[abstract_op_id].concrete_op_ids {
                let slot = &mut saturated_costs[op_id.index()];
                *slot = slot.max(needed);
            }
        }
    }
    Ok(saturated_costs)
}
