use super::*;

/// One abstraction's turn in a saturated cost partitioning: the costs earlier
/// abstractions in the order have left unspent, which abstraction of the
/// collection this is, and the two states that can cut the build short -- the
/// state being evaluated, whose distance reaching zero means there is nothing
/// left for this abstraction to contribute, and the perimeter cap beyond which
/// distances are discarded.
#[derive(Clone, Copy)]
pub struct SaturationStep<'a> {
    pub residual_costs: &'a TransitionResidualCosts,
    pub abstraction_id: usize,
    pub current_state_id: Option<usize>,
    pub cap_state_id: Option<usize>,
}

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
            let concrete_cost = *operator_costs.get(concrete_op_id).with_context(|| {
                format!("missing residual cost for concrete operator {concrete_op_id}")
            })?;
            ensure!(
                concrete_cost.is_finite(),
                "residual cost for concrete operator {concrete_op_id} must be finite"
            );
            cost = cost.min(concrete_cost);
        }
        op.cost = cost;
    }
    Ok(())
}

fn apply_abstract_operator_costs(
    operators: &mut [AbstractOperator],
    operator_costs: &[f64],
) -> Result<()> {
    ensure!(
        operators.len() == operator_costs.len(),
        "abstract operator/cost vector size mismatch: {} vs {}",
        operators.len(),
        operator_costs.len()
    );
    for (abstract_op_id, op) in operators.iter_mut().enumerate() {
        let cost = operator_costs[abstract_op_id];
        ensure!(
            cost.is_finite(),
            "residual cost for abstract operator {abstract_op_id} must be finite"
        );
        op.cost = cost;
    }
    Ok(())
}

pub(super) fn abstract_operator_costs_from_operator_regions(
    num_operators: usize,
    operator_regions: &[AbstractOperatorRegions],
    residual_costs: &TransitionResidualCosts,
    abstraction_id: usize,
    deadline: Option<Instant>,
) -> Result<Vec<f64>> {
    let has_reductions = residual_costs.has_reductions();
    let mut operator_costs = vec![f64::INFINITY; num_operators];
    for (abstract_op_id, operator_cost) in operator_costs.iter_mut().enumerate() {
        if abstract_op_id % 64 == 0 {
            ensure_online_scp_deadline(deadline)?;
        }
        let operator_region = operator_regions.get(abstract_op_id).with_context(|| {
            format!("missing operator region for abstract operator {abstract_op_id}")
        })?;
        ensure!(
            !operator_region.labels.is_empty(),
            "abstract operator {abstract_op_id} has no concrete operator-region labels"
        );
        *operator_cost = operator_region
            .labels
            .iter()
            .map(|label| {
                let residual = if has_reductions {
                    residual_costs.cost_for_operator_region(abstraction_id, abstract_op_id, label)
                } else {
                    residual_costs.base_cost(label.concrete_op_id)
                };
                residual.min(residual_costs.base_cost(label.concrete_op_id))
            })
            .fold(f64::INFINITY, f64::min);
        ensure!(
            operator_cost.is_finite(),
            "residual cost for abstract operator {abstract_op_id} is not finite"
        );
    }
    Ok(operator_costs)
}

pub(super) fn get_comparison_preconditions(
    op: &AbstractOperator,
    comparison_var_ids: &[usize],
) -> Vec<ExplicitFact> {
    op.preconditions
        .iter()
        .copied()
        .filter(|f| comparison_var_ids.contains(&f.var()))
        .collect()
}

pub(super) fn comparison_preconditions_by_operator(
    operators: &[AbstractOperator],
    comparison_var_ids: &[usize],
) -> Vec<Vec<ExplicitFact>> {
    operators
        .iter()
        .map(|op| get_comparison_preconditions(op, comparison_var_ids))
        .collect()
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
        let computed_goal_facts;
        let goal_facts = if let Some(goal_facts) = options.goal_facts {
            goal_facts
        } else {
            computed_goal_facts = self.compute_abstract_goals(task);
            &computed_goal_facts
        };
        let deadline = options.deadline;
        ensure_online_scp_deadline(deadline)?;
        let mut generator = self.make_operator_generator(task, combine_labels)?;
        let mut operators = generator.build_abstract_operators(task)?;
        ensure_online_scp_deadline(deadline)?;
        apply_operator_costs(&mut operators, operator_costs)?;
        let table = self.build_distance_table_with_operators_for_goals_inner(
            task,
            &generator,
            &operators,
            goal_facts,
            DistanceTableOptions {
                goal_facts: None,
                ..options
            },
        )?;
        ensure_online_scp_deadline(deadline)?;
        let saturated_costs = self.compute_saturated_costs(task, &generator, &operators, &table)?;
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
        self.compute_saturated_costs(task, &generator, operators, table)
    }

    pub fn build_precise_regional_cost_partitioned_distance_table(
        &self,
        transition_system: &AbstractTransitionSystem,
        abstract_operator_regions: &[AbstractOperatorRegions],
        residual_costs: &TransitionResidualCosts,
        abstraction_id: usize,
        options: DistanceTableOptions<'_>,
    ) -> Result<(AbstractDistanceTable, RegionalCostAllocation)> {
        let deadline = options.deadline;
        let cap_state_id = options.cap_state_id;
        ensure_online_scp_deadline(deadline)?;
        ensure!(
            transition_system.state_regions.is_empty(),
            "precise domain regional SCP expects lazy state regions"
        );

        let mut timings = RegionalTableTimings::default();
        let mut operator_pairs = 0usize;
        let phase_start = std::time::Instant::now();

        let transition_costs = transition_system
            .transitions
            .iter()
            .enumerate()
            .map(|(transition_id, transition)| {
                if transition_id.is_multiple_of(1024) {
                    ensure_online_scp_deadline(deadline)?;
                }
                let source_region = self.state_region_from_hash(
                    transition.source_hash,
                    &transition_system.numeric_domain_sizes,
                    &transition_system.hash_multipliers,
                )?;
                operator_pairs += transition.concrete_op_ids.len();
                transition
                    .concrete_op_ids
                    .iter()
                    .map(|&concrete_op_id| {
                        let operator_region = precise_operator_region_for_transition(
                            transition,
                            concrete_op_id,
                            &source_region,
                            abstract_operator_regions,
                        )?;
                        Ok(residual_costs.cost_for_operator_region(
                            abstraction_id,
                            transition_id,
                            &operator_region,
                        ))
                    })
                    .collect::<Result<Vec<_>>>()
                    .map(|costs| costs.into_iter().fold(f64::INFINITY, f64::min))
            })
            .collect::<Result<Vec<_>>>()?;
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
                ensure_online_scp_deadline(deadline)?;
            }
            let saturated = tcf.transition_costs[transition_id];
            if !saturated.is_finite() || saturated <= 1e-9 {
                continue;
            }
            let source_region = self.state_region_from_hash(
                transition.source_hash,
                &transition_system.numeric_domain_sizes,
                &transition_system.hash_multipliers,
            )?;
            for &concrete_op_id in &transition.concrete_op_ids {
                let operator_region = precise_operator_region_for_transition(
                    transition,
                    concrete_op_id,
                    &source_region,
                    abstract_operator_regions,
                )?;
                let current_residual = residual_costs.cost_for_operator_region(
                    abstraction_id,
                    transition_id,
                    &operator_region,
                );
                ensure!(
                    saturated <= current_residual + 1e-7,
                    "regional transition allocation {saturated} exceeds residual {current_residual} for transition {transition_id}, operator {concrete_op_id}"
                );
                entries.push(RegionalCostAllocationEntry {
                    operator_region,
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

        Ok((lookup_table, RegionalCostAllocation::new(entries)))
    }

    pub fn build_abstract_operator_cost_partitioned_distance_table_with_operators_and_operator_regions(
        &self,
        task: &dyn AbstractNumericTask,
        combine_labels: bool,
        operators: &[AbstractOperator],
        operator_regions: &[AbstractOperatorRegions],
        step: SaturationStep<'_>,
        options: DistanceTableOptions<'_>,
    ) -> Result<(AbstractDistanceTable, AbstractOperatorCostFunction)> {
        let SaturationStep {
            residual_costs,
            abstraction_id,
            current_state_id,
            cap_state_id,
        } = step;
        let deadline = options.deadline;
        ensure_online_scp_deadline(deadline)?;
        ensure!(
            operator_regions.len() >= operators.len(),
            "abstract-operator region/operator size mismatch: {} vs {}",
            operator_regions.len(),
            operators.len()
        );

        let operator_costs = abstract_operator_costs_from_operator_regions(
            operators.len(),
            operator_regions,
            residual_costs,
            abstraction_id,
            deadline,
        )?;
        let mut operators = operators.to_vec();
        apply_abstract_operator_costs(&mut operators, &operator_costs)?;
        let generator = self.make_operator_generator(task, combine_labels)?;
        if operator_costs
            .iter()
            .all(|&cost| cost <= float_tolerance::DIJKSTRA_EPSILON)
        {
            let table = self.zero_distance_table_for_generator(task, &generator)?;
            let tcf = AbstractOperatorCostFunction {
                operator_costs: vec![0.0; operator_costs.len()],
            };
            return Ok((table, tcf));
        }
        if residual_costs.has_reductions()
            && let Some(current_state_id) = current_state_id
        {
            let current_distance = self.compute_distance_to_goal_state_with_operators(
                task,
                &generator,
                &operators,
                current_state_id,
                deadline,
            )?;
            if current_distance <= float_tolerance::DIJKSTRA_EPSILON {
                let table = self.zero_distance_table_for_generator(task, &generator)?;
                let tcf = AbstractOperatorCostFunction {
                    operator_costs: vec![0.0; operator_costs.len()],
                };
                return Ok((table, tcf));
            }
        }
        // Build the match tree once. It depends only on
        // (domain_sizes, numeric_domain_sizes, hash_multipliers, regression
        // preconditions of `operators`); none of those change when we re-apply
        // costs below. Reusing it avoids 2x (or 4x for perimstar) rebuilds.
        let comparison_var_ids_for_tree = self.comparison_var_ids();
        let match_tree = MatchTree::build(
            generator.domain_sizes(),
            generator.numeric_domain_sizes(),
            generator.hash_multipliers(),
            &operators,
            &comparison_var_ids_for_tree,
        );

        let goal_facts = self.compute_abstract_goals(task);
        let table = self.build_distance_table_with_operators_for_goals_inner(
            task,
            &generator,
            &operators,
            &goal_facts,
            DistanceTableOptions {
                prebuilt_match_tree: Some(&match_tree),
                deadline,
                ..DistanceTableOptions::default()
            },
        )?;

        if let Some(state_id) = cap_state_id
            && let Some(&h_cap) = table.distances.get(state_id)
            && h_cap.is_finite()
        {
            let mut perim_table = table.clone();
            for h in &mut perim_table.distances {
                if !h.is_finite() || *h > h_cap {
                    *h = f64::NEG_INFINITY;
                }
            }
            let tcf = self.compute_saturated_abstract_operator_costs_from_operators_inner(
                SolvedAbstraction {
                    task,
                    generator: &generator,
                    operators: &operators,
                    table: &perim_table,
                    match_tree: &match_tree,
                    comparison_var_ids: &comparison_var_ids_for_tree,
                },
                &operator_costs,
                deadline,
            )?;
            let mut saturated_operators = operators;
            apply_abstract_operator_costs(&mut saturated_operators, &tcf.operator_costs)?;
            let global_table = self.build_distance_table_with_operators_for_goals_inner(
                task,
                &generator,
                &saturated_operators,
                &goal_facts,
                DistanceTableOptions {
                    prebuilt_match_tree: Some(&match_tree),
                    deadline,
                    ..DistanceTableOptions::default()
                },
            )?;
            return Ok((global_table, tcf));
        }

        let tcf = self.compute_saturated_abstract_operator_costs_from_operators_inner(
            SolvedAbstraction {
                task,
                generator: &generator,
                operators: &operators,
                table: &table,
                match_tree: &match_tree,
                comparison_var_ids: &comparison_var_ids_for_tree,
            },
            &operator_costs,
            deadline,
        )?;
        // For Saturator::All, the saturated abstract-operator costs are tight
        // wrt `table.distances`: by construction every transition (u,v) using
        // operator op has saturated[op] >= h(u) - h(v), so any path from s to
        // the goal has length >= h(s) under saturated costs (telescoping), and
        // the original shortest path remains feasible. Therefore distances
        // under saturated costs equal `table.distances`, and the historic
        // second Dijkstra over saturated_operators was redundant.
        Ok((table, tcf))
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

    pub(super) fn compute_saturated_abstract_operator_costs_from_operators_inner(
        &self,
        abstraction: SolvedAbstraction<'_>,
        operator_costs: &[f64],
        deadline: Option<Instant>,
    ) -> Result<AbstractOperatorCostFunction> {
        let SolvedAbstraction {
            task,
            generator,
            operators,
            table,
            match_tree,
            comparison_var_ids,
        } = abstraction;
        ensure!(
            operators.len() == operator_costs.len(),
            "abstract operator/cost vector size mismatch: {} vs {}",
            operators.len(),
            operator_costs.len()
        );

        let num_states = table.distances.len();
        let comparison_branching = !comparison_var_ids.is_empty();
        let mut saturated = vec![0.0_f64; operators.len()];
        let mut applicable_operator_ids = Vec::new();

        // Mirror `compute_distances_and_generating_ops`: under
        // comparison-branching the regression Dijkstra clears each popped
        // state's comparison-axiom digits before consulting the match tree,
        // computes the natural predecessor from the cleared form, and then
        // expands every wildcard-consistent predecessor.
        // Comparison-axiom prop vars are not directly written by
        // operators (their values are derived from the predecessor's
        // numeric intervals), so the "hash_effect alone determines the
        // source" rationale that was here previously was wrong — under
        // comparison-branching it both pulls in transitions Dijkstra
        // never took (those with mismatched comparison bits on the
        // target) and skips the very transitions Dijkstra used (the
        // wildcard-enumerated predecessors). The mismatch made
        // `saturated[abstract_op_id]` reflect `src_h - target_h` for
        // transitions outside the cost partition that produced the
        // distance table, so `sum_k h_k > h*` on plant-watering
        // AOCP-fillSCP (prob_4_2_2: 34 vs 33, prob_5_1_2: 30 vs 24,
        // prob_4_2_3: 32 vs 29, etc.).
        let comparison_preconditions = if comparison_branching {
            comparison_preconditions_by_operator(operators, comparison_var_ids)
        } else {
            Vec::new()
        };
        let mut comparison_enumeration_memo = ComparisonEnumerationMemo::default();

        for target_hash in 0..num_states {
            if target_hash % 64 == 0 {
                ensure_online_scp_deadline(deadline)?;
            }
            let target_h = table.distances[target_hash];
            if !target_h.is_finite() {
                continue;
            }

            let base_target = if comparison_branching {
                self.clear_comparison_vars_except(
                    target_hash,
                    generator.hash_multipliers(),
                    comparison_var_ids,
                    &[],
                )?
            } else {
                target_hash
            };

            match_tree.get_applicable_operator_ids(base_target, &mut applicable_operator_ids);
            for &abstract_op_id in &applicable_operator_ids {
                let op = &operators[abstract_op_id];
                let predecessor_i64 = base_target as i64 + op.hash_effect as i64;
                if predecessor_i64 < 0 || predecessor_i64 >= num_states as i64 {
                    continue;
                }
                let base_predecessor = predecessor_i64 as usize;

                let consider_source = |source_hash: usize, saturated: &mut [f64]| -> Result<()> {
                    let source_h = table.distances[source_hash];
                    let Some(needed) = saturation_need(
                        source_h,
                        target_h,
                        operator_costs[abstract_op_id],
                        "saturated abstract-operator cost",
                    )?
                    else {
                        return Ok(());
                    };
                    saturated[abstract_op_id] = saturated[abstract_op_id].max(needed);
                    Ok(())
                };

                if comparison_branching {
                    let possible_predecessors = self
                        .enumerate_states_with_evaluated_comparisons_cached(
                            base_predecessor,
                            task,
                            ComparisonBranchingLayout {
                                numeric_domain_sizes: generator.numeric_domain_sizes(),
                                hash_multipliers: generator.hash_multipliers(),
                                comparison_var_ids,
                            },
                            &comparison_preconditions[abstract_op_id],
                            &mut comparison_enumeration_memo,
                        )?;
                    for &source_hash in possible_predecessors.iter() {
                        consider_source(source_hash, &mut saturated)?;
                    }
                } else {
                    consider_source(base_predecessor, &mut saturated)?;
                }
            }
        }

        Ok(AbstractOperatorCostFunction {
            operator_costs: saturated,
        })
    }

    pub(super) fn compute_saturated_costs(
        &self,
        task: &dyn AbstractNumericTask,
        generator: &AbstractOperatorGenerator,
        operators: &[AbstractOperator],
        table: &AbstractDistanceTable,
    ) -> Result<Vec<f64>> {
        let num_operators = task.get_operators().len();
        let num_states = table.distances.len();
        let mut saturated_costs = vec![f64::NEG_INFINITY; num_operators];

        let comparison_var_ids = self.comparison_var_ids();
        let comparison_branching = !comparison_var_ids.is_empty();
        let match_tree = MatchTree::build(
            generator.domain_sizes(),
            generator.numeric_domain_sizes(),
            generator.hash_multipliers(),
            operators,
            &comparison_var_ids,
        );
        // Mirror `compute_distances_and_generating_ops`: when comparison-axiom
        // vars are refined, an operator's predecessor set is enumerated via
        // wildcard expansion on the comparison bits, not just the single
        // `target + hash_effect` hash. Without this, cascade-only transitions
        // (e.g. an op that flips a comparison-axiom prop var only via its
        // effect on a numeric dependency) are missed during saturation, the
        // residual stays inflated, subsequent abstractions over-saturate the
        // same operator, and `sum_a h_a > h*` — inadmissibility.
        let comparison_preconditions = if comparison_branching {
            comparison_preconditions_by_operator(operators, &comparison_var_ids)
        } else {
            Vec::new()
        };
        let mut comparison_enumeration_memo = ComparisonEnumerationMemo::default();

        let mut applicable_operator_ids = Vec::new();
        for target_hash in 0..num_states {
            let target_h = table.distances[target_hash];
            if !target_h.is_finite() {
                continue;
            }

            // Mirror `compute_distances_and_generating_ops`: under
            // comparison-branching, Dijkstra clears the target state's
            // comparison-axiom prop vars before consulting `match_tree` for
            // applicable operators, and computes the predecessor hash from
            // that cleared state. The reason is that
            // ops have no direct effect on comparison-axiom prop vars —
            // those values are derived from the predecessor's numeric
            // intervals, then enumerated via wildcard expansion. If we
            // consult `match_tree` at `target_hash` directly (with
            // specific comparison bits), we'd saturate over transitions
            // Dijkstra never used (because Dijkstra applied them at the
            // base form) and miss transitions Dijkstra did use,
            // diverging from the cost partition that produced the
            // distance table. The divergence inflates `needed` for ops
            // not actually on Dijkstra's paths and under-saturates ops
            // that were, leaving cost in `remaining_costs` for the next
            // abstraction to re-charge — the symptom is `sum_k h_k > h*`
            // on plant-watering/prob_4_2_2 (cost 34 vs optimal 33 across
            // a fraction of seeds).
            let base_target = if comparison_branching {
                self.clear_comparison_vars_except(
                    target_hash,
                    generator.hash_multipliers(),
                    &comparison_var_ids,
                    &[],
                )?
            } else {
                target_hash
            };

            match_tree.get_applicable_operator_ids(base_target, &mut applicable_operator_ids);
            for &abstract_op_id in &applicable_operator_ids {
                let op = &operators[abstract_op_id];
                let predecessor_i64 = base_target as i64 + op.hash_effect as i64;
                if predecessor_i64 < 0 || predecessor_i64 >= num_states as i64 {
                    continue;
                }
                let base_predecessor = predecessor_i64 as usize;

                // Saturate over ALL applicable transitions, not just those chosen by
                // Dijkstra as the generating op for the source state. Mirrors
                // numeric-fd `DomainAbstraction::compute_saturated_costs`
                // (cost_saturation/domain_abstraction.cc:872-902) which iterates
                // every label transition with `for_each_label_transition`.
                //
                // The previous "generator-only" filter under-saturated: an
                // operator applicable at a non-generating source still needs cost
                // ≥ (src_h - target_h) to be admissible in the next abstraction's
                // residual, but skipping it left that cost in the residual and let
                // subsequent abstractions over-allocate, producing sum h > h* on
                // plant-watering/prob_6_2_2 (32-47 across seeds vs optimal 32).
                let consider_source = |source_hash: usize, saturated_costs: &mut [f64]| {
                    if let Some(&src_h) = table.distances.get(source_hash)
                        && src_h.is_finite()
                    {
                        let needed = (src_h - target_h).max(0.0);
                        for &op_id in &op.concrete_op_ids {
                            if let Some(slot) = saturated_costs.get_mut(op_id) {
                                *slot = slot.max(needed);
                            }
                        }
                    }
                };

                if comparison_branching {
                    let possible_predecessors = self
                        .enumerate_states_with_evaluated_comparisons_cached(
                            base_predecessor,
                            task,
                            ComparisonBranchingLayout {
                                numeric_domain_sizes: generator.numeric_domain_sizes(),
                                hash_multipliers: generator.hash_multipliers(),
                                comparison_var_ids: &comparison_var_ids,
                            },
                            &comparison_preconditions[abstract_op_id],
                            &mut comparison_enumeration_memo,
                        )?;
                    for &source_hash in possible_predecessors.iter() {
                        consider_source(source_hash, &mut saturated_costs);
                    }
                } else {
                    consider_source(base_predecessor, &mut saturated_costs);
                }
            }
        }

        for cost in &mut saturated_costs {
            if *cost == f64::NEG_INFINITY {
                *cost = 0.0;
            }
        }

        Ok(saturated_costs)
    }
}
