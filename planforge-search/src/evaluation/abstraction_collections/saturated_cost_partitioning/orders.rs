use super::*;

pub(super) struct CandidateCostPartitions {
    pub(super) partitions: Vec<CostPartitioningHeuristic>,
    pub(super) best_index: usize,
}

impl<'task> SaturatedCostPartitioningHeuristic<'task> {
    pub(super) fn compute_order_for_state(
        &self,
        task: &dyn AbstractNumericTask,
        state: &mut ScpState,
        abstract_state_ids: &[Option<usize>],
        components: &[AbstractionComponent<'_>],
        deadline: Option<Instant>,
    ) -> Result<Vec<usize>, EvaluationError> {
        let order = self.compute_state_dependent_order(
            task,
            state,
            abstract_state_ids,
            components,
            deadline,
        )?;
        if self.config.partitioning.uses_regions()
            && let Some(goal_cover_order) = self.goal_cover_order(
                &order,
                components,
                &abstraction_heuristic_values(state, abstract_state_ids),
                true,
            )
        {
            if state.evaluated_states == 0 {
                info!(
                    "scp: using goal-cover regional-SCP order, first_components={:?}",
                    goal_cover_order.iter().take(24).collect::<Vec<_>>()
                );
            }
            return Ok(goal_cover_order);
        }
        Ok(order)
    }

    pub(super) fn compute_state_dependent_order(
        &self,
        task: &dyn AbstractNumericTask,
        state: &mut ScpState,
        abstract_state_ids: &[Option<usize>],
        components: &[AbstractionComponent<'_>],
        deadline: Option<Instant>,
    ) -> Result<Vec<usize>, EvaluationError> {
        match self.config.order_generator {
            OrderGenerator::Greedy => Ok(Self::compute_greedy_order_for_state(
                state,
                abstract_state_ids,
                self.config.scoring_function,
            )),
            OrderGenerator::Random => {
                let total = state.goal_distances_by_abstraction.len();
                let mut order: Vec<usize> = (0..total).collect();
                order.shuffle(&mut state.rng);
                Ok(order)
            }
            OrderGenerator::DynamicGreedy => self.compute_dynamic_greedy_order_for_state(
                task,
                state,
                abstract_state_ids,
                components,
                deadline,
            ),
            OrderGenerator::Diverse => Ok(Self::compute_greedy_order_for_state(
                state,
                abstract_state_ids,
                self.config.scoring_function,
            )),
        }
    }

    pub(super) fn diverse_orders_for_sample(
        &self,
        task: &dyn AbstractNumericTask,
        state: &mut ScpState,
        abstract_state_ids: &[Option<usize>],
        components: &[AbstractionComponent<'_>],
        deadline: Option<Instant>,
    ) -> Result<Vec<Vec<usize>>, EvaluationError> {
        if self.config.order_generator != OrderGenerator::Diverse {
            return self
                .compute_state_dependent_order(
                    task,
                    state,
                    abstract_state_ids,
                    components,
                    deadline,
                )
                .map(|order| vec![order]);
        }

        let greedy = Self::compute_greedy_order_for_state(
            state,
            abstract_state_ids,
            self.config.scoring_function,
        );
        let mut random = (0..state.goal_distances_by_abstraction.len()).collect::<Vec<_>>();
        random.shuffle(&mut state.rng);
        Ok(deduplicate_orders(vec![greedy, random]))
    }

    pub(super) fn compute_greedy_order_for_state(
        state: &mut ScpState,
        abstract_state_ids: &[Option<usize>],
        scoring_function: ScoringFunction,
    ) -> Vec<usize> {
        let total = state.goal_distances_by_abstraction.len();
        let mut order: Vec<usize> = (0..total).collect();
        order.shuffle(&mut state.rng);
        let current_h: Vec<f64> = (0..total)
            .map(|abs_id| {
                abstract_state_ids
                    .get(abs_id)
                    .copied()
                    .flatten()
                    .and_then(|sid| {
                        state
                            .goal_distances_by_abstraction
                            .get(abs_id)
                            .and_then(|values| values.get(sid))
                            .copied()
                    })
                    .unwrap_or(0.0)
            })
            .collect();

        let scores: Vec<f64> = (0..total)
            .map(|abs_id| {
                let h = current_h[abs_id];
                let stolen = state
                    .stolen_costs_by_abstraction
                    .get(abs_id)
                    .copied()
                    .unwrap_or(0.0);
                compute_score(h, stolen, scoring_function)
            })
            .collect();

        order.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        order
    }

    pub(super) fn compute_dynamic_greedy_order_for_state(
        &self,
        task: &dyn AbstractNumericTask,
        state: &mut ScpState,
        abstract_state_ids: &[Option<usize>],
        components: &[AbstractionComponent<'_>],
        deadline: Option<Instant>,
    ) -> Result<Vec<usize>, EvaluationError> {
        if self.config.partitioning.uses_regions() {
            return Err(EvaluationError::ComputationFailed(
                "dynamic_greedy_orders is only implemented for label SCP; regional SCP needs residual regional-cost scoring, not label-order scoring".to_string(),
            ));
        }

        let total = state.goal_distances_by_abstraction.len();
        let mut remaining_components: Vec<usize> = (0..total).collect();
        let mut remaining_costs = self.original_operator_costs.clone();
        let mut order = Vec::with_capacity(total);

        'selection: while !remaining_components.is_empty() {
            if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                break;
            }

            remaining_components.shuffle(&mut state.rng);
            let mut candidate_saturated_costs = Vec::with_capacity(remaining_components.len());
            let mut candidate_h_values = Vec::with_capacity(remaining_components.len());

            for &pos in &remaining_components {
                if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                    break;
                }
                let component = components.get(pos).ok_or_else(|| {
                    EvaluationError::ComputationFailed(format!(
                        "dynamic order requested missing component {pos}"
                    ))
                })?;
                let result = match component {
                    AbstractionComponent::Domain(heuristic) => {
                        let abstraction = heuristic.abstraction();
                        let abstraction_task = abstraction.task_for_factory(task);
                        Self::saturate_domain(
                            abstraction,
                            abstraction_task,
                            self.config.combine_labels,
                            &remaining_costs,
                            deadline,
                        )
                    }
                    AbstractionComponent::Cartesian(heuristic) => {
                        build_explicit_label_cost_partitioning_table(
                            &heuristic.abstraction().transition_system,
                            &remaining_costs,
                            None,
                            deadline,
                        )
                        .map_err(|error| {
                            Self::construction_error(
                                &format!(
                                    "failed to compute Cartesian dynamic-order table for component {pos}"
                                ),
                                error,
                            )
                        })
                    }
                    AbstractionComponent::PatternDatabase(pdb) => pdb
                        .build_cost_partitioned_distance_table(&remaining_costs)
                        .map_err(|error| {
                            EvaluationError::ComputationFailed(format!(
                                "failed to compute PDB dynamic-order table {pos}: {error}"
                            ))
                        }),
                };
                let (distances, saturated) = match result {
                    Ok(entry) => entry,
                    Err(error) if Self::is_deadline_exceeded(&error) => {
                        info!("scp: dynamic greedy order stopped at deadline");
                        break 'selection;
                    }
                    Err(error) => return Err(error),
                };
                let h = current_h_for_distances(pos, &distances, abstract_state_ids);
                candidate_h_values.push(h);
                candidate_saturated_costs.push(saturated);
            }

            if candidate_saturated_costs.len() != remaining_components.len() {
                break;
            }

            let surplus_costs =
                compute_all_surplus_costs(&remaining_costs, &candidate_saturated_costs);
            let mut best_index = None;
            let mut best_score = f64::NEG_INFINITY;
            for rem_index in 0..remaining_components.len() {
                let stolen = compute_costs_stolen_by_heuristic(
                    &candidate_saturated_costs[rem_index],
                    &surplus_costs,
                );
                let score = compute_score(
                    candidate_h_values[rem_index],
                    stolen,
                    self.config.scoring_function,
                );
                if best_index.is_none() || score > best_score {
                    best_index = Some(rem_index);
                    best_score = score;
                }
            }
            let best_index = best_index.ok_or_else(|| {
                EvaluationError::ComputationFailed(
                    "dynamic order generator had no remaining candidate".to_string(),
                )
            })?;
            order.push(remaining_components[best_index]);
            reduce_costs(&mut remaining_costs, &candidate_saturated_costs[best_index])?;
            remaining_components.swap_remove(best_index);
        }

        order.extend(remaining_components);
        Ok(order)
    }

    /// The cost partitionings of the incumbent order and of the candidate
    /// orders derived from it, built until `optimization_max_time` runs out,
    /// with the incumbent replaced by the order of the best one at the
    /// evaluated state. Interval cost partitionings remember the goal their
    /// goal-cover order puts first.
    pub(super) fn build_best_cp_from_candidate_orders(
        &self,
        collection: PartitionedCollection<'_, '_>,
        incumbent_order: &mut Vec<usize>,
        baseline_deadline: Option<Instant>,
        optimization_max_time: f64,
    ) -> Result<CandidateCostPartitions, EvaluationError> {
        let PartitionedCollection {
            components,
            abstract_state_ids,
            abstraction_h,
            ..
        } = collection;
        let interval = self.config.partitioning.uses_regions();
        let goal_cover_goal = |order: &[usize]| {
            interval
                .then(|| self.goal_cover_goal_of_order(order, components))
                .flatten()
        };
        let mut best_order = incumbent_order.clone();
        let mut baseline = self.build_cp(collection, &best_order, baseline_deadline)?;
        baseline.goal_cover_goal_id = goal_cover_goal(&best_order);
        let mut best_h = baseline.compute_heuristic(abstract_state_ids);
        let mut partitions = vec![baseline];
        let mut best_index = 0;
        let candidate_deadline = earliest_deadline(
            baseline_deadline,
            optimization_deadline(optimization_max_time),
        );
        let candidates = if !optimization_max_time.is_sign_positive() {
            Vec::new()
        } else if interval {
            self.candidate_interval_orders(incumbent_order, components, abstraction_h)
        } else {
            self.candidate_label_orders(incumbent_order, abstraction_h)
                .into_iter()
                .map(|order| (None, order))
                .collect()
        };

        for (goal_id, candidate) in candidates {
            if candidate_deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                break;
            }
            if candidate == best_order {
                continue;
            }
            let mut candidate_cp = match self.build_cp(collection, &candidate, candidate_deadline) {
                Ok(cp) => cp,
                Err(error) if Self::is_deadline_exceeded(&error) => {
                    info!("scp: candidate order stopped while computing table (deadline)");
                    break;
                }
                Err(error) => return Err(error),
            };
            candidate_cp.goal_cover_goal_id = goal_id.or_else(|| goal_cover_goal(&candidate));
            let candidate_h = candidate_cp.compute_heuristic(abstract_state_ids);
            let candidate_index = partitions.len();
            partitions.push(candidate_cp);
            if candidate_h > best_h
                || (candidate_h == best_h
                    && partitions[best_index].is_empty()
                    && !partitions[candidate_index].is_empty())
            {
                if self.debug_diagnostics {
                    info!("scp: candidate order improved h {best_h} -> {candidate_h}");
                }
                best_h = candidate_h;
                best_order = candidate;
                best_index = candidate_index;
            }
        }

        *incumbent_order = best_order;
        Ok(CandidateCostPartitions {
            partitions,
            best_index,
        })
    }

    pub(super) fn candidate_interval_orders(
        &self,
        base_order: &[usize],
        components: &[AbstractionComponent<'_>],
        abstraction_h: &[f64],
    ) -> Vec<(Option<usize>, Vec<usize>)> {
        let mut orders = Vec::new();
        orders.push((None, base_order.to_vec()));

        let mut declaration_order = base_order.to_vec();
        declaration_order.sort_unstable();
        orders.push((None, declaration_order));

        if let Some(prefixed_goal_cover_order) =
            self.prefixed_goal_cover_order(base_order, components, abstraction_h)
        {
            orders.push((None, prefixed_goal_cover_order));
        }

        orders.extend(
            self.compact_goal_cover_orders(base_order, components, abstraction_h)
                .into_iter()
                .map(|(goal_id, order)| (Some(goal_id), order)),
        );

        if let Some(goal_cover_order) =
            self.goal_cover_order(base_order, components, abstraction_h, false)
        {
            orders.push((None, goal_cover_order));
        }

        orders.push((None, max_heuristic_greedy_order(base_order, abstraction_h)));

        let mut by_collection = base_order.to_vec();
        by_collection.sort_by_key(|&id| abstraction_collection_iteration(components, id));
        orders.push((None, by_collection));

        let mut concrete_splits_first = base_order.to_vec();
        concrete_splits_first.sort_by(|&left, &right| {
            uses_wanted_splits(components, left)
                .cmp(&uses_wanted_splits(components, right))
                .then_with(|| abstraction_h[right].total_cmp(&abstraction_h[left]))
                .then_with(|| left.cmp(&right))
        });
        orders.push((None, concrete_splits_first));

        let mut wanted_splits_first = base_order.to_vec();
        wanted_splits_first.sort_by(|&left, &right| {
            uses_wanted_splits(components, right)
                .cmp(&uses_wanted_splits(components, left))
                .then_with(|| abstraction_h[right].total_cmp(&abstraction_h[left]))
                .then_with(|| left.cmp(&right))
        });
        orders.push((None, wanted_splits_first));

        for seed_offset in 0..3 {
            let mut random_order = base_order.to_vec();
            random_order.shuffle(&mut SmallRng::seed_from_u64(
                self.config
                    .random_seed
                    .unwrap_or(0x5C9_0A11)
                    .wrapping_add(seed_offset),
            ));
            orders.push((None, random_order));
        }

        deduplicate_goal_cover_orders(orders)
    }

    pub(super) fn candidate_label_orders(
        &self,
        base_order: &[usize],
        abstraction_h: &[f64],
    ) -> Vec<Vec<usize>> {
        deduplicate_orders(vec![
            base_order.to_vec(),
            max_heuristic_greedy_order(base_order, abstraction_h),
        ])
    }

    pub(super) fn optimize_order_with_hill_climbing(
        &self,
        collection: PartitionedCollection<'_, '_>,
        incumbent_order: &mut [usize],
        incumbent_cp: &mut CostPartitioningHeuristic,
        optimization_deadline: Option<Instant>,
    ) -> Result<(), EvaluationError> {
        let abstract_state_ids = collection.abstract_state_ids;
        let mut incumbent_h = incumbent_cp.compute_heuristic(abstract_state_ids);
        if self.debug_diagnostics {
            info!("scp: order optimization incumbent_h={incumbent_h}");
        }

        loop {
            if optimization_deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                break;
            }
            let mut improved = false;
            for i in 0..incumbent_order.len() {
                for j in (i + 1)..incumbent_order.len() {
                    if optimization_deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                        return Ok(());
                    }

                    incumbent_order.swap(i, j);
                    let neighbor_result =
                        self.build_cp(collection, incumbent_order, optimization_deadline);
                    let neighbor_cp = match neighbor_result {
                        Ok(cp) => cp,
                        Err(error) if Self::is_deadline_exceeded(&error) => {
                            incumbent_order.swap(i, j);
                            info!(
                                "scp: order optimization stopped while computing table (deadline)"
                            );
                            return Ok(());
                        }
                        Err(error) => {
                            incumbent_order.swap(i, j);
                            return Err(error);
                        }
                    };
                    let neighbor_h = neighbor_cp.compute_heuristic(abstract_state_ids);
                    if neighbor_h > incumbent_h {
                        if self.debug_diagnostics {
                            info!(
                                "scp: order optimization swapped positions {i}/{j}, h {incumbent_h} -> {neighbor_h}"
                            );
                        }
                        *incumbent_cp = neighbor_cp;
                        incumbent_h = neighbor_h;
                        improved = true;
                        break;
                    }

                    incumbent_order.swap(i, j);
                }
                if improved {
                    break;
                }
            }
            if !improved {
                break;
            }
        }
        Ok(())
    }
}

pub(super) fn abstraction_collection_iteration(
    components: &[AbstractionComponent<'_>],
    component_id: usize,
) -> usize {
    components
        .get(component_id)
        .and_then(AbstractionComponent::as_domain)
        .and_then(|abstraction| abstraction.metadata.collection_iteration)
        .unwrap_or(usize::MAX)
}

pub(super) fn max_heuristic_greedy_order(
    base_order: &[usize],
    abstraction_h: &[f64],
) -> Vec<usize> {
    let mut order = base_order.to_vec();
    order.sort_by(|&left, &right| {
        let left_h = abstraction_h[left];
        let right_h = abstraction_h[right];
        right_h.total_cmp(&left_h).then_with(|| left.cmp(&right))
    });
    order
}

pub(super) fn uses_wanted_splits(
    components: &[AbstractionComponent<'_>],
    component_id: usize,
) -> bool {
    components
        .get(component_id)
        .and_then(AbstractionComponent::as_domain)
        .and_then(|abstraction| abstraction.metadata.flaw_kind.as_deref())
        .is_some_and(|flaw_kind| flaw_kind == "target_centered")
}

pub(super) fn deduplicate_orders(orders: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    let mut seen = HashSet::with_capacity(orders.len());
    orders
        .into_iter()
        .filter(|order| seen.insert(order.clone()))
        .collect()
}
