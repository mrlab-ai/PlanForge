use super::*;

pub(super) struct CandidateCostPartitions {
    pub(super) partitions: Vec<CostPartitioningHeuristic>,
    pub(super) best_index: usize,
}

impl<'task> SaturatedCostPartitioningHeuristic<'task> {
    /// The order SCP starts from at the evaluated state: the greedy order, or
    /// for interval SCP over a goal-focused collection, its goal-cover order.
    pub(super) fn compute_order_for_state(
        &self,
        state: &mut ScpState,
        abstract_state_ids: &[Option<usize>],
        components: &[AbstractionComponent<'_>],
    ) -> Vec<usize> {
        let order = Self::compute_greedy_order_for_state(state, abstract_state_ids);
        if self.config.partitioning.uses_regions()
            && let Some(goal_cover_order) = self.goal_cover_order(
                &order,
                components,
                &abstraction_heuristic_values(state, abstract_state_ids),
                true,
            )
        {
            info!(
                "scp: using the goal-cover order, first abstractions {:?}",
                goal_cover_order.iter().take(24).collect::<Vec<_>>()
            );
            return goal_cover_order;
        }
        order
    }

    /// The orders tried at a sample: its greedy order, and for diverse orders
    /// also a random permutation.
    pub(super) fn diverse_orders_for_sample(
        &self,
        state: &mut ScpState,
        abstract_state_ids: &[Option<usize>],
    ) -> Vec<Vec<usize>> {
        let greedy = Self::compute_greedy_order_for_state(state, abstract_state_ids);
        if self.config.order_generator != OrderGenerator::Diverse {
            return vec![greedy];
        }
        let mut random = (0..state.goal_distances_by_abstraction.len()).collect::<Vec<_>>();
        random.shuffle(&mut state.rng);
        deduplicate_orders(vec![greedy, random])
    }

    pub(super) fn compute_greedy_order_for_state(
        state: &mut ScpState,
        abstract_state_ids: &[Option<usize>],
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
                compute_score(h, stolen)
            })
            .collect();

        order.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        order
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

pub(super) fn deduplicate_orders(orders: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    let mut seen = HashSet::with_capacity(orders.len());
    orders
        .into_iter()
        .filter(|order| seen.insert(order.clone()))
        .collect()
}
