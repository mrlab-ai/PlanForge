use super::*;

impl<'task> SaturatedCostPartitioningOnlineHeuristic<'task> {
    pub(super) fn build_offline_diversified_portfolio(
        &self,
        collection: PartitionedCollection<'_, '_>,
        state: &mut ScpOnlineState,
        initial_candidates: Vec<CostPartitioningHeuristic>,
        table_deadline: Option<Instant>,
    ) -> Result<Vec<CostPartitioningHeuristic>, EvaluationError> {
        let PartitionedCollection {
            task,
            components,
            abstract_state_ids: initial_abstract_state_ids,
            ..
        } = collection;
        assert!(!self.config.online);
        assert!(self.config.diversify);
        assert!(!initial_candidates.is_empty());
        assert!(!initial_candidates[0].is_empty());

        let diversification_deadline = self
            .config
            .max_time
            .is_finite()
            .then(|| Instant::now() + Duration::from_secs_f64(self.config.max_time.max(0.0)));
        let deadline = earliest_deadline(table_deadline, diversification_deadline);
        let initial_h = initial_candidates[0]
            .compute_heuristic(initial_abstract_state_ids)
            .max(standalone_envelope_value(state, initial_abstract_state_ids));
        self.generate_offline_samples(state, initial_h, deadline)?;
        assert!(
            !state.offline_sample_ids.is_empty(),
            "offline diversification must retain at least the initial state sample"
        );

        let mut sample_best = state
            .offline_sample_ids
            .iter()
            .map(|ids| standalone_envelope_value(state, ids))
            .collect::<Vec<_>>();
        let mut portfolio = Vec::new();
        // The standalone envelope and the mandatory orders below are part of
        // the heuristic, not optional extras: max_orders and max_size only
        // bound the diversified orders added after them.
        let standalone_size_kb = standalone_lookup_values_size_kb(&state.h_values_by_abstraction);
        let mut portfolio_size_kb = standalone_size_kb;
        let mut evaluated_orders = initial_candidates.len();
        let mandatory_indices =
            mandatory_goal_specialist_indices(&initial_candidates, &state.offline_sample_ids);
        let represented_goal_count = mandatory_indices
            .iter()
            .filter_map(|&index| initial_candidates[index].specialist_goal_id)
            .collect::<HashSet<_>>()
            .len();
        let specialist_count = mandatory_indices
            .iter()
            .filter(|&&index| initial_candidates[index].specialist_goal_id.is_some())
            .count();
        let mut initial_candidates = initial_candidates.into_iter().map(Some).collect::<Vec<_>>();
        let mut mandatory_partitions = Vec::new();
        for index in mandatory_indices {
            let candidate = initial_candidates[index]
                .take()
                .expect("mandatory SCP candidate indices must be unique");
            if !mandatory_partitions.contains(&candidate) {
                mandatory_partitions.push(candidate);
            }
        }
        if mandatory_partitions.len() > self.config.max_orders {
            info!(
                "scp_online: retaining all {} mandatory orders (global best and goal specialists), more than max_orders={}; no diversified orders are added",
                mandatory_partitions.len(),
                self.config.max_orders,
            );
        }
        for candidate in mandatory_partitions {
            retain_mandatory_partition(
                candidate,
                &state.offline_sample_ids,
                &mut sample_best,
                &mut portfolio,
                &mut portfolio_size_kb,
            );
        }
        info!(
            "scp_online: reserved {standalone_size_kb} KiB for the standalone envelope and retained {specialist_count} specialists across {represented_goal_count} goals plus the global best"
        );

        for candidate in initial_candidates.into_iter().flatten() {
            if portfolio.len() >= self.config.max_orders
                || portfolio_size_kb >= self.config.max_size
                || deadline.is_some_and(|end| Instant::now() >= end)
            {
                break;
            }
            retain_if_sample_improving(
                candidate,
                &state.offline_sample_ids,
                &mut sample_best,
                &mut portfolio,
                &mut portfolio_size_kb,
                self.config.max_size,
            );
        }
        assert!(
            !portfolio.is_empty(),
            "the global best SCP must be retained"
        );

        for sample_index in 1..state.offline_sample_ids.len() {
            if portfolio.len() >= self.config.max_orders
                || portfolio_size_kb >= self.config.max_size
                || deadline.is_some_and(|end| Instant::now() >= end)
            {
                break;
            }
            let sample_ids = state.offline_sample_ids[sample_index].clone();
            let standalone_h = standalone_current_h_values(state, &sample_ids);
            // The same collection, evaluated at this sample rather than at the
            // state the portfolio was seeded from.
            let sample_collection = PartitionedCollection {
                abstract_state_ids: &sample_ids,
                standalone_current_h: &standalone_h,
                ..collection
            };
            let orders = self.compute_diversification_orders(
                task,
                state,
                &sample_ids,
                components,
                deadline,
            )?;
            for mut order in orders {
                if portfolio.len() >= self.config.max_orders
                    || portfolio_size_kb >= self.config.max_size
                    || deadline.is_some_and(|end| Instant::now() >= end)
                {
                    break;
                }
                let candidate = if self.config.partitioning.uses_regions() {
                    self.build_abstract_operator_cp(
                        sample_collection,
                        &order,
                        deadline,
                        self.config.saturator,
                    )
                } else {
                    self.build_label_cp(sample_collection, &order, deadline)
                };
                let mut candidate = match candidate {
                    Ok(candidate) => candidate,
                    Err(error) if Self::is_online_deadline_error_eval(&error) => break,
                    Err(error) => return Err(error),
                };
                evaluated_orders += 1;

                if self.config.order_optimization_max_time > 0.0 {
                    let local_deadline =
                        optimization_deadline(self.config.order_optimization_max_time);
                    self.optimize_order_with_hill_climbing(
                        sample_collection,
                        &mut order,
                        &mut candidate,
                        earliest_deadline(deadline, local_deadline),
                    )?;
                }

                retain_if_sample_improving(
                    candidate,
                    &state.offline_sample_ids,
                    &mut sample_best,
                    &mut portfolio,
                    &mut portfolio_size_kb,
                    self.config.max_size,
                );
            }
        }

        let sample_count = state.offline_sample_ids.len();
        info!(
            "scp_online: offline diversification retained {} of {} evaluated partitions over {} samples ({} KiB)",
            portfolio.len(),
            evaluated_orders,
            sample_count,
            portfolio_size_kb,
        );
        state.offline_sample_ids.clear();
        state.offline_sample_ids.shrink_to_fit();
        Ok(portfolio)
    }
}

pub(super) fn retain_if_sample_improving(
    candidate: CostPartitioningHeuristic,
    sample_ids: &[Vec<Option<usize>>],
    sample_best: &mut [f64],
    portfolio: &mut Vec<CostPartitioningHeuristic>,
    portfolio_size_kb: &mut usize,
    max_size_kb: usize,
) -> bool {
    assert_eq!(sample_ids.len(), sample_best.len());
    if portfolio.contains(&candidate) {
        return false;
    }
    let candidate_size = candidate.estimate_size_in_kb();
    if !portfolio.is_empty() && portfolio_size_kb.saturating_add(candidate_size) > max_size_kb {
        return false;
    }
    let values = sample_ids
        .iter()
        .map(|ids| candidate.compute_heuristic(ids))
        .collect::<Vec<_>>();
    if !values
        .iter()
        .zip(sample_best.iter())
        .any(|(&value, &best)| value > best)
    {
        return false;
    }
    for (best, value) in sample_best.iter_mut().zip(values) {
        *best = (*best).max(value);
    }
    *portfolio_size_kb = portfolio_size_kb.saturating_add(candidate_size);
    portfolio.push(candidate);
    true
}

pub(super) fn mandatory_goal_specialist_indices(
    candidates: &[CostPartitioningHeuristic],
    sample_ids: &[Vec<Option<usize>>],
) -> Vec<usize> {
    assert!(!candidates.is_empty());
    pub(super) const SPECIALISTS_PER_GOAL: usize = 4;
    let mut candidates_by_goal = HashMap::<usize, Vec<(usize, f64)>>::new();
    for (index, candidate) in candidates.iter().enumerate() {
        let Some(goal_id) = candidate.specialist_goal_id else {
            continue;
        };
        let score = sample_ids
            .iter()
            .map(|ids| candidate.compute_heuristic(ids))
            .sum::<f64>();
        assert!(!score.is_nan(), "goal-specialist SCP score must not be NaN");
        candidates_by_goal
            .entry(goal_id)
            .or_default()
            .push((index, score));
    }

    let mut goals = candidates_by_goal.into_iter().collect::<Vec<_>>();
    goals.sort_by_key(|(goal_id, _)| *goal_id);
    let mut indices = vec![0];
    for (_, mut goal_candidates) in goals {
        goal_candidates.sort_by(|left, right| {
            (right.0 == 0)
                .cmp(&(left.0 == 0))
                .then_with(|| right.1.total_cmp(&left.1))
                .then_with(|| left.0.cmp(&right.0))
        });
        indices.extend(
            goal_candidates
                .into_iter()
                .take(SPECIALISTS_PER_GOAL)
                .map(|(index, _)| index)
                .filter(|&index| index != 0),
        );
    }
    indices
}

/// Adds an order the heuristic requires. Unlike the diversified orders, it is
/// not subject to max_orders or max_size.
pub(super) fn retain_mandatory_partition(
    candidate: CostPartitioningHeuristic,
    sample_ids: &[Vec<Option<usize>>],
    sample_best: &mut [f64],
    portfolio: &mut Vec<CostPartitioningHeuristic>,
    portfolio_size_kb: &mut usize,
) {
    assert_eq!(sample_ids.len(), sample_best.len());
    for (best, ids) in sample_best.iter_mut().zip(sample_ids) {
        *best = (*best).max(candidate.compute_heuristic(ids));
    }
    *portfolio_size_kb = portfolio_size_kb.saturating_add(candidate.estimate_size_in_kb());
    portfolio.push(candidate);
}
