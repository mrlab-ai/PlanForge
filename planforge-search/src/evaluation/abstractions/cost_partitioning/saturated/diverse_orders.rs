use super::*;

impl<'task> SaturatedCostPartitioningHeuristic<'task> {
    /// The cost partitionings diversification retains, starting from
    /// `initial_candidates`, whose first is the best order's at the
    /// evaluated state.
    pub(super) fn select_diverse_orders(
        &self,
        collection: PartitionedCollection<'_, '_>,
        state: &mut OrderSelection<'task>,
        initial_candidates: Vec<CostPartitioningHeuristic>,
        budget: Deadline,
    ) -> Result<Vec<CostPartitioningHeuristic>, EvaluationError> {
        let initial_abstract_state_ids = collection.abstract_state_ids;
        assert!(self.config.diversify);
        assert!(!initial_candidates.is_empty());
        assert!(!initial_candidates[0].is_empty());

        let deadline = budget.earliest(deadline_from_now("max_time", self.config.max_time)?);
        let initial_h = initial_candidates[0].compute_heuristic(initial_abstract_state_ids);
        let samples = self.sample_states(collection.components, state, initial_h, deadline)?;
        assert!(
            !samples.is_empty(),
            "offline diversification must retain at least the initial state sample"
        );

        // The heuristic is the maximum over the retained orders, so every
        // sample starts at 0 and the mandatory orders below are part of it:
        // max_orders only bounds the diversified orders added after them.
        let mut sample_best = vec![0.0; samples.len()];
        let mut selected = Vec::new();
        let mut selected_size_kb = 0;
        let mut evaluated_orders = initial_candidates.len();
        let required_indices = best_goal_cover_order_indices(
            &initial_candidates,
            &samples,
            self.config.goal_cover_orders_per_goal,
        );
        let represented_goal_count = required_indices
            .iter()
            .filter_map(|&index| initial_candidates[index].goal_cover_goal_id)
            .collect::<HashSet<_>>()
            .len();
        let goal_cover_order_count = required_indices
            .iter()
            .filter(|&&index| initial_candidates[index].goal_cover_goal_id.is_some())
            .count();
        let mut initial_candidates = initial_candidates.into_iter().map(Some).collect::<Vec<_>>();
        let mut required_orders = Vec::new();
        for index in required_indices {
            let candidate = initial_candidates[index]
                .take()
                .expect("mandatory SCP candidate indices must be unique");
            if !required_orders.contains(&candidate) {
                required_orders.push(candidate);
            }
        }
        if required_orders.len() > self.config.max_orders {
            info!(
                "scp: retaining all {} mandatory orders (global best and goal specialists), more than max_orders={}; no diversified orders are added",
                required_orders.len(),
                self.config.max_orders,
            );
        }
        for candidate in required_orders {
            retain_required_order(
                candidate,
                &samples,
                &mut sample_best,
                &mut selected,
                &mut selected_size_kb,
            );
        }
        info!(
            "scp: retained the best order plus {goal_cover_order_count} goal-cover orders of {represented_goal_count} goals"
        );

        for candidate in initial_candidates.into_iter().flatten() {
            if selected.len() >= self.config.max_orders || deadline.passed() {
                break;
            }
            retain_if_improving_some_sample(
                candidate,
                &samples,
                &mut sample_best,
                &mut selected,
                &mut selected_size_kb,
            );
        }
        assert!(!selected.is_empty(), "the global best SCP must be retained");

        for sample_index in 1..samples.len() {
            if selected.len() >= self.config.max_orders || deadline.passed() {
                break;
            }
            let sample_ids = &samples[sample_index];
            let abstraction_h = abstraction_heuristic_values(state, sample_ids);
            // The same collection, evaluated at this sample rather than at the
            // state the selected was seeded from.
            let sample_collection = PartitionedCollection {
                abstract_state_ids: sample_ids,
                abstraction_h: &abstraction_h,
                ..collection
            };
            for order in self.diverse_orders_for_sample(state, sample_ids) {
                if selected.len() >= self.config.max_orders || deadline.passed() {
                    break;
                }
                let (candidate, completion) = self.build_cp(sample_collection, &order, deadline)?;
                evaluated_orders += 1;

                retain_if_improving_some_sample(
                    candidate,
                    &samples,
                    &mut sample_best,
                    &mut selected,
                    &mut selected_size_kb,
                );
                if completion == Completion::Truncated {
                    break;
                }
            }
        }

        info!(
            "scp: retained {} of {} evaluated orders over {} samples ({} KiB)",
            selected.len(),
            evaluated_orders,
            samples.len(),
            selected_size_kb,
        );
        Ok(selected)
    }
}

pub(super) fn retain_if_improving_some_sample(
    candidate: CostPartitioningHeuristic,
    sample_ids: &[Vec<Option<usize>>],
    sample_best: &mut [f64],
    selected: &mut Vec<CostPartitioningHeuristic>,
    selected_size_kb: &mut usize,
) -> bool {
    assert_eq!(sample_ids.len(), sample_best.len());
    // A candidate equal to a selected one improves no sample, so the
    // selected stay pairwise distinct.
    let candidate_size = candidate.estimate_size_in_kb();
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
    *selected_size_kb = selected_size_kb.saturating_add(candidate_size);
    selected.push(candidate);
    true
}

pub(super) fn best_goal_cover_order_indices(
    candidates: &[CostPartitioningHeuristic],
    sample_ids: &[Vec<Option<usize>>],
    orders_per_goal: usize,
) -> Vec<usize> {
    assert!(!candidates.is_empty());
    let mut candidates_by_goal = HashMap::<usize, Vec<(usize, f64)>>::new();
    for (index, candidate) in candidates.iter().enumerate() {
        let Some(goal_id) = candidate.goal_cover_goal_id else {
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
                .take(orders_per_goal)
                .map(|(index, _)| index)
                .filter(|&index| index != 0),
        );
    }
    indices
}

/// Adds an order the heuristic requires. Unlike the diversified orders, it is
/// not subject to max_orders.
pub(super) fn retain_required_order(
    candidate: CostPartitioningHeuristic,
    sample_ids: &[Vec<Option<usize>>],
    sample_best: &mut [f64],
    selected: &mut Vec<CostPartitioningHeuristic>,
    selected_size_kb: &mut usize,
) {
    assert_eq!(sample_ids.len(), sample_best.len());
    for (best, ids) in sample_best.iter_mut().zip(sample_ids) {
        *best = (*best).max(candidate.compute_heuristic(ids));
    }
    *selected_size_kb = selected_size_kb.saturating_add(candidate.estimate_size_in_kb());
    selected.push(candidate);
}
