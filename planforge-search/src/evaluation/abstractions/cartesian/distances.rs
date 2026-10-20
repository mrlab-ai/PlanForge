use super::*;

#[derive(Debug)]
pub(super) struct ShortestPaths {
    distances: Vec<f64>,
    generating_transition: Vec<Option<TransitionKey>>,
    dependents: Vec<Vec<usize>>,
    dependent_positions: Vec<Option<usize>>,
    is_goal: Vec<bool>,
    invalid: Vec<bool>,
}

impl ShortestPaths {
    #[cfg(test)]
    pub(super) fn for_test(
        distances: Vec<f64>,
        generating_transition: Vec<Option<TransitionKey>>,
        dependents: Vec<Vec<usize>>,
        dependent_positions: Vec<Option<usize>>,
        is_goal: Vec<bool>,
    ) -> Self {
        let invalid = vec![false; distances.len()];
        Self {
            distances,
            generating_transition,
            dependents,
            dependent_positions,
            is_goal,
            invalid,
        }
    }

    #[cfg(test)]
    pub(super) fn dependents(&self, state_id: usize) -> &[usize] {
        &self.dependents[state_id]
    }

    #[cfg(test)]
    pub(super) fn dependent_position(&self, state_id: usize) -> Option<usize> {
        self.dependent_positions[state_id]
    }

    pub(super) fn distances(&self) -> &[f64] {
        &self.distances
    }

    pub(super) fn distance(&self, state_id: usize) -> f64 {
        self.distances[state_id]
    }

    pub(super) fn is_goal(&self, state_id: usize) -> bool {
        self.is_goal[state_id]
    }

    pub(super) fn generating_transition(&self, state_id: usize) -> Option<TransitionKey> {
        self.generating_transition[state_id]
    }
}

impl ShortestPaths {
    pub(super) fn remove_generating_transition(&mut self, source: usize) {
        let Some(old) = self.generating_transition[source].take() else {
            assert!(
                self.dependent_positions[source].is_none(),
                "Cartesian state without a generating transition has a dependency position"
            );
            return;
        };
        let position = self.dependent_positions[source]
            .take()
            .expect("Cartesian generating transition has no dependency position");
        let removed = self.dependents[old.target].swap_remove(position);
        assert_eq!(
            removed, source,
            "Cartesian dependency position references another state"
        );
        if position < self.dependents[old.target].len() {
            let moved = self.dependents[old.target][position];
            self.dependent_positions[moved] = Some(position);
        }
    }

    fn set_generating_transition(&mut self, source: usize, transition: TransitionKey) {
        assert_eq!(transition.source, source);
        assert_ne!(
            transition.target, source,
            "self-loop cannot generate a shortest path with nonnegative costs"
        );
        self.remove_generating_transition(source);
        let position = self.dependents[transition.target].len();
        self.dependents[transition.target].push(source);
        self.dependent_positions[source] = Some(position);
        self.generating_transition[source] = Some(transition);
    }
}

pub(super) fn compute_shortest_paths(
    working: &WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
) -> Result<ShortestPaths> {
    let mut is_goal = vec![false; working.states().len()];
    for (state_id, region) in working.states().iter().enumerate() {
        if semantics.regions().is_goal(region)? {
            is_goal[state_id] = true;
        }
    }
    compute_shortest_paths_with_goals(working, semantics, is_goal)
}

pub(super) fn compute_shortest_paths_with_goals(
    working: &WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    is_goal: Vec<bool>,
) -> Result<ShortestPaths> {
    ensure!(is_goal.len() == working.states().len());
    ensure!(
        is_goal.iter().any(|is_goal| *is_goal),
        "Cartesian abstraction has no abstract goal state"
    );
    let goal_states = is_goal
        .iter()
        .enumerate()
        .filter_map(|(state_id, &state_is_goal)| state_is_goal.then_some(state_id))
        .collect::<Vec<_>>();
    let mut generating_transition = vec![None; working.states().len()];
    let distances = backward_goal_distances(
        &mut WorkingGraph { working, semantics },
        &goal_states,
        || Ok(()),
        Some(&mut generating_transition),
    )?;
    let mut dependents = vec![Vec::new(); working.states().len()];
    let mut dependent_positions = vec![None; working.states().len()];
    for (source, transition) in generating_transition.iter().enumerate() {
        if let Some(transition) = transition {
            let position = dependents[transition.target].len();
            dependents[transition.target].push(source);
            dependent_positions[source] = Some(position);
        }
    }
    Ok(ShortestPaths {
        distances,
        generating_transition,
        dependents,
        dependent_positions,
        is_goal,
        invalid: vec![false; working.states().len()],
    })
}

/// The working abstraction under the operator costs; labels are transitions.
struct WorkingGraph<'a> {
    working: &'a WorkingAbstraction,
    semantics: &'a CartesianSemantics<'a>,
}

impl BackwardGraph for WorkingGraph<'_> {
    type Label = TransitionKey;

    fn num_states(&self) -> usize {
        self.working.states().len()
    }

    fn incoming(
        &mut self,
        target: usize,
        incoming: &mut Vec<(usize, TransitionKey, f64)>,
    ) -> Result<()> {
        incoming.clear();
        let operator_costs = self.semantics.cegar().operator_costs();
        for &transition_id in &self.working.incoming()[target] {
            let transition = self.working.transition(transition_id);
            // A self-loop never shortens a path.
            if transition.source == target {
                continue;
            }
            incoming.push((
                transition.source,
                TransitionKey {
                    source: transition.source,
                    concrete_op_id: transition.concrete_op_id,
                    target,
                },
                operator_costs[transition.concrete_op_id.index()].value(),
            ));
        }
        Ok(())
    }
}

pub(super) fn update_shortest_paths_after_split(
    working: &WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    shortest_paths: &mut ShortestPaths,
    split_state_id: usize,
    new_state_id: usize,
) -> Result<()> {
    let old_num_states = shortest_paths.distances.len();
    ensure!(
        new_state_id == old_num_states && working.states().len() == old_num_states + 1,
        "Cartesian incremental shortest-path update requires one appended split state"
    );

    let mut queue = std::collections::VecDeque::new();
    let mut invalid_states = Vec::new();
    let invalidate = |state_id: usize,
                      shortest_paths: &mut ShortestPaths,
                      invalid_states: &mut Vec<usize>,
                      queue: &mut std::collections::VecDeque<usize>| {
        if !shortest_paths.invalid[state_id] {
            shortest_paths.invalid[state_id] = true;
            invalid_states.push(state_id);
            queue.push_back(state_id);
        }
    };
    let parent_distance = shortest_paths.distances[split_state_id];
    shortest_paths.distances.push(parent_distance);
    shortest_paths.generating_transition.push(None);
    shortest_paths.dependents.push(Vec::new());
    shortest_paths.dependent_positions.push(None);
    shortest_paths.is_goal[split_state_id] = semantics
        .regions()
        .is_goal(&working.states()[split_state_id])?;
    shortest_paths.is_goal.push(
        semantics
            .regions()
            .is_goal(&working.states()[new_state_id])?,
    );
    shortest_paths.invalid.push(false);

    invalidate(
        split_state_id,
        shortest_paths,
        &mut invalid_states,
        &mut queue,
    );
    invalidate(
        new_state_id,
        shortest_paths,
        &mut invalid_states,
        &mut queue,
    );
    while let Some(target) = queue.pop_front() {
        shortest_paths.remove_generating_transition(target);
        let dependents = std::mem::take(&mut shortest_paths.dependents[target]);
        for source in dependents {
            let transition = shortest_paths.generating_transition[source]
                .take()
                .expect("Cartesian shortest-path dependent has no generating transition");
            assert_eq!(transition.target, target);
            shortest_paths.dependent_positions[source] = None;
            invalidate(source, shortest_paths, &mut invalid_states, &mut queue);
        }
    }

    for &state_id in &invalid_states {
        shortest_paths.distances[state_id] = f64::INFINITY;
    }

    let mut heap = BinaryHeap::new();
    for &state_id in &invalid_states {
        if shortest_paths.is_goal[state_id] {
            shortest_paths.distances[state_id] = 0.0;
            heap.push((Reverse(NotNan::new(0.0).unwrap()), state_id));
        }
    }

    for &source in &invalid_states {
        for &transition_id in &working.outgoing()[source] {
            let transition = working.transition(transition_id);
            if transition.source == transition.target || shortest_paths.invalid[transition.target] {
                continue;
            }
            let target_distance = shortest_paths.distances[transition.target];
            if !target_distance.is_finite() {
                continue;
            }
            let candidate = target_distance
                + semantics.cegar().operator_costs()[transition.concrete_op_id.index()]
                    .value()
                    .max(0.0);
            if candidate + float_tolerance::SEARCH_EPSILON < shortest_paths.distances[source] {
                shortest_paths.distances[source] = candidate;
                shortest_paths.set_generating_transition(
                    source,
                    TransitionKey {
                        source,
                        concrete_op_id: transition.concrete_op_id,
                        target: transition.target,
                    },
                );
                heap.push((Reverse(NotNan::new(candidate).unwrap()), source));
            }
        }
    }

    while let Some((Reverse(distance), target)) = heap.pop() {
        let distance = distance.into_inner();
        if distance > shortest_paths.distances[target] + float_tolerance::SEARCH_EPSILON {
            continue;
        }
        for &transition_id in &working.incoming()[target] {
            let transition = working.transition(transition_id);
            if transition.source == target || !shortest_paths.invalid[transition.source] {
                continue;
            }
            let alternative = distance
                + semantics.cegar().operator_costs()[transition.concrete_op_id.index()]
                    .value()
                    .max(0.0);
            if alternative + float_tolerance::SEARCH_EPSILON
                < shortest_paths.distances[transition.source]
            {
                shortest_paths.distances[transition.source] = alternative;
                shortest_paths.set_generating_transition(
                    transition.source,
                    TransitionKey {
                        source: transition.source,
                        concrete_op_id: transition.concrete_op_id,
                        target,
                    },
                );
                heap.push((
                    Reverse(NotNan::new(alternative).unwrap()),
                    transition.source,
                ));
            }
        }
    }

    #[cfg(debug_assertions)]
    if working.states().len() <= 512 {
        let reference = compute_shortest_paths(working, semantics)?;
        for state_id in 0..working.states().len() {
            let actual = shortest_paths.distances[state_id];
            let expected = reference.distances[state_id];
            assert!(
                (actual == expected) || (actual - expected).abs() <= 1e-7,
                "incremental Cartesian distance mismatch at state {state_id}: {actual} vs {expected}"
            );
        }
    }

    for state_id in invalid_states {
        shortest_paths.invalid[state_id] = false;
    }
    Ok(())
}
