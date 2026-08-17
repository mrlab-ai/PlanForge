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

    pub(super) fn goal_flags(&self) -> &[bool] {
        &self.is_goal
    }

    pub(super) fn generating_transition(&self, state_id: usize) -> Option<TransitionKey> {
        self.generating_transition[state_id]
    }
}

pub(super) struct StableAbstractSearch {
    h_values: Vec<f64>,
    g_values: Vec<f64>,
    predecessors: Vec<Option<TransitionKey>>,
    open: BinaryHeap<(Reverse<NotNan<f64>>, usize, usize)>,
}

impl StableAbstractSearch {
    pub(super) fn trivial() -> Self {
        Self {
            h_values: vec![0.0],
            g_values: vec![f64::INFINITY],
            predecessors: vec![None],
            open: BinaryHeap::new(),
        }
    }

    pub(super) fn inherit_split_state(&mut self, split_state_id: usize, new_state_id: usize) {
        assert_eq!(new_state_id, self.h_values.len());
        self.h_values.push(self.h_values[split_state_id]);
        self.g_values.push(f64::INFINITY);
        self.predecessors.push(None);
    }

    pub(super) fn find_plan(
        &mut self,
        working: &WorkingAbstraction,
        semantics: &CartesianSemantics<'_>,
        initial_state: usize,
        is_goal: &[bool],
    ) -> Result<Option<Vec<TransitionKey>>> {
        ensure!(self.h_values.len() == working.states().len());
        ensure!(is_goal.len() == working.states().len());
        ensure!(self.g_values.len() == working.states().len());
        ensure!(self.predecessors.len() == working.states().len());
        self.g_values.fill(f64::INFINITY);
        self.predecessors.fill(None);
        self.open.clear();
        self.g_values[initial_state] = 0.0;
        let mut sequence = 0usize;
        self.open.push((
            Reverse(NotNan::new(self.h_values[initial_state]).unwrap()),
            sequence,
            initial_state,
        ));

        let mut abstract_goal = None;
        while let Some((Reverse(old_f), _, state_id)) = self.open.pop() {
            let current_f = self.g_values[state_id] + self.h_values[state_id];
            if current_f + float_tolerance::SEARCH_EPSILON < old_f.into_inner() {
                continue;
            }
            if is_goal[state_id] {
                abstract_goal = Some(state_id);
                break;
            }
            for &transition_id in &working.outgoing()[state_id] {
                let transition = working.transition(transition_id);
                let candidate =
                    self.g_values[state_id] + semantics.operator_costs()[transition.concrete_op_id];
                if candidate < self.g_values[transition.target] {
                    self.g_values[transition.target] = candidate;
                    self.predecessors[transition.target] = Some(TransitionKey {
                        source: transition.source,
                        concrete_op_id: transition.concrete_op_id,
                        target: transition.target,
                    });
                    sequence = sequence
                        .checked_add(1)
                        .context("ICAPS Cartesian abstract-search insertion counter overflow")?;
                    let f = candidate + self.h_values[transition.target];
                    self.open.push((
                        Reverse(NotNan::new(f).context(
                            "ICAPS Cartesian abstract search produced a non-finite key",
                        )?),
                        sequence,
                        transition.target,
                    ));
                }
            }
        }

        let Some(mut state_id) = abstract_goal else {
            return Ok(None);
        };
        let mut plan = Vec::new();
        while state_id != initial_state {
            let transition = self.predecessors[state_id].with_context(|| {
                format!(
                    "ICAPS Cartesian abstract goal state {state_id} has no predecessor from initial state {initial_state}"
                )
            })?;
            plan.push(transition);
            state_id = transition.source;
        }
        plan.reverse();

        for transition in plan.iter().rev() {
            let path_h = self.h_values[transition.target]
                + semantics.operator_costs()[transition.concrete_op_id];
            ensure!(
                path_h + float_tolerance::SEARCH_EPSILON >= self.h_values[transition.source],
                "ICAPS Cartesian inherited h-value decreased along selected abstract plan"
            );
            self.h_values[transition.source] = path_h;
        }
        Ok(Some(plan))
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
        if semantics.region_is_goal(region)? {
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
    let mut distances = vec![f64::INFINITY; working.states().len()];
    let mut generating_transition = vec![None; working.states().len()];
    let mut heap = BinaryHeap::new();
    for (state_id, &state_is_goal) in is_goal.iter().enumerate() {
        if state_is_goal {
            distances[state_id] = 0.0;
            heap.push((Reverse(NotNan::new(0.0).unwrap()), state_id));
        }
    }
    while let Some((Reverse(distance), target)) = heap.pop() {
        let distance = distance.into_inner();
        if distance > distances[target] + float_tolerance::SEARCH_EPSILON {
            continue;
        }
        for &transition_id in &working.incoming()[target] {
            let transition = working.transition(transition_id);
            if transition.source == target {
                continue;
            }
            let cost = semantics.operator_costs()[transition.concrete_op_id];
            ensure!(
                cost >= -float_tolerance::SEARCH_EPSILON && cost.is_finite(),
                "invalid operator cost {cost}"
            );
            let alternative = distance + cost.max(0.0);
            let source = transition.source;
            if alternative + float_tolerance::SEARCH_EPSILON < distances[source] {
                distances[source] = alternative;
                generating_transition[source] = Some(TransitionKey {
                    source,
                    concrete_op_id: transition.concrete_op_id,
                    target,
                });
                heap.push((Reverse(NotNan::new(alternative).unwrap()), source));
            }
        }
    }
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
    if matches!(
        semantics.split_selection(),
        CartesianSplitSelection::Icaps26(_)
    ) {
        let parent_was_goal = shortest_paths.is_goal[split_state_id];
        shortest_paths.is_goal[split_state_id] = false;
        shortest_paths.is_goal.push(parent_was_goal);
    } else {
        shortest_paths.is_goal[split_state_id] =
            semantics.region_is_goal(&working.states()[split_state_id])?;
        shortest_paths
            .is_goal
            .push(semantics.region_is_goal(&working.states()[new_state_id])?);
    }
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
            let candidate =
                target_distance + semantics.operator_costs()[transition.concrete_op_id].max(0.0);
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
            let alternative =
                distance + semantics.operator_costs()[transition.concrete_op_id].max(0.0);
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
