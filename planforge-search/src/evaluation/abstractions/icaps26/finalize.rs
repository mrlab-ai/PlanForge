use super::*;

pub(super) fn finalize_abstraction(
    working: &WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    initial_state: usize,
    combine_labels: bool,
    compute_source_sets: bool,
) -> Result<(CartesianTransitions, DistanceTable, Vec<OperatorIndex>)> {
    let mut grouped: HashMap<(usize, usize), Vec<OperatorIndex>> = HashMap::new();
    let mut raw = Vec::new();
    for transition_id in working.active_transition_ids() {
        let transition = working.transition(transition_id);
        if combine_labels {
            grouped
                .entry((transition.source, transition.target))
                .or_default()
                .push(transition.concrete_op_id);
        } else {
            raw.push((
                transition.source,
                transition.target,
                vec![transition.concrete_op_id],
            ));
        }
    }
    // Self loops have zero shortest-path and saturated-cost requirements. Keep
    // them only while refining, where a later split can turn one into an exact
    // cross-child transition; materializing them here wastes memory without
    // changing standalone, canonical, label-SCP, or regional-SCP values.
    if combine_labels {
        raw.extend(grouped.into_iter().map(|((source, target), mut labels)| {
            labels.sort_unstable();
            labels.dedup();
            (source, target, labels)
        }));
    }
    raw.sort();
    let mut transitions = Vec::with_capacity(raw.len());
    let mut backward = vec![Vec::new(); working.states().len()];
    let mut source_sets = if compute_source_sets {
        Vec::with_capacity(raw.len())
    } else {
        Vec::new()
    };
    let shared_cartesian_sets = working
        .states
        .iter()
        .cloned()
        .map(Arc::new)
        .collect::<Vec<_>>();
    let mut relevant = HashSet::new();
    for (transition_id, (source, target, labels)) in raw.into_iter().enumerate() {
        if source != target {
            for &label in &labels {
                relevant.insert(label);
            }
        }
        if compute_source_sets {
            source_sets.push(AbstractOperatorSourceSets {
                labels: labels
                    .iter()
                    .copied()
                    .map(|concrete_op_id| {
                        let source_set = semantics.source_set_source_for_transition(
                            &shared_cartesian_sets[source],
                            concrete_op_id,
                            &shared_cartesian_sets[target],
                        )?
                        .with_context(|| {
                            format!(
                                "emitted Cartesian transition {source} --{concrete_op_id}--> {target} has an empty source operator region"
                            )
                        })?;
                        let source_region = if source_set == *shared_cartesian_sets[source] {
                            Arc::clone(&shared_cartesian_sets[source])
                        } else {
                            Arc::new(source_set)
                        };
                        Ok(LabelSourceSet {
                            concrete_op_id,
                            source: source_region,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
            });
        }
        transitions.push(AbstractTransition {
            transition_id,
            abstract_op_id: transition_id,
            concrete_op_ids: labels,
            source_state: source,
            target_state: target,
        });
        backward[target].push(transition_id);
    }
    let mut goal_states = Vec::new();
    for (state_id, region) in working.states().iter().enumerate() {
        if semantics.region_is_goal(region)? {
            goal_states.push(state_id);
        }
    }
    let transition_system = AbstractTransitionSystem {
        transitions,
        backward,
        goal_states,
    };
    let transition_costs = transition_system
        .transitions
        .iter()
        .map(|transition| {
            transition
                .concrete_op_ids
                .iter()
                .map(|&op_id| semantics.operator_costs()[op_id.index()].value())
                .fold(f64::INFINITY, f64::min)
        })
        .collect::<Vec<_>>();
    let distances = backward_goal_distances(
        &mut CostedTransitionSystem {
            system: &transition_system,
            transition_costs: &transition_costs,
        },
        &transition_system.goal_states,
        || Ok(()),
        None,
    )?;
    let distance_table = DistanceTable {
        distances,
        initial_state,
    };
    let mut relevant_operator_ids: Vec<_> = relevant.into_iter().collect();
    relevant_operator_ids.sort_unstable();
    let transitions = CartesianTransitions {
        system: transition_system,
        regions: shared_cartesian_sets,
        source_sets: compute_source_sets.then_some(source_sets),
    };
    Ok((transitions, distance_table, relevant_operator_ids))
}

pub(super) fn finalize_standalone_abstraction(
    working: &WorkingAbstraction,
    shortest_paths: &ShortestPaths,
    initial_state: usize,
) -> Result<(DistanceTable, Vec<OperatorIndex>)> {
    ensure!(
        shortest_paths.distances().len() == working.states().len(),
        "Cartesian shortest-path/state count mismatch during standalone finalization"
    );
    let mut relevant_operator_ids = working
        .active_transition_ids()
        .map(|transition_id| working.transition(transition_id).concrete_op_id)
        .collect::<Vec<_>>();
    relevant_operator_ids.sort_unstable();
    relevant_operator_ids.dedup();
    let distance_table = DistanceTable {
        distances: shortest_paths.distances().to_vec(),
        initial_state,
    };
    Ok((distance_table, relevant_operator_ids))
}

/// The goal distances of an ICAPS26 abstraction.
pub(super) struct DistanceTable {
    pub(super) distances: Vec<f64>,
    pub(super) initial_state: usize,
}
