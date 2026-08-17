use super::*;

pub(super) fn finalize_abstraction(
    working: &WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    initial_state_hash: usize,
    combine_labels: bool,
    compute_operator_footprints: bool,
) -> Result<(
    AbstractTransitionSystem,
    AbstractDistanceTable,
    Vec<usize>,
    Vec<AbstractOperatorFootprint>,
)> {
    let mut grouped: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
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
    let mut forward = vec![Vec::new(); working.states().len()];
    let mut backward = vec![Vec::new(); working.states().len()];
    let mut footprints = if compute_operator_footprints {
        Vec::with_capacity(raw.len())
    } else {
        Vec::new()
    };
    let shared_state_regions = working
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
        if compute_operator_footprints {
            footprints.push(AbstractOperatorFootprint {
                labels: labels
                    .iter()
                    .copied()
                    .map(|concrete_op_id| {
                        let footprint = semantics.transition_source_footprint(
                            &shared_state_regions[source],
                            concrete_op_id,
                            &shared_state_regions[target],
                        )?
                        .with_context(|| {
                            format!(
                                "emitted Cartesian transition {source} --{concrete_op_id}--> {target} has an empty source footprint"
                            )
                        })?;
                        let source_region = if footprint == *shared_state_regions[source] {
                            Arc::clone(&shared_state_regions[source])
                        } else {
                            Arc::new(footprint)
                        };
                        Ok(ConcreteOperatorFootprint {
                            concrete_op_id,
                            source_region,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
            });
        }
        transitions.push(AbstractTransition {
            transition_id,
            abstract_op_id: transition_id,
            concrete_op_ids: labels,
            source_hash: source,
            target_hash: target,
        });
        forward[source].push(transition_id);
        backward[target].push(transition_id);
    }
    let mut goal_state_hashes = Vec::new();
    for (state_id, region) in working.states().iter().enumerate() {
        if semantics.region_is_goal(region)? {
            goal_state_hashes.push(state_id);
        }
    }
    let transition_system = AbstractTransitionSystem {
        transitions,
        duplicate_transition_attempts: 0,
        backward,
        forward,
        goal_facts: (0..semantics.task().get_num_goals())
            .map(|goal_id| *semantics.task().get_goal_fact(goal_id))
            .collect(),
        goal_state_hashes,
        initial_state_hash,
        hash_multipliers: Vec::new(),
        numeric_domain_sizes: Vec::new(),
        state_regions: shared_state_regions,
    };
    let transition_costs = transition_system
        .transitions
        .iter()
        .map(|transition| {
            transition
                .concrete_op_ids
                .iter()
                .map(|&op_id| semantics.operator_costs()[op_id])
                .fold(f64::INFINITY, f64::min)
        })
        .collect::<Vec<_>>();
    let mut generating_op_ids = vec![None; transition_system.backward.len()];
    let distances = build_explicit_goal_distances(
        &transition_system,
        &transition_costs,
        None,
        Some(&mut generating_op_ids),
    )?;
    let distance_table = AbstractDistanceTable {
        distances,
        generating_op_ids,
        initial_state_hash,
        goal_facts: transition_system.goal_facts.clone(),
        hash_multipliers: Vec::new(),
        numeric_domain_sizes: Vec::new(),
    };
    let mut relevant_operator_ids: Vec<_> = relevant.into_iter().collect();
    relevant_operator_ids.sort_unstable();
    Ok((
        transition_system,
        distance_table,
        relevant_operator_ids,
        footprints,
    ))
}

pub(super) fn finalize_standalone_abstraction(
    working: &WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    shortest_paths: &ShortestPaths,
    initial_state_hash: usize,
) -> Result<(
    AbstractTransitionSystem,
    AbstractDistanceTable,
    Vec<usize>,
    Vec<AbstractOperatorFootprint>,
)> {
    ensure!(
        shortest_paths.distances().len() == working.states().len(),
        "Cartesian shortest-path/state count mismatch during standalone finalization"
    );
    let goal_facts = (0..semantics.task().get_num_goals())
        .map(|goal_id| *semantics.task().get_goal_fact(goal_id))
        .collect::<Vec<_>>();
    let mut relevant_operator_ids = working
        .active_transition_ids()
        .map(|transition_id| working.transition(transition_id).concrete_op_id)
        .collect::<Vec<_>>();
    relevant_operator_ids.sort_unstable();
    relevant_operator_ids.dedup();

    let distance_table = AbstractDistanceTable {
        distances: shortest_paths.distances().to_vec(),
        generating_op_ids: vec![None; working.states().len()],
        initial_state_hash,
        goal_facts: goal_facts.clone(),
        hash_multipliers: Vec::new(),
        numeric_domain_sizes: Vec::new(),
    };
    let transition_system = AbstractTransitionSystem {
        transitions: Vec::new(),
        duplicate_transition_attempts: 0,
        backward: Vec::new(),
        forward: Vec::new(),
        goal_facts,
        goal_state_hashes: Vec::new(),
        initial_state_hash,
        hash_multipliers: Vec::new(),
        numeric_domain_sizes: Vec::new(),
        state_regions: Vec::new(),
    };
    Ok((
        transition_system,
        distance_table,
        relevant_operator_ids,
        Vec::new(),
    ))
}
