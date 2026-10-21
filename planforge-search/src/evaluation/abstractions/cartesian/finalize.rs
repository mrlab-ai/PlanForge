use super::*;

pub(super) fn finalize_abstraction(
    working: &WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    initial_state: usize,
    combine_labels: bool,
) -> Result<(
    AbstractTransitionSystem,
    AbstractDistanceTable,
    Vec<OperatorIndex>,
)> {
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
        if semantics.regions().is_goal(region)? {
            goal_states.push(state_id);
        }
    }
    let transition_system = AbstractTransitionSystem {
        transitions,
        backward,
        goal_facts: semantics.task().goals().to_vec(),
        goal_states,
        initial_state,
        cartesian_sets: shared_cartesian_sets,
    };
    let transition_costs = transition_system
        .transitions
        .iter()
        .map(|transition| {
            transition
                .concrete_op_ids
                .iter()
                .map(|&op_id| semantics.cegar().operator_costs()[op_id.index()].value())
                .fold(f64::INFINITY, f64::min)
        })
        .collect::<Vec<_>>();
    let mut generating_op_ids = vec![None; transition_system.backward.len()];
    let distances = backward_goal_distances(
        &mut CostedTransitionSystem {
            system: &transition_system,
            transition_costs: &transition_costs,
        },
        &transition_system.goal_states,
        || Ok(()),
        Some(&mut generating_op_ids),
    )?;
    let distance_table = AbstractDistanceTable {
        distances,
        generating_op_ids,
        initial_state,
        goal_facts: transition_system.goal_facts.clone(),
    };
    let mut relevant_operator_ids: Vec<_> = relevant.into_iter().collect();
    relevant_operator_ids.sort_unstable();
    Ok((transition_system, distance_table, relevant_operator_ids))
}

pub(super) fn finalize_standalone_abstraction(
    working: &WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    shortest_paths: &ShortestPaths,
    initial_state: usize,
) -> Result<(
    AbstractTransitionSystem,
    AbstractDistanceTable,
    Vec<OperatorIndex>,
)> {
    ensure!(
        shortest_paths.distances().len() == working.states().len(),
        "Cartesian shortest-path/state count mismatch during standalone finalization"
    );
    let goal_facts = semantics.task().goals().to_vec();
    let mut relevant_operator_ids = working
        .active_transition_ids()
        .map(|transition_id| working.transition(transition_id).concrete_op_id)
        .collect::<Vec<_>>();
    relevant_operator_ids.sort_unstable();
    relevant_operator_ids.dedup();

    let distance_table = AbstractDistanceTable {
        distances: shortest_paths.distances().to_vec(),
        generating_op_ids: vec![None; working.states().len()],
        initial_state,
        goal_facts: goal_facts.clone(),
    };
    let transition_system = AbstractTransitionSystem {
        transitions: Vec::new(),
        backward: Vec::new(),
        goal_facts,
        goal_states: Vec::new(),
        initial_state,
        cartesian_sets: Vec::new(),
    };
    Ok((transition_system, distance_table, relevant_operator_ids))
}
