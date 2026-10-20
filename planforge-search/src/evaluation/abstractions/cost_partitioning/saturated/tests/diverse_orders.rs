use super::*;

#[test]
fn diverse_orders_retain_available_goal_cover_orders_per_cartesian_goal() {
    // Standalone bounds live in one envelope outside the SCP order budget.
    let task = std::sync::Arc::new(independent_goals_task());
    let (heuristic, expected_goals) = goal_cover_scp(&task, |goals| 1 + 4 * goals);
    evaluate_initial(&task, &heuristic).unwrap();
    let retained_per_goal = retained_goal_cover_orders_per_goal(&heuristic);
    assert_eq!(
        retained_per_goal.keys().copied().collect::<HashSet<_>>(),
        expected_goals
    );
    assert!(
        retained_per_goal
            .values()
            .all(|&count| (2..=4).contains(&count)),
        "retained specialists by goal: {retained_per_goal:?}"
    );
}

/// The global best and the goal specialists are part of the heuristic, so a
/// max_orders below their number limits only the diversified orders.
#[test]
fn required_orders_are_retained_beyond_max_orders() {
    let task = std::sync::Arc::new(independent_goals_task());
    let (heuristic, expected_goals) = goal_cover_scp(&task, |_| 1);
    evaluate_initial(&task, &heuristic).unwrap();
    let retained_per_goal = retained_goal_cover_orders_per_goal(&heuristic);
    assert_eq!(
        retained_per_goal.keys().copied().collect::<HashSet<_>>(),
        expected_goals
    );
    assert!(heuristic.state.borrow().cp_heuristics.len() > 1);
}

#[test]
fn diverse_orders_support_mixed_abstraction_representations() {
    let task = std::sync::Arc::new(independent_goals_task());
    let domain_config = CegarConfig {
        max_states: 16,
        combine_labels: false,
        compute_source_sets: true,
        ..Default::default()
    };
    let domain = DomainAbstractionGenerator::new(domain_config)
        .unwrap()
        .generate(&*task)
        .unwrap();
    let pattern = Pattern::new(vec![VariableIndex::from_usize(1)], vec![]);
    let pdb = PatternDatabase::new(ProjectedTask::new(&*task, &pattern).unwrap(), 32).unwrap();
    let components = vec![
        AbstractionComponent::domain(None, domain),
        AbstractionComponent::cartesian(None, cartesian_abstraction(&task)),
        AbstractionComponent::pattern_database(pdb),
    ];
    let mut config = scp_config(Saturator::All, true);
    config.diversify = true;
    config.samples = 16;
    config.max_orders = 8;
    let heuristic = SaturatedCostPartitioningHeuristic::new(
        None,
        components,
        config,
        &*task,
        ScpBuildOptions {
            sampling_task: Some(task.clone()),
            ..Default::default()
        },
    )
    .unwrap();

    assert_eq!(evaluate_initial(&task, &heuristic).unwrap(), 5.0);
    let state = heuristic.state.borrow();
    assert!(state.built);
    assert!(state.sample_abstract_state_ids.is_empty());
    assert!(!state.cp_heuristics.is_empty());
    assert!(state.cp_heuristics.len() <= 8);
    drop(state);
    assert!(heuristic.sampling_task.borrow().is_none());
    assert!(heuristic.components.borrow().iter().all(|component| {
        component
            .as_cartesian()
            .is_none_or(|abstraction| abstraction.transition_system.transitions.is_empty())
    }));
}

#[test]
fn diverse_order_generator_requires_diversify() {
    let task = independent_goals_task();
    let component = AbstractionComponent::cartesian(None, cartesian_abstraction(&task));
    let mut config = scp_config(Saturator::All, true);
    config.order_generator = OrderGenerator::Diverse;

    let result = SaturatedCostPartitioningHeuristic::new(
        None,
        vec![component],
        config,
        &task,
        Default::default(),
    );
    let error = match result {
        Ok(_) => panic!("diverse orders without diversification must be rejected"),
        Err(error) => error,
    };

    assert!(
        error
            .to_string()
            .contains("diverse SCP orders require diversify=true")
    );
}

#[test]
fn diverse_orders_reject_nan_initial_order_budget() {
    let task = std::sync::Arc::new(independent_goals_task());
    let component = AbstractionComponent::cartesian(None, cartesian_abstraction(&task));
    let mut config = scp_config(Saturator::All, true);
    config.diversify = true;
    config.initial_order_generation_max_time = f64::NAN;

    let result = SaturatedCostPartitioningHeuristic::new(
        None,
        vec![component],
        config,
        &*task,
        ScpBuildOptions {
            sampling_task: Some(task.clone()),
            ..Default::default()
        },
    );
    let error = match result {
        Ok(_) => panic!("NaN initial-order budget must be rejected"),
        Err(error) => error,
    };

    assert!(
        error
            .to_string()
            .contains("initial_order_generation_max_time >= 0")
    );
}

#[test]
fn diverse_orders_keep_partitionings_that_improve_different_samples() {
    let partition = |distances| CostPartitioningHeuristic {
        lookup_tables: vec![LookupTable {
            abstraction_id: 0,
            distances,
            unknown_value: f64::INFINITY,
        }],
        goal_cover_goal_id: None,
    };
    let samples = vec![vec![Some(0)], vec![Some(1)]];
    let mut best = vec![f64::NEG_INFINITY; samples.len()];
    let mut selected = Vec::new();
    let mut size_kb = 0;

    assert!(retain_if_improving_some_sample(
        partition(vec![5.0, 1.0]),
        &samples,
        &mut best,
        &mut selected,
        &mut size_kb,
    ));
    assert!(retain_if_improving_some_sample(
        partition(vec![4.0, 6.0]),
        &samples,
        &mut best,
        &mut selected,
        &mut size_kb,
    ));
    assert!(!retain_if_improving_some_sample(
        partition(vec![5.0, 5.0]),
        &samples,
        &mut best,
        &mut selected,
        &mut size_kb,
    ));

    assert_eq!(selected.len(), 2);
    assert_eq!(best, vec![5.0, 6.0]);
}

#[test]
fn abstraction_maximum_keeps_every_abstraction_without_orders() {
    let mut state = ScpState::new(Some(1));
    state.goal_distances_by_abstraction = vec![vec![5.0, 1.0], vec![2.0, 6.0], vec![0.0, 0.0]];
    let samples = [vec![Some(0), Some(0), None], vec![Some(1), Some(1), None]];

    assert_eq!(
        samples
            .iter()
            .map(|ids| SaturatedCostPartitioningHeuristic::compute_max_h(&state, ids))
            .collect::<Vec<_>>(),
        vec![5.0, 6.0]
    );

    SaturatedCostPartitioningHeuristic::retain_abstraction_lookup_tables(&mut state);
    state.lookup_mask = SaturatedCostPartitioningHeuristic::compute_lookup_mask(&state, 3);

    assert!(state.goal_distances_by_abstraction.is_empty());
    assert_eq!(state.abstraction_lookup_tables.len(), 2);
    assert!(state.cp_heuristics.is_empty());
    assert_eq!(state.lookup_mask, vec![true, true, false]);
    assert_eq!(
        samples
            .iter()
            .map(|ids| SaturatedCostPartitioningHeuristic::compute_max_h(&state, ids))
            .collect::<Vec<_>>(),
        vec![5.0, 6.0]
    );
}

#[test]
fn abstraction_maximum_size_excludes_tables_that_are_not_retained() {
    let useful = vec![1.0; 128];
    let zero = vec![0.0; 128];

    assert_eq!(abstraction_lookup_tables_size_kb(&[useful, zero]), 1);
}

#[test]
fn diverse_orders_keep_tables_that_are_zero_only_at_the_initial_state() {
    let distances = [0.0, 4.0];
    let initial_state = [Some(0)];

    assert!(should_skip_zero_current_table(
        false,
        "test",
        0,
        &distances,
        &initial_state,
    ));
    assert!(!should_skip_zero_current_table(
        true,
        "test",
        0,
        &distances,
        &initial_state,
    ));
}
