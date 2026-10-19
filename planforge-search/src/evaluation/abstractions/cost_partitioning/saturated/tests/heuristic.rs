use super::*;

#[test]
fn scp_retains_an_order_that_is_weaker_only_at_the_initial_state() {
    let partition = |distances| CostPartitioningHeuristic {
        lookup_tables: vec![LookupTable {
            abstraction_id: 0,
            distances,
            unknown_value: f64::INFINITY,
        }],
        goal_cover_goal_id: None,
    };
    let mut state = ScpState::new(Some(1));
    SaturatedCostPartitioningHeuristic::retain_cp(&mut state, partition(vec![5.0, 1.0]));
    SaturatedCostPartitioningHeuristic::retain_cp(&mut state, partition(vec![4.0, 4.0]));
    SaturatedCostPartitioningHeuristic::retain_cp(&mut state, partition(vec![4.0, 4.0]));

    assert_eq!(state.cp_heuristics.len(), 2, "duplicates are discarded");
    assert_eq!(
        SaturatedCostPartitioningHeuristic::compute_max_h(&state, &[Some(0)]),
        5.0
    );
    assert_eq!(
        SaturatedCostPartitioningHeuristic::compute_max_h(&state, &[Some(1)]),
        4.0
    );
}

#[test]
fn cartesian_scp_supports_every_saturator_in_both_cost_modes() {
    let task = independent_goals_task();
    for saturator in [Saturator::All, Saturator::Perim, Saturator::Perimstar] {
        for abstract_operator in [false, true] {
            let component = AbstractionComponent::cartesian(None, cartesian_abstraction(&task));
            let heuristic = SaturatedCostPartitioningHeuristic::new(
                None,
                vec![component],
                scp_config(saturator, abstract_operator),
                &task,
                Default::default(),
            )
            .unwrap();
            assert_eq!(evaluate_initial(&task, &heuristic).unwrap(), 5.0);
        }
    }
}

#[test]
fn offline_scp_releases_cartesian_construction_data_after_first_evaluation() {
    let task = independent_goals_task();
    let component = AbstractionComponent::cartesian(None, cartesian_abstraction(&task));
    let config = scp_config(Saturator::All, true);
    let heuristic = SaturatedCostPartitioningHeuristic::new(
        None,
        vec![component],
        config,
        &task,
        Default::default(),
    )
    .unwrap();

    assert!(
        !heuristic.components.borrow()[0]
            .as_cartesian()
            .unwrap()
            .transition_system
            .transitions
            .is_empty()
    );
    assert_eq!(evaluate_initial(&task, &heuristic).unwrap(), 5.0);
    assert!(
        heuristic.components.borrow()[0]
            .as_cartesian()
            .unwrap()
            .transition_system
            .transitions
            .is_empty()
    );
    assert!(heuristic.state.borrow().built);
    assert_eq!(evaluate_initial(&task, &heuristic).unwrap(), 5.0);
}

#[test]
fn interval_scp_combines_all_representations() {
    let task = independent_goals_task();
    let domain_config = CegarConfig {
        max_abstraction_size: 16,
        combine_labels: false,
        compute_source_sets: true,
        ..Default::default()
    };
    let domain = DomainAbstractionGenerator::new(domain_config)
        .unwrap()
        .generate(&task)
        .unwrap();
    let pattern = Pattern::new(vec![VariableIndex::from_usize(1)], vec![]);
    let pdb = PatternDatabase::new(ProjectedTask::new(&task, &pattern).unwrap(), 32).unwrap();
    let components = vec![
        AbstractionComponent::domain(None, domain),
        AbstractionComponent::cartesian(None, cartesian_abstraction(&task)),
        AbstractionComponent::pattern_database(pdb),
    ];
    let heuristic = SaturatedCostPartitioningHeuristic::new(
        None,
        components,
        scp_config(Saturator::All, true),
        &task,
        Default::default(),
    )
    .unwrap();

    assert_eq!(evaluate_initial(&task, &heuristic).unwrap(), 5.0);
}
