use super::*;

#[test]
fn interval_stolen_costs_accept_transition_free_components() {
    let task = independent_goals_task();
    let mut cartesian = cartesian_abstraction(&task);
    cartesian.transition_system.transitions.clear();
    cartesian.abstract_source_sets.clear();
    let domain_config = CegarConfig {
        max_states: 16,
        compute_source_sets: true,
        ..Default::default()
    };
    let mut domain = DomainAbstractionGenerator::new(domain_config)
        .unwrap()
        .generate(&task)
        .unwrap();
    domain.abstract_operators.clear();
    domain.abstract_source_sets.clear();

    let components = vec![
        AbstractionComponent::domain(None, domain),
        AbstractionComponent::cartesian(None, cartesian),
    ];
    assert_eq!(
        interval_stolen_costs(
            &components,
            &[
                vec![f64::NEG_INFINITY; task.get_num_operators()],
                vec![f64::NEG_INFINITY; task.get_num_operators()],
            ],
            &[2.0, 3.0],
        )
        .unwrap(),
        vec![Some(0.0), Some(0.0)]
    );
}

#[test]
fn interval_stolen_costs_reject_missing_source_sets() {
    let task = independent_goals_task();
    let mut abstraction = cartesian_abstraction(&task);
    assert!(!abstraction.transition_system.transitions.is_empty());
    abstraction.abstract_source_sets.clear();

    let components = vec![AbstractionComponent::cartesian(None, abstraction)];
    let error = interval_stolen_costs(
        &components,
        &[vec![f64::NEG_INFINITY; task.get_num_operators()]],
        &[2.0, 3.0],
    )
    .unwrap_err();
    assert!(error.to_string().contains("has 0 operator regions for"));
}
