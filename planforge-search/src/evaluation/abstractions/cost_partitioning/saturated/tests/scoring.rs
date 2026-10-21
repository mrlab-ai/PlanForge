use super::*;
use crate::evaluation::abstractions::collections::generator::AbstractionGenerator;
use crate::evaluation::abstractions::common::cegar::options::CegarOptions;
use crate::evaluation::abstractions::common::regions::RegionSemantics;

#[test]
fn interval_stolen_costs_accept_transition_free_components() {
    let task = independent_goals_task();
    let regions = RegionSemantics::new(&task).unwrap();
    let mut cartesian = CartesianAbstractionGenerator::new(CartesianAbstractionConfig {
        cegar: CegarOptions {
            max_states: 1,
            ..CartesianAbstractionConfig::default().cegar
        },
        ..Default::default()
    })
    .unwrap()
    .generate(&task)
    .unwrap();
    cartesian
        .prepare_regional_saturation(&regions, Deadline::NONE)
        .unwrap();
    assert_eq!(cartesian.num_abstract_operators(), 0);
    let mut domain = DomainAbstractionGenerator::new(DomainAbstractionConfig {
        cegar: CegarOptions {
            max_states: 1,
            ..DomainAbstractionConfig::default().cegar
        },
        ..Default::default()
    })
    .unwrap()
    .generate(&task)
    .unwrap();
    domain
        .prepare_regional_saturation(&regions, Deadline::NONE)
        .unwrap();

    let components = vec![
        AbstractionComponent::from(domain),
        AbstractionComponent::from(cartesian),
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
    abstraction.discard_transition_data();
    assert!(abstraction.abstract_source_sets().is_none());

    let components = vec![AbstractionComponent::from(abstraction)];
    let error = interval_stolen_costs(
        &components,
        &[vec![f64::NEG_INFINITY; task.get_num_operators()]],
        &[2.0, 3.0],
    )
    .unwrap_err();
    assert!(error.to_string().contains("has no source sets"));
}
