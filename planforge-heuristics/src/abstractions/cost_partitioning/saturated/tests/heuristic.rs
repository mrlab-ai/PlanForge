use super::*;
use crate::abstractions::collections::generator::AbstractionGenerator;
use crate::abstractions::common::abstraction::CegarAbstraction;
use crate::abstractions::common::cegar::options::CegarOptions;

#[test]
fn cartesian_scp_supports_every_saturator_in_both_cost_modes() {
    let task = independent_goals_task();
    for saturator in [Saturator::All, Saturator::Perim, Saturator::Perimstar] {
        for abstract_operator in [false, true] {
            let component = AbstractionComponent::from(cartesian_abstraction(&task));
            let mut heuristic = SaturatedCostPartitioningHeuristic::new(
                None,
                vec![component],
                scp_config(saturator, abstract_operator),
                std::sync::Arc::new(task.clone()),
                Default::default(),
            )
            .unwrap();
            assert_eq!(evaluate_initial(&task, &mut heuristic).unwrap(), 5.0);
        }
    }
}

#[test]
fn offline_scp_releases_cartesian_construction_data_after_construction() {
    let task = independent_goals_task();
    let component = AbstractionComponent::from(cartesian_abstraction(&task));
    assert!(component.cegar().unwrap().retains_transitions());
    let config = scp_config(Saturator::All, true);
    let mut heuristic = SaturatedCostPartitioningHeuristic::new(
        None,
        vec![component],
        config,
        std::sync::Arc::new(task.clone()),
        Default::default(),
    )
    .unwrap();

    assert!(
        !heuristic.components[0]
            .cegar()
            .unwrap()
            .retains_transitions()
    );
    assert_eq!(evaluate_initial(&task, &mut heuristic).unwrap(), 5.0);
}

#[test]
fn region_scp_combines_all_representations() {
    let task = independent_goals_task();
    let domain_config = DomainAbstractionConfig {
        cegar: CegarOptions {
            max_states: 16,
            combine_labels: false,
            ..DomainAbstractionConfig::default().cegar
        },
        ..Default::default()
    };
    let mut domain = DomainAbstractionGenerator::new(domain_config)
        .unwrap()
        .generate(&task)
        .unwrap();
    domain
        .prepare_regional_saturation(&RegionSemantics::new(&task).unwrap(), Deadline::NONE)
        .unwrap();
    let pattern = Pattern::new(vec![VariableIndex::from_usize(1)], vec![]);
    let pdb = PatternDatabase::new(ProjectedTask::new(&task, &pattern).unwrap(), 32).unwrap();
    let components = vec![
        AbstractionComponent::from(domain),
        AbstractionComponent::from(cartesian_abstraction(&task)),
        AbstractionComponent::from(pdb),
    ];
    let mut heuristic = SaturatedCostPartitioningHeuristic::new(
        None,
        components,
        scp_config(Saturator::All, true),
        std::sync::Arc::new(task.clone()),
        Default::default(),
    )
    .unwrap();

    assert_eq!(evaluate_initial(&task, &mut heuristic).unwrap(), 5.0);
}
