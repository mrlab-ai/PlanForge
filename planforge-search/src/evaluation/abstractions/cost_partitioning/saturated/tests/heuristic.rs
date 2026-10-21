use super::*;
use crate::evaluation::abstractions::collections::generator::AbstractionGenerator;
use crate::evaluation::abstractions::common::abstraction::CegarAbstraction;
use crate::evaluation::abstractions::common::cegar::options::CegarOptions;

#[test]
fn scp_retains_an_order_that_is_weaker_only_at_the_initial_state() {
    let partition = |distances| CostPartitioningHeuristic {
        lookup_tables: vec![LookupTable {
            abstraction_id: 0,
            distances,
        }],
        goal_cover_goal_id: None,
    };
    let mut retained = Vec::new();
    SaturatedCostPartitioningHeuristic::retain_cp(&mut retained, partition(vec![5.0, 1.0]));
    SaturatedCostPartitioningHeuristic::retain_cp(&mut retained, partition(vec![4.0, 4.0]));
    SaturatedCostPartitioningHeuristic::retain_cp(&mut retained, partition(vec![4.0, 4.0]));

    assert_eq!(retained.len(), 2, "duplicates are discarded");
    assert_eq!(compute_max_h(&retained, &[Some(0)]), 5.0);
    assert_eq!(compute_max_h(&retained, &[Some(1)]), 4.0);
}

#[test]
fn cartesian_scp_supports_every_saturator_in_both_cost_modes() {
    let task = independent_goals_task();
    for saturator in [Saturator::All, Saturator::Perim, Saturator::Perimstar] {
        for abstract_operator in [false, true] {
            let component = AbstractionComponent::from(cartesian_abstraction(&task));
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
    let component = AbstractionComponent::from(cartesian_abstraction(&task));
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
        heuristic.state.borrow().components[0]
            .cegar()
            .unwrap()
            .retains_transitions()
    );
    assert_eq!(evaluate_initial(&task, &heuristic).unwrap(), 5.0);
    assert!(
        !heuristic.state.borrow().components[0]
            .cegar()
            .unwrap()
            .retains_transitions()
    );
    assert!(matches!(heuristic.state.borrow().phase, ScpPhase::Built(_)));
    assert_eq!(evaluate_initial(&task, &heuristic).unwrap(), 5.0);
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
