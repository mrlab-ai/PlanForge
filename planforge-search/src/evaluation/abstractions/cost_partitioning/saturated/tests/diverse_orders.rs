use super::*;
use crate::evaluation::abstractions::collections::generator::AbstractionGenerator;
use crate::evaluation::abstractions::common::abstraction::CegarAbstraction;
use crate::evaluation::abstractions::common::cegar::options::CegarOptions;

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
    assert!(retained_cost_partitionings(&heuristic).len() > 1);
}

#[test]
fn diverse_orders_support_mixed_abstraction_representations() {
    let task = std::sync::Arc::new(independent_goals_task());
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
        .generate(&*task)
        .unwrap();
    domain
        .prepare_regional_saturation(&RegionSemantics::new(&*task).unwrap(), Deadline::NONE)
        .unwrap();
    let pattern = Pattern::new(vec![VariableIndex::from_usize(1)], vec![]);
    let pdb = PatternDatabase::new(ProjectedTask::new(&*task, &pattern).unwrap(), 32).unwrap();
    let components = vec![
        AbstractionComponent::from(domain),
        AbstractionComponent::from(cartesian_abstraction(&task)),
        AbstractionComponent::from(pdb),
    ];
    let mut config = scp_config(Saturator::All, true);
    config.diversify = true;
    config.samples = 16;
    config.max_orders = 8;
    let heuristic = SaturatedCostPartitioningHeuristic::new(
        None,
        components,
        config,
        task.clone(),
        Default::default(),
    )
    .unwrap();

    assert_eq!(evaluate_initial(&task, &heuristic).unwrap(), 5.0);
    let retained = retained_cost_partitionings(&heuristic);
    assert!(!retained.is_empty());
    assert!(retained.len() <= 8);
    drop(retained);
    assert!(heuristic.state.borrow().components.iter().all(|component| {
        component
            .cegar()
            .is_none_or(|abstraction| !abstraction.retains_transitions())
    }));
}

#[test]
fn diverse_order_generator_requires_diversify() {
    let task = independent_goals_task();
    let component = AbstractionComponent::from(cartesian_abstraction(&task));
    let mut config = scp_config(Saturator::All, true);
    config.order_generator = OrderGenerator::Diverse;

    let result = SaturatedCostPartitioningHeuristic::new(
        None,
        vec![component],
        config,
        std::sync::Arc::new(&task),
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
fn diverse_orders_keep_partitionings_that_improve_different_samples() {
    let partition = |distances| CostPartitioningHeuristic {
        lookup_tables: vec![LookupTable {
            abstraction_id: 0,
            distances,
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
        partition(vec![5.0, 1.0]),
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

/// The heuristic is the maximum over the retained orders only (Seipp et al.
/// 2020), which look up only the components their tables read.
#[test]
fn the_heuristic_is_the_maximum_over_the_retained_orders_only() {
    let samples = [vec![Some(0), Some(0), None], vec![Some(1), Some(1), None]];
    assert!(samples.iter().all(|ids| compute_max_h(&[], ids) == 0.0));
    assert_eq!(compute_lookup_mask(&[], 3), vec![false, false, false]);

    let retained = [CostPartitioningHeuristic {
        lookup_tables: vec![LookupTable {
            abstraction_id: 1,
            distances: vec![2.0, 6.0],
        }],
        goal_cover_goal_id: None,
    }];
    assert_eq!(
        samples
            .iter()
            .map(|ids| compute_max_h(&retained, ids))
            .collect::<Vec<_>>(),
        vec![2.0, 6.0]
    );
    assert_eq!(compute_lookup_mask(&retained, 3), vec![false, true, false]);
}

#[test]
fn a_table_is_zero_at_the_evaluated_state_only_where_it_estimates_nothing() {
    let distances = [0.0, 4.0];

    assert!(is_zero_at_evaluated_state(0, &distances, &[Some(0)]));
    assert!(!is_zero_at_evaluated_state(0, &distances, &[Some(1)]));
}
