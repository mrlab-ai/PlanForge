use super::*;
use crate::evaluation::abstraction_collections::portfolio::CollectionStrategy;
use planforge_sas::numeric_task::{
    Effect, ExplicitFact, ExplicitVariable, Metric, NumericRootTask, NumericRootTaskParts, Operator,
};
use planforge_sas::state_registry::StateRegistry;

use crate::evaluation::cartesian_abstractions::{
    CartesianAbstractionCollectionConfig, CartesianAbstractionCollectionGenerator,
    CartesianAbstractionConfig, CartesianAbstractionGenerator,
};
use crate::evaluation::domain_abstractions::cegar::CegarConfig;
use crate::evaluation::domain_abstractions::domain_abstraction_generator::DomainAbstractionGenerator;
use crate::evaluation::pattern_databases::pattern_database::PatternDatabase;
use crate::evaluation::pattern_databases::projected_task::{Pattern, ProjectedTask};

fn binary_variable(name: &str) -> ExplicitVariable {
    ExplicitVariable::new(
        2,
        name.to_string(),
        vec![format!("{name}=0"), format!("{name}=1")],
        None,
        1,
    )
}

fn independent_goals_task() -> NumericRootTask {
    NumericRootTask::new(NumericRootTaskParts {
        version: 1,
        metric: Metric::new(true, None),
        variables: vec![binary_variable("p"), binary_variable("q")],
        numeric_variables: vec![],
        goals: vec![
            ExplicitFact::propositional(0, 1),
            ExplicitFact::propositional(1, 1),
        ],
        mutexes: vec![],
        state: vec![0, 0],
        numeric_state: vec![],
        operators: vec![
            Operator::new(
                "set-p".to_string(),
                vec![],
                vec![Effect::new(vec![], 0, Some(0), 1)],
                vec![],
                2,
            ),
            Operator::new(
                "set-q".to_string(),
                vec![],
                vec![Effect::new(vec![], 1, Some(0), 1)],
                vec![],
                3,
            ),
        ],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    })
}

fn cartesian_abstraction(task: &NumericRootTask) -> CartesianAbstraction {
    CartesianAbstractionGenerator::new(CartesianAbstractionConfig {
        max_states: 16,
        max_time: None,
        combine_labels: false,
        compute_operator_regions: true,
        random_seed: None,
        debug: false,
        ..Default::default()
    })
    .unwrap()
    .generate(task)
    .unwrap()
}

fn scp_config(saturator: Saturator, abstract_operator: bool) -> ScpOnlineConfig {
    ScpOnlineConfig {
        max_time: 10.0,
        table_construction_max_time: 10.0,
        interval: usize::MAX,
        order_optimization_max_time: 0.0,
        saturator,
        random_seed: Some(1),
        partitioning: if abstract_operator {
            CostPartitioningMethod::Region
        } else {
            CostPartitioningMethod::Label
        },
        ..ScpOnlineConfig::default()
    }
}

fn evaluate_initial(
    task: &NumericRootTask,
    heuristic: &dyn Heuristic,
) -> Result<f64, EvaluationError> {
    let mut registry = StateRegistry::for_task(std::sync::Arc::new(task));
    let initial_state = registry.get_initial_state();
    let eval_state = EvaluationState::new(&initial_state, task, &registry);
    heuristic.compute_heuristic(&eval_state)
}

#[test]
fn reduce_costs_rejects_significant_underflow() {
    let mut remaining = vec![1.0];
    let saturated = vec![1.5];

    let err = reduce_costs(&mut remaining, &saturated).unwrap_err();
    assert!(format!("{err}").contains("underflow"));
}

#[test]
fn reduce_costs_rejects_non_finite_saturated_costs() {
    let mut remaining = vec![1.0];

    let error = reduce_costs(&mut remaining, &[f64::NEG_INFINITY]).unwrap_err();

    assert!(error.to_string().contains("must be finite"));
}

#[test]
fn regional_conflict_scoring_accepts_transition_free_components() {
    let task = independent_goals_task();
    let mut cartesian = cartesian_abstraction(&task);
    cartesian.transition_system.transitions.clear();
    cartesian.abstract_operator_regions.clear();
    let domain_config = CegarConfig {
        max_abstraction_size: 16,
        compute_operator_regions: true,
        ..Default::default()
    };
    let mut domain = DomainAbstractionGenerator::new(domain_config)
        .unwrap()
        .generate(&task)
        .unwrap();
    domain.abstract_operators.clear();
    domain.abstract_operator_regions.clear();

    let components = vec![
        AbstractionComponent::domain(None, domain),
        AbstractionComponent::cartesian(None, cartesian),
    ];
    assert_eq!(
        compute_regional_conflict_scores(
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
fn regional_conflict_scoring_rejects_missing_operator_regions() {
    let task = independent_goals_task();
    let mut abstraction = cartesian_abstraction(&task);
    assert!(!abstraction.transition_system.transitions.is_empty());
    abstraction.abstract_operator_regions.clear();

    let components = vec![AbstractionComponent::cartesian(None, abstraction)];
    let error = compute_regional_conflict_scores(
        &components,
        &[vec![f64::NEG_INFINITY; task.get_num_operators()]],
        &[2.0, 3.0],
    )
    .unwrap_err();
    assert!(error.to_string().contains("has 0 operator regions for"));
}

#[test]
fn reduce_costs_clamps_tiny_negative_roundoff() {
    let mut remaining = vec![1.0];
    let saturated = vec![1.0 + 1e-12];

    reduce_costs(&mut remaining, &saturated).unwrap();
    assert_eq!(remaining, vec![0.0]);
}

#[test]
fn label_candidates_always_include_max_heuristic_greedy_order() {
    let base_order = vec![2, 0, 1];
    let h_values = vec![3.0, 7.0, 5.0];

    let max_order = max_heuristic_greedy_order(&base_order, &h_values);

    assert_eq!(max_order, vec![1, 2, 0]);
}

#[test]
fn compact_goal_cover_orders_pair_complementary_anchor_variants() {
    let task = independent_goals_task();
    let abstractions =
        CartesianAbstractionCollectionGenerator::new(CartesianAbstractionCollectionConfig {
            abstraction: CartesianAbstractionConfig {
                max_states: 16,
                max_time: None,
                combine_labels: false,
                compute_operator_regions: true,
                random_seed: Some(1),
                debug: false,
                ..Default::default()
            },
            collection_strategy: CollectionStrategy::Complementary,
            variants_per_goal: 4,
            max_collection_states: 128,
            total_max_time: None,
            progressive_goal_roots: false,
        })
        .unwrap()
        .generate(&task)
        .unwrap();
    let base_order = (0..abstractions.len()).collect::<Vec<_>>();
    let standalone_h = abstractions
        .iter()
        .map(|abstraction| {
            abstraction.distance_table.distances[abstraction.distance_table.initial_state_hash]
        })
        .collect::<Vec<_>>();
    let components = abstractions
        .iter()
        .cloned()
        .map(|abstraction| AbstractionComponent::cartesian(None, abstraction))
        .collect::<Vec<_>>();
    let order = |variant| {
        cartesian_goal_cover_order(&base_order, &components, &standalone_h, true, variant).unwrap()
    };
    let goal = |component_id: usize| {
        abstractions[component_id]
            .metadata
            .collection_goal_id
            .unwrap()
    };
    let baseline = order(GoalCoverOrderVariant {
        compact: true,
        ..Default::default()
    });
    assert_eq!(baseline.len(), 3);
    assert_eq!(goal(baseline[0]), goal(baseline[1]));
    assert_ne!(goal(baseline[0]), goal(baseline[2]));
    let first = &abstractions[baseline[0]].metadata;
    let complement = &abstractions[baseline[1]].metadata;
    assert_ne!(first.refinement_direction, complement.refinement_direction);
    assert_eq!(first.split_selection_rank, complement.split_selection_rank);

    let other_goal = order(GoalCoverOrderVariant {
        anchor_goal_offset: 1,
        compact: true,
        ..Default::default()
    });
    assert_ne!(goal(other_goal[0]), goal(baseline[0]));

    let other_anchor = order(GoalCoverOrderVariant {
        anchor_offset: 1,
        compact: true,
        ..Default::default()
    });
    assert_ne!(other_anchor[0], baseline[0]);

    let other_representative = order(GoalCoverOrderVariant {
        representative_round: 1,
        compact: true,
        ..Default::default()
    });
    assert_eq!(&other_representative[..2], &baseline[..2]);
    assert_ne!(other_representative[2], baseline[2]);

    let other_complement = order(GoalCoverOrderVariant {
        complementary_round: 1,
        compact: true,
        ..Default::default()
    });
    assert_eq!(other_complement[0], baseline[0]);
    assert_ne!(other_complement[1], baseline[1]);

    let mut mixed = abstractions.clone();
    mixed.push(cartesian_abstraction(&task));
    let mixed_order = (0..mixed.len()).collect::<Vec<_>>();
    let mixed_h = mixed
        .iter()
        .map(|abstraction| {
            abstraction.distance_table.distances[abstraction.distance_table.initial_state_hash]
        })
        .collect::<Vec<_>>();
    let structural_id = mixed.len() - 1;
    let mixed_components = mixed
        .iter()
        .cloned()
        .map(|abstraction| AbstractionComponent::cartesian(None, abstraction))
        .collect::<Vec<_>>();
    let prefixed = cartesian_goal_cover_order(
        &mixed_order,
        &mixed_components,
        &mixed_h,
        false,
        GoalCoverOrderVariant {
            non_goal_prefix: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(prefixed[0], structural_id);
}

#[test]
fn compact_goal_cover_schedule_visits_every_anchor_goal_and_four_variants() {
    let variants = compact_goal_cover_variants(18, 8, true);
    let anchor_goals = variants
        .iter()
        .map(|variant| variant.anchor_goal_offset)
        .collect::<HashSet<_>>();

    assert_eq!(variants.len(), 72);
    assert_eq!(anchor_goals, (0..18).collect());
}

#[test]
fn compact_goal_cover_schedule_supports_one_abstraction_per_goal() {
    let variants = compact_goal_cover_variants(18, 1, true);
    let anchor_goals = variants
        .iter()
        .map(|variant| variant.anchor_goal_offset)
        .collect::<HashSet<_>>();

    assert_eq!(variants.len(), 18);
    assert_eq!(anchor_goals, (0..18).collect());
}

#[test]
fn offline_diversification_retains_available_specialists_per_cartesian_goal() {
    let task = std::sync::Arc::new(independent_goals_task());
    let abstractions =
        CartesianAbstractionCollectionGenerator::new(CartesianAbstractionCollectionConfig {
            abstraction: CartesianAbstractionConfig {
                max_states: 16,
                max_time: None,
                combine_labels: false,
                compute_operator_regions: true,
                random_seed: Some(1),
                debug: false,
                ..Default::default()
            },
            collection_strategy: CollectionStrategy::Complementary,
            variants_per_goal: 4,
            max_collection_states: 128,
            total_max_time: None,
            progressive_goal_roots: true,
        })
        .unwrap()
        .generate(&*task)
        .unwrap();
    let expected_goals = abstractions
        .iter()
        .filter_map(|abstraction| abstraction.metadata.collection_goal_id)
        .collect::<HashSet<_>>();
    let components = abstractions
        .into_iter()
        .map(|abstraction| AbstractionComponent::cartesian(None, abstraction))
        .collect();
    let mut config = scp_config(Saturator::All, true);
    config.online = false;
    config.diversify = true;
    config.samples = 16;
    // Standalone bounds live in one envelope outside the SCP order budget.
    config.max_orders = 1 + 4 * expected_goals.len();
    config.initial_order_generation_max_time = 10.0;
    let heuristic = SaturatedCostPartitioningOnlineHeuristic::from_components_with_sampling_task(
        None,
        components,
        config,
        &*task,
        task.clone(),
    )
    .unwrap();

    evaluate_initial(&task, &heuristic).unwrap();
    let retained_goals = heuristic
        .state
        .borrow()
        .cp_heuristics
        .iter()
        .filter_map(|cp| cp.specialist_goal_id)
        .collect::<HashSet<_>>();
    assert_eq!(retained_goals, expected_goals);
    let mut retained_per_goal = HashMap::<usize, usize>::new();
    for goal_id in heuristic
        .state
        .borrow()
        .cp_heuristics
        .iter()
        .filter_map(|cp| cp.specialist_goal_id)
    {
        *retained_per_goal.entry(goal_id).or_default() += 1;
    }
    assert!(
        retained_per_goal
            .values()
            .all(|&count| (2..=4).contains(&count)),
        "retained specialists by goal: {retained_per_goal:?}"
    );
}

#[test]
fn offline_scp_retains_an_order_that_is_weaker_only_at_the_initial_state() {
    let partition = |distances| CostPartitioningHeuristic {
        lookup_tables: vec![LookupTable {
            abstraction_id: 0,
            distances,
            unknown_value: f64::INFINITY,
        }],
        specialist_goal_id: None,
    };
    let mut state = ScpOnlineState::new(Some(1));
    let mut initial_h = 0.0;

    SaturatedCostPartitioningOnlineHeuristic::retain_cp(
        &mut state,
        partition(vec![5.0, 1.0]),
        &[Some(0)],
        &mut initial_h,
        true,
        usize::MAX,
    );
    SaturatedCostPartitioningOnlineHeuristic::retain_cp(
        &mut state,
        partition(vec![4.0, 4.0]),
        &[Some(0)],
        &mut initial_h,
        true,
        usize::MAX,
    );

    assert_eq!(state.cp_heuristics.len(), 2);
    assert_eq!(initial_h, 5.0);
    assert_eq!(
        SaturatedCostPartitioningOnlineHeuristic::compute_max_h(&state, &[Some(1)]),
        4.0
    );
}

#[test]
fn online_scp_rejects_a_non_improving_state_specific_order() {
    let partition = |distances| CostPartitioningHeuristic {
        lookup_tables: vec![LookupTable {
            abstraction_id: 0,
            distances,
            unknown_value: f64::INFINITY,
        }],
        specialist_goal_id: None,
    };
    let mut state = ScpOnlineState::new(Some(1));
    let mut current_h = 0.0;

    SaturatedCostPartitioningOnlineHeuristic::retain_cp(
        &mut state,
        partition(vec![5.0]),
        &[Some(0)],
        &mut current_h,
        false,
        usize::MAX,
    );
    SaturatedCostPartitioningOnlineHeuristic::retain_cp(
        &mut state,
        partition(vec![4.0]),
        &[Some(0)],
        &mut current_h,
        false,
        usize::MAX,
    );

    assert_eq!(state.cp_heuristics.len(), 1);
    assert_eq!(current_h, 5.0);
}

#[test]
fn cartesian_scp_supports_every_saturator_in_both_cost_modes() {
    let task = independent_goals_task();
    for saturator in [Saturator::All, Saturator::Perim, Saturator::Perimstar] {
        for abstract_operator in [false, true] {
            let component = AbstractionComponent::cartesian(None, cartesian_abstraction(&task));
            let heuristic = SaturatedCostPartitioningOnlineHeuristic::from_components(
                None,
                vec![component],
                scp_config(saturator, abstract_operator),
                &task,
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
    let mut config = scp_config(Saturator::All, true);
    config.online = false;
    config.interval = 1;
    let heuristic = SaturatedCostPartitioningOnlineHeuristic::from_components(
        None,
        vec![component],
        config,
        &task,
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
    assert!(heuristic.state.borrow().improvement_ended);
    assert_eq!(evaluate_initial(&task, &heuristic).unwrap(), 5.0);
}

#[test]
fn abstract_operator_scp_combines_all_backend_types() {
    let task = independent_goals_task();
    let domain_config = CegarConfig {
        max_abstraction_size: 16,
        combine_labels: false,
        compute_operator_regions: true,
        ..Default::default()
    };
    let domain = DomainAbstractionGenerator::new(domain_config)
        .unwrap()
        .generate(&task)
        .unwrap();
    let pattern = Pattern::new(vec![1], vec![]);
    let pdb = PatternDatabase::new(ProjectedTask::new(&task, &pattern).unwrap(), 32).unwrap();
    let components = vec![
        AbstractionComponent::domain(None, domain),
        AbstractionComponent::cartesian(None, cartesian_abstraction(&task)),
        AbstractionComponent::pattern_database(pdb),
    ];
    let heuristic = SaturatedCostPartitioningOnlineHeuristic::from_components(
        None,
        components,
        scp_config(Saturator::All, true),
        &task,
    )
    .unwrap();

    assert_eq!(evaluate_initial(&task, &heuristic).unwrap(), 5.0);
}

#[test]
fn offline_diversification_supports_mixed_abstraction_backends() {
    let task = std::sync::Arc::new(independent_goals_task());
    let domain_config = CegarConfig {
        max_abstraction_size: 16,
        combine_labels: false,
        compute_operator_regions: true,
        ..Default::default()
    };
    let domain = DomainAbstractionGenerator::new(domain_config)
        .unwrap()
        .generate(&*task)
        .unwrap();
    let pattern = Pattern::new(vec![1], vec![]);
    let pdb = PatternDatabase::new(ProjectedTask::new(&*task, &pattern).unwrap(), 32).unwrap();
    let components = vec![
        AbstractionComponent::domain(None, domain),
        AbstractionComponent::cartesian(None, cartesian_abstraction(&task)),
        AbstractionComponent::pattern_database(pdb),
    ];
    let mut config = scp_config(Saturator::All, true);
    config.online = false;
    config.diversify = true;
    config.samples = 16;
    config.max_orders = 8;
    let heuristic = SaturatedCostPartitioningOnlineHeuristic::from_components_with_sampling_task(
        None,
        components,
        config,
        &*task,
        task.clone(),
    )
    .unwrap();

    assert_eq!(evaluate_initial(&task, &heuristic).unwrap(), 5.0);
    let state = heuristic.state.borrow();
    assert!(state.improvement_ended);
    assert!(state.offline_sample_ids.is_empty());
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
fn offline_diversification_rejects_online_construction() {
    let task = std::sync::Arc::new(independent_goals_task());
    let component = AbstractionComponent::cartesian(None, cartesian_abstraction(&task));
    let mut config = scp_config(Saturator::All, true);
    config.online = true;
    config.diversify = true;

    let result = SaturatedCostPartitioningOnlineHeuristic::from_components_with_sampling_task(
        None,
        vec![component],
        config,
        &*task,
        task.clone(),
    );
    let error = match result {
        Ok(_) => panic!("online diversified construction must be rejected"),
        Err(error) => error,
    };

    assert!(error.to_string().contains("requires online=false"));
}

#[test]
fn diverse_orders_require_offline_diversification() {
    let task = independent_goals_task();
    let component = AbstractionComponent::cartesian(None, cartesian_abstraction(&task));
    let mut config = scp_config(Saturator::All, true);
    config.order_generator = OrderGenerator::Diverse;

    let result = SaturatedCostPartitioningOnlineHeuristic::from_components(
        None,
        vec![component],
        config,
        &task,
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
fn offline_diversification_rejects_nan_initial_order_budget() {
    let task = std::sync::Arc::new(independent_goals_task());
    let component = AbstractionComponent::cartesian(None, cartesian_abstraction(&task));
    let mut config = scp_config(Saturator::All, true);
    config.online = false;
    config.diversify = true;
    config.initial_order_generation_max_time = f64::NAN;

    let result = SaturatedCostPartitioningOnlineHeuristic::from_components_with_sampling_task(
        None,
        vec![component],
        config,
        &*task,
        task.clone(),
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
fn offline_diversifier_keeps_partitions_that_improve_different_samples() {
    let partition = |distances| CostPartitioningHeuristic {
        lookup_tables: vec![LookupTable {
            abstraction_id: 0,
            distances,
            unknown_value: f64::INFINITY,
        }],
        specialist_goal_id: None,
    };
    let samples = vec![vec![Some(0)], vec![Some(1)]];
    let mut best = vec![f64::NEG_INFINITY; samples.len()];
    let mut portfolio = Vec::new();
    let mut size_kb = 0;

    assert!(retain_if_sample_improving(
        partition(vec![5.0, 1.0]),
        &samples,
        &mut best,
        &mut portfolio,
        &mut size_kb,
        usize::MAX,
    ));
    assert!(retain_if_sample_improving(
        partition(vec![4.0, 6.0]),
        &samples,
        &mut best,
        &mut portfolio,
        &mut size_kb,
        usize::MAX,
    ));
    assert!(!retain_if_sample_improving(
        partition(vec![5.0, 5.0]),
        &samples,
        &mut best,
        &mut portfolio,
        &mut size_kb,
        usize::MAX,
    ));

    assert_eq!(portfolio.len(), 2);
    assert_eq!(best, vec![5.0, 6.0]);
}

#[test]
fn retained_standalone_envelope_preserves_every_component_without_cp_orders() {
    let mut state = ScpOnlineState::new(Some(1));
    state.h_values_by_abstraction = vec![vec![5.0, 1.0], vec![2.0, 6.0], vec![0.0, 0.0]];
    let samples = [vec![Some(0), Some(0), None], vec![Some(1), Some(1), None]];

    assert_eq!(
        samples
            .iter()
            .map(|ids| SaturatedCostPartitioningOnlineHeuristic::compute_max_h(&state, ids))
            .collect::<Vec<_>>(),
        vec![5.0, 6.0]
    );

    SaturatedCostPartitioningOnlineHeuristic::retain_standalone_envelope(&mut state, 3);

    assert!(state.h_values_by_abstraction.is_empty());
    assert_eq!(state.standalone_lookup_tables.len(), 2);
    assert!(state.cp_heuristics.is_empty());
    assert_eq!(state.required_mask, vec![true, true, false]);
    assert_eq!(
        samples
            .iter()
            .map(|ids| SaturatedCostPartitioningOnlineHeuristic::compute_max_h(&state, ids))
            .collect::<Vec<_>>(),
        vec![5.0, 6.0]
    );
}

#[test]
fn standalone_envelope_size_excludes_tables_that_are_not_retained() {
    let useful = vec![1.0; 128];
    let zero = vec![0.0; 128];

    assert_eq!(standalone_lookup_values_size_kb(&[useful, zero]), 1);
}

#[test]
fn diversified_scp_keeps_tables_that_are_zero_only_at_the_initial_state() {
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

#[test]
fn offline_random_walk_lengths_are_deterministic() {
    let mut left = SmallRng::seed_from_u64(7);
    let mut right = SmallRng::seed_from_u64(7);
    let left_lengths = (0..20)
        .map(|_| random_walk_length(110.0, 1.0, &mut left).unwrap())
        .collect::<Vec<_>>();
    let right_lengths = (0..20)
        .map(|_| random_walk_length(110.0, 1.0, &mut right).unwrap())
        .collect::<Vec<_>>();

    assert_eq!(left_lengths, right_lengths);
    assert!(left_lengths.iter().all(|&length| length <= 440));
}

mod handcrafted_sailing_tests {

    use std::collections::HashSet;
    use std::path::{Path, PathBuf};
    use std::time::Duration;

    use planforge_sas::numeric_task::{
        AbstractNumericTask, ExplicitFact, NumericRootTask, NumericType,
    };
    use planforge_translate::translate_to_sas_to_path_fast;

    use super::*;
    use crate::evaluation::abstraction_task::SingleGoalTask;
    use crate::evaluation::cartesian_abstractions::{
        CartesianAbstraction, CartesianAbstractionCollectionConfig,
        CartesianAbstractionCollectionGenerator, CartesianAbstractionConfig,
        CartesianRefinementDirection,
    };
    use crate::evaluation::domain_abstractions::cegar::InitialSeedSplit;
    use crate::evaluation::domain_abstractions::domain_abstraction::NumericPartitions;
    use crate::evaluation::domain_abstractions::domain_abstraction_factory::DomainAbstractionFactory;
    use crate::evaluation::domain_abstractions::domain_abstraction_generator::{
        DomainAbstractionMetadata, compute_hash_multipliers,
    };
    use crate::task_restriction::build_restricted_task;
    use planforge_sas::utils::interval::Interval;

    #[test]
    fn regional_order_conflict_preserves_disjoint_infinite_tails() {
        let region = |lower, upper, lower_closed, upper_closed| {
            StateRegion::with_all_props_constrained(
                Vec::new(),
                vec![Interval::new(lower, upper, lower_closed, upper_closed)],
            )
        };
        let left_tail = region(f64::NEG_INFINITY, 0.0, false, true);
        let right_tail = region(0.0, f64::INFINITY, false, false);
        let overlapping_tail = region(-1.0, f64::INFINITY, true, false);

        assert_eq!(
            pair_regional_conflict(&[&left_tail], 1.0, &[&right_tail], 1.0, 1.0),
            0.0
        );
        assert_eq!(
            pair_regional_conflict(&[&left_tail], 1.0, &[&overlapping_tail], 1.0, 1.0),
            1.0
        );
    }

    #[test]
    #[ignore = "oracle collection report over translated sailing benchmark"]
    fn sailing_perfect_complementary_cartesian_collection_reaches_optimum_with_regional_scp() {
        let task = translated_sailing_2_2_task();
        let restricted_task = build_restricted_task(&task)
            .expect("sailing task should support restricted task")
            .expect("sailing task should have promotable roots")
            .into_task();
        let abstractions =
            CartesianAbstractionCollectionGenerator::new(CartesianAbstractionCollectionConfig {
                abstraction: CartesianAbstractionConfig {
                    max_states: 1_000,
                    max_time: Some(Duration::from_secs(5)),
                    combine_labels: false,
                    compute_operator_regions: true,
                    random_seed: Some(1),
                    debug: false,
                    ..Default::default()
                },
                collection_strategy: CollectionStrategy::Complementary,
                variants_per_goal: 4,
                max_collection_states: 10_000,
                total_max_time: Some(Duration::from_secs(15)),
                progressive_goal_roots: false,
            })
            .expect("valid oracle Cartesian collection config")
            .generate(&restricted_task)
            .expect("failed to build oracle Cartesian collection");

        assert_eq!(abstractions.len(), 8);
        assert_eq!(
            abstractions
                .iter()
                .map(CartesianAbstraction::num_states)
                .sum::<usize>(),
            8_000
        );
        for goal_id in 0..2 {
            let mut modes = abstractions
                .iter()
                .filter(|abstraction| abstraction.metadata.collection_goal_id == Some(goal_id))
                .map(|abstraction| {
                    (
                        abstraction.metadata.refinement_direction,
                        abstraction.metadata.split_selection_rank,
                    )
                })
                .collect::<Vec<_>>();
            modes.sort_by_key(|(direction, rank)| {
                (
                    match direction {
                        CartesianRefinementDirection::Progression => 0,
                        CartesianRefinementDirection::Regression => 1,
                    },
                    *rank,
                )
            });
            assert_eq!(
                modes,
                vec![
                    (CartesianRefinementDirection::Progression, Some(0)),
                    (CartesianRefinementDirection::Progression, Some(1)),
                    (CartesianRefinementDirection::Regression, Some(0)),
                    (CartesianRefinementDirection::Regression, Some(1)),
                ]
            );
        }

        let standalone_max = abstractions
            .iter()
            .map(|abstraction| {
                abstraction.distance_table.distances[abstraction.distance_table.initial_state_hash]
            })
            .fold(0.0f64, f64::max);
        let standalone_h = abstractions
            .iter()
            .map(|abstraction| {
                abstraction.distance_table.distances[abstraction.distance_table.initial_state_hash]
            })
            .collect::<Vec<_>>();
        let components = abstractions
            .iter()
            .cloned()
            .map(|abstraction| AbstractionComponent::cartesian(None, abstraction))
            .collect::<Vec<_>>();
        let goal_cover_order = cartesian_goal_cover_order(
            &(0..abstractions.len()).collect::<Vec<_>>(),
            &components,
            &standalone_h,
            true,
            GoalCoverOrderVariant::default(),
        )
        .expect("the complementary sailing collection must have a goal-cover order");
        assert_eq!(&goal_cover_order[..3], &[5, 0, 1]);
        let regional_h = initial_cartesian_scp_value(&restricted_task, abstractions, true, 10.0);

        println!(
            "SAILING_PERFECT_COLLECTION standalone_max={standalone_max} regional_scp={regional_h}"
        );
        assert_eq!(standalone_max, 40.0);
        assert_eq!(regional_h, 76.0);
    }

    fn initial_cartesian_scp_value(
        task: &dyn AbstractNumericTask,
        abstractions: Vec<CartesianAbstraction>,
        partitioning: bool,
        order_optimization_max_time: f64,
    ) -> f64 {
        let initial_prop = task.get_initial_propositional_state_values();
        let initial_numeric = task.get_initial_numeric_state_values();
        let abstract_state_ids = abstractions
            .iter()
            .map(|abstraction| {
                Some(
                    abstraction
                        .hierarchy
                        .map_state(initial_prop, initial_numeric)
                        .expect("failed to map sailing initial state"),
                )
            })
            .collect::<Vec<_>>();
        let config = ScpOnlineConfig {
            online: true,
            table_construction_max_time: 30.0,
            order_optimization_max_time,
            max_size: 10_000_000,
            combine_labels: false,
            saturator: Saturator::All,
            residual_sweeps: 0,
            random_seed: Some(1),
            partitioning: if partitioning {
                CostPartitioningMethod::Region
            } else {
                CostPartitioningMethod::Label
            },
            ..Default::default()
        };
        let heuristic = SaturatedCostPartitioningOnlineHeuristic::new_with_cartesian(
            None,
            vec![],
            abstractions,
            vec![],
            config,
            task,
        )
        .expect("failed to construct oracle SCP heuristic");
        let mut state = heuristic.state.borrow_mut();
        let mut partitions = heuristic
            .maybe_build_cp(task, &mut state, &abstract_state_ids)
            .expect("oracle SCP construction failed");
        assert_eq!(partitions.len(), 1);
        partitions
            .pop()
            .expect("one oracle SCP partition")
            .compute_heuristic(&abstract_state_ids)
    }

    #[test]
    #[ignore = "diagnostic full-task handcrafted sailing regional SCP report"]
    fn sailing_handcrafted_four_abstractions_full_task_regional_scp_initial_h_report() {
        let task = translated_sailing_2_2_task();
        let restricted_task = build_restricted_task(&task)
            .expect("sailing task should support restricted task")
            .expect("sailing task should have promotable roots")
            .into_task();
        let transformed_task = &restricted_task;
        let specs = handcrafted_full_task_specs(transformed_task);
        assert_eq!(specs.len(), 4);

        let mut abstractions = Vec::new();
        for (index, spec) in specs.iter().enumerate() {
            let single_goal_task = SingleGoalTask::new(transformed_task, spec.goal);
            let mut abstraction = build_handcrafted_abstraction(&single_goal_task, spec)
                .unwrap_or_else(|error| panic!("failed to build {}: {error:#}", spec.name));
            abstraction.metadata = DomainAbstractionMetadata {
                collection_iteration: Some(index + 1),
                collection_strategy: Some("handcrafted_full_task_sailing".to_string()),
                flaw_kind: None,
                full_goal_task: Some(false),
                initial_seed_splits: spec.seed_splits.iter().map(seed_description).collect(),
                max_abstraction_size: Some(10_000),
                ..DomainAbstractionMetadata::default()
            };
            let states = abstraction_state_count(&abstraction);
            assert!(
                states <= 10_000,
                "{} has {states} states, expected at most 10000",
                spec.name
            );
            abstractions.push(abstraction);
        }

        let config = ScpOnlineConfig {
            online: true,
            max_time: 300.0,
            table_construction_max_time: 30.0,
            max_size: 10_000_000,
            diversify: false,
            samples: 1_000,
            max_orders: usize::MAX,
            interval: usize::MAX,
            combine_labels: false,
            scoring_function: ScoringFunction::MaxHeuristic,
            order_generator: OrderGenerator::Greedy,
            initial_order_generation_max_time: 0.0,
            order_optimization_max_time: 0.0,
            saturator: Saturator::All,
            random_seed: Some(1),
            partitioning: CostPartitioningMethod::Region,
            residual_sweeps: 1,
        };

        let heuristic = SaturatedCostPartitioningOnlineHeuristic::new(
            None,
            abstractions,
            vec![],
            config,
            transformed_task,
        )
        .expect("failed to construct SCP heuristic");
        let abstract_state_ids = initial_abstract_state_ids(&heuristic, transformed_task);
        {
            let mut state = heuristic.state.borrow_mut();
            let mut max_h = SaturatedCostPartitioningOnlineHeuristic::compute_max_h(
                &state,
                &abstract_state_ids,
            );
            let mut partitions = heuristic
                .maybe_build_cp(transformed_task, &mut state, &abstract_state_ids)
                .expect("initial SCP construction failed");
            assert_eq!(partitions.len(), 1);
            let cp = partitions.pop().unwrap();
            SaturatedCostPartitioningOnlineHeuristic::retain_cp(
                &mut state,
                cp,
                &abstract_state_ids,
                &mut max_h,
                true,
                usize::MAX,
            );
        }

        let state = heuristic.state.borrow();
        let initial_h =
            SaturatedCostPartitioningOnlineHeuristic::compute_max_h(&state, &abstract_state_ids);
        let components = heuristic.components.borrow();

        println!("HANDCRAFTED_FULL_TASK_INITIAL_H {initial_h}");
        let mut contributions = vec![0.0; specs.len()];
        for cp in &state.cp_heuristics {
            for table in &cp.lookup_tables {
                let contribution = abstract_state_ids
                    .get(table.abstraction_id)
                    .copied()
                    .flatten()
                    .and_then(|state_id| table.distances.get(state_id).copied())
                    .unwrap_or(table.unknown_value);
                if let Some(total) = contributions.get_mut(table.abstraction_id) {
                    *total += contribution;
                }
            }
        }

        for (index, (spec, component)) in specs.iter().zip(components.iter()).enumerate() {
            let abstraction = component
                .as_domain()
                .expect("handcrafted diagnostic uses domain components");
            let standalone_h = current_h_for_distances(
                index,
                &abstraction.distance_table.distances,
                &abstract_state_ids,
            );
            println!(
                "HANDCRAFTED_FULL_TASK_ABS index={index} name={} standalone_h={standalone_h} scp_contribution={} states={} abstract_ops={} views={}",
                spec.name,
                contributions[index],
                abstraction_state_count(abstraction),
                abstraction.abstract_operators.len(),
                partition_report(transformed_task, abstraction, &spec.view_ids),
            );
        }
        println!("HANDCRAFTED_FULL_TASK_CONTRIBUTIONS {contributions:?}");

        assert!(
            initial_h.is_finite(),
            "initial h must be finite, got {initial_h}"
        );
        assert!(initial_h > 0.0, "initial h should be positive");
        assert!(
            initial_h <= 76.0,
            "full-task diagnostic should not exceed the known optimal cost 76, got {initial_h}"
        );
    }

    #[derive(Debug, Clone)]
    struct HandcraftedSpec {
        name: String,
        goal: ExplicitFact,
        view_ids: Vec<usize>,
        seed_splits: Vec<InitialSeedSplit>,
    }

    fn handcrafted_full_task_specs(task: &dyn AbstractNumericTask) -> Vec<HandcraftedSpec> {
        [
            ("p1-u", "p1", ViewKind::Sum),
            ("p1-v", "p1", ViewKind::Difference),
            ("p0-u", "p0", ViewKind::Sum),
            ("p0-v", "p0", ViewKind::Difference),
        ]
        .into_iter()
        .map(|(name, person, view_kind)| {
            let view_ids = ["b0", "b1"]
                .into_iter()
                .map(|boat| find_sailing_view(task, boat, person, view_kind))
                .collect::<Vec<_>>();
            let seed_splits = route_seed_splits(task, &view_ids);
            HandcraftedSpec {
                name: name.to_string(),
                goal: find_saved_goal_fact(task, person),
                view_ids,
                seed_splits,
            }
        })
        .collect()
    }

    fn build_handcrafted_abstraction(
        transformed_task: &dyn AbstractNumericTask,
        spec: &HandcraftedSpec,
    ) -> anyhow::Result<DomainAbstraction> {
        let mut partitions = NumericPartitions::trivial(transformed_task);
        for seed in &spec.seed_splits {
            let InitialSeedSplit::Numeric {
                numeric_var_id,
                value,
                include_in_lower,
            } = seed
            else {
                continue;
            };
            partitions.split_at(*numeric_var_id, *value, *include_in_lower);
        }

        let goal_vars = (0..transformed_task.get_num_goals())
            .map(|goal_id| transformed_task.get_goal_fact(goal_id).var())
            .collect::<HashSet<_>>();
        let domain_mapping = (0..transformed_task.get_num_variables())
            .map(|var_id| {
                let domain_size = transformed_task
                    .get_variable_domain_size(var_id)
                    .expect("valid transformed prop var id");
                if goal_vars.contains(&var_id) {
                    (0..domain_size).collect::<Vec<_>>()
                } else {
                    vec![0; domain_size]
                }
            })
            .collect::<Vec<_>>();
        let domain_sizes = domain_mapping
            .iter()
            .map(|mapping| mapping.iter().copied().max().map_or(0, |value| value + 1))
            .collect::<Vec<_>>();
        let numeric_domain_sizes = (0..transformed_task.numeric_variables().len())
            .map(|numeric_var_id| {
                partitions
                    .partitions(numeric_var_id)
                    .expect("trivial partitions contain every numeric variable")
                    .len()
            })
            .collect::<Vec<_>>();
        let factory = DomainAbstractionFactory::new(
            transformed_task,
            domain_mapping,
            domain_sizes,
            partitions,
            numeric_domain_sizes,
        )?;
        let mut operator_generator = factory.make_operator_generator(transformed_task, false)?;
        let abstract_operators = operator_generator.build_abstract_operators(transformed_task)?;
        let abstract_operator_regions =
            factory.build_abstract_operator_regions(transformed_task, &abstract_operators)?;
        let distance_table = factory.build_distance_table_with_operators(
            transformed_task,
            &operator_generator,
            &abstract_operators,
            false,
        )?;
        let relevant_operator_ids = factory.relevant_operator_ids_from_operators(
            transformed_task,
            false,
            &abstract_operators,
            DistanceTableOptions::default(),
        )?;
        let hash_multipliers =
            compute_hash_multipliers(factory.domain_sizes(), factory.numeric_domain_sizes())?;
        Ok(DomainAbstraction {
            factory,
            distance_table,
            hash_multipliers,
            combine_labels: false,
            relevant_operator_ids,
            abstract_operators,
            abstract_operator_regions,
            regional_transition_system: RefCell::new(None),
            metadata: DomainAbstractionMetadata::default(),
        })
    }

    fn initial_abstract_state_ids(
        heuristic: &SaturatedCostPartitioningOnlineHeuristic<'_>,
        task: &dyn AbstractNumericTask,
    ) -> Vec<Option<usize>> {
        let prop = task.get_initial_propositional_state_values();
        let numeric = task.get_initial_numeric_state_values();
        heuristic
            .components
            .borrow()
            .iter()
            .map(|component| match component {
                AbstractionComponent::Domain(domain) => Some(
                    domain
                        .abstract_state_hash_from_state_values(prop, numeric)
                        .expect("failed to hash initial state for handcrafted abstraction"),
                ),
                AbstractionComponent::Cartesian(cartesian) => Some(
                    cartesian
                        .abstraction()
                        .abstract_state_id(prop, numeric)
                        .expect("failed to map initial Cartesian state"),
                ),
                AbstractionComponent::PatternDatabase(pdb) => pdb
                    .abstract_state_id_from_source_state_values(prop, numeric)
                    .expect("failed to map initial PDB state"),
            })
            .collect()
    }

    #[derive(Debug, Clone, Copy)]
    enum ViewKind {
        Sum,
        Difference,
    }

    fn find_sailing_view(
        task: &dyn AbstractNumericTask,
        boat: &str,
        person: &str,
        view_kind: ViewKind,
    ) -> usize {
        let tuple = format!("({boat}, {boat}, {person})");
        let candidates = task
            .numeric_variables()
            .iter()
            .enumerate()
            .filter(|(_, variable)| variable.get_type() == &NumericType::Regular)
            .filter(|(_, variable)| {
                let name = variable.name();
                name.contains(&tuple)
                    && !name.contains("25.0")
                    && match view_kind {
                        ViewKind::Sum => name.contains("derived!sum_PNE x"),
                        ViewKind::Difference => name.contains("derived!difference_PNE y"),
                    }
            })
            .map(|(id, _)| id)
            .collect::<Vec<_>>();
        assert_eq!(
            candidates.len(),
            1,
            "expected one {view_kind:?} view for {boat}/{person}, got {candidates:?}"
        );
        candidates[0]
    }

    fn find_saved_goal_fact(task: &dyn AbstractNumericTask, person: &str) -> ExplicitFact {
        let suffix = format!(" {person}");
        let mut candidates = task
            .get_operators()
            .iter()
            .filter(|operator| operator.name().starts_with("save_person "))
            .filter(|operator| operator.name().ends_with(&suffix))
            .flat_map(|operator| operator.effects().iter())
            .filter(|effect| effect.conditions().is_empty())
            .map(|effect| ExplicitFact::propositional(effect.var_id(), effect.value()))
            .collect::<Vec<_>>();
        candidates.sort();
        candidates.dedup();
        assert_eq!(
            candidates.len(),
            1,
            "expected one saved fact from save_person operators for {person}, got {candidates:?}"
        );
        candidates[0]
    }

    fn route_seed_splits(
        task: &dyn AbstractNumericTask,
        view_ids: &[usize],
    ) -> Vec<InitialSeedSplit> {
        let initial = task.get_initial_numeric_state_values();
        let mut seeds = Vec::new();
        for &view_id in view_ids {
            add_split(&mut seeds, view_id, initial[view_id], false);
            add_split(&mut seeds, view_id, 0.0, true);
            add_split(&mut seeds, view_id, 25.0, true);
            add_route_grid_values(&mut seeds, view_id, initial[view_id], 25.0, 3.0);
        }
        seeds.sort_by_key(seed_description);
        seeds.dedup();
        seeds
    }

    fn add_route_grid_values(
        seeds: &mut Vec<InitialSeedSplit>,
        numeric_var_id: usize,
        start: f64,
        end: f64,
        step: f64,
    ) {
        assert!(start.is_finite() && end.is_finite() && step.is_finite() && step > 0.0);
        let direction = if start <= end { 1.0 } else { -1.0 };
        let mut value = start;
        while (end - value) * direction > step {
            value += direction * step;
            add_split(seeds, numeric_var_id, value, true);
        }
    }

    fn add_split(
        seeds: &mut Vec<InitialSeedSplit>,
        numeric_var_id: usize,
        value: f64,
        include_in_lower: bool,
    ) {
        seeds.push(InitialSeedSplit::Numeric {
            numeric_var_id,
            value,
            include_in_lower,
        });
    }

    fn partition_report(
        task: &dyn AbstractNumericTask,
        abstraction: &DomainAbstraction,
        view_ids: &[usize],
    ) -> String {
        view_ids
            .iter()
            .map(|&view_id| {
                let name = task.numeric_variables()[view_id].name();
                let num_parts = abstraction
                    .factory
                    .partitions()
                    .partitions(view_id)
                    .expect("missing partition for handcrafted view")
                    .len();
                format!("n{view_id}:{name} parts={num_parts}")
            })
            .collect::<Vec<_>>()
            .join(" || ")
    }

    fn seed_description(seed: &InitialSeedSplit) -> String {
        match seed {
            InitialSeedSplit::Propositional { var_id, value } => format!("p{var_id}={value}"),
            InitialSeedSplit::Numeric {
                numeric_var_id,
                value,
                include_in_lower,
            } => format!(
                "n{numeric_var_id}{}{}",
                if *include_in_lower { "<=" } else { "<" },
                value
            ),
        }
    }

    fn translated_sailing_2_2_task() -> NumericRootTask {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let domain = root.join("others/sailing/domain.pddl");
        let problem = root.join("others/sailing/prob_2_2_1229.pddl");
        assert!(domain.is_file(), "missing {}", domain.display());
        assert!(problem.is_file(), "missing {}", problem.display());
        let temp_dir = unique_temp_dir("sailing_handcrafted_full_task_scp")
            .expect("failed to create sailing diagnostic temp dir");
        let preprocessed = temp_dir.join("output");
        translate_to_sas_to_path_fast(
            domain.to_str().expect("non-utf8 sailing domain path"),
            problem.to_str().expect("non-utf8 sailing problem path"),
            &preprocessed,
        )
        .expect("sailing translation failed");
        NumericRootTask::from_file(&preprocessed)
    }

    fn unique_temp_dir(prefix: &str) -> std::io::Result<PathBuf> {
        let base = std::env::temp_dir().join("numeric_planneRS");
        std::fs::create_dir_all(&base)?;
        let dir = base.join(format!(
            "{prefix}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        std::fs::create_dir(&dir)?;
        Ok(dir)
    }
}
