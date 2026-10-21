use super::*;
use crate::evaluation::abstractions::common::cegar::options::CegarOptions;

#[test]
fn label_candidates_always_include_max_heuristic_greedy_order() {
    let base_order = vec![2, 0, 1];
    let h_values = vec![3.0, 7.0, 5.0];

    let max_order = max_heuristic_greedy_order(&base_order, &h_values);

    assert_eq!(max_order, vec![1, 2, 0]);
}

#[test]
fn compact_goal_cover_orders_pair_second_anchor_variants() {
    let task = independent_goals_task();
    let abstractions =
        CartesianAbstractionCollectionGenerator::new(CartesianAbstractionCollectionConfig {
            member: CartesianAbstractionConfig {
                cegar: CegarOptions {
                    max_states: 16,
                    max_time: None,
                    combine_labels: false,
                    random_seed: Some(1),
                    debug: false,
                    ..CartesianAbstractionConfig::default().cegar
                },
                compute_source_sets: true,
                ..Default::default()
            },
            collection_strategy: CollectionStrategy::GoalFocused,
            variants_per_goal: 4,
            max_collection_size: 128,
            total_max_time: None,
            progress_states: false,
            alternate_split_kinds: Some(true),
            split_kind: None,
            random_seed: Some(1),
            single_goal_tasks: true,
            ..Default::default()
        })
        .unwrap()
        .generate(&task)
        .unwrap();
    let base_order = (0..abstractions.len()).collect::<Vec<_>>();
    let abstraction_h = abstractions
        .iter()
        .map(|abstraction| {
            abstraction.distance_table.distances[abstraction.distance_table.initial_state]
        })
        .collect::<Vec<_>>();
    let components = abstractions
        .iter()
        .cloned()
        .map(AbstractionComponent::from)
        .collect::<Vec<_>>();
    let order = |variant| {
        goal_cover_order(&base_order, &components, &abstraction_h, true, variant).unwrap()
    };
    let goal = |component_id: usize| {
        abstractions[component_id]
            .metadata
            .collection_member
            .map(|member| member.goal_id)
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
    assert_ne!(first.split_kind, complement.split_kind);
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
        second_anchor_round: 1,
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
            abstraction.distance_table.distances[abstraction.distance_table.initial_state]
        })
        .collect::<Vec<_>>();
    let structural_id = mixed.len() - 1;
    let mixed_components = mixed
        .iter()
        .cloned()
        .map(AbstractionComponent::from)
        .collect::<Vec<_>>();
    let prefixed = goal_cover_order(
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
fn compact_goal_cover_schedule_visits_every_anchor_goal_with_every_variant_pair() {
    let variants = compact_goal_cover_variants(18, 8, Some(4));
    let anchor_goals = variants
        .iter()
        .map(|variant| variant.anchor_goal_offset)
        .collect::<HashSet<_>>();

    assert_eq!(variants.len(), 18 * 8 * 8);
    assert_eq!(anchor_goals, (0..18).collect());
}

#[test]
fn compact_goal_cover_schedule_supports_one_abstraction_per_goal() {
    let variants = compact_goal_cover_variants(18, 1, Some(4));
    let anchor_goals = variants
        .iter()
        .map(|variant| variant.anchor_goal_offset)
        .collect::<HashSet<_>>();

    assert_eq!(variants.len(), 18);
    assert_eq!(anchor_goals, (0..18).collect());
}
