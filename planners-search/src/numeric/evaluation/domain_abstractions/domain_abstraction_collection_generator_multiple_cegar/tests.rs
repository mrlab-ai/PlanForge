use super::*;

#[test]
fn single_init_split_selection_uses_round_robin_iteration_order() {
    let candidates = [0usize, 1, 2, 3, 4];
    let selected = (1..=8)
        .map(|iteration| select_single_init_split_var(&candidates, iteration).unwrap())
        .collect::<Vec<_>>();

    assert_eq!(selected, vec![1, 2, 3, 4, 0, 1, 2, 3]);
}

#[test]
fn single_init_split_selection_handles_empty_candidates() {
    assert_eq!(select_single_init_split_var(&[], 1), None);
}

#[test]
fn standard_uses_configured_full_goal_flaw_kind() {
    let config = DomainAbstractionCollectionGeneratorMultipleCegarConfig {
        portfolio_strategy: PortfolioStrategy::Standard,
        flaw_kind: FlawKind::SequenceBidirectional,
        ..Default::default()
    };
    let generator = DomainAbstractionCollectionGeneratorMultipleCegar::new(config);

    assert!(generator.uses_full_goal_task(11, 1));
    assert!(generator.uses_full_goal_task(11, 2));
    assert_eq!(
        generator.flaw_kind_for_goal_count(11, 1),
        FlawKind::SequenceBidirectional
    );
}

#[test]
fn complementary_uses_full_goal_then_single_goal_schedule() {
    let config = DomainAbstractionCollectionGeneratorMultipleCegarConfig {
        portfolio_strategy: PortfolioStrategy::Complementary,
        flaw_kind: FlawKind::SequenceBidirectional,
        ..Default::default()
    };
    let generator = DomainAbstractionCollectionGeneratorMultipleCegar::new(config);

    assert!(generator.uses_full_goal_task(2, 1));
    assert!(!generator.uses_full_goal_task(2, 2));
    assert!(!generator.uses_full_goal_task(2, 3));
    assert!(generator.uses_full_goal_task(0, 2));
    assert_eq!(
        generator.flaw_kind_for_goal_count(2, 1),
        FlawKind::SequenceBidirectional
    );
    assert_eq!(
        generator.flaw_kind_for_goal_count(2, 2),
        FlawKind::SequenceBidirectional
    );
    assert_eq!(
        generator.flaw_kind_for_goal_count(2, 3),
        FlawKind::TargetCentered
    );
    assert_eq!(
        generator.flaw_kind_for_goal_count(11, 1),
        FlawKind::SequenceBidirectional
    );
    assert_eq!(
        generator.flaw_kind_for_goal_count(11, 2),
        FlawKind::TargetCentered
    );
    assert_eq!(
        generator.flaw_kind_for_goal_count(11, 3),
        FlawKind::TargetCentered
    );
}
