use super::*;
use planforge_search::evaluation::domain_abstractions::cegar::{CegarConfig, FlawKind};
use planforge_search::evaluation::domain_abstractions::domain_abstraction_collection_generator_multiple_cegar::{
    DomainAbstractionCollectionGeneratorMultipleCegarConfig, InitSplitQuantity, PortfolioStrategy,
    VariableSubset,
};
use planforge_search::evaluation::abstraction_collections::saturated_cost_partitioning_online_heuristic::{
    FillScpConfig, OrderGenerator, Saturator, ScoringFunction, ScpOnlineConfig,
};
use planforge_search::evaluation::numeric_landmarks::lm_cut_numeric_heuristic::LmCutNumericConfig;
use planforge_search::evaluation::pattern_databases::canonical_pdb_heuristic::CanonicalNumericPdbConfig;
use planforge_search::evaluation::pattern_databases::pattern_database::PdbInternalHeuristic;
use planforge_search::evaluation::pattern_databases::pattern_generator_greedy::GreedyPatternGeneratorConfig;
use planforge_search::evaluation::pattern_databases::variable_order_finder::GreedyVariableOrderType;

fn astar_heuristic(input: &str) -> HeuristicSpec {
    match parse_search_spec(input).unwrap() {
        SearchSpec::Astar(h) => h,
        other => panic!("expected astar(...), got {other:?}"),
    }
}

#[test]
fn parses_heuristic_spec_ff_call() {
    let h = parse_heuristic_spec("ff()").unwrap();
    assert_eq!(h.name, "ff");
    assert!(h.args.is_empty());
}

#[test]
fn parses_heuristic_spec_blind_bare_identifier() {
    let h = parse_heuristic_spec("blind").unwrap();
    assert_eq!(h.name, "blind");
    assert!(h.args.is_empty());
}

#[test]
fn parses_astar_blind_with_or_without_unit_parens() {
    let h = astar_heuristic("astar(blind)");
    assert_eq!(h.name, "blind");
    assert!(h.args.is_empty());

    let h = astar_heuristic("astar(blind())");
    assert_eq!(h.name, "blind");
    assert!(h.args.is_empty());
}

#[test]
fn parses_astar_domain_abstraction_with_or_without_unit_parens() {
    let h = astar_heuristic("astar(domain_abstraction)");
    assert_eq!(h.name, "domain_abstraction");
    assert!(h.args.is_empty());

    let h = astar_heuristic("astar(domain_abstraction())");
    assert_eq!(h.name, "domain_abstraction");
    assert!(h.args.is_empty());
}

#[test]
fn parses_astar_domain_abstraction_with_named_options() {
    let h = astar_heuristic(
        "astar(domain_abstraction(max_abstraction_size=10000, use_wildcard_plans=false, combine_labels=true, random_seed=7))",
    );
    assert_eq!(h.name, "domain_abstraction");

    let mut cfg = CegarConfig::default();
    apply_da_options(&mut cfg, &h.args).unwrap();
    assert_eq!(cfg.max_abstraction_size, 10_000);
    assert!(!cfg.use_wildcard_plans);
    assert!(cfg.combine_labels);
    assert_eq!(cfg.random_seed, Some(7));
}

#[test]
fn parses_astar_canonical_domain_abstractions_with_or_without_parens() {
    let h = astar_heuristic("astar(canonical_domain_abstractions)");
    assert_eq!(h.name, "canonical_domain_abstractions");
    assert!(h.args.is_empty());

    let h = astar_heuristic("astar(canonical_domain_abstractions())");
    assert_eq!(h.name, "canonical_domain_abstractions");
    assert!(h.args.is_empty());
}

#[test]
fn parses_astar_canonical_domain_abstractions_with_named_options() {
    let h = astar_heuristic(
        "astar(canonical_domain_abstractions(max_collection_size=123, total_max_time=4.5, blacklist_option=non_goals, init_split_quantity=all, use_wildcard_plans=false, combine_labels=true, flaw_kind=sequence_progression, random_seed=7))",
    );
    let mut cfg = DomainAbstractionCollectionGeneratorMultipleCegarConfig::default();
    ApplyOptions::apply_options(&mut cfg, &h.args).unwrap();

    assert_eq!(cfg.max_collection_size, 123);
    assert_eq!(cfg.total_max_time, 4.5);
    assert_eq!(cfg.blacklist_option, VariableSubset::NonGoals);
    assert_eq!(cfg.init_split_quantity, InitSplitQuantity::All);
    assert!(!cfg.use_wildcard_plans);
    assert!(cfg.combine_labels);
    assert_eq!(cfg.flaw_kind, FlawKind::SequenceProgression);
    assert_eq!(cfg.random_seed, Some(7));
}

#[test]
fn parses_hierarchical_canonical_abstraction_sources() {
    let h = astar_heuristic(
        "astar(canonical(domain(max_abstraction_size=100), cartesian(max_states=100), pdb(max_pdb_states=100)))",
    );
    assert_eq!(h.name, "canonical");
    let source_names: Vec<_> = h
        .args
        .iter()
        .map(|arg| arg.value().as_call().unwrap().name())
        .collect();
    assert_eq!(source_names, ["domain", "cartesian", "pdb"]);
}

#[test]
fn parses_hierarchical_scp_options_and_sources() {
    let h = astar_heuristic(
        "astar(scp(domain(max_collection_size=1000), cartesian(max_states=100), pdb(max_pdb_states=100), saturator=perimstar, use_abstract_operator_cost_partitioning=true))",
    );
    assert_eq!(h.name, "scp");
    let (sources, options) = crate::abstraction_config::split_component_sources(&h.args).unwrap();
    assert_eq!(sources.len(), 3);
    assert_eq!(options.len(), 2);
}

#[test]
fn parses_execute_entire_plan_flaw_kind() {
    let spec =
        parse_search_spec("astar(canonical_domain_abstractions(flaw_kind=execute_entire_plan))")
            .unwrap();
    let h = match &spec {
        SearchSpec::Astar(h) => h,
        _ => panic!("expected astar(...)"),
    };
    let mut cfg = DomainAbstractionCollectionGeneratorMultipleCegarConfig::default();
    ApplyOptions::apply_options(&mut cfg, &h.args).unwrap();
    assert_eq!(cfg.flaw_kind, FlawKind::ExecuteEntirePlan);

    assert_eq!(parse_search_spec(&spec.to_string()).unwrap(), spec);
}

#[test]
fn parses_forward_partition_deviation_split_direction() {
    let spec = parse_search_spec(
        "astar(canonical_domain_abstractions(split_direction=forward_partition_deviation))",
    )
    .unwrap();
    let h = match &spec {
        SearchSpec::Astar(h) => h,
        _ => panic!("expected astar(...)"),
    };
    let mut cfg = DomainAbstractionCollectionGeneratorMultipleCegarConfig::default();
    ApplyOptions::apply_options(&mut cfg, &h.args).unwrap();
    assert_eq!(
        cfg.split_direction,
        Some(planforge_search::evaluation::domain_abstractions::cegar::SplitDirection::ForwardPartitionDeviation)
    );
    assert_eq!(parse_search_spec(&spec.to_string()).unwrap(), spec);
}

#[test]
fn parses_astar_scp_online_with_or_without_unit_parens() {
    let h = astar_heuristic("astar(scp_online)");
    assert_eq!(h.name, "scp_online");
    assert!(h.args.is_empty());

    let h = astar_heuristic("astar(scp_online())");
    assert_eq!(h.name, "scp_online");
    assert!(h.args.is_empty());
}

#[test]
fn parses_astar_fill_scp_with_named_options() {
    // LMcut params are now nested via `lmcut=lmcutnumeric(...)` rather than
    // flat at the fillSCP level.
    let h = astar_heuristic(
        "astar(fillSCP(table_construction_max_time=34.5, use_abstract_operator_cost_partitioning=true, saturator=perimstar, scoring_function=max_heuristic, orders=random_orders, order_optimization_max_time=1.5, max_collection_size=123, total_max_time=4.5, blacklist_option=non_goals, init_split_quantity=all, use_wildcard_plans=false, combine_labels=true, flaw_kind=sequence_progression, split_direction=backward, random_seed=7, debug=true, lmcut=lmcutnumeric(precision=0.5, epsilon=0.25)))",
    );
    assert_eq!(h.name, "fillscp");

    let mut cfg = FillScpConfig::default();
    ApplyOptions::apply_options(&mut cfg, &h.args).unwrap();
    assert_eq!(cfg.table_construction_max_time, 34.5);
    assert!(cfg.use_abstract_operator_cost_partitioning);
    assert_eq!(cfg.saturator, Saturator::Perimstar);
    assert_eq!(cfg.scoring_function, ScoringFunction::MaxHeuristic);
    assert_eq!(cfg.order_generator, OrderGenerator::Random);
    assert_eq!(cfg.order_optimization_max_time, 1.5);
    assert!(cfg.combine_labels);
    assert_eq!(cfg.collection_config.max_collection_size, 123);
    assert_eq!(cfg.collection_config.total_max_time, 4.5);
    assert_eq!(
        cfg.collection_config.blacklist_option,
        VariableSubset::NonGoals
    );
    assert_eq!(
        cfg.collection_config.init_split_quantity,
        InitSplitQuantity::All
    );
    assert!(!cfg.collection_config.use_wildcard_plans);
    assert_eq!(
        cfg.collection_config.flaw_kind,
        FlawKind::SequenceProgression
    );
    assert_eq!(
        cfg.collection_config.split_direction,
        Some(planforge_search::evaluation::domain_abstractions::cegar::SplitDirection::Backward)
    );
    assert_eq!(cfg.collection_config.random_seed, Some(7));
    assert_eq!(cfg.random_seed, Some(7));
    assert!(cfg.collection_config.debug);
    assert_eq!(cfg.lmcut_config.precision, 0.5);
    assert_eq!(cfg.lmcut_config.epsilon, 0.25);
}

#[test]
fn parses_astar_scp_online_with_named_options() {
    let h = astar_heuristic(
        "astar(scp_online(max_time=12.5, table_construction_max_time=34.5, max_size=2048, interval=3, use_abstract_operator_cost_partitioning=true, saturator=perimstar, scoring_function=max_heuristic, orders=dynamic_greedy_orders, order_optimization_max_time=1.5, max_collection_size=123, total_max_time=4.5, blacklist_option=non_goals, init_split_quantity=all, use_wildcard_plans=false, combine_labels=true, flaw_kind=sequence_progression, portfolio_strategy=complementary, random_seed=7, debug=true))",
    );
    let mut cfg = ScpOnlineConfig::default();
    ApplyOptions::apply_options(&mut cfg, &h.args).unwrap();
    assert_eq!(cfg.max_time, 12.5);
    assert_eq!(cfg.table_construction_max_time, 34.5);
    assert_eq!(cfg.max_size, 2048);
    assert_eq!(cfg.interval, 3);
    assert!(cfg.use_abstract_operator_cost_partitioning);
    assert_eq!(cfg.saturator, Saturator::Perimstar);
    assert_eq!(cfg.scoring_function, ScoringFunction::MaxHeuristic);
    assert_eq!(cfg.order_generator, OrderGenerator::DynamicGreedy);
    assert_eq!(cfg.order_optimization_max_time, 1.5);
    assert!(cfg.combine_labels);
    assert_eq!(cfg.collection_config.max_collection_size, 123);
    assert_eq!(cfg.collection_config.total_max_time, 4.5);
    assert_eq!(
        cfg.collection_config.blacklist_option,
        VariableSubset::NonGoals
    );
    assert_eq!(
        cfg.collection_config.init_split_quantity,
        InitSplitQuantity::All
    );
    assert!(!cfg.collection_config.use_wildcard_plans);
    assert!(cfg.collection_config.combine_labels);
    assert_eq!(
        cfg.collection_config.flaw_kind,
        FlawKind::SequenceProgression
    );
    assert_eq!(
        cfg.collection_config.portfolio_strategy,
        PortfolioStrategy::Complementary
    );
    assert_eq!(cfg.collection_config.random_seed, Some(7));
    assert_eq!(cfg.random_seed, Some(7));
    assert!(cfg.collection_config.debug);
}

#[test]
fn parses_astar_greedy_numeric_pdb_with_or_without_unit_parens() {
    let h = astar_heuristic("astar(greedy_numeric_pdb)");
    assert_eq!(h.name, "greedy_numeric_pdb");
    assert!(h.args.is_empty());

    let h = astar_heuristic("astar(greedy_numeric_pdb())");
    assert_eq!(h.name, "greedy_numeric_pdb");
    assert!(h.args.is_empty());
}

#[test]
fn parses_astar_greedy_numeric_pdb_with_named_options() {
    let h = astar_heuristic(
        "astar(greedy_numeric_pdb(max_pdb_states=321, numeric_first=false, random_seed=7, variable_order_type=cg_goal_random, exploration_heuristic=lmcut, frontier_heuristic=blind, failed_lookup_heuristic=lmcut))",
    );
    let mut cfg = GreedyPatternGeneratorConfig::default();
    ApplyOptions::apply_options(&mut cfg, &h.args).unwrap();
    assert_eq!(cfg.max_pdb_states, 321);
    assert!(!cfg.numeric_first);
    assert_eq!(cfg.random_seed, 7);
    assert_eq!(
        cfg.variable_order_type,
        GreedyVariableOrderType::CgGoalRandom
    );
    assert_eq!(cfg.exploration_heuristic, PdbInternalHeuristic::Lmcut);
    assert_eq!(cfg.frontier_heuristic, PdbInternalHeuristic::Blind);
    assert_eq!(cfg.failed_lookup_heuristic, PdbInternalHeuristic::Lmcut);
}

#[test]
fn positional_args_map_to_canonical_order() {
    // greedy_numeric_pdb's ORDER starts with max_pdb_states, numeric_first, random_seed
    let h = astar_heuristic("astar(greedy_numeric_pdb(321, false, 7))");
    let mut cfg = GreedyPatternGeneratorConfig::default();
    ApplyOptions::apply_options(&mut cfg, &h.args).unwrap();
    assert_eq!(cfg.max_pdb_states, 321);
    assert!(!cfg.numeric_first);
    assert_eq!(cfg.random_seed, 7);
}

#[test]
fn mixed_positional_and_named_args_work() {
    // First positional → max_pdb_states; the named ones are explicit.
    let h = astar_heuristic("astar(greedy_numeric_pdb(321, numeric_first=false, random_seed=7))");
    let mut cfg = GreedyPatternGeneratorConfig::default();
    ApplyOptions::apply_options(&mut cfg, &h.args).unwrap();
    assert_eq!(cfg.max_pdb_states, 321);
    assert!(!cfg.numeric_first);
    assert_eq!(cfg.random_seed, 7);
}

#[test]
fn positional_then_named_for_same_slot_errors() {
    // Positional 321 → max_pdb_states, then max_pdb_states=999 collides.
    let h = astar_heuristic("astar(greedy_numeric_pdb(321, max_pdb_states=999))");
    let err = ApplyOptions::apply_options(&mut GreedyPatternGeneratorConfig::default(), &h.args)
        .unwrap_err();
    assert!(
        err.contains("duplicate option `max_pdb_states`"),
        "got `{err}`"
    );
}

#[test]
fn too_many_positional_args_errors() {
    // greedy_numeric_pdb has 7 positional slots; 8 should error.
    let h = astar_heuristic(
        "astar(greedy_numeric_pdb(1, false, 2, cg_goal_level, blind, blind, blind, EXTRA))",
    );
    let err = ApplyOptions::apply_options(&mut GreedyPatternGeneratorConfig::default(), &h.args)
        .unwrap_err();
    assert!(err.contains("too many positional"), "got `{err}`");
}

#[test]
fn scp_online_accepts_nested_collection_call() {
    let h = astar_heuristic(
        "astar(scp_online(collection=multi_domain_abstractions(max_collection_size=99, total_max_time=2.5), saturator=perimstar))",
    );
    let mut cfg = ScpOnlineConfig::default();
    ApplyOptions::apply_options(&mut cfg, &h.args).unwrap();
    assert_eq!(cfg.collection_config.max_collection_size, 99);
    assert_eq!(cfg.collection_config.total_max_time, 2.5);
    assert_eq!(cfg.saturator, Saturator::Perimstar);
}

#[test]
fn fill_scp_accepts_nested_collection_and_lmcut_calls() {
    let h = astar_heuristic(
        "astar(fillSCP(collection=canonical_domain_abstractions(max_collection_size=7), lmcut=lmcutnumeric(precision=0.5)))",
    );
    let mut cfg = FillScpConfig::default();
    ApplyOptions::apply_options(&mut cfg, &h.args).unwrap();
    assert_eq!(cfg.collection_config.max_collection_size, 7);
    assert_eq!(cfg.lmcut_config.precision, 0.5);
}

#[test]
fn nested_collection_ignores_inner_call_name() {
    // The derived nested arm consumes the inner call's args without
    // validating its name — `collection=anything(max_collection_size=1)` is
    // equivalent. The wrapping name is treated as a free-form label.
    let h = astar_heuristic("astar(scp_online(collection=bogus(max_collection_size=1)))");
    let mut cfg = ScpOnlineConfig::default();
    ApplyOptions::apply_options(&mut cfg, &h.args).unwrap();
    assert_eq!(cfg.collection_config.max_collection_size, 1);
}

#[test]
fn parses_registry_style_search_with_keyed_heuristic() {
    let h = astar_heuristic(
        "search(astar(heuristic=greedy_numeric_pdb(max_pdb_states=321, numeric_first=false)))",
    );
    assert_eq!(h.name, "greedy_numeric_pdb");
    let mut cfg = GreedyPatternGeneratorConfig::default();
    ApplyOptions::apply_options(&mut cfg, &h.args).unwrap();
    assert_eq!(cfg.max_pdb_states, 321);
    assert!(!cfg.numeric_first);
}

#[test]
fn parses_astar_canonical_numeric_pdb_with_or_without_unit_parens() {
    let h = astar_heuristic("astar(canonical_numeric_pdb)");
    assert_eq!(h.name, "canonical_numeric_pdb");
    assert!(h.args.is_empty());

    let h = astar_heuristic("astar(canonical_numeric_pdb())");
    assert_eq!(h.name, "canonical_numeric_pdb");
    assert!(h.args.is_empty());
}

#[test]
fn parses_astar_canonical_numeric_pdb_with_named_options() {
    let h = astar_heuristic(
        "astar(canonical_numeric_pdb(max_pdb_states=321, max_pattern_size=3, only_interesting_patterns=false, exploration_heuristic=blind, frontier_heuristic=lmcut, failed_lookup_heuristic=lmcut))",
    );
    let mut cfg = CanonicalNumericPdbConfig::default();
    ApplyOptions::apply_options(&mut cfg, &h.args).unwrap();
    assert_eq!(cfg.max_pdb_states, 321);
    assert_eq!(cfg.max_pattern_size, 3);
    assert!(!cfg.only_interesting_patterns);
    assert_eq!(cfg.exploration_heuristic, PdbInternalHeuristic::Blind);
    assert_eq!(cfg.frontier_heuristic, PdbInternalHeuristic::Lmcut);
    assert_eq!(cfg.failed_lookup_heuristic, PdbInternalHeuristic::Lmcut);
}

#[test]
fn parses_astar_lmcutnumeric_with_or_without_unit_parens() {
    let h = astar_heuristic("astar(lmcutnumeric)");
    assert_eq!(h.name, "lmcutnumeric");
    assert!(h.args.is_empty());

    let h = astar_heuristic("astar(lmcutnumeric())");
    assert_eq!(h.name, "lmcutnumeric");
    assert!(h.args.is_empty());
}

#[test]
fn parses_astar_lmcutnumeric_with_named_options() {
    let h = astar_heuristic(
        "astar(lmcutnumeric(ceiling_less_than_one=true, ignore_numeric=true, random_pcf=true, irmax=true, disable_ma=true, use_second_order_simple=true, use_constant_assignment=true, bound_iterations=7, precision=0.5, epsilon=0.25))",
    );
    let mut cfg = LmCutNumericConfig::default();
    ApplyOptions::apply_options(&mut cfg, &h.args).unwrap();
    assert!(cfg.ceiling_less_than_one);
    assert!(cfg.ignore_numeric);
    assert!(cfg.random_pcf);
    assert!(cfg.irmax);
    assert!(cfg.disable_ma);
    assert!(cfg.use_second_order_simple);
    assert!(cfg.use_constant_assignment);
    assert_eq!(cfg.bound_iterations, 7);
    assert_eq!(cfg.precision, 0.5);
    assert_eq!(cfg.epsilon, 0.25);
}

#[test]
fn parses_astar_multi_domain_abstractions_with_or_without_parens() {
    let h = astar_heuristic("astar(multi_domain_abstractions)");
    assert_eq!(h.name, "multi_domain_abstractions");
    assert!(h.args.is_empty());

    let h = astar_heuristic("astar(multi_domain_abstractions())");
    assert_eq!(h.name, "multi_domain_abstractions");
    assert!(h.args.is_empty());
}

#[test]
fn parses_astar_multi_domain_abstractions_with_named_options() {
    let h = astar_heuristic(
        "astar(multi_domain_abstractions(max_collection_size=123, total_max_time=4.5, blacklist_option=non_goals, init_split_quantity=all, use_wildcard_plans=false, combine_labels=true, flaw_kind=sequence_bidirectional, portfolio_strategy=complementary, random_seed=7, debug=true))",
    );
    let mut cfg = DomainAbstractionCollectionGeneratorMultipleCegarConfig::default();
    ApplyOptions::apply_options(&mut cfg, &h.args).unwrap();
    assert_eq!(cfg.max_collection_size, 123);
    assert_eq!(cfg.total_max_time, 4.5);
    assert_eq!(cfg.blacklist_option, VariableSubset::NonGoals);
    assert_eq!(cfg.init_split_quantity, InitSplitQuantity::All);
    assert!(!cfg.use_wildcard_plans);
    assert!(cfg.combine_labels);
    assert_eq!(cfg.flaw_kind, FlawKind::SequenceBidirectional);
    assert_eq!(cfg.portfolio_strategy, PortfolioStrategy::Complementary);
    assert_eq!(cfg.random_seed, Some(7));
    assert!(cfg.debug);
}

#[test]
fn parses_astar_multi_domain_abstractions_with_trailing_comma() {
    let h = astar_heuristic("astar(multi_domain_abstractions(max_collection_size=123,))");
    let mut cfg = DomainAbstractionCollectionGeneratorMultipleCegarConfig::default();
    ApplyOptions::apply_options(&mut cfg, &h.args).unwrap();
    assert_eq!(cfg.max_collection_size, 123);
}

#[test]
fn display_round_trips_multi_domain_abstractions() {
    let parsed = parse_search_spec(
        "astar(multi_domain_abstractions(max_abstraction_size=42, abstraction_generation_max_time=infinity))",
    )
    .unwrap();
    let reparsed = parse_search_spec(&parsed.to_string()).unwrap();
    assert_eq!(parsed, reparsed);
}

#[test]
fn display_round_trips_canonical_domain_abstractions() {
    let parsed = parse_search_spec(
        "astar(canonical_domain_abstractions(max_abstraction_size=42, abstraction_generation_max_time=infinity))",
    )
    .unwrap();
    let reparsed = parse_search_spec(&parsed.to_string()).unwrap();
    assert_eq!(parsed, reparsed);
}

#[test]
fn display_round_trips_hierarchical_abstraction_collection() {
    let parsed = parse_search_spec(
        "astar(canonical(domain(max_abstraction_size=42), cartesian(max_states=42), pdb(max_pdb_states=42)))",
    )
    .unwrap();
    let reparsed = parse_search_spec(&parsed.to_string()).unwrap();
    assert_eq!(parsed, reparsed);
}

#[test]
fn display_round_trips_scp_online() {
    let parsed = parse_search_spec(
        "astar(scp_online(max_time=12.5, max_abstraction_size=42, abstraction_generation_max_time=infinity, use_abstract_operator_cost_partitioning=true, saturator=perimstar, scoring_function=min_stolen_costs, orders=random_orders, order_optimization_max_time=0.25))",
    )
    .unwrap();
    let reparsed = parse_search_spec(&parsed.to_string()).unwrap();
    assert_eq!(parsed, reparsed);
}

#[test]
fn display_round_trips_scp_online_with_nested_collection() {
    let parsed = parse_search_spec(
        "astar(scp_online(collection=multi_domain_abstractions(max_collection_size=99, total_max_time=2.5), saturator=perimstar))",
    )
    .unwrap();
    let reparsed = parse_search_spec(&parsed.to_string()).unwrap();
    assert_eq!(parsed, reparsed);
}

#[test]
fn display_round_trips_positional_args() {
    let parsed = parse_search_spec("astar(greedy_numeric_pdb(321, false, 7))").unwrap();
    let reparsed = parse_search_spec(&parsed.to_string()).unwrap();
    assert_eq!(parsed, reparsed);
}

#[test]
fn rejects_unknown_options_inside_known_heuristics() {
    let h = astar_heuristic("astar(scp_online(deviation_flaws=false))");
    let err = ApplyOptions::apply_options(&mut ScpOnlineConfig::default(), &h.args).unwrap_err();
    assert!(err.contains("deviation_flaws"), "got `{err}`");

    let h = astar_heuristic("astar(canonical_domain_abstractions(deviation_flaws=false))");
    let err = ApplyOptions::apply_options(
        &mut DomainAbstractionCollectionGeneratorMultipleCegarConfig::default(),
        &h.args,
    )
    .unwrap_err();
    assert!(err.contains("deviation_flaws"), "got `{err}`");
}

#[test]
fn display_round_trips_greedy_numeric_pdb() {
    let parsed = parse_search_spec(
        "astar(greedy_numeric_pdb(max_pdb_states=42, numeric_first=false, random_seed=9, variable_order_type=cg_goal_random, exploration_heuristic=lmcut, frontier_heuristic=blind, failed_lookup_heuristic=lmcut))",
    )
    .unwrap();
    let reparsed = parse_search_spec(&parsed.to_string()).unwrap();
    assert_eq!(parsed, reparsed);
}

#[test]
fn display_round_trips_canonical_numeric_pdb() {
    let parsed = parse_search_spec(
        "astar(canonical_numeric_pdb(max_pdb_states=42, max_pattern_size=3, only_interesting_patterns=false, exploration_heuristic=blind, frontier_heuristic=lmcut, failed_lookup_heuristic=lmcut))",
    )
    .unwrap();
    let reparsed = parse_search_spec(&parsed.to_string()).unwrap();
    assert_eq!(parsed, reparsed);
}

#[test]
fn display_round_trips_lmcutnumeric() {
    let parsed = parse_search_spec(
        "astar(lmcutnumeric(ceiling_less_than_one=true, disable_ma=true, bound_iterations=4, precision=0.5, epsilon=0.25))",
    )
    .unwrap();
    let reparsed = parse_search_spec(&parsed.to_string()).unwrap();
    assert_eq!(parsed, reparsed);
}

#[test]
fn rejects_removed_exec_entire_plan_randomize_option() {
    let h = astar_heuristic("astar(multi_domain_abstractions(exec_entire_plan=randomize))");
    let err = ApplyOptions::apply_options(
        &mut DomainAbstractionCollectionGeneratorMultipleCegarConfig::default(),
        &h.args,
    )
    .unwrap_err();
    assert!(err.contains("exec_entire_plan"));
}

#[test]
fn trims_trailing_punctuation() {
    assert_eq!(astar_heuristic("astar(blind()).").name, "blind");
    assert_eq!(
        astar_heuristic("astar(domain_abstraction());").name,
        "domain_abstraction"
    );
    assert_eq!(
        astar_heuristic("astar(greedy_numeric_pdb());").name,
        "greedy_numeric_pdb"
    );
    assert_eq!(
        astar_heuristic("astar(canonical_numeric_pdb());").name,
        "canonical_numeric_pdb"
    );
    assert_eq!(
        astar_heuristic("astar(lmcutnumeric());").name,
        "lmcutnumeric"
    );
    assert_eq!(
        astar_heuristic("astar(multi_domain_abstractions());").name,
        "multi_domain_abstractions"
    );
    assert_eq!(
        astar_heuristic("astar(canonical_domain_abstractions());").name,
        "canonical_domain_abstractions"
    );

    assert_eq!(
        parse_search_spec("da_debug();").unwrap(),
        SearchSpec::DaDebug
    );
    assert_eq!(
        parse_search_spec("astar_da_debug();").unwrap(),
        SearchSpec::AstarDaDebug
    );
}

#[test]
fn parses_top_level_da_debug_with_or_without_unit_parens() {
    assert_eq!(parse_search_spec("da_debug").unwrap(), SearchSpec::DaDebug);
    assert_eq!(
        parse_search_spec("da_debug()").unwrap(),
        SearchSpec::DaDebug
    );
}

#[test]
fn parses_top_level_astar_da_debug_with_or_without_unit_parens() {
    assert_eq!(
        parse_search_spec("astar_da_debug").unwrap(),
        SearchSpec::AstarDaDebug
    );
    assert_eq!(
        parse_search_spec("astar_da_debug()").unwrap(),
        SearchSpec::AstarDaDebug
    );
}

#[test]
fn errors_are_human_readable() {
    let err = parse_search_spec("astar(").unwrap_err();
    assert!(err.to_lowercase().contains("invalid"));
}

#[test]
fn unknown_heuristic_name_propagates() {
    let h = astar_heuristic("astar(does_not_exist)");
    assert_eq!(h.name, "does_not_exist");
}
