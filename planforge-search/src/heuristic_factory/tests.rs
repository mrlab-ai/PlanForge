//! What each `--search` option name means, tested through the text a user
//! actually types.
//!
//! The option language is parsed in this crate too, so these tests need nothing
//! from `planforge-searcher` and the factory's internals stay private.

use super::abstraction_config;
use crate::config::{ApplyOptions, HeuristicSpec, parse_heuristic_spec};
use crate::evaluation::abstraction_collections::portfolio::CollectionStrategy;
use crate::evaluation::abstraction_collections::saturated_cost_partitioning_online_heuristic::{
    CostPartitioningMethod, FillScpConfig, OrderGenerator, Saturator, ScoringFunction,
    ScpOnlineConfig,
};
use crate::evaluation::cartesian_abstractions::CartesianSplitSelection;
use crate::evaluation::cartesian_abstractions::icaps26::Icaps26SplitSelection;
use crate::evaluation::domain_abstractions::cegar::{CegarConfig, FlawKind};
use crate::evaluation::domain_abstractions::domain_abstraction_collection_generator_multiple_cegar::{
    DomainAbstractionCollectionGeneratorMultipleCegarConfig, FlawTreatmentVariants,
    InitSplitQuantity, VariableSubset,
};
use crate::evaluation::numeric_landmarks::lm_cut_numeric_heuristic::LmCutNumericConfig;
use crate::evaluation::pattern_databases::canonical_pdb_heuristic::CanonicalNumericPdbConfig;
use crate::evaluation::pattern_databases::pattern_database::PdbInternalHeuristic;
use crate::evaluation::pattern_databases::pattern_generator_greedy::GreedyPatternGeneratorConfig;
use crate::evaluation::pattern_databases::variable_order_finder::GreedyVariableOrderType;
/// The heuristic inside an `astar(...)` spec. Written as the full search spec
/// because that is what a user types, and `astar(` / `)` is all the engine
/// grammar these tests need.
fn astar_heuristic(input: &str) -> HeuristicSpec {
    let inner = input
        .strip_prefix("astar(")
        .and_then(|rest| rest.strip_suffix(')'))
        .unwrap_or_else(|| panic!("expected `astar(<heuristic>)`, got `{input}`"));
    parse_heuristic_spec(inner).unwrap()
}

#[test]
fn parses_astar_domain_abstraction_with_named_options() {
    let h = astar_heuristic(
        "astar(domain_abstraction(max_abstraction_size=10000, use_wildcard_plans=false, combine_labels=true, random_seed=7))",
    );
    assert_eq!(h.name, "domain_abstraction");

    let mut cfg = CegarConfig::default();
    cfg.apply_options(&h.args).unwrap();
    assert_eq!(cfg.max_abstraction_size, 10_000);
    assert!(!cfg.use_wildcard_plans);
    assert!(cfg.combine_labels);
    assert_eq!(cfg.random_seed, Some(7));
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
fn parses_hierarchical_scp_options_and_sources() {
    let h = astar_heuristic(
        "astar(scp(domain(max_collection_size=1000), cartesian(max_states=100), pdb(max_pdb_states=100), online=false, diversify=true, samples=123, max_orders=17, orders=diverse_orders, initial_order_generation_max_time=9, saturator=perimstar, residual_sweeps=2, partitioning=region))",
    );
    assert_eq!(h.name, "scp");
    let (sources, options) = abstraction_config::split_component_sources(&h.args).unwrap();
    assert_eq!(sources.len(), 3);
    assert_eq!(options.len(), 9);
    abstraction_config::validate_scp_combinator_options(&options).unwrap();
    let mut config = ScpOnlineConfig::default();
    ApplyOptions::apply_options(&mut config, &options).unwrap();
    assert!(!config.online);
    assert!(config.diversify);
    assert_eq!(config.samples, 123);
    assert_eq!(config.max_orders, 17);
    assert_eq!(config.order_generator, OrderGenerator::Diverse);
    assert_eq!(config.initial_order_generation_max_time, 9.0);
    assert_eq!(config.residual_sweeps, 2);
}

#[test]
fn parses_hierarchical_cartesian_collection_source() {
    let h = astar_heuristic(
        "astar(scp(cartesian_collection(max_states=1000, flaw_kind=execute_entire_plan, collection_strategy=complementary, variants_per_goal=8, progressive_goal_roots=true, max_collection_size=100000, total_max_time=60, random_seed=1), saturator=all, partitioning=region))",
    );
    let (sources, options) = abstraction_config::split_component_sources(&h.args).unwrap();
    assert_eq!(sources.len(), 1);
    assert_eq!(sources[0].name(), "cartesian_collection");
    assert_eq!(options.len(), 2);
    let config = abstraction_config::apply_cartesian_collection_options(
        sources[0].args(),
        abstraction_config::ComponentUse::RegionalCostPartitioning,
    )
    .unwrap();
    assert_eq!(
        config.collection_strategy,
        CollectionStrategy::Complementary
    );
    assert_eq!(config.variants_per_goal, 8);
    assert_eq!(config.abstraction.flaw_kind, FlawKind::ExecuteEntirePlan);
    assert!(config.abstraction.compute_operator_footprints);
}

#[test]
fn parses_strict_icaps26_cartesian_source() {
    let h = astar_heuristic(
        "astar(scp(icaps26_cartesian(pick=min_unwanted,max_time=900,random_seed=7),online=false,partitioning=region))",
    );
    let (sources, options) = abstraction_config::split_component_sources(&h.args).unwrap();
    assert_eq!(sources.len(), 1);
    assert_eq!(sources[0].name(), "icaps26_cartesian");
    assert_eq!(options.len(), 2);
    let config = abstraction_config::apply_icaps26_cartesian_options(
        sources[0].args(),
        abstraction_config::ComponentUse::RegionalCostPartitioning,
    )
    .unwrap();
    assert_eq!(
        config.split_selection,
        CartesianSplitSelection::Icaps26(Icaps26SplitSelection::MinUnwanted)
    );
    assert_eq!(config.max_time, Some(std::time::Duration::from_secs(900)));
    assert_eq!(config.random_seed, Some(7));
    assert_eq!(
        config.refinement_direction,
        crate::evaluation::cartesian_abstractions::CartesianRefinementDirection::Regression
    );
    assert_eq!(
        config.abstract_plan_selection,
        crate::evaluation::cartesian_abstractions::CartesianAbstractPlanSelection::StableAStar
    );
    assert_eq!(
        config.flaw_candidate_generation,
        crate::evaluation::cartesian_abstractions::CartesianFlawCandidateGeneration::DesiredRegion
    );
    assert!(config.compute_operator_footprints);
    assert!(config.retain_transition_system);

    let defaults = abstraction_config::apply_icaps26_cartesian_options(
        &[],
        abstraction_config::ComponentUse::Standalone,
    )
    .unwrap();
    assert_eq!(defaults.random_seed, Some(2011));
}

#[test]
fn parses_matched_single_abstraction_icaps26_configuration() {
    let h = astar_heuristic("astar(canonical(icaps26_cartesian(pick=random,max_time=900)))");
    let (sources, construction_deadline) =
        abstraction_config::canonical_sources_and_deadline(&h.args).unwrap();
    assert_eq!(sources.len(), 1);
    assert_eq!(sources[0].name(), "icaps26_cartesian");
    assert!(
        construction_deadline.is_none(),
        "the published 900-second limit bounds CEGAR, not the outer combinator"
    );
    let config = abstraction_config::apply_icaps26_cartesian_options(
        sources[0].args(),
        abstraction_config::ComponentUse::Standalone,
    )
    .unwrap();
    assert_eq!(config.max_states, usize::MAX);
    assert_eq!(config.max_time, Some(std::time::Duration::from_secs(900)));
    assert_eq!(
        config.split_selection,
        CartesianSplitSelection::Icaps26(Icaps26SplitSelection::Random)
    );
    assert!(!config.retain_transition_system);
}

#[test]
fn parses_shared_construction_deadlines_for_canonical_and_scp() {
    let canonical =
        astar_heuristic("astar(canonical(cartesian(max_states=100),construction_max_time=900))");
    let (sources, deadline) =
        abstraction_config::canonical_sources_and_deadline(&canonical.args).unwrap();
    assert_eq!(sources.len(), 1);
    assert!(deadline.is_some());

    let scp = astar_heuristic(
        "astar(scp(cartesian(max_states=100),online=false,construction_max_time=900))",
    );
    let source_config = abstraction_config::scp_sources_options_and_deadline(&scp.args).unwrap();
    assert_eq!(source_config.sources.len(), 1);
    assert_eq!(source_config.options.len(), 1);
    assert_eq!(source_config.options[0].key(), Some("online"));
    assert!(source_config.construction_deadline.is_some());
}

#[test]
fn rejects_invalid_shared_construction_deadlines() {
    for value in ["0", "-1", "infinity"] {
        let h = astar_heuristic(&format!(
            "astar(canonical(cartesian(),construction_max_time={value}))"
        ));
        let error = abstraction_config::canonical_sources_and_deadline(&h.args)
            .expect_err("invalid construction deadline must fail");
        assert!(error.contains("construction_max_time"));
    }
}

#[test]
fn icaps26_cartesian_rejects_native_refinement_options() {
    let h = astar_heuristic("astar(canonical(icaps26_cartesian(flaw_kind=execute_entire_plan)))");
    let (sources, _) = abstraction_config::split_component_sources(&h.args).unwrap();
    let error = abstraction_config::apply_icaps26_cartesian_options(
        sources[0].args(),
        abstraction_config::ComponentUse::Standalone,
    )
    .unwrap_err();
    assert!(error.contains("unknown option `flaw_kind`"));
}

#[test]
fn hierarchical_cartesian_collection_preserves_first_flaw_default() {
    let h = astar_heuristic("astar(canonical(cartesian_collection(max_states=1000)))");
    let (sources, _) = abstraction_config::split_component_sources(&h.args).unwrap();
    let config = abstraction_config::apply_cartesian_collection_options(
        sources[0].args(),
        abstraction_config::ComponentUse::Standalone,
    )
    .unwrap();
    assert_eq!(config.abstraction.flaw_kind, FlawKind::Progression);
}

#[test]
fn hierarchical_cartesian_collection_rejects_unsupported_flaw_kind() {
    let h = astar_heuristic("astar(canonical(cartesian_collection(flaw_kind=regression)))");
    let (sources, _) = abstraction_config::split_component_sources(&h.args).unwrap();
    let error = abstraction_config::apply_cartesian_collection_options(
        sources[0].args(),
        abstraction_config::ComponentUse::Standalone,
    )
    .expect_err("unsupported Cartesian flaw kind must fail during parsing");
    assert!(error.contains("flaw_kind=regression"));
}

#[test]
fn direct_cartesian_abstraction_rejects_unsupported_flaw_kind() {
    let h = astar_heuristic("astar(cartesian_abstraction(flaw_kind=regression))");
    let mut cegar = CegarConfig::default();
    cegar.apply_options(&h.args).unwrap();

    let result = crate::evaluation::cartesian_abstractions::CartesianAbstractionGenerator::new(
        super::cartesian_config_from_cegar(&cegar),
    );
    assert!(
        result.is_err(),
        "flaw_kind=regression must not silently become progression"
    );
}

#[test]
fn direct_cartesian_abstraction_rejects_cegar_only_options() {
    for option in [
        "max_iterations=10",
        "use_wildcard_plans=true",
        "flaw_treatment=first",
        "init_split_method=identity",
    ] {
        let h = astar_heuristic(&format!("astar(cartesian_abstraction({option}))"));
        let error = super::validate_cartesian_cegar_options(&h.args).unwrap_err();
        assert!(
            error.contains("not supported for Cartesian"),
            "got `{error}`"
        );
    }
}

#[test]
fn legacy_cartesian_collection_conversion_preserves_flaw_kind() {
    let mut config = DomainAbstractionCollectionGeneratorMultipleCegarConfig {
        flaw_kind: FlawKind::ExecuteEntirePlan,
        ..Default::default()
    };
    let cartesian = super::cartesian_config_from_collection(&config, false).unwrap();
    assert_eq!(cartesian.flaw_kind, FlawKind::ExecuteEntirePlan);

    config.flaw_kind = FlawKind::Regression;
    let error = super::cartesian_config_from_collection(&config, false).unwrap_err();
    assert!(error.contains("flaw_kind=regression"), "got `{error}`");
}

#[test]
fn legacy_cartesian_collection_rejects_domain_only_options() {
    for option in [
        "max_collection_size=100",
        "use_wildcard_plans=true",
        "flaw_treatment=first",
        "init_split_method=identity",
    ] {
        let h = astar_heuristic(&format!("astar(scp_online_cartesian({option}))"));
        let error = super::validate_legacy_cartesian_collection_options(&h.args).unwrap_err();
        assert!(error.contains("not supported"), "got `{error}`");
    }
}

#[test]
fn rejects_boolean_cost_partitioning_mode() {
    let h = astar_heuristic("astar(scp(domain(), partitioning=label))");
    let (_, options) = abstraction_config::split_component_sources(&h.args).unwrap();
    let mut config = ScpOnlineConfig::default();
    ApplyOptions::apply_options(&mut config, &options).unwrap();
    assert_eq!(config.partitioning, CostPartitioningMethod::Label);

    let h = astar_heuristic("astar(scp(domain(), partitioning=true))");
    let (_, options) = abstraction_config::split_component_sources(&h.args).unwrap();
    let mut config = ScpOnlineConfig::default();
    assert!(ApplyOptions::apply_options(&mut config, &options).is_err());
}

#[test]
fn parses_execute_entire_plan_flaw_kind() {
    let h = astar_heuristic("astar(canonical_domain_abstractions(flaw_kind=execute_entire_plan))");
    let mut cfg = DomainAbstractionCollectionGeneratorMultipleCegarConfig::default();
    ApplyOptions::apply_options(&mut cfg, &h.args).unwrap();
    assert_eq!(cfg.flaw_kind, FlawKind::ExecuteEntirePlan);
}

#[test]
fn parses_single_domain_abstraction_time_limit() {
    let h = astar_heuristic(
        "astar(domain_abstraction(max_abstraction_size=1000,max_time=900,flaw_kind=execute_entire_plan))",
    );
    let mut cfg = CegarConfig::default();
    cfg.apply_options(&h.args).unwrap();
    assert_eq!(cfg.max_abstraction_size, 1000);
    assert_eq!(cfg.max_time, Some(std::time::Duration::from_secs(900)));
    assert_eq!(cfg.flaw_kind, FlawKind::ExecuteEntirePlan);

    let h = astar_heuristic("astar(domain_abstraction(max_time=infinity))");
    cfg.apply_options(&h.args).unwrap();
    assert_eq!(cfg.max_time, None);
}

#[test]
fn parses_forward_partition_deviation_split_direction() {
    let h = astar_heuristic(
        "astar(canonical_domain_abstractions(split_direction=forward_partition_deviation))",
    );
    let mut cfg = DomainAbstractionCollectionGeneratorMultipleCegarConfig::default();
    ApplyOptions::apply_options(&mut cfg, &h.args).unwrap();
    assert_eq!(
        cfg.split_direction,
        Some(crate::evaluation::domain_abstractions::cegar::SplitDirection::ForwardPartitionDeviation)
    );
}

#[test]
fn parses_full_task_interleaved_domain_collection() {
    let h = astar_heuristic(
        "astar(canonical(domain(collection_strategy=standard,interleave_split_directions=true,flaw_kind=execute_entire_plan,flaw_treatment=min_growth_single_atom)))",
    );
    let (sources, options) = abstraction_config::split_component_sources(&h.args).unwrap();
    assert_eq!(sources.len(), 1);
    assert!(options.is_empty());
    let mut source = DomainAbstractionCollectionGeneratorMultipleCegarConfig::default();
    ApplyOptions::apply_options(&mut source, sources[0].args()).unwrap();

    assert_eq!(source.collection_strategy, CollectionStrategy::Standard);
    assert!(source.interleave_split_directions);
    assert_eq!(source.flaw_kind, FlawKind::ExecuteEntirePlan);
    assert_eq!(
        source.flaw_treatment,
        FlawTreatmentVariants::MinGrowthSingleAtom
    );
}

#[test]
fn parses_astar_fill_scp_with_named_options() {
    // LMcut params are now nested via `lmcut=lmcutnumeric(...)` rather than
    // flat at the fillSCP level.
    let h = astar_heuristic(
        "astar(fillSCP(table_construction_max_time=34.5, partitioning=region, saturator=perimstar, scoring_function=max_heuristic, orders=random_orders, order_optimization_max_time=1.5, max_collection_size=123, total_max_time=4.5, blacklist_option=non_goals, init_split_quantity=all, use_wildcard_plans=false, combine_labels=true, flaw_kind=sequence_progression, split_direction=backward, random_seed=7, debug=true, lmcut=lmcutnumeric(precision=0.5, epsilon=0.25)))",
    );
    assert_eq!(h.name, "fillscp");

    let mut cfg = FillScpConfig::default();
    ApplyOptions::apply_options(&mut cfg, &h.args).unwrap();
    assert_eq!(cfg.table_construction_max_time, 34.5);
    assert_eq!(cfg.partitioning, CostPartitioningMethod::Region);
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
        Some(crate::evaluation::domain_abstractions::cegar::SplitDirection::Backward)
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
        "astar(scp_online(max_time=12.5, table_construction_max_time=34.5, max_size=2048, interval=3, partitioning=region, saturator=perimstar, scoring_function=max_heuristic, orders=dynamic_greedy_orders, order_optimization_max_time=1.5, max_collection_size=123, total_max_time=4.5, blacklist_option=non_goals, init_split_quantity=all, use_wildcard_plans=false, combine_labels=true, flaw_kind=sequence_progression, collection_strategy=complementary, random_seed=7, debug=true))",
    );
    let mut cfg = ScpOnlineConfig::default();
    ApplyOptions::apply_options(&mut cfg, &h.args).unwrap();
    assert_eq!(cfg.max_time, 12.5);
    assert_eq!(cfg.table_construction_max_time, 34.5);
    assert_eq!(cfg.max_size, 2048);
    assert_eq!(cfg.interval, 3);
    assert_eq!(cfg.partitioning, CostPartitioningMethod::Region);
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
        cfg.collection_config.collection_strategy,
        CollectionStrategy::Complementary
    );
    assert_eq!(cfg.collection_config.random_seed, Some(7));
    assert_eq!(cfg.random_seed, Some(7));
    assert!(cfg.collection_config.debug);
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
fn parses_astar_multi_domain_abstractions_with_named_options() {
    let h = astar_heuristic(
        "astar(multi_domain_abstractions(max_collection_size=123, total_max_time=4.5, blacklist_option=non_goals, init_split_quantity=all, use_wildcard_plans=false, combine_labels=true, flaw_kind=sequence_bidirectional, collection_strategy=complementary, random_seed=7, debug=true))",
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
    assert_eq!(cfg.collection_strategy, CollectionStrategy::Complementary);
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
fn parses_explicit_offline_scp() {
    let h = astar_heuristic("astar(scp_online(online=false))");
    let mut config = ScpOnlineConfig::default();
    ApplyOptions::apply_options(&mut config, &h.args).unwrap();
    assert!(!config.online);
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
fn rejects_removed_exec_entire_plan_randomize_option() {
    let h = astar_heuristic("astar(multi_domain_abstractions(exec_entire_plan=randomize))");
    let err = ApplyOptions::apply_options(
        &mut DomainAbstractionCollectionGeneratorMultipleCegarConfig::default(),
        &h.args,
    )
    .unwrap_err();
    assert!(err.contains("exec_entire_plan"));
}
