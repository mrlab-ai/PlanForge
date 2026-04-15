use super::*;
use planners_search::numeric::evaluation::domain_abstractions::domain_abstraction_collection_generator_multiple_cegar::{
    DomainAbstractionCollectionGeneratorMultipleCegarConfig, ExecEntirePlanMode,
    InitSplitQuantity, VariableSubset,
};
use planners_search::numeric::evaluation::numeric_landmarks::lm_cut_numeric_heuristic::LmCutNumericConfig;
use planners_search::numeric::evaluation::pattern_databases::canonical_pdb_heuristic::CanonicalNumericPdbConfig;
use planners_search::numeric::evaluation::pattern_databases::pattern_database::PdbInternalHeuristic;
use planners_search::numeric::evaluation::pattern_databases::pattern_generator_greedy::GreedyPatternGeneratorConfig;
use planners_search::numeric::evaluation::pattern_databases::variable_order_finder::GreedyVariableOrderType;

#[test]
fn parses_astar_blind_with_or_without_unit_parens() {
    assert_eq!(
        parse_search_spec("astar(blind)").unwrap(),
        SearchSpec::Astar(HeuristicSpec::Blind)
    );
    assert_eq!(
        parse_search_spec("astar(blind())").unwrap(),
        SearchSpec::Astar(HeuristicSpec::Blind)
    );
}

#[test]
fn parses_astar_domain_abstraction_with_or_without_unit_parens() {
    assert_eq!(
        parse_search_spec("astar(domain_abstraction)").unwrap(),
        SearchSpec::Astar(HeuristicSpec::DomainAbstraction(
            DomainAbstractionConfig::default()
        ))
    );
    assert_eq!(
        parse_search_spec("astar(domain_abstraction())").unwrap(),
        SearchSpec::Astar(HeuristicSpec::DomainAbstraction(
            DomainAbstractionConfig::default()
        ))
    );
}

#[test]
fn parses_astar_domain_abstraction_with_named_options() {
    let spec = parse_search_spec(
        "astar(domain_abstraction(max_abstraction_size=10000, use_wildcard_plans=false, combine_labels=true, random_seed=7, exec_entire_plan=execute_entire_plan))",
    )
    .unwrap();

    let SearchSpec::Astar(HeuristicSpec::DomainAbstraction(config)) = spec else {
        panic!("expected domain_abstraction config");
    };

    assert_eq!(config.max_abstraction_size, 10_000);
    assert!(!config.use_wildcard_plans);
    assert!(config.combine_labels);
    assert_eq!(config.random_seed, 7);
    assert_eq!(config.exec_entire_plan, ExecEntirePlanMode::ExecuteEntirePlan);
}

#[test]
fn parses_astar_canonical_domain_abstractions_with_or_without_parens() {
    assert_eq!(
        parse_search_spec("astar(canonical_domain_abstractions)").unwrap(),
        SearchSpec::Astar(HeuristicSpec::CanonicalDomainAbstractions(
            DomainAbstractionCollectionGeneratorMultipleCegarConfig::default()
        ))
    );
    assert_eq!(
        parse_search_spec("astar(canonical_domain_abstractions())").unwrap(),
        SearchSpec::Astar(HeuristicSpec::CanonicalDomainAbstractions(
            DomainAbstractionCollectionGeneratorMultipleCegarConfig::default()
        ))
    );
}

#[test]
fn parses_astar_canonical_domain_abstractions_with_named_options() {
    let spec = parse_search_spec(
        "astar(canonical_domain_abstractions(max_collection_size=123, total_max_time=4.5, blacklist_option=non_goals, init_split_quantity=all, exec_entire_plan=execute_entire_plan, use_wildcard_plans=false, combine_labels=true, random_seed=7))",
    )
    .unwrap();

    let SearchSpec::Astar(HeuristicSpec::CanonicalDomainAbstractions(config)) = spec else {
        panic!("expected canonical_domain_abstractions config");
    };

    assert_eq!(config.max_collection_size, 123);
    assert_eq!(config.total_max_time, 4.5);
    assert_eq!(config.blacklist_option, VariableSubset::NonGoals);
    assert_eq!(config.init_split_quantity, InitSplitQuantity::All);
    assert_eq!(
        config.exec_entire_plan,
        ExecEntirePlanMode::ExecuteEntirePlan
    );
    assert!(!config.use_wildcard_plans);
    assert!(config.combine_labels);
    assert_eq!(config.random_seed, 7);
}

#[test]
fn parses_astar_greedy_numeric_pdb_with_or_without_unit_parens() {
    assert_eq!(
        parse_search_spec("astar(greedy_numeric_pdb)").unwrap(),
        SearchSpec::Astar(HeuristicSpec::GreedyNumericPdb(
            GreedyPatternGeneratorConfig::default()
        ))
    );
    assert_eq!(
        parse_search_spec("astar(greedy_numeric_pdb())").unwrap(),
        SearchSpec::Astar(HeuristicSpec::GreedyNumericPdb(
            GreedyPatternGeneratorConfig::default()
        ))
    );
}

#[test]
fn parses_astar_greedy_numeric_pdb_with_named_options() {
    let spec = parse_search_spec(
        "astar(greedy_numeric_pdb(max_pdb_states=321, numeric_first=false, random_seed=7, variable_order_type=cg_goal_random, exploration_heuristic=lmcut, frontier_heuristic=blind, failed_lookup_heuristic=lmcut))",
    )
    .unwrap();

    let SearchSpec::Astar(HeuristicSpec::GreedyNumericPdb(config)) = spec else {
        panic!("expected greedy_numeric_pdb config");
    };

    assert_eq!(config.max_pdb_states, 321);
    assert!(!config.numeric_first);
    assert_eq!(config.random_seed, 7);
    assert_eq!(
        config.variable_order_type,
        GreedyVariableOrderType::CgGoalRandom
    );
    assert_eq!(config.exploration_heuristic, PdbInternalHeuristic::Lmcut);
    assert_eq!(config.frontier_heuristic, PdbInternalHeuristic::Blind);
    assert_eq!(config.failed_lookup_heuristic, PdbInternalHeuristic::Lmcut);
}

#[test]
fn parses_astar_canonical_numeric_pdb_with_or_without_unit_parens() {
    assert_eq!(
        parse_search_spec("astar(canonical_numeric_pdb)").unwrap(),
        SearchSpec::Astar(HeuristicSpec::CanonicalNumericPdb(
            CanonicalNumericPdbConfig::default()
        ))
    );
    assert_eq!(
        parse_search_spec("astar(canonical_numeric_pdb())").unwrap(),
        SearchSpec::Astar(HeuristicSpec::CanonicalNumericPdb(
            CanonicalNumericPdbConfig::default()
        ))
    );
}

#[test]
fn parses_astar_canonical_numeric_pdb_with_named_options() {
    let spec = parse_search_spec(
        "astar(canonical_numeric_pdb(max_pdb_states=321, max_pattern_size=3, only_interesting_patterns=false, random_seed=7, variable_order_type=cg_goal_random, exploration_heuristic=blind, frontier_heuristic=lmcut, failed_lookup_heuristic=lmcut))",
    )
    .unwrap();

    let SearchSpec::Astar(HeuristicSpec::CanonicalNumericPdb(config)) = spec else {
        panic!("expected canonical_numeric_pdb config");
    };

    assert_eq!(config.max_pdb_states, 321);
    assert_eq!(config.max_pattern_size, 3);
    assert!(!config.only_interesting_patterns);
    assert_eq!(config.random_seed, 7);
    assert_eq!(
        config.variable_order_type,
        GreedyVariableOrderType::CgGoalRandom
    );
    assert_eq!(config.exploration_heuristic, PdbInternalHeuristic::Blind);
    assert_eq!(config.frontier_heuristic, PdbInternalHeuristic::Lmcut);
    assert_eq!(config.failed_lookup_heuristic, PdbInternalHeuristic::Lmcut);
}

#[test]
fn parses_astar_lmcutnumeric_with_or_without_unit_parens() {
    assert_eq!(
        parse_search_spec("astar(lmcutnumeric)").unwrap(),
        SearchSpec::Astar(HeuristicSpec::Lmcutnumeric(LmCutNumericConfig::default()))
    );
    assert_eq!(
        parse_search_spec("astar(lmcutnumeric())").unwrap(),
        SearchSpec::Astar(HeuristicSpec::Lmcutnumeric(LmCutNumericConfig::default()))
    );
}

#[test]
fn parses_astar_lmcutnumeric_with_named_options() {
    let spec = parse_search_spec(
        "astar(lmcutnumeric(ceiling_less_than_one=true, ignore_numeric=true, random_pcf=true, irmax=true, disable_ma=true, use_second_order_simple=true, use_constant_assignment=true, bound_iterations=7, precision=0.5, epsilon=0.25))",
    )
    .unwrap();

    let SearchSpec::Astar(HeuristicSpec::Lmcutnumeric(config)) = spec else {
        panic!("expected lmcutnumeric config");
    };

    assert!(config.ceiling_less_than_one);
    assert!(config.ignore_numeric);
    assert!(config.random_pcf);
    assert!(config.irmax);
    assert!(config.disable_ma);
    assert!(config.use_second_order_simple);
    assert!(config.use_constant_assignment);
    assert_eq!(config.bound_iterations, 7);
    assert_eq!(config.precision, 0.5);
    assert_eq!(config.epsilon, 0.25);
}

#[test]
fn parses_astar_multi_domain_abstractions_with_or_without_parens() {
    assert_eq!(
        parse_search_spec("astar(multi_domain_abstractions)").unwrap(),
        SearchSpec::Astar(HeuristicSpec::MultiDomainAbstractions(
            DomainAbstractionCollectionGeneratorMultipleCegarConfig::default()
        ))
    );
    assert_eq!(
        parse_search_spec("astar(multi_domain_abstractions())").unwrap(),
        SearchSpec::Astar(HeuristicSpec::MultiDomainAbstractions(
            DomainAbstractionCollectionGeneratorMultipleCegarConfig::default()
        ))
    );
}

#[test]
fn parses_astar_multi_domain_abstractions_with_named_options() {
    let spec = parse_search_spec(
        "astar(multi_domain_abstractions(max_collection_size=123, total_max_time=4.5, blacklist_option=non_goals, init_split_quantity=all, exec_entire_plan=execute_entire_plan, use_wildcard_plans=false, combine_labels=true, random_seed=7))",
    )
    .unwrap();

    let SearchSpec::Astar(HeuristicSpec::MultiDomainAbstractions(config)) = spec else {
        panic!("expected multi_domain_abstractions config");
    };

    assert_eq!(config.max_collection_size, 123);
    assert_eq!(config.total_max_time, 4.5);
    assert_eq!(config.blacklist_option, VariableSubset::NonGoals);
    assert_eq!(config.init_split_quantity, InitSplitQuantity::All);
    assert_eq!(
        config.exec_entire_plan,
        ExecEntirePlanMode::ExecuteEntirePlan
    );
    assert!(!config.use_wildcard_plans);
    assert!(config.combine_labels);
    assert_eq!(config.random_seed, 7);
}

#[test]
fn parses_astar_multi_domain_abstractions_with_trailing_comma() {
    let spec = parse_search_spec(
        "astar(multi_domain_abstractions(max_collection_size=123, exec_entire_plan=stop_at_first_flaw,))",
    )
    .unwrap();

    let SearchSpec::Astar(HeuristicSpec::MultiDomainAbstractions(config)) = spec else {
        panic!("expected multi_domain_abstractions config");
    };

    assert_eq!(config.max_collection_size, 123);
    assert_eq!(config.exec_entire_plan, ExecEntirePlanMode::StopAtFirstFlaw);
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
        "astar(canonical_numeric_pdb(max_pdb_states=42, max_pattern_size=3, only_interesting_patterns=false, random_seed=9, variable_order_type=cg_goal_random, exploration_heuristic=blind, frontier_heuristic=lmcut, failed_lookup_heuristic=lmcut))",
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
    assert!(
        parse_search_spec("astar(multi_domain_abstractions(exec_entire_plan=randomize))",).is_err()
    );
}

#[test]
fn trims_trailing_punctuation() {
    assert_eq!(
        parse_search_spec("astar(blind()).").unwrap(),
        SearchSpec::Astar(HeuristicSpec::Blind)
    );

    assert_eq!(
        parse_search_spec("astar(domain_abstraction());").unwrap(),
        SearchSpec::Astar(HeuristicSpec::DomainAbstraction(
            DomainAbstractionConfig::default()
        ))
    );

    assert_eq!(
        parse_search_spec("astar(greedy_numeric_pdb());").unwrap(),
        SearchSpec::Astar(HeuristicSpec::GreedyNumericPdb(
            GreedyPatternGeneratorConfig::default()
        ))
    );

    assert_eq!(
        parse_search_spec("astar(canonical_numeric_pdb());").unwrap(),
        SearchSpec::Astar(HeuristicSpec::CanonicalNumericPdb(
            CanonicalNumericPdbConfig::default()
        ))
    );

    assert_eq!(
        parse_search_spec("astar(lmcutnumeric());").unwrap(),
        SearchSpec::Astar(HeuristicSpec::Lmcutnumeric(LmCutNumericConfig::default()))
    );

    assert!(matches!(
        parse_search_spec("astar(multi_domain_abstractions());").unwrap(),
        SearchSpec::Astar(HeuristicSpec::MultiDomainAbstractions(_))
    ));

    assert!(matches!(
        parse_search_spec("astar(canonical_domain_abstractions());").unwrap(),
        SearchSpec::Astar(HeuristicSpec::CanonicalDomainAbstractions(_))
    ));

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
