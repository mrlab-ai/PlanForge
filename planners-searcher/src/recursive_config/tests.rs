use super::*;
use planners_search::numeric::evaluation::domain_abstractions::domain_abstraction_collection_generator_multiple_cegar::{
    DomainAbstractionCollectionGeneratorMultipleCegarConfig, ExecEntirePlanMode,
    InitSplitQuantity, VariableSubset,
};
use planners_search::numeric::evaluation::pattern_databases::canonical_pdb_heuristic::CanonicalNumericPdbConfig;
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
        SearchSpec::Astar(HeuristicSpec::DomainAbstraction)
    );
    assert_eq!(
        parse_search_spec("astar(domain_abstraction())").unwrap(),
        SearchSpec::Astar(HeuristicSpec::DomainAbstraction)
    );
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
        "astar(greedy_numeric_pdb(max_pdb_states=321, numeric_first=false, random_seed=7, variable_order_type=reverse_level))",
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
        GreedyVariableOrderType::ReverseLevel
    );
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
        "astar(canonical_numeric_pdb(max_pdb_states=321, max_pattern_size=3, random_seed=7, variable_order_type=reverse_level))",
    )
    .unwrap();

    let SearchSpec::Astar(HeuristicSpec::CanonicalNumericPdb(config)) = spec else {
        panic!("expected canonical_numeric_pdb config");
    };

    assert_eq!(config.max_pdb_states, 321);
    assert_eq!(config.max_pattern_size, 3);
    assert_eq!(config.random_seed, 7);
    assert_eq!(
        config.variable_order_type,
        GreedyVariableOrderType::ReverseLevel
    );
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
        "astar(multi_domain_abstractions(max_collection_size=123, total_max_time=4.5, blacklist_option=non_goals, init_split_quantity=all, exec_entire_plan=execute_entire_plan, use_wildcard_plans=false, random_seed=7))",
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
fn display_round_trips_greedy_numeric_pdb() {
    let parsed = parse_search_spec(
        "astar(greedy_numeric_pdb(max_pdb_states=42, numeric_first=false, random_seed=9, variable_order_type=random))",
    )
    .unwrap();
    let reparsed = parse_search_spec(&parsed.to_string()).unwrap();
    assert_eq!(parsed, reparsed);
}

#[test]
fn display_round_trips_canonical_numeric_pdb() {
    let parsed = parse_search_spec(
        "astar(canonical_numeric_pdb(max_pdb_states=42, max_pattern_size=3, random_seed=9, variable_order_type=random))",
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
        SearchSpec::Astar(HeuristicSpec::DomainAbstraction)
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

    assert!(matches!(
        parse_search_spec("astar(multi_domain_abstractions());").unwrap(),
        SearchSpec::Astar(HeuristicSpec::MultiDomainAbstractions(_))
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
