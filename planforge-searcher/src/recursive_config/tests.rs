use super::*;

fn astar_heuristic(input: &str) -> HeuristicSpec {
    match parse_search_spec(input).unwrap() {
        SearchSpec::Astar(heuristic, _) => heuristic,
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
fn parses_and_round_trips_astar_mpd() {
    let parsed = parse_search_spec("astar(blind(), mpd=true)").unwrap();
    assert!(matches!(&parsed, SearchSpec::Astar(_, true)));
    assert_eq!(parsed.to_string(), "astar(blind(), mpd=true)");
    assert_eq!(parse_search_spec(&parsed.to_string()).unwrap(), parsed);

    let default = parse_search_spec("astar(blind())").unwrap();
    assert!(matches!(default, SearchSpec::Astar(_, false)));
}

#[test]
fn rejects_invalid_or_duplicate_astar_mpd() {
    let error = parse_search_spec("astar(blind(), mpd=yes)").unwrap_err();
    assert!(error.contains("expects true or false"), "got `{error}`");

    let error = parse_search_spec("astar(blind(), mpd=true, mpd=false)").unwrap_err();
    assert!(error.contains("duplicate option `mpd`"), "got `{error}`");
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
fn parses_astar_canonical_domain_abstractions_with_or_without_parens() {
    let h = astar_heuristic("astar(canonical_domain_abstractions)");
    assert_eq!(h.name, "canonical_domain_abstractions");
    assert!(h.args.is_empty());

    let h = astar_heuristic("astar(canonical_domain_abstractions())");
    assert_eq!(h.name, "canonical_domain_abstractions");
    assert!(h.args.is_empty());
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
fn parses_astar_scp_online_with_or_without_unit_parens() {
    let h = astar_heuristic("astar(scp_online)");
    assert_eq!(h.name, "scp_online");
    assert!(h.args.is_empty());

    let h = astar_heuristic("astar(scp_online())");
    assert_eq!(h.name, "scp_online");
    assert!(h.args.is_empty());
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
fn parses_astar_canonical_numeric_pdb_with_or_without_unit_parens() {
    let h = astar_heuristic("astar(canonical_numeric_pdb)");
    assert_eq!(h.name, "canonical_numeric_pdb");
    assert!(h.args.is_empty());

    let h = astar_heuristic("astar(canonical_numeric_pdb())");
    assert_eq!(h.name, "canonical_numeric_pdb");
    assert!(h.args.is_empty());
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
fn parses_astar_multi_domain_abstractions_with_or_without_parens() {
    let h = astar_heuristic("astar(multi_domain_abstractions)");
    assert_eq!(h.name, "multi_domain_abstractions");
    assert!(h.args.is_empty());

    let h = astar_heuristic("astar(multi_domain_abstractions())");
    assert_eq!(h.name, "multi_domain_abstractions");
    assert!(h.args.is_empty());
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
        "astar(scp_online(max_time=12.5, max_abstraction_size=42, abstraction_generation_max_time=infinity, partitioning=region, saturator=perimstar, scoring_function=min_stolen_costs, orders=random_orders, order_optimization_max_time=0.25))",
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
}

#[test]
fn parses_check_admissible_around_a_nested_heuristic() {
    let spec = astar_heuristic("astar(check_admissible(domain_abstraction()));");
    assert_eq!(spec.name, "check_admissible");
    assert_eq!(spec.args.len(), 1);
    let inner = HeuristicSpec::from_value(spec.args[0].value());
    assert_eq!(inner.name, "domain_abstraction");
    assert!(spec.contains_call("domain_abstraction"));
}

#[test]
fn check_admissible_round_trips_through_display() {
    let spec = parse_search_spec("astar(check_admissible(blind()))").unwrap();
    // `parse_value` normalizes a zero-argument call into a bare name, so the
    // round-trip is over the spec, not over the exact input text.
    assert_eq!(spec.to_string(), "astar(check_admissible(blind))");
    assert_eq!(parse_search_spec(&spec.to_string()).unwrap(), spec);

    let with_options =
        parse_search_spec("astar(check_admissible(domain_abstraction(max_time=1.5)))").unwrap();
    assert_eq!(
        with_options.to_string(),
        "astar(check_admissible(domain_abstraction(max_time=1.5)))"
    );
    assert_eq!(
        parse_search_spec(&with_options.to_string()).unwrap(),
        with_options
    );
}

#[test]
fn debug_search_engines_are_gone() {
    for removed in ["da_debug()", "astar_da_debug()"] {
        let error = parse_search_spec(removed).unwrap_err();
        assert!(error.contains("unknown search engine"), "{error}");
    }
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

#[test]
fn contains_call_finds_only_nested_icaps_cartesian_sources() {
    let icaps = parse_search_spec(
        "astar(scp(cartesian_collection(source=icaps26_cartesian(pick=min_unwanted))))",
    )
    .unwrap();
    assert!(icaps.contains_call("icaps26_cartesian"));

    let native =
        parse_search_spec("astar(scp(cartesian_collection(source=cartesian(max_states=1000))))")
            .unwrap();
    assert!(!native.contains_call("icaps26_cartesian"));
}

// =============================================================================
// `sgd(...)`
// =============================================================================

/// The spec must survive a round trip through `Display`, because `planforge`
/// re-execs itself and passes `--search` back as a string. A variant that
/// printed its own name twice would silently produce `sgd(sgd(...))` and fail
/// only in the child process.
#[test]
fn sgd_spec_round_trips_through_display() {
    for raw in [
        "sgd()",
        "sgd(horizon=12)",
        "sgd(horizon=dovetail)",
        "sgd(horizon=dovetail(8, 2.0, 512))",
        "sgd(horizon=12, particles=8, seed=7, refresh=true)",
    ] {
        let spec = parse_search_spec(raw).expect("spec parses");
        let printed = spec.to_string();
        let reparsed = parse_search_spec(&printed).expect("printed spec re-parses");
        assert_eq!(
            spec, reparsed,
            "round trip changed the spec: {raw} -> {printed}"
        );
        assert!(
            printed.starts_with("sgd("),
            "printed spec should start with `sgd(`, got {printed}"
        );
        assert!(
            !printed.contains("sgd(sgd("),
            "the spec printed its name twice: {printed}"
        );
    }
}

#[test]
fn sgd_keeps_its_arguments_in_order() {
    let spec = parse_search_spec("sgd(horizon=12, particles=4)").expect("parses");
    match spec {
        SearchSpec::Sgd(args) => {
            assert_eq!(args.len(), 2);
            assert_eq!(args[0].key(), Some("horizon"));
            assert_eq!(args[1].key(), Some("particles"));
        }
        other => panic!("expected an sgd spec, got {other:?}"),
    }
}

#[test]
fn sgd_never_reports_a_nested_heuristic() {
    // The engine is not allowed a heuristic, so `contains_call` must not claim
    // one however the arguments are written.
    let spec = parse_search_spec("sgd(horizon=12)").expect("parses");
    assert!(!spec.contains_call("ff"));
    assert!(!spec.contains_call("icaps26_cartesian"));
}

#[test]
fn display_round_trips_enum_valued_options() {
    for spec in [
        "astar(canonical_domain_abstractions(flaw_kind=execute_entire_plan))",
        "astar(canonical_domain_abstractions(split_direction=forward_partition_deviation))",
    ] {
        let parsed = parse_search_spec(spec).unwrap();
        assert_eq!(parse_search_spec(&parsed.to_string()).unwrap(), parsed);
    }
}

#[test]
fn parses_registry_style_search_with_keyed_heuristic() {
    let spec = parse_search_spec("search(astar(heuristic=greedy_numeric_pdb(max_pdb_states=321)))")
        .unwrap();
    let SearchSpec::Astar(heuristic, false) = &spec else {
        panic!("expected astar(...), got {spec:?}");
    };
    assert_eq!(heuristic.name, "greedy_numeric_pdb");
    assert_eq!(heuristic.args.len(), 1);
}
