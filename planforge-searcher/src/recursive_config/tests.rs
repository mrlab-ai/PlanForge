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
fn parses_hierarchical_canonical_abstraction_sources() {
    let h = astar_heuristic(
        "astar(canonical(domain_collection(max_states=100), cartesian(max_states=100), pdb(max_pdb_states=100)))",
    );
    assert_eq!(h.name, "canonical");
    let source_names: Vec<_> = h
        .args
        .iter()
        .map(|arg| arg.value().as_call().unwrap().name())
        .collect();
    assert_eq!(source_names, ["domain_collection", "cartesian", "pdb"]);
}

#[test]
fn parses_and_round_trips_collection_source_lists() {
    let spec = parse_search_spec(
        "astar(scp([domain_collection(max_states=100), cartesian(), pdb(),], partitioning=region))",
    )
    .unwrap();
    assert_eq!(parse_search_spec(&spec.to_string()).unwrap(), spec);
}

#[test]
fn rejects_nested_collection_source_lists() {
    let error = parse_search_spec("astar(scp([[domain_collection()]]))").unwrap_err();
    assert!(
        error.contains("nested lists are not supported"),
        "got `{error}`"
    );
}

#[test]
fn parses_astar_scp_with_or_without_unit_parens() {
    let h = astar_heuristic("astar(scp)");
    assert_eq!(h.name, "scp");
    assert!(h.args.is_empty());

    let h = astar_heuristic("astar(scp())");
    assert_eq!(h.name, "scp");
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
fn display_round_trips_max_domain_collection() {
    let parsed =
        parse_search_spec("astar(max(domain_collection(max_states=42, max_time=infinity)))")
            .unwrap();
    let reparsed = parse_search_spec(&parsed.to_string()).unwrap();
    assert_eq!(parsed, reparsed);
}

#[test]
fn display_round_trips_canonical_domain_collection() {
    let parsed =
        parse_search_spec("astar(canonical(domain_collection(max_states=42, max_time=infinity)))")
            .unwrap();
    let reparsed = parse_search_spec(&parsed.to_string()).unwrap();
    assert_eq!(parsed, reparsed);
}

#[test]
fn display_round_trips_hierarchical_abstraction_collection() {
    let parsed = parse_search_spec(
        "astar(canonical(domain_collection(max_states=42), cartesian(max_states=42), pdb(max_pdb_states=42)))",
    )
    .unwrap();
    let reparsed = parse_search_spec(&parsed.to_string()).unwrap();
    assert_eq!(parsed, reparsed);
}

#[test]
fn display_round_trips_scp_options() {
    let parsed = parse_search_spec(
        "astar(scp([domain_collection(max_states=42, max_time=infinity)], max_time=12.5, partitioning=region, saturator=perimstar, orders=diverse_orders, max_orders=3))",
    )
    .unwrap();
    let reparsed = parse_search_spec(&parsed.to_string()).unwrap();
    assert_eq!(parsed, reparsed);
}

#[test]
fn display_round_trips_scp_with_nested_collection() {
    let parsed = parse_search_spec(
        "astar(scp(max(domain_collection(max_collection_size=99, total_max_time=2.5)), saturator=perimstar))",
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
        astar_heuristic("astar(max(domain_collection()));").name,
        "max"
    );
    assert_eq!(
        astar_heuristic("astar(canonical(domain_collection()));").name,
        "canonical"
    );
}

#[test]
fn parses_check_admissible_around_a_nested_heuristic() {
    let spec = astar_heuristic("astar(check_admissible(domain_abstraction()));");
    assert_eq!(spec.name, "check_admissible");
    assert_eq!(spec.args.len(), 1);
    let inner = HeuristicSpec::from_value(spec.args[0].value()).unwrap();
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
fn unknown_heuristic_name_is_rejected_from_the_registry() {
    let error = parse_search_spec("astar(does_not_exist)").unwrap_err();
    assert!(error.contains("unknown heuristic `does_not_exist`"));
    assert!(error.contains("blind"));
    assert!(error.contains("lmcutnumeric"));
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

#[test]
fn contains_call_recurses_into_source_lists() {
    let spec = parse_search_spec("astar(scp([domain_collection(), pdb()]))").unwrap();
    assert!(spec.contains_call("pdb"));
    assert!(!spec.contains_call("numeric_pdb"));
}

#[test]
fn lists_are_typed_errors_where_heuristics_are_expected() {
    let error = parse_search_spec("astar([blind()])").unwrap_err();
    assert_eq!(error, "expected heuristic, got a list");
}

#[test]
fn display_round_trips_enum_valued_options() {
    for spec in [
        "astar(canonical(domain_collection(flaw_scope=whole_plan)))",
        "astar(canonical(domain_collection(split_kind=wanted)))",
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
