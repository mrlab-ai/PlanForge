//! The search configuration grammar over this crate's heuristics.

use planforge_search::config::*;
use planforge_search::search::*;

fn astar_heuristic(input: &str) -> ConfigCall {
    let parsed = planforge_heuristics::features()
        .parse_search(input)
        .unwrap();
    let config = parsed
        .config::<AStarConfig>()
        .unwrap_or_else(|| panic!("expected astar(...), got {parsed:?}"));
    parse_call(&config.heuristic.to_string()).unwrap()
}

#[test]
fn parses_heuristic_spec_ff_call() {
    let h = parse_call("ff()").unwrap();
    assert_eq!(h.name, "ff");
    assert!(h.args.is_empty());
}

#[test]
fn parses_heuristic_spec_blind_bare_identifier() {
    let h = parse_call("blind").unwrap();
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
    let features = planforge_heuristics::features();
    let parsed = features.parse_search("astar(blind(), mpd=true)").unwrap();
    assert!(parsed.config::<AStarConfig>().unwrap().mpd);
    assert_eq!(parsed.to_string(), "astar(blind(), mpd=true)");
    assert_eq!(features.parse_search(&parsed.to_string()).unwrap(), parsed);

    let default = features.parse_search("astar(blind())").unwrap();
    assert!(!default.config::<AStarConfig>().unwrap().mpd);
}

#[test]
fn rejects_invalid_or_duplicate_astar_mpd() {
    let features = planforge_heuristics::features();
    let error = features
        .parse_search("astar(blind(), mpd=yes)")
        .unwrap_err()
        .to_string();
    assert_eq!(error, "astar.mpd: expected boolean, got `yes`");

    let error = features
        .parse_search("astar(blind(), mpd=true, mpd=false)")
        .unwrap_err()
        .to_string();
    assert!(error.contains("duplicate option `mpd`"), "got `{error}`");
}

#[test]
fn parses_hierarchical_canonical_abstraction_sources() {
    let h = astar_heuristic(
        "astar(canonical([domain_collection(max_states=100), cartesian(max_states=100), pdb(patterns=systematic(max_pdb_states=100))]))",
    );
    assert_eq!(h.name, "canonical");
    let [arg] = &h.args[..] else {
        panic!("one source list");
    };
    let ConfigValue::List(sources) = arg.value() else {
        panic!("a source list");
    };
    let source_names: Vec<_> = sources
        .iter()
        .map(|source| source.as_call().unwrap().name())
        .collect();
    assert_eq!(source_names, ["domain_collection", "cartesian", "pdb"]);
}

#[test]
fn parses_and_round_trips_collection_source_lists() {
    let features = planforge_heuristics::features();
    let spec = features.parse_search(
        "astar(scp([domain_collection(max_states=100), cartesian(), pdb(),], partitioning=region))",
    )
    .unwrap();
    assert_eq!(features.parse_search(&spec.to_string()).unwrap(), spec);
}

#[test]
fn rejects_nested_collection_source_lists() {
    let features = planforge_heuristics::features();
    let error = features
        .parse_search("astar(scp([[domain_collection()]]))")
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("expected AbstractionSource, got a list"),
        "got `{error}`"
    );
}

#[test]
fn rejects_scp_without_abstraction_sources() {
    let features = planforge_heuristics::features();
    assert!(features.parse_search("astar(scp)").is_err());
    assert!(features.parse_search("astar(scp())").is_err());
}

#[test]
fn display_round_trips_max_domain_collection() {
    let features = planforge_heuristics::features();
    let parsed = features
        .parse_search("astar(max(domain_collection(max_states=42, max_time=infinity)))")
        .unwrap();
    let reparsed = features.parse_search(&parsed.to_string()).unwrap();
    assert_eq!(parsed, reparsed);
}

#[test]
fn display_round_trips_canonical_domain_collection() {
    let features = planforge_heuristics::features();
    let parsed = features
        .parse_search("astar(canonical(domain_collection(max_states=42, max_time=infinity)))")
        .unwrap();
    let reparsed = features.parse_search(&parsed.to_string()).unwrap();
    assert_eq!(parsed, reparsed);
}

#[test]
fn display_round_trips_hierarchical_abstraction_collection() {
    let features = planforge_heuristics::features();
    let parsed = features.parse_search(
        "astar(canonical([domain_collection(max_states=42), cartesian(max_states=42), pdb(patterns=systematic(max_pdb_states=42))]))",
    )
    .unwrap();
    let reparsed = features.parse_search(&parsed.to_string()).unwrap();
    assert_eq!(parsed, reparsed);
}

#[test]
fn display_round_trips_scp_options() {
    let features = planforge_heuristics::features();
    let parsed = features.parse_search(
        "astar(scp([domain_collection(max_states=42, max_time=infinity)], max_time=12.5, partitioning=region, saturator=perimstar, orders=diverse_orders, diversify=true, max_orders=3))",
    )
    .unwrap();
    let reparsed = features.parse_search(&parsed.to_string()).unwrap();
    assert_eq!(parsed, reparsed);
}

#[test]
fn rejects_heuristic_combinators_in_abstraction_source_positions() {
    let features = planforge_heuristics::features();
    let error = features
        .parse_search("astar(scp(max(domain_collection()),saturator=perimstar))")
        .unwrap_err()
        .to_string();
    assert!(error.contains("unknown AbstractionSource `max`"), "{error}");
}

#[test]
fn display_round_trips_positional_args() {
    let features = planforge_heuristics::features();
    let parsed = features
        .parse_search("astar(pdb(patterns=greedy(321, false, 7)))")
        .unwrap();
    let reparsed = features.parse_search(&parsed.to_string()).unwrap();
    assert_eq!(parsed, reparsed);
}

#[test]
fn display_round_trips_greedy_pdb() {
    let features = planforge_heuristics::features();
    let parsed = features.parse_search(
        "astar(pdb(patterns=greedy(max_pdb_states=42, numeric_first=false, random_seed=9, variable_order_type=cg_goal_random), exploration_heuristic=lmcut, frontier_heuristic=blind, failed_lookup_heuristic=lmcut))",
    )
    .unwrap();
    let reparsed = features.parse_search(&parsed.to_string()).unwrap();
    assert_eq!(parsed, reparsed);
}

#[test]
fn display_round_trips_canonical_systematic_pdb() {
    let features = planforge_heuristics::features();
    let parsed = features.parse_search(
        "astar(canonical(pdb(patterns=systematic(max_pdb_states=42, max_pattern_size=3), exploration_heuristic=blind, frontier_heuristic=lmcut, failed_lookup_heuristic=lmcut)))",
    )
    .unwrap();
    let reparsed = features.parse_search(&parsed.to_string()).unwrap();
    assert_eq!(parsed, reparsed);
}

#[test]
fn display_round_trips_lmcutnumeric() {
    let features = planforge_heuristics::features();
    let parsed = features.parse_search(
        "astar(lmcutnumeric(ceiling_less_than_one=true, disable_ma=true, bound_iterations=4, precision=0.5, epsilon=0.25))",
    )
    .unwrap();
    let reparsed = features.parse_search(&parsed.to_string()).unwrap();
    assert_eq!(parsed, reparsed);
}

#[test]
fn parses_check_admissible_around_a_nested_heuristic() {
    let spec = astar_heuristic("astar(check_admissible(domain()))");
    assert_eq!(spec.name, "check_admissible");
    assert_eq!(spec.args.len(), 1);
    let inner = ConfigCall::from_value(spec.args[0].value()).unwrap();
    assert_eq!(inner.name, "domain");
}

#[test]
fn check_admissible_round_trips_through_display() {
    let features = planforge_heuristics::features();
    let spec = features
        .parse_search("astar(check_admissible(blind()))")
        .unwrap();
    // `parse_value` normalizes a zero-argument call into a bare name, so the
    // round-trip is over the spec, not over the exact input text.
    assert_eq!(spec.to_string(), "astar(check_admissible(blind()))");
    assert_eq!(features.parse_search(&spec.to_string()).unwrap(), spec);

    let with_options = features
        .parse_search("astar(check_admissible(domain(max_time=1.5)))")
        .unwrap();
    assert_eq!(
        with_options.to_string(),
        "astar(check_admissible(domain(max_time=1.5)))"
    );
    assert_eq!(
        features.parse_search(&with_options.to_string()).unwrap(),
        with_options
    );
}

#[test]
fn errors_are_human_readable() {
    let features = planforge_heuristics::features();
    let err = features.parse_search("astar(").unwrap_err().to_string();
    assert!(err.to_lowercase().contains("invalid"));
}

#[test]
fn unknown_heuristic_name_is_rejected_from_the_registry() {
    let features = planforge_heuristics::features();
    let error = features
        .parse_search("astar(does_not_exist)")
        .unwrap_err()
        .to_string();
    assert!(error.contains("unknown heuristic `does_not_exist`"));
    assert!(error.contains("blind"));
    assert!(error.contains("lmcutnumeric"));
}

#[test]
fn lists_are_typed_errors_where_heuristics_are_expected() {
    let features = planforge_heuristics::features();
    let error = features
        .parse_search("astar([blind()])")
        .unwrap_err()
        .to_string();
    assert_eq!(error, "astar.heuristic: expected heuristic, got a list");
}

#[test]
fn display_round_trips_enum_valued_options() {
    let features = planforge_heuristics::features();
    for spec in [
        "astar(canonical(domain_collection(flaw_scope=whole_plan)))",
        "astar(canonical(domain_collection(split_kind=wanted)))",
    ] {
        let parsed = features.parse_search(spec).unwrap();
        assert_eq!(features.parse_search(&parsed.to_string()).unwrap(), parsed);
    }
}

#[test]
fn lazy_astar_round_trips_and_none_disables_it() {
    let features = planforge_heuristics::features();
    let astar = features
        .parse_search("astar(blind(), lazy=lmcutnumeric())")
        .unwrap();
    assert_eq!(features.parse_search(&astar.to_string()).unwrap(), astar);
    assert_eq!(
        features.parse_search("astar(blind(), lazy=none)").unwrap(),
        features.parse_search("astar(blind())").unwrap()
    );
}

#[test]
fn gbfs_has_no_mpd_or_lazy_options() {
    let features = planforge_heuristics::features();
    assert!(features.parse_search("gbfs(blind(), mpd=true)").is_err());
    assert!(features.parse_search("gbfs(blind(), lazy=ff())").is_err());
    assert!(
        features
            .parse_search("astar(blind(), lazy=none, lazy=ff())")
            .is_err()
    );
}

#[test]
fn task_requirements_follow_typed_sources_and_wrappers() {
    let features = planforge_heuristics::features();
    let spec = features
        .parse_search("astar(check_admissible(scp([domain(),pdb()])))")
        .unwrap();
    let requirements = spec.requirements();
    assert!(requirements.requires("restricted"));
    assert!(requirements.requires("abstractable_goal"));
    assert!(requirements.transformations.is_empty());

    let spec = features
        .parse_search("astar(blind(), lazy=canonical([icaps26_cartesian()]))")
        .unwrap();
    let requirements = spec.requirements();
    assert!(requirements.requires("restricted"));
    assert_eq!(requirements.transformations, ["icaps26_restrict"]);
}

#[test]
fn potential_seconds_beyond_duration_range_are_still_valid_configuration() {
    let features = planforge_heuristics::features();
    features
        .parse_search("astar(numeric_potential(max_ray_generation_time=1e300))")
        .unwrap();
}

#[test]
fn maximum_accepts_any_heuristic_and_unions_nested_requirements() {
    let features = planforge_heuristics::features();
    for text in [
        "astar(max(ff()))",
        "astar(max([blind(),ff()]))",
        "astar(max([domain(),lmcutnumeric()]))",
    ] {
        let config = features.parse_search(text).unwrap();
        assert_eq!(features.parse_search(&config.to_string()).unwrap(), config);
    }
    let config = features
        .parse_search("astar(max([ff(),pdb(),numeric_potential()]))")
        .unwrap();
    assert!(config.requirements().requires("restricted"));
    assert!(config.requirements().requires("abstractable_goal"));
    #[cfg(not(feature = "cplex"))]
    assert!(config.preflight().is_err(), "numeric_potential needs CPLEX");
    let error = features
        .parse_search("astar(max([ff(),scp(domian())]))")
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("max.heuristics[1].scp.sources: unknown AbstractionSource `domian`"),
        "{error}"
    );
    assert!(features.parse_search("astar(max([]))").is_err());
    assert!(features.parse_search("astar(max([ff()],blind()))").is_err());
    assert!(features.parse_search("astar(max(ff(),blind()))").is_err());
}

#[test]
#[cfg(target_os = "linux")]
fn construction_clock_overflow_is_rejected_without_rejecting_capped_source_budgets() {
    let features = planforge_heuristics::features();
    // This fits Duration but exceeds the Linux Instant range.
    assert!(std::time::Duration::try_from_secs_f64(1e19).is_ok());
    for combinator in ["canonical", "scp", "posthoc_optimization"] {
        let text = format!("astar({combinator}(domain(),construction_max_time=1e19))");
        let error = features.parse_search(&text).unwrap_err().to_string();
        assert!(error.contains("construction_max_time"), "{error}");
        assert!(error.contains("platform clock"), "{error}");
    }
    // Unlike the construction budget itself, a source budget is capped by it.
    features
        .parse_search("astar(scp(domain(max_time=1e19),construction_max_time=900))")
        .unwrap();
    // Potential optimization uses elapsed seconds rather than an Instant deadline.
    features
        .parse_search("astar(numeric_potential(max_diverse_generation_time=1e100))")
        .unwrap();
}

#[test]
fn rejects_nan_floats_quoted_heuristics_and_zero_budgets() {
    let features = planforge_heuristics::features();
    for spec in [
        "astar(lmcutnumeric(precision=nan))",
        "astar(\"blind\")",
        "astar(5)",
        "astar(canonical(icaps26_cartesian(max_time=-infinity)))",
        "astar(canonical(domain(), construction_max_time=1e-100))",
    ] {
        assert!(features.parse_search(spec).is_err(), "{spec}");
    }
}
