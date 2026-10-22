//! What each `--search` option name means, tested through the text a user
//! actually types.
//!
//! The option language is parsed in this crate too, so the factory's internals
//! stay private.

use super::{AbstractionSource, SourcesConfig};
use super::{abstraction_config, heuristic_names, register_external_heuristics};
use crate::config::{ApplyOptions, ConfigArg, ConfigCall, configure, parse_call};
use crate::evaluation::abstractions::cartesian::CartesianAbstractionConfig;
use crate::evaluation::abstractions::collections::strategy::CollectionStrategy;
use crate::evaluation::abstractions::common::cegar::FlawKind;
use crate::evaluation::abstractions::common::cegar::{FlawScope, SplitKind};
use crate::evaluation::abstractions::domain::config::DomainAbstractionCollectionConfig;
use crate::evaluation::abstractions::domain::config::DomainAbstractionConfig;
use crate::evaluation::abstractions::pdb::pattern_generator_greedy::GreedyPatternGeneratorConfig;
use crate::evaluation::abstractions::pdb::variable_order_finder::GreedyVariableOrderType;
use crate::evaluation::numeric_landmarks::lm_cut_numeric_heuristic::LmCutNumericConfig;

fn astar_heuristic(input: &str) -> ConfigCall {
    parse_call(
        input
            .strip_prefix("astar(")
            .unwrap()
            .strip_suffix(')')
            .unwrap(),
    )
    .unwrap()
}

fn source_config(input: &str) -> Result<SourcesConfig, String> {
    let spec = astar_heuristic(input);
    Ok(configure(&spec.name, SourcesConfig::default(), &spec.args)?)
}

fn domain_collection_source(input: &str) -> DomainAbstractionCollectionConfig {
    let mut config = source_config(input).unwrap();
    assert_eq!(config.sources.len(), 1);
    let AbstractionSource::DomainCollection(config) = config.sources.remove(0) else {
        panic!("expected domain collection");
    };
    config
}

fn domain_collection_source_error(input: &str) -> String {
    source_config(input).unwrap_err()
}

fn cartesian_source_config(
    args: &[ConfigArg],
    purpose: super::AbstractionPurpose,
) -> Result<CartesianAbstractionConfig, String> {
    let mut config = CartesianAbstractionConfig::default();
    config.cegar.keep_transitions = purpose.needs_transition_system();
    Ok(configure("cartesian", config, args)?)
}

#[test]
fn parses_astar_domain_with_named_options() {
    let h = astar_heuristic(
        "astar(domain(max_states=10000, use_wildcard_plans=false, combine_labels=true, random_seed=7))",
    );
    assert_eq!(h.name, "domain");

    let mut cfg = DomainAbstractionConfig::default();
    cfg.apply_options(&h.args).unwrap();
    assert_eq!(cfg.cegar.max_states, 10_000);
    assert!(!cfg.use_wildcard_plans);
    assert!(cfg.cegar.combine_labels);
    assert_eq!(cfg.cegar.random_seed, Some(7));
}

#[test]
fn parses_canonical_domain_collection_with_named_options() {
    let cfg = domain_collection_source(
        "astar(canonical(domain_collection(max_collection_size=123, total_max_time=4.5, use_wildcard_plans=false, combine_labels=true, flaw_kind=progression, random_seed=7)))",
    );

    assert_eq!(cfg.max_collection_size, 123);
    assert_eq!(
        cfg.total_max_time,
        Some(std::time::Duration::from_secs_f64(4.5)).into()
    );
    assert!(!cfg.member.use_wildcard_plans);
    assert!(cfg.member.cegar.combine_labels);
    assert_eq!(cfg.member.cegar.flaw_kind, FlawKind::Progression);
    assert_eq!(cfg.member.cegar.random_seed, Some(7));
}

#[test]
fn hierarchical_scp_rejects_legacy_pdb_switches() {
    let h = astar_heuristic("astar(scp([domain_collection()], use_numeric_pdbs=true))");
    let error = configure(
        &h.name,
        abstraction_config::ScpHeuristicConfig::default(),
        &h.args,
    )
    .unwrap_err()
    .to_string();
    assert_eq!(
        error,
        "scp.use_numeric_pdbs: unknown option `use_numeric_pdbs`"
    );
}

#[test]
fn hierarchical_scp_sets_every_scp_option() {
    let h = astar_heuristic(
        "astar(scp([domain_collection()], goal_cover_orders_per_goal=2, random_orders=5))",
    );
    let config = configure(
        &h.name,
        abstraction_config::ScpHeuristicConfig::default(),
        &h.args,
    )
    .unwrap()
    .scp;
    assert_eq!(config.goal_cover_orders_per_goal, 2);
    assert_eq!(config.random_orders, 5);
}

#[test]
fn hierarchical_scp_rejects_positional_options() {
    let h = astar_heuristic("astar(scp([domain_collection()], 5))");
    let error = configure(
        &h.name,
        abstraction_config::ScpHeuristicConfig::default(),
        &h.args,
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("abstraction source"), "got `{error}`");
}

#[test]
fn list_is_a_typed_error_for_scalar_and_wrapped_heuristic_options() {
    let h = astar_heuristic("astar(scp([domain_collection()], max_time=[5]))");
    let error = configure(
        &h.name,
        abstraction_config::ScpHeuristicConfig::default(),
        &h.args,
    )
    .unwrap_err()
    .to_string();
    assert_eq!(error, "scp.max_time: expected scalar value, got a list");

    let wrapper = astar_heuristic("astar(check_admissible([domain_collection()]))");
    let error =
        super::single_wrapped_heuristic_spec("check_admissible", &wrapper.args).unwrap_err();
    assert_eq!(error, "expected heuristic, got a list");
}

#[test]
fn cartesian_rejects_domain_only_options() {
    let h = astar_heuristic("astar(cartesian(use_wildcard_plans=true))");
    let error =
        cartesian_source_config(&h.args, super::AbstractionPurpose::Standalone).unwrap_err();
    assert_eq!(
        error,
        "cartesian.use_wildcard_plans: unknown option `use_wildcard_plans`"
    );
}

#[test]
fn both_representations_take_the_same_cegar_options() {
    let options = "max_states=7, max_iterations=3, max_time=2, combine_labels=true, \
                   random_seed=5, flaw_kind=regression, flaw_scope=whole_plan, split_kind=wanted, \
                   split_selection=random, split_selection_rank=1, debug=true";
    let domain = astar_heuristic(&format!("astar(domain({options}))"));
    let domain = configure("domain", DomainAbstractionConfig::default(), &domain.args).unwrap();
    let cartesian = astar_heuristic(&format!("astar(cartesian({options}))"));
    let cartesian = configure(
        "cartesian",
        CartesianAbstractionConfig::default(),
        &cartesian.args,
    )
    .unwrap();
    assert_eq!(domain.cegar, cartesian.cegar);
    assert_eq!(
        domain.cegar.max_time,
        Some(std::time::Duration::from_secs(2)).into()
    );
}

#[test]
fn configs_with_shared_options_take_named_options_only() {
    let h = astar_heuristic("astar(cartesian(10))");
    let error =
        cartesian_source_config(&h.args, super::AbstractionPurpose::Standalone).unwrap_err();
    assert_eq!(error, "cartesian: options must be named, as in `key=value`");
}

#[test]
fn parses_whole_plan_regression_flaws() {
    let cfg = domain_collection_source(
        "astar(canonical(domain_collection(flaw_kind=regression, flaw_scope=whole_plan)))",
    );
    assert_eq!(cfg.member.cegar.flaw_kind, FlawKind::Regression);
    assert_eq!(cfg.member.cegar.flaw_scope, FlawScope::WholePlan);
}

#[test]
fn parses_single_domain_time_limit() {
    let h = astar_heuristic("astar(domain(max_states=1000,max_time=900,flaw_scope=whole_plan))");
    let mut cfg = DomainAbstractionConfig::default();
    cfg.apply_options(&h.args).unwrap();
    assert_eq!(cfg.cegar.max_states, 1000);
    assert_eq!(
        cfg.cegar.max_time,
        Some(std::time::Duration::from_secs(900)).into()
    );
    assert_eq!(cfg.cegar.flaw_scope, FlawScope::WholePlan);

    let h = astar_heuristic("astar(domain(max_time=infinity))");
    cfg.apply_options(&h.args).unwrap();
    assert_eq!(cfg.cegar.max_time, None.into());
}

#[test]
fn parses_greedy_pattern_options() {
    let h = parse_call(
        "greedy(max_pdb_states=321, numeric_first=false, random_seed=7, variable_order_type=cg_goal_random)",
    )
    .unwrap();
    let mut cfg = GreedyPatternGeneratorConfig::default();
    ApplyOptions::apply_options(&mut cfg, &h.args).unwrap();
    assert_eq!(cfg.max_pdb_states, 321);
    assert!(!cfg.numeric_first);
    assert_eq!(cfg.random_seed, 7);
    assert_eq!(
        cfg.variable_order_type,
        GreedyVariableOrderType::CgGoalRandom
    );
}

#[test]
fn positional_args_map_to_canonical_order() {
    // greedy's positional order starts with max_pdb_states, numeric_first, random_seed
    let h = parse_call("greedy(321, false, 7)").unwrap();
    let mut cfg = GreedyPatternGeneratorConfig::default();
    ApplyOptions::apply_options(&mut cfg, &h.args).unwrap();
    assert_eq!(cfg.max_pdb_states, 321);
    assert!(!cfg.numeric_first);
    assert_eq!(cfg.random_seed, 7);
}

#[test]
fn mixed_positional_and_named_args_work() {
    // First positional → max_pdb_states; the named ones are explicit.
    let h = parse_call("greedy(321, numeric_first=false, random_seed=7)").unwrap();
    let mut cfg = GreedyPatternGeneratorConfig::default();
    ApplyOptions::apply_options(&mut cfg, &h.args).unwrap();
    assert_eq!(cfg.max_pdb_states, 321);
    assert!(!cfg.numeric_first);
    assert_eq!(cfg.random_seed, 7);
}

#[test]
fn positional_then_named_for_same_slot_errors() {
    // Positional 321 → max_pdb_states, then max_pdb_states=999 collides.
    let h = parse_call("greedy(321, max_pdb_states=999)").unwrap();
    let err = ApplyOptions::apply_options(&mut GreedyPatternGeneratorConfig::default(), &h.args)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("duplicate option `max_pdb_states`"),
        "got `{err}`"
    );
}

#[test]
fn too_many_positional_args_errors() {
    // greedy has 4 positional slots; 5 should error.
    let h = parse_call("greedy(1, false, 2, cg_goal_level, extra)").unwrap();
    let err = ApplyOptions::apply_options(&mut GreedyPatternGeneratorConfig::default(), &h.args)
        .unwrap_err()
        .to_string();
    assert!(err.contains("too many positional"), "got `{err}`");
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
fn parses_max_domain_collection_with_named_options() {
    let cfg = domain_collection_source(
        "astar(max(domain_collection(max_collection_size=123, total_max_time=4.5, use_wildcard_plans=false, combine_labels=true, flaw_kind=progression, flaw_scope=whole_plan, collection_strategy=goal_focused, random_seed=7, debug=true)))",
    );
    assert_eq!(cfg.max_collection_size, 123);
    assert_eq!(
        cfg.total_max_time,
        Some(std::time::Duration::from_secs_f64(4.5)).into()
    );
    assert!(!cfg.member.use_wildcard_plans);
    assert!(cfg.member.cegar.combine_labels);
    assert_eq!(cfg.member.cegar.flaw_kind, FlawKind::Progression);
    assert_eq!(cfg.member.cegar.flaw_scope, FlawScope::WholePlan);
    assert_eq!(cfg.collection_strategy, CollectionStrategy::GoalFocused);
    assert_eq!(cfg.member.cegar.random_seed, Some(7));
    assert!(cfg.member.cegar.debug);
}

#[test]
fn parses_goal_focused_domain_collection_options() {
    let h = astar_heuristic(
        "astar(domain_collection(collection_strategy=goal_focused, variants_per_goal=10, progress_states=true, alternate_split_kinds=false, split_kind=wanted))",
    );
    let mut cfg = DomainAbstractionCollectionConfig::default();
    ApplyOptions::apply_options(&mut cfg, &h.args).unwrap();
    assert_eq!(cfg.collection_strategy, CollectionStrategy::GoalFocused);
    assert_eq!(cfg.variants_per_goal, 10);
    assert!(cfg.progress_states);
    assert_eq!(cfg.alternate_split_kinds, Some(false));
    assert!(cfg.single_goal_tasks);
    assert_eq!(cfg.split_kind, Some(SplitKind::Wanted));
}

#[test]
fn parses_domain_collection_with_trailing_comma() {
    let cfg = domain_collection_source("astar(max(domain_collection(max_collection_size=123,)))");
    assert_eq!(cfg.max_collection_size, 123);
}

#[test]
fn rejects_removed_exec_entire_plan_randomize_option() {
    let err =
        domain_collection_source_error("astar(max(domain_collection(exec_entire_plan=randomize)))");
    assert!(err.contains("exec_entire_plan"));
}

#[test]
fn heuristic_registry_names_are_unique_and_complete() {
    let names = heuristic_names().collect::<Vec<_>>();
    let unique = names
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        names.len(),
        unique.len(),
        "duplicate registry names: {names:?}"
    );
    for required in [
        "blind",
        "ff",
        "domain",
        "cartesian",
        "scp",
        "pdb",
        "lmcutnumeric",
        "numeric_potential",
    ] {
        assert!(
            unique.contains(required),
            "missing registry entry {required}"
        );
    }
}

#[test]
fn external_registration_rejects_builtin_name_collision() {
    let error = register_external_heuristics(vec![super::entry::<
        crate::evaluation::heuristic::BlindFeature,
    >()])
    .unwrap_err();
    assert!(error.contains("collides with a built-in"), "got `{error}`");
}

#[test]
fn external_registration_rejects_duplicate_external_names() {
    let error = register_external_heuristics(vec![
        super::entry::<Duplicate>(),
        super::entry::<Duplicate>(),
    ])
    .unwrap_err();
    assert!(error.contains("registered more than once"), "got `{error}`");
}

#[test]
fn rejects_unknown_options_inside_known_heuristics() {
    let h = astar_heuristic("astar(scp(domain_collection(), deviation_flaws=false))");
    let err = configure(
        &h.name,
        abstraction_config::ScpHeuristicConfig::default(),
        &h.args,
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("deviation_flaws"), "got `{err}`");

    let err = domain_collection_source_error(
        "astar(canonical(domain_collection(deviation_flaws=false)))",
    );
    assert!(err.contains("deviation_flaws"), "got `{err}`");
}

#[test]
fn collection_seeds_split_kinds_and_stagnation_limits_parse_once() {
    let cfg = domain_collection_source(
        "astar(max(domain_collection(random_seed=none, split_kind=wanted, stagnation_limit=infinity)))",
    );
    assert_eq!(cfg.member.cegar.random_seed, None);
    assert_eq!(cfg.split_kind, Some(SplitKind::Wanted));
    assert_eq!(cfg.member.cegar.split_kind, SplitKind::Concrete);
    assert_eq!(cfg.stagnation_limit, None.into());

    let cfg = domain_collection_source("astar(max(domain_collection(stagnation_limit=2.5)))");
    assert_eq!(
        cfg.stagnation_limit,
        Some(std::time::Duration::from_millis(2500)).into()
    );

    let error =
        domain_collection_source_error("astar(max(domain_collection(stagnation_limit=-1)))");
    assert!(error.contains("non-negative"), "{error}");
}

struct Duplicate;
impl super::HeuristicFeature for Duplicate {
    const NAME: &'static str = "duplicate";
    type Config = super::EmptyConfig;
    fn build<'a>(
        _: &Self::Config,
        context: &super::BuildContext<'a>,
    ) -> Result<Box<dyn crate::evaluation::Heuristic + 'a>, super::HeuristicBuildError> {
        Ok(Box::new(
            crate::evaluation::heuristic::BlindHeuristic::for_task(&**context.task),
        ))
    }
}

#[test]
fn typed_sources_take_one_list_or_a_single_source() {
    let listed = parse_call("scp([domain(max_states=7), pdb(patterns=greedy(321))], random_orders=5, goal_cover_orders_per_goal=2)").unwrap();
    let single =
        parse_call("scp(domain(max_states=7), random_orders=5, goal_cover_orders_per_goal=2)")
            .unwrap();
    let parse = |spec: &ConfigCall| {
        configure(
            "scp",
            abstraction_config::ScpHeuristicConfig::default(),
            &spec.args,
        )
        .unwrap()
    };
    let listed = parse(&listed);
    let single = parse(&single);
    assert_eq!(
        listed
            .sources
            .sources
            .iter()
            .map(AbstractionSource::name)
            .collect::<Vec<_>>(),
        ["domain", "pdb"]
    );
    assert_eq!(listed.scp, single.scp);
    assert_eq!(listed.scp.random_orders, 5);
    assert_eq!(listed.scp.goal_cover_orders_per_goal, 2);
    let AbstractionSource::Domain(config) = &single.sources.sources[0] else {
        panic!("domain source");
    };
    assert_eq!(config.cegar.max_states, 7);
    assert!(listed.sources.sources[1].requirements().restricted);
}

#[test]
fn typed_sources_reject_misspellings_unknown_options_and_mixed_lists() {
    for spec in [
        "astar(scp(domian()))",
        "astar(scp(domain(max_collection_size=7)))",
        "astar(scp([domain()], cartesian()))",
        "astar(scp([domain()], [cartesian()]))",
        "astar(scp([]))",
        "astar(scp(domain(), [cartesian()]))",
        "astar(scp(domain(), cartesian()))",
        "astar(scp(numeric_pdb()))",
    ] {
        assert!(source_config(spec).is_err(), "{spec}");
    }
}

#[test]
fn shared_construction_limits_remain_positive_and_finite() {
    for combinator in ["canonical", "scp"] {
        for seconds in ["0", "-1", "infinity"] {
            let spec = astar_heuristic(&format!(
                "astar({combinator}(cartesian(),construction_max_time={seconds}))"
            ));
            let error = if combinator == "scp" {
                configure(
                    combinator,
                    abstraction_config::ScpHeuristicConfig::default(),
                    &spec.args,
                )
                .unwrap_err()
            } else {
                configure(combinator, SourcesConfig::default(), &spec.args).unwrap_err()
            };
            assert!(
                error.to_string().contains("construction_max_time"),
                "{error}"
            );
        }
    }
    let sources = source_config("astar(canonical(cartesian(),construction_max_time=900))").unwrap();
    assert!(sources.deadline().unwrap().remaining().unwrap().as_secs() <= 900);
}

#[test]
fn strict_icaps26_options_and_requirements_are_typed() {
    let mut sources =
        source_config("astar(canonical(icaps26_cartesian(pick=random,max_time=900)))").unwrap();
    assert!(sources.sources[0].requirements().restricted);
    let AbstractionSource::Icaps26Cartesian(config) = sources.sources.remove(0) else {
        panic!("ICAPS26 source");
    };
    assert_eq!(config.random_seed, 2011);
    assert_eq!(config.max_time, Some(std::time::Duration::from_secs(900)));
    assert!(source_config("astar(canonical(icaps26_cartesian(flaw_kind=regression)))").is_err());
}

#[test]
fn pdb_sources_take_patterns_and_heuristics_as_separate_options() {
    for spec in [
        "pdb(patterns=greedy(max_pdb_states=321), exploration_heuristic=lmcut)",
        "pdb(patterns=systematic(max_pdb_states=321, max_pattern_size=2), failed_lookup_heuristic=lmcut)",
        "pdb()",
    ] {
        let spec = parse_call(spec).unwrap();
        configure("pdb", super::PdbSourceConfig::default(), &spec.args).unwrap();
    }
    for spec in [
        "pdb(max_pdb_states=321)",
        "pdb(patterns=greedy(exploration_heuristic=lmcut))",
    ] {
        let spec = parse_call(spec).unwrap();
        configure("pdb", super::PdbSourceConfig::default(), &spec.args).unwrap_err();
    }
    // A positive budget that rounds to no time at all is no budget.
    source_config("astar(canonical(domain(),construction_max_time=1e-100))").unwrap_err();
}

#[test]
fn config_errors_keep_the_nested_option_path() {
    let spec = parse_call("scp([domain(max_states=bogus)])").unwrap();
    let error = configure(
        "scp",
        abstraction_config::ScpHeuristicConfig::default(),
        &spec.args,
    )
    .unwrap_err();
    assert_eq!(
        error.path(),
        ["scp", "sources", "[0]", "domain", "max_states"]
    );
    assert!(error.message().contains("non-negative integer"));
    assert!(matches!(
        super::HeuristicBuildError::from(error),
        super::HeuristicBuildError::Config(_)
    ));
}
