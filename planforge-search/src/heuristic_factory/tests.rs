//! What each `--search` option name means, tested through the text a user
//! actually types.
//!
//! The option language is parsed in this crate too, so these tests need nothing
//! from `planforge-searcher` and the factory's internals stay private.

use super::{
    ExternalHeuristic, RequiredBackend, TaskRequirements, abstraction_config, heuristic_names,
    register_external_heuristics,
};
use crate::config::{ApplyOptions, ConfigArg, HeuristicSpec, configure, parse_heuristic_spec};
use crate::evaluation::abstractions::cartesian::{
    CartesianAbstractionCollectionConfig, CartesianAbstractionConfig,
};
use crate::evaluation::abstractions::collections::strategy::CollectionStrategy;
use crate::evaluation::abstractions::common::cegar::FlawKind;
use crate::evaluation::abstractions::common::cegar::{FlawScope, SplitKind, SplitSelection};
use crate::evaluation::abstractions::cost_partitioning::saturated::{
    CostPartitioningMethod, OrderGenerator, ScpConfig,
};
use crate::evaluation::abstractions::domain::config::DomainAbstractionCollectionConfig;
use crate::evaluation::abstractions::domain::config::DomainAbstractionConfig;
use crate::evaluation::abstractions::icaps26::Icaps26SplitSelection;
use crate::evaluation::abstractions::pdb::canonical_pdb_heuristic::CanonicalNumericPdbConfig;
use crate::evaluation::abstractions::pdb::pattern_database::PdbInternalHeuristic;
use crate::evaluation::abstractions::pdb::pattern_generator_greedy::GreedyPatternGeneratorConfig;
use crate::evaluation::abstractions::pdb::variable_order_finder::GreedyVariableOrderType;
use crate::evaluation::numeric_landmarks::lm_cut_numeric_heuristic::LmCutNumericConfig;
use crate::resource_limits::Deadline;
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
        "astar(domain_abstraction(max_states=10000, use_wildcard_plans=false, combine_labels=true, random_seed=7))",
    );
    assert_eq!(h.name, "domain_abstraction");

    let mut cfg = DomainAbstractionConfig::default();
    cfg.apply_options(&h.args).unwrap();
    assert_eq!(cfg.cegar.max_states, 10_000);
    assert!(!cfg.use_wildcard_plans);
    assert!(cfg.cegar.combine_labels);
    assert_eq!(cfg.cegar.random_seed, Some(7));
}

/// The domain collection options of a combinator's single source.
fn domain_collection_source(spec: &str) -> DomainAbstractionCollectionConfig {
    let h = astar_heuristic(spec);
    let (sources, _) = abstraction_config::split_component_sources(&h.name, &h.args).unwrap();
    assert_eq!(sources.len(), 1);
    assert_eq!(sources[0].name(), "domain_collection");
    configure(
        "domain_collection",
        DomainAbstractionCollectionConfig::default(),
        sources[0].args(),
    )
    .unwrap()
}

/// The error the domain collection options of a combinator's single source give.
fn domain_collection_source_error(spec: &str) -> String {
    let h = astar_heuristic(spec);
    let (sources, _) = abstraction_config::split_component_sources(&h.name, &h.args).unwrap();
    configure(
        "domain_collection",
        DomainAbstractionCollectionConfig::default(),
        sources[0].args(),
    )
    .unwrap_err()
}

/// The `cartesian_collection(...)` options `args` for a component used as
/// `component_use`.
fn cartesian_collection_config(
    args: &[ConfigArg],
    component_use: abstraction_config::ComponentUse,
) -> Result<CartesianAbstractionCollectionConfig, String> {
    let mut config = CartesianAbstractionCollectionConfig::default();
    config.member.cegar.keep_transition_system = component_use.needs_transition_system();
    configure("cartesian_collection", config, args)
}

/// The `cartesian(...)` options `args` for a component used as
/// `component_use`.
fn cartesian_source_config(
    args: &[ConfigArg],
    component_use: abstraction_config::ComponentUse,
) -> Result<CartesianAbstractionConfig, String> {
    let mut config = CartesianAbstractionConfig::default();
    config.cegar.keep_transition_system = component_use.needs_transition_system();
    configure("cartesian", config, args)
}

#[test]
fn parses_canonical_domain_collection_with_named_options() {
    let cfg = domain_collection_source(
        "astar(canonical(domain_collection(max_collection_size=123, total_max_time=4.5, use_wildcard_plans=false, combine_labels=true, flaw_kind=progression, random_seed=7)))",
    );

    assert_eq!(cfg.max_collection_size, 123);
    assert_eq!(
        cfg.total_max_time,
        Some(std::time::Duration::from_secs_f64(4.5))
    );
    assert!(!cfg.member.use_wildcard_plans);
    assert!(cfg.member.cegar.combine_labels);
    assert_eq!(cfg.member.cegar.flaw_kind, FlawKind::Progression);
    assert_eq!(cfg.member.cegar.random_seed, Some(7));
}

#[test]
fn parses_hierarchical_scp_options_and_sources() {
    let h = astar_heuristic(
        "astar(scp(domain_collection(max_collection_size=1000), cartesian(max_states=100), pdb(max_pdb_states=100), diversify=true, samples=123, max_orders=17, orders=diverse_orders, initial_order_generation_max_time=9, saturator=perimstar, partitioning=region))",
    );
    assert_eq!(h.name, "scp");
    let (sources, options) = abstraction_config::split_component_sources(&h.name, &h.args).unwrap();
    assert_eq!(sources.len(), 3);
    assert_eq!(options.len(), 7);
    abstraction_config::validate_scp_combinator_options(&options).unwrap();
    let mut config = ScpConfig::default();
    ApplyOptions::apply_options(&mut config, &options).unwrap();
    assert!(config.diversify);
    assert_eq!(config.samples, 123);
    assert_eq!(config.max_orders, 17);
    assert_eq!(config.order_generator, OrderGenerator::Diverse);
    assert_eq!(config.initial_order_generation_max_time, 9.0);
}

#[test]
fn hierarchical_scp_rejects_legacy_pdb_switches() {
    let h = astar_heuristic("astar(scp([domain_collection()], use_numeric_pdbs=true))");
    let error = abstraction_config::scp_sources_options_and_deadline(&h.args).unwrap_err();
    assert!(error.contains("unknown `scp` combinator option `use_numeric_pdbs`"));
    assert!(error.contains("pdb(...)"));
}

#[test]
fn list_is_a_typed_error_for_scalar_and_wrapped_heuristic_options() {
    let h = astar_heuristic("astar(scp([domain_collection()], max_time=[5]))");
    let source_config = abstraction_config::scp_sources_options_and_deadline(&h.args).unwrap();
    let error =
        ApplyOptions::apply_options(&mut ScpConfig::default(), &source_config.options).unwrap_err();
    assert_eq!(error, "expected scalar value, got a list");

    let wrapper = astar_heuristic("astar(check_admissible([domain_collection()]))");
    let error =
        super::single_wrapped_heuristic_spec("check_admissible", &wrapper.args).unwrap_err();
    assert_eq!(error, "expected heuristic, got a list");
}

#[test]
fn parses_hierarchical_cartesian_collection_source() {
    let h = astar_heuristic(
        "astar(scp(cartesian_collection(max_states=1000, flaw_scope=whole_plan, collection_strategy=goal_focused, variants_per_goal=8, progress_states=true, alternate_split_kinds=false, max_collection_size=100000, total_max_time=60, random_seed=1), saturator=all, partitioning=region))",
    );
    let (sources, options) = abstraction_config::split_component_sources(&h.name, &h.args).unwrap();
    assert_eq!(sources.len(), 1);
    assert_eq!(sources[0].name(), "cartesian_collection");
    assert_eq!(options.len(), 2);
    let config = cartesian_collection_config(
        sources[0].args(),
        abstraction_config::ComponentUse::RegionalCostPartitioning,
    )
    .unwrap();
    assert_eq!(config.collection_strategy, CollectionStrategy::GoalFocused);
    assert_eq!(config.variants_per_goal, 8);
    assert!(config.progress_states);
    assert_eq!(config.alternate_split_kinds, Some(false));
    assert!(config.single_goal_tasks);
    assert_eq!(config.member.cegar.flaw_kind, FlawKind::Progression);
    assert_eq!(config.member.cegar.flaw_scope, FlawScope::WholePlan);
    assert!(config.member.cegar.keep_transition_system);
}

#[test]
fn parses_strict_icaps26_cartesian_source() {
    let h = astar_heuristic(
        "astar(scp(icaps26_cartesian(pick=min_unwanted,max_time=900,random_seed=7),partitioning=region))",
    );
    let (sources, options) = abstraction_config::split_component_sources(&h.name, &h.args).unwrap();
    assert_eq!(sources.len(), 1);
    assert_eq!(sources[0].name(), "icaps26_cartesian");
    assert_eq!(options.len(), 1);
    let config = abstraction_config::apply_icaps26_cartesian_options(
        sources[0].args(),
        abstraction_config::ComponentUse::RegionalCostPartitioning,
    )
    .unwrap();
    assert_eq!(config.pick, Icaps26SplitSelection::MinUnwanted);
    assert_eq!(config.max_time, Some(std::time::Duration::from_secs(900)));
    assert_eq!(config.random_seed, 7);
    assert!(config.compute_source_sets);
    assert!(config.retain_transition_system);

    let defaults = abstraction_config::apply_icaps26_cartesian_options(
        &[],
        abstraction_config::ComponentUse::Standalone,
    )
    .unwrap();
    assert_eq!(defaults.random_seed, 2011);
}

#[test]
fn parses_matched_single_abstraction_icaps26_configuration() {
    let h = astar_heuristic("astar(canonical(icaps26_cartesian(pick=random,max_time=900)))");
    let (sources, construction_deadline) =
        abstraction_config::canonical_sources_and_deadline(&h.args).unwrap();
    assert_eq!(sources.len(), 1);
    assert_eq!(sources[0].name(), "icaps26_cartesian");
    assert!(
        construction_deadline == Deadline::NONE,
        "the published 900-second limit bounds CEGAR, not the outer combinator"
    );
    let config = abstraction_config::apply_icaps26_cartesian_options(
        sources[0].args(),
        abstraction_config::ComponentUse::Standalone,
    )
    .unwrap();
    assert_eq!(config.max_states, usize::MAX);
    assert_eq!(config.max_time, Some(std::time::Duration::from_secs(900)));
    assert_eq!(config.pick, Icaps26SplitSelection::Random);
    assert!(!config.retain_transition_system);
}

#[test]
fn parses_shared_construction_deadlines_for_canonical_and_scp() {
    let canonical =
        astar_heuristic("astar(canonical(cartesian(max_states=100),construction_max_time=900))");
    let (sources, deadline) =
        abstraction_config::canonical_sources_and_deadline(&canonical.args).unwrap();
    assert_eq!(sources.len(), 1);
    assert_ne!(deadline, Deadline::NONE);

    let scp = astar_heuristic("astar(scp(cartesian(max_states=100),construction_max_time=900))");
    let source_config = abstraction_config::scp_sources_options_and_deadline(&scp.args).unwrap();
    assert_eq!(source_config.sources.len(), 1);
    assert_eq!(source_config.options.len(), 0);
    assert_ne!(source_config.construction_deadline, Deadline::NONE);
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
    let (sources, _) = abstraction_config::split_component_sources(&h.name, &h.args).unwrap();
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
    let (sources, _) = abstraction_config::split_component_sources(&h.name, &h.args).unwrap();
    let config = cartesian_collection_config(
        sources[0].args(),
        abstraction_config::ComponentUse::Standalone,
    )
    .unwrap();
    assert_eq!(config.member.cegar.flaw_kind, FlawKind::Progression);
}

#[test]
fn cartesian_sources_reject_unknown_flaw_kinds() {
    let h =
        astar_heuristic("astar(canonical(cartesian_collection(flaw_kind=sequence_regression)))");
    let (sources, _) = abstraction_config::split_component_sources(&h.name, &h.args).unwrap();
    let error = cartesian_collection_config(
        sources[0].args(),
        abstraction_config::ComponentUse::Standalone,
    )
    .expect_err("an unknown flaw kind must fail during parsing");
    assert!(
        error.contains("invalid flaw_kind `sequence_regression`"),
        "got `{error}`"
    );

    let h = astar_heuristic("astar(cartesian_abstraction(flaw_kind=sequence_regression))");
    let error =
        cartesian_source_config(&h.args, abstraction_config::ComponentUse::Standalone).unwrap_err();
    assert!(
        error.contains("invalid flaw_kind `sequence_regression`"),
        "got `{error}`"
    );
}

#[test]
fn cartesian_abstraction_rejects_domain_abstraction_options() {
    let h = astar_heuristic("astar(cartesian_abstraction(use_wildcard_plans=true))");
    let error =
        cartesian_source_config(&h.args, abstraction_config::ComponentUse::Standalone).unwrap_err();
    assert_eq!(error, "cartesian(...): unknown option `use_wildcard_plans`");
}

#[test]
fn both_representations_take_the_same_cegar_options() {
    let options = "max_states=7, max_iterations=3, max_time=2, combine_labels=true, \
                   random_seed=5, flaw_kind=regression, flaw_scope=whole_plan, split_kind=wanted, \
                   split_selection=random, split_selection_rank=1, debug=true";
    let domain = astar_heuristic(&format!("astar(domain_abstraction({options}))"));
    let domain = configure("domain", DomainAbstractionConfig::default(), &domain.args).unwrap();
    let cartesian = astar_heuristic(&format!("astar(cartesian_abstraction({options}))"));
    let cartesian = configure(
        "cartesian",
        CartesianAbstractionConfig::default(),
        &cartesian.args,
    )
    .unwrap();
    assert_eq!(domain.cegar, cartesian.cegar);
    assert_eq!(
        domain.cegar.max_time,
        Some(std::time::Duration::from_secs(2))
    );
}

#[test]
fn configs_with_shared_options_take_named_options_only() {
    let h = astar_heuristic("astar(cartesian_abstraction(10))");
    let error =
        cartesian_source_config(&h.args, abstraction_config::ComponentUse::Standalone).unwrap_err();
    assert_eq!(
        error,
        "cartesian(...): options must be named, as in `key=value`"
    );
}

#[test]
fn rejects_boolean_cost_partitioning_mode() {
    let h = astar_heuristic("astar(scp(domain_collection(), partitioning=label))");
    let (_, options) = abstraction_config::split_component_sources(&h.name, &h.args).unwrap();
    let mut config = ScpConfig::default();
    ApplyOptions::apply_options(&mut config, &options).unwrap();
    assert_eq!(config.partitioning, CostPartitioningMethod::Label);

    let h = astar_heuristic("astar(scp(domain_collection(), partitioning=true))");
    let (_, options) = abstraction_config::split_component_sources(&h.name, &h.args).unwrap();
    let mut config = ScpConfig::default();
    assert!(ApplyOptions::apply_options(&mut config, &options).is_err());
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
fn parses_single_domain_abstraction_time_limit() {
    let h = astar_heuristic(
        "astar(domain_abstraction(max_states=1000,max_time=900,flaw_scope=whole_plan))",
    );
    let mut cfg = DomainAbstractionConfig::default();
    cfg.apply_options(&h.args).unwrap();
    assert_eq!(cfg.cegar.max_states, 1000);
    assert_eq!(
        cfg.cegar.max_time,
        Some(std::time::Duration::from_secs(900))
    );
    assert_eq!(cfg.cegar.flaw_scope, FlawScope::WholePlan);

    let h = astar_heuristic("astar(domain_abstraction(max_time=infinity))");
    cfg.apply_options(&h.args).unwrap();
    assert_eq!(cfg.cegar.max_time, None);
}

#[test]
fn parses_full_task_interleaved_domain_collection() {
    let h = astar_heuristic(
        "astar(canonical(domain_collection(collection_strategy=standard,alternate_split_kinds=true,flaw_scope=whole_plan,split_selection=min_growth)))",
    );
    let (sources, options) = abstraction_config::split_component_sources(&h.name, &h.args).unwrap();
    assert_eq!(sources.len(), 1);
    assert!(options.is_empty());
    let mut source = DomainAbstractionCollectionConfig::default();
    ApplyOptions::apply_options(&mut source, sources[0].args()).unwrap();

    assert_eq!(source.collection_strategy, CollectionStrategy::Standard);
    assert_eq!(source.alternate_split_kinds, Some(true));
    assert_eq!(source.member.cegar.flaw_kind, FlawKind::Progression);
    assert_eq!(source.member.cegar.flaw_scope, FlawScope::WholePlan);
    assert_eq!(
        source.member.cegar.split_selection,
        SplitSelection::MinGrowth
    );
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
fn parses_max_domain_collection_with_named_options() {
    let cfg = domain_collection_source(
        "astar(max(domain_collection(max_collection_size=123, total_max_time=4.5, use_wildcard_plans=false, combine_labels=true, flaw_kind=progression, flaw_scope=whole_plan, collection_strategy=goal_focused, random_seed=7, debug=true)))",
    );
    assert_eq!(cfg.max_collection_size, 123);
    assert_eq!(
        cfg.total_max_time,
        Some(std::time::Duration::from_secs_f64(4.5))
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
fn parses_full_task_goal_focused_options() {
    let h = astar_heuristic(
        "astar(scp(domain_collection(collection_strategy=goal_focused, single_goal_tasks=false), cartesian_collection(collection_strategy=goal_focused, single_goal_tasks=false)))",
    );
    let (sources, _) = abstraction_config::split_component_sources(&h.name, &h.args).unwrap();
    let mut domain = DomainAbstractionCollectionConfig::default();
    ApplyOptions::apply_options(&mut domain, sources[0].args()).unwrap();
    assert!(!domain.single_goal_tasks);
    let cartesian = cartesian_collection_config(
        sources[1].args(),
        abstraction_config::ComponentUse::RegionalCostPartitioning,
    )
    .unwrap();
    assert!(!cartesian.single_goal_tasks);
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
        "domain_abstraction",
        "cartesian_abstraction",
        "scp",
        "greedy_numeric_pdb",
        "lmcutnumeric",
        "numeric_potential",
    ] {
        assert!(
            unique.contains(required),
            "missing registry entry {required}"
        );
    }
}

fn test_external(name: &'static str) -> ExternalHeuristic {
    ExternalHeuristic {
        name,
        backend: RequiredBackend::None,
        requirements: |_| Ok(TaskRequirements::ANY),
        nested_heuristics: |_| Ok(Vec::new()),
        build: |_, _, _| Ok(None::<Box<dyn crate::evaluation::Heuristic>>),
    }
}

#[test]
fn external_registration_rejects_builtin_name_collision() {
    let error = register_external_heuristics(vec![test_external("blind")]).unwrap_err();
    assert!(error.contains("collides with a built-in"), "got `{error}`");
}

#[test]
fn external_registration_rejects_duplicate_external_names() {
    let error =
        register_external_heuristics(vec![test_external("duplicate"), test_external("duplicate")])
            .unwrap_err();
    assert!(error.contains("registered more than once"), "got `{error}`");
}

#[test]
fn rejects_unknown_options_inside_known_heuristics() {
    let h = astar_heuristic("astar(scp(domain_collection(), deviation_flaws=false))");
    let (_, options) = abstraction_config::split_component_sources(&h.name, &h.args).unwrap();
    let err = abstraction_config::validate_scp_combinator_options(&options).unwrap_err();
    assert!(err.contains("deviation_flaws"), "got `{err}`");

    let err = domain_collection_source_error(
        "astar(canonical(domain_collection(deviation_flaws=false)))",
    );
    assert!(err.contains("deviation_flaws"), "got `{err}`");
}

#[test]
fn domain_source_is_one_domain_abstraction_with_its_own_options() {
    let h = astar_heuristic(
        "astar(canonical(domain(max_states=500, flaw_kind=regression, flaw_scope=whole_plan, split_kind=wanted)))",
    );
    let (sources, _) = abstraction_config::split_component_sources(&h.name, &h.args).unwrap();
    assert_eq!(sources[0].name(), "domain");
    let mut config = DomainAbstractionConfig::default();
    config.apply_options(sources[0].args()).unwrap();
    assert_eq!(config.cegar.max_states, 500);
    assert_eq!(config.cegar.flaw_kind, FlawKind::Regression);
    assert_eq!(config.cegar.flaw_scope, FlawScope::WholePlan);
    assert_eq!(config.cegar.split_kind, SplitKind::Wanted);

    let h = astar_heuristic("astar(canonical(domain(max_collection_size=10)))");
    let (sources, _) = abstraction_config::split_component_sources(&h.name, &h.args).unwrap();
    let error = DomainAbstractionConfig::default()
        .apply_options(sources[0].args())
        .unwrap_err();
    assert!(
        error.contains("unknown option `max_collection_size`"),
        "{error}"
    );
}

#[test]
fn collection_seeds_split_kinds_and_stagnation_limits_parse_once() {
    let cfg = domain_collection_source(
        "astar(max(domain_collection(random_seed=none, split_kind=wanted, stagnation_limit=infinity)))",
    );
    assert_eq!(cfg.member.cegar.random_seed, None);
    assert_eq!(cfg.split_kind, Some(SplitKind::Wanted));
    assert_eq!(cfg.member.cegar.split_kind, SplitKind::Concrete);
    assert_eq!(cfg.stagnation_limit, None);

    let cfg = domain_collection_source("astar(max(domain_collection(stagnation_limit=2.5)))");
    assert_eq!(
        cfg.stagnation_limit,
        Some(std::time::Duration::from_millis(2500))
    );

    let error =
        domain_collection_source_error("astar(max(domain_collection(stagnation_limit=-1)))");
    assert!(error.contains("non-negative"), "{error}");
}
