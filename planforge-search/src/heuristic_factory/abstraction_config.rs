use std::collections::HashSet;
use std::time::{Duration, Instant};

use crate::config::{ConfigArg, ConfigCall, ConfigValue, FromOptionValue, configure};
use crate::evaluation::abstractions::cartesian::{
    CartesianAbstractionCollectionConfig, CartesianAbstractionCollectionGenerator,
    CartesianAbstractionConfig, CartesianAbstractionGenerator,
};
use crate::evaluation::abstractions::collections::generator::AbstractionGenerator;
use crate::evaluation::abstractions::component::AbstractionComponent;
use crate::evaluation::abstractions::domain::collection::DomainAbstractionCollectionGenerator;
use crate::evaluation::abstractions::domain::config::{
    DomainAbstractionCollectionConfig, DomainAbstractionConfig,
};
use crate::evaluation::abstractions::domain::generator::DomainAbstractionGenerator;
use crate::evaluation::abstractions::icaps26::{Icaps26CartesianConfig, Icaps26CartesianGenerator};
use crate::evaluation::abstractions::pdb::canonical_pdb_heuristic::CanonicalNumericPdbConfig;
use crate::evaluation::abstractions::pdb::pattern_collection::PatternCollection;
use crate::evaluation::abstractions::pdb::pattern_database::PdbHeuristicConfig;
use crate::evaluation::abstractions::pdb::pattern_generator_greedy::{
    GreedyPatternGeneratorConfig, generate_greedy_pattern,
};
use crate::evaluation::abstractions::pdb::pattern_generator_systematic::{
    SystematicPatternGeneratorConfig, generate_systematic_patterns,
};
use crate::evaluation::abstractions::pdb::pdb_collection::PdbCollection;
use crate::evaluation::abstractions::pdb::validate_restricted_task;
use planforge_sas::numeric_task::AbstractNumericTask;
use tracing::info;

use super::HeuristicBuildError;

pub(crate) const COMPONENT_SOURCE_NAMES: &[&str] = &[
    "domain",
    "domain_collection",
    "cartesian",
    "cartesian_collection",
    "icaps26_cartesian",
    "pdb",
    "numeric_pdb",
];

const SCP_COMBINATOR_OPTIONS: &[&str] = &[
    "max_time",
    "table_construction_max_time",
    "diversify",
    "samples",
    "max_orders",
    "combine_labels",
    "orders",
    "initial_order_generation_max_time",
    "saturator",
    "random_seed",
    "partitioning",
];

pub(crate) fn component_source_help() -> String {
    COMPONENT_SOURCE_NAMES
        .iter()
        .map(|name| format!("{name}(...)"))
        .collect::<Vec<_>>()
        .join(", ")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ComponentUse {
    Standalone,
    LabelCostPartitioning,
    RegionalCostPartitioning,
}

#[derive(Debug)]
enum PdbPatternsConfig {
    Systematic(SystematicPatternGeneratorConfig),
    Greedy(GreedyPatternGeneratorConfig),
    // Phase 4 adds the third closed strategy, CEGAR, here.
}

#[derive(Debug)]
struct PdbSourceConfig {
    patterns: PdbPatternsConfig,
    systematic_heuristic: PdbHeuristicConfig,
}

impl PdbSourceConfig {
    fn from_args(args: &[ConfigArg]) -> Result<Self, String> {
        let Some(patterns_arg) = args.iter().find(|arg| arg.key() == Some("patterns")) else {
            let legacy = configure("pdb", CanonicalNumericPdbConfig::default(), args)?;
            return Ok(Self {
                patterns: PdbPatternsConfig::Systematic(SystematicPatternGeneratorConfig {
                    max_pdb_states: legacy.max_pdb_states,
                    max_pattern_size: legacy.max_pattern_size,
                    only_interesting_patterns: legacy.only_interesting_patterns,
                }),
                systematic_heuristic: legacy.pdb_heuristic_config(),
            });
        };

        if args
            .iter()
            .filter(|arg| arg.key() == Some("patterns"))
            .count()
            > 1
        {
            return Err("duplicate option `patterns` for `pdb` source".to_string());
        }
        let patterns = Self::parse_patterns(patterns_arg.value())?;
        let mut systematic_heuristic = PdbHeuristicConfig::default();
        let mut seen = HashSet::new();
        for arg in args {
            let key = arg
                .key()
                .ok_or_else(|| "options for `pdb(patterns=...)` must be named".to_string())?;
            if !seen.insert(key) {
                return Err(format!("duplicate option `{key}` for `pdb` source"));
            }
            match key {
                "patterns" => {}
                "exploration_heuristic" => {
                    systematic_heuristic.exploration_heuristic =
                        FromOptionValue::from_option_value(arg.value())?;
                }
                "frontier_heuristic" => {
                    systematic_heuristic.frontier_heuristic =
                        FromOptionValue::from_option_value(arg.value())?;
                }
                "failed_lookup_heuristic" => {
                    systematic_heuristic.failed_lookup_heuristic =
                        FromOptionValue::from_option_value(arg.value())?;
                }
                other => {
                    return Err(format!(
                        "unknown option `{other}` for `pdb` source with `patterns=...`; put pattern-generator options inside `patterns=systematic(...)` or `patterns=greedy(...)`"
                    ));
                }
            }
        }
        Ok(Self {
            patterns,
            systematic_heuristic,
        })
    }

    fn parse_patterns(value: &ConfigValue) -> Result<PdbPatternsConfig, String> {
        let (name, args) = match value {
            ConfigValue::Atom(name) => (name.as_str(), &[][..]),
            ConfigValue::Call(call) => (call.name(), call.args()),
            ConfigValue::List(_) => {
                return Err("expected PDB pattern strategy, got a list".to_string());
            }
        };
        match name {
            "systematic" => Ok(PdbPatternsConfig::Systematic(configure(
                name,
                SystematicPatternGeneratorConfig::default(),
                args,
            )?)),
            "greedy" => Ok(PdbPatternsConfig::Greedy(configure(
                name,
                GreedyPatternGeneratorConfig::default(),
                args,
            )?)),
            other => Err(format!(
                "unknown PDB pattern strategy `{other}`; expected `systematic(...)` or `greedy(...)`"
            )),
        }
    }

    fn build_patterns(
        self,
        task: &dyn AbstractNumericTask,
    ) -> (PatternCollection, usize, PdbHeuristicConfig) {
        match self.patterns {
            PdbPatternsConfig::Systematic(config) => (
                generate_systematic_patterns(task, config),
                config.max_pdb_states,
                self.systematic_heuristic,
            ),
            PdbPatternsConfig::Greedy(config) => {
                let max_pdb_states = config.max_pdb_states;
                let heuristic = config.pdb_heuristic_config();
                let pattern = generate_greedy_pattern(task, config);
                (
                    PatternCollection::new(vec![pattern]),
                    max_pdb_states,
                    heuristic,
                )
            }
        }
    }
}

impl ComponentUse {
    fn needs_source_sets(self) -> bool {
        matches!(self, Self::RegionalCostPartitioning)
    }

    fn needs_transition_system(self) -> bool {
        !matches!(self, Self::Standalone)
    }
}

pub(crate) fn split_component_sources(
    combinator: &str,
    args: &[ConfigArg],
) -> Result<(Vec<ConfigCall>, Vec<ConfigArg>), String> {
    let mut sources = Vec::new();
    let mut options = Vec::new();
    let mut uses_source_list = false;

    for (index, arg) in args.iter().enumerate() {
        if let ConfigValue::List(values) = arg.value()
            && arg.key().is_none()
        {
            if uses_source_list {
                return Err(format!("duplicate source list for `{combinator}`"));
            }
            if index != 0 {
                return Err(format!(
                    "the `[...]` source list for `{combinator}` must be the first argument"
                ));
            }
            uses_source_list = true;
            if values.is_empty() {
                return Err(format!(
                    "`{combinator}` requires at least one abstraction source inside `[...]`"
                ));
            }
            for value in values {
                if let Some(source) = component_source_call(value) {
                    sources.push(source);
                    continue;
                }
                if combinator == "scp"
                    && let ConfigValue::Atom(option) = value
                    && let Some((key, _)) = option.split_once('=')
                    && SCP_COMBINATOR_OPTIONS.contains(&key)
                {
                    return Err(format!(
                        "`{option}` is a collection option and cannot appear inside the source list; move it after `]`"
                    ));
                }
                let name = match value {
                    ConfigValue::Atom(name) => name.as_str(),
                    ConfigValue::Call(call) => call.name(),
                    ConfigValue::List(_) => {
                        return Err("nested lists are not supported".to_string());
                    }
                };
                return Err(format!(
                    "unknown abstraction source `{name}`; expected one of: {}",
                    component_source_help()
                ));
            }
            continue;
        }

        let Some(source) = component_source_call(arg.value()) else {
            options.push(arg.clone());
            continue;
        };
        if let Some(key) = arg.key() {
            return Err(format!(
                "abstraction source `{}` must be positional, not `{key}=...`",
                source.name()
            ));
        }
        if uses_source_list {
            return Err(format!(
                "abstraction sources must all appear inside the `[...]` list; found stray source `{}`",
                source.name()
            ));
        }
        sources.push(source);
    }

    Ok((sources, options))
}

fn take_construction_deadline(
    options: Vec<ConfigArg>,
) -> Result<(Vec<ConfigArg>, Option<Instant>), String> {
    let mut remaining_options = Vec::with_capacity(options.len());
    let mut max_time = None;
    for option in options {
        if option.key() != Some("construction_max_time") {
            remaining_options.push(option);
            continue;
        }
        if max_time.is_some() {
            return Err("duplicate option `construction_max_time`".to_string());
        }
        let seconds = f64::from_option_value(option.value())?;
        if !seconds.is_finite() || seconds <= 0.0 {
            return Err(format!(
                "construction_max_time must be finite and > 0, got {seconds}"
            ));
        }
        let duration = Duration::try_from_secs_f64(seconds)
            .map_err(|error| format!("invalid construction_max_time {seconds}: {error}"))?;
        max_time = Some(duration);
    }
    let deadline = max_time
        .map(|duration| {
            Instant::now()
                .checked_add(duration)
                .ok_or_else(|| "construction_max_time is too large".to_string())
        })
        .transpose()?;
    Ok((remaining_options, deadline))
}

pub(crate) fn canonical_sources_and_deadline(
    args: &[ConfigArg],
) -> Result<(Vec<ConfigCall>, Option<Instant>), String> {
    let (sources, options) = split_component_sources("canonical", args)?;
    let (options, deadline) = take_construction_deadline(options)?;
    if let Some(option) = options.first() {
        let key = option.key().unwrap_or("<positional>");
        return Err(format!("unknown `canonical` combinator option `{key}`"));
    }
    if sources.is_empty() {
        return Err("`canonical` requires at least one abstraction source".to_string());
    }
    Ok((sources, deadline))
}

#[derive(Debug)]
pub(crate) struct ScpSourceConfig {
    pub sources: Vec<ConfigCall>,
    pub options: Vec<ConfigArg>,
    pub construction_deadline: Option<Instant>,
}

pub(crate) fn scp_sources_options_and_deadline(
    args: &[ConfigArg],
) -> Result<ScpSourceConfig, String> {
    let (sources, options) = split_component_sources("scp", args)?;
    let (options, construction_deadline) = take_construction_deadline(options)?;
    validate_scp_combinator_options(&options)?;
    Ok(ScpSourceConfig {
        sources,
        options,
        construction_deadline,
    })
}

pub(crate) fn require_only_component_sources(
    combinator: &str,
    args: &[ConfigArg],
) -> Result<Vec<ConfigCall>, String> {
    let (sources, options) = split_component_sources(combinator, args)?;
    if let Some(option) = options.first() {
        let description = option.key().map_or_else(
            || format_config_value(option.value()),
            |key| format!("{key}={}", format_config_value(option.value())),
        );
        return Err(format!(
            "`{combinator}` accepts only these abstraction sources: {}; got `{description}`",
            component_source_help(),
        ));
    }
    if sources.is_empty() {
        return Err(format!(
            "`{combinator}` requires at least one abstraction source: {}",
            component_source_help(),
        ));
    }
    Ok(sources)
}

pub(crate) fn validate_scp_combinator_options(args: &[ConfigArg]) -> Result<(), String> {
    let mut seen = HashSet::new();
    for arg in args {
        let key = arg.key().ok_or_else(|| {
            "options in hierarchical `scp(...)` must be named; abstraction sources are the only positional arguments"
                .to_string()
        })?;
        if !SCP_COMBINATOR_OPTIONS.contains(&key) {
            return Err(format!(
                "unknown `scp` combinator option `{key}`; abstraction-generation options belong inside one of: {}",
                component_source_help(),
            ));
        }
        if !seen.insert(key) {
            return Err(format!("duplicate option `{key}` for `scp`"));
        }
    }
    Ok(())
}

pub(crate) fn build_components<'task>(
    task: &'task dyn AbstractNumericTask,
    sources: &[ConfigCall],
    component_use: ComponentUse,
    construction_deadline: Option<Instant>,
) -> Result<Vec<AbstractionComponent<'task>>, HeuristicBuildError> {
    if sources.is_empty() {
        return Err("an abstraction collection requires at least one source"
            .to_string()
            .into());
    }

    let construction_start = Instant::now();
    let mut components = Vec::new();
    for (source_index, source) in sources.iter().enumerate() {
        let remaining = remaining_construction_time(construction_deadline)?;
        let before = components.len();
        match source.name() {
            "domain" => {
                let mut config = configure(
                    "domain",
                    DomainAbstractionConfig {
                        compute_source_sets: component_use.needs_source_sets(),
                        ..DomainAbstractionConfig::default()
                    },
                    source.args(),
                )?;
                cap_duration(&mut config.cegar.max_time, remaining);
                info!("Building domain abstraction source {source_index}...");
                let abstraction = DomainAbstractionGenerator::new(config)
                    .and_then(|generator| generator.generate(task))
                    .map_err(|error| {
                        HeuristicBuildError::from_source(
                            error,
                            format!("failed to build domain abstraction source {source_index}"),
                        )
                    })?;
                components.push(AbstractionComponent::from(abstraction));
            }
            "domain_collection" => {
                let mut config = configure(
                    "domain_collection",
                    DomainAbstractionCollectionConfig::default(),
                    source.args(),
                )?;
                if let Some(remaining) = remaining {
                    config.cap_construction_time(remaining);
                }
                // Operator regions do not influence CEGAR. Build them after the collection so
                // canonical, label SCP, and region SCP receive the same generation budget.
                config.member.compute_source_sets = false;
                info!("Building domain abstraction collection source {source_index}...");
                let mut abstractions = DomainAbstractionCollectionGenerator::new(config)
                    .and_then(|generator| generator.generate(task))
                    .map_err(|error| {
                    HeuristicBuildError::from_source(
                        error,
                        format!(
                            "failed to build domain abstraction collection source {source_index}"
                        ),
                    )
                })?;
                if component_use.needs_source_sets() {
                    for (component_index, abstraction) in abstractions.iter_mut().enumerate() {
                        abstraction
                            .ensure_abstract_source_sets(task)
                            .map_err(|error| {
                    HeuristicBuildError::from_source(error, format!("failed to build regional operator regions for domain abstraction source {source_index}, component {component_index}"))
                })?;
                    }
                }
                components.extend(abstractions.into_iter().map(AbstractionComponent::from));
            }
            "cartesian" => {
                let mut config =
                    configure("cartesian", cartesian_config(component_use), source.args())?;
                cap_duration(&mut config.cegar.max_time, remaining);
                info!("Building Cartesian abstraction source {source_index}...");
                let abstraction = CartesianAbstractionGenerator::new(config)
                    .map_err(|error| {
                        HeuristicBuildError::from_source(
                            error,
                            format!(
                                "failed to construct Cartesian abstraction source {source_index}"
                            ),
                        )
                    })?
                    .generate(task)
                    .map_err(|error| {
                        HeuristicBuildError::from_source(
                            error,
                            format!("failed to build Cartesian abstraction source {source_index}"),
                        )
                    })?;
                components.push(AbstractionComponent::from(abstraction));
            }
            "cartesian_collection" => {
                let mut config = configure(
                    "cartesian_collection",
                    CartesianAbstractionCollectionConfig {
                        member: cartesian_config(component_use),
                        ..CartesianAbstractionCollectionConfig::default()
                    },
                    source.args(),
                )?;
                if let Some(remaining) = remaining {
                    config.cap_construction_time(remaining);
                }
                info!("Building Cartesian abstraction collection source {source_index}...");
                let abstractions = CartesianAbstractionCollectionGenerator::new(config)
                    .map_err(|error| {
                    HeuristicBuildError::from_source(error, format!("failed to construct Cartesian abstraction collection source {source_index}"))
                })?
                    .generate(task)
                    .map_err(|error| {
                    HeuristicBuildError::from_source(error, format!("failed to build Cartesian abstraction collection source {source_index}"))
                })?;
                components.extend(abstractions.into_iter().map(AbstractionComponent::from));
            }
            "icaps26_cartesian" => {
                validate_restricted_task(task)?;
                let mut config = apply_icaps26_cartesian_options(source.args(), component_use)?;
                cap_duration(&mut config.max_time, remaining);
                info!("Building ICAPS 2026 Cartesian abstraction source {source_index}...");
                let abstraction = Icaps26CartesianGenerator::new(config)
                    .map_err(|error| {
                        format!(
                            "failed to construct ICAPS 2026 Cartesian source {source_index}: {error:#}"
                        )
                    })?
                    .generate(task)
                    .map_err(|error| {
                        format!(
                            "failed to build ICAPS 2026 Cartesian source {source_index}: {error:#}"
                        )
                    })?;
                components.push(AbstractionComponent::from(abstraction));
            }
            "pdb" | "numeric_pdb" => {
                validate_restricted_task(task)?;
                let config = PdbSourceConfig::from_args(source.args())?;
                info!("Building numeric PDB source {source_index}...");
                let (patterns, max_pdb_states, heuristic_config) = config.build_patterns(task);
                let pdbs = PdbCollection::with_heuristic_config(
                    task,
                    patterns,
                    max_pdb_states,
                    heuristic_config,
                )
                .map_err(|error| {
                    format!("failed to build numeric PDB source {source_index}: {error}")
                })?
                .into_pdbs();
                components.extend(pdbs.into_iter().map(AbstractionComponent::from));
            }
            other => {
                return Err(format!(
                    "unknown abstraction source `{other}`; expected one of {}",
                    component_source_help(),
                )
                .into());
            }
        }
        if components.len() == before {
            return Err(format!(
                "abstraction source `{}` produced no components",
                source.name()
            )
            .into());
        }
    }

    let states = components
        .iter()
        .map(AbstractionComponent::num_states)
        .collect::<Vec<_>>();
    let total_states = states.iter().try_fold(0usize, |total, states| {
        total.checked_add(*states).ok_or_else(|| {
            "abstraction collection state count exceeds addressable memory".to_string()
        })
    })?;
    let domain_abstract_operators = components
        .iter()
        .filter_map(AbstractionComponent::as_domain)
        .map(|abstraction| abstraction.abstract_operators.len())
        .try_fold(0usize, |total, operators| {
            total.checked_add(operators).ok_or_else(|| {
                "domain abstract-operator count exceeds addressable memory".to_string()
            })
        })?;
    let cartesian_transitions = components
        .iter()
        .filter_map(AbstractionComponent::as_cartesian)
        .map(|abstraction| abstraction.transition_count)
        .try_fold(0usize, |total, transitions| {
            total
                .checked_add(transitions)
                .ok_or_else(|| "Cartesian transition count exceeds addressable memory".to_string())
        })?;
    info!(
        "Abstraction collection: abstractions={}, total_states={}, states={:?}, domain_abstract_operators={}, cartesian_transitions={}, construction_time={:.3}s",
        components.len(),
        total_states,
        states,
        domain_abstract_operators,
        cartesian_transitions,
        construction_start.elapsed().as_secs_f64(),
    );

    Ok(components)
}

pub(crate) fn remaining_construction_time(
    deadline: Option<Instant>,
) -> Result<Option<Duration>, HeuristicBuildError> {
    let Some(deadline) = deadline else {
        return Ok(None);
    };
    deadline
        .checked_duration_since(Instant::now())
        .filter(|remaining| !remaining.is_zero())
        .map(Some)
        .ok_or(HeuristicBuildError::ConstructionTimeout)
}

fn cap_duration(limit: &mut Option<Duration>, remaining: Option<Duration>) {
    if let Some(remaining) = remaining {
        *limit = Some(limit.map_or(remaining, |current| current.min(remaining)));
    }
}

fn component_source_call(value: &ConfigValue) -> Option<ConfigCall> {
    match value {
        ConfigValue::Call(call) if is_component_source_name(call.name()) => Some(call.clone()),
        ConfigValue::Atom(name) if is_component_source_name(name) => {
            Some(ConfigCall::new(name.clone(), Vec::new()))
        }
        ConfigValue::List(_) => None,
        _ => None,
    }
}

fn is_component_source_name(name: &str) -> bool {
    COMPONENT_SOURCE_NAMES.contains(&name)
}

pub(crate) fn apply_icaps26_cartesian_options(
    args: &[ConfigArg],
    component_use: ComponentUse,
) -> Result<Icaps26CartesianConfig, String> {
    configure(
        "icaps26_cartesian",
        Icaps26CartesianConfig {
            compute_source_sets: component_use.needs_source_sets(),
            retain_transition_system: component_use.needs_transition_system(),
            ..Icaps26CartesianConfig::default()
        },
        args,
    )
}

/// The Cartesian defaults for a component used as `component_use`.
pub(crate) fn cartesian_config(component_use: ComponentUse) -> CartesianAbstractionConfig {
    CartesianAbstractionConfig {
        compute_source_sets: component_use.needs_source_sets(),
        retain_transition_system: component_use.needs_transition_system(),
        ..CartesianAbstractionConfig::default()
    }
}

fn format_config_value(value: &ConfigValue) -> String {
    match value {
        ConfigValue::Atom(atom) => atom.clone(),
        ConfigValue::Call(call) => format!("{}(...)", call.name()),
        ConfigValue::List(_) => value.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use crate::config::{ConfigArg, ConfigCall, ConfigValue};

    use super::HeuristicBuildError;

    use super::{
        ComponentUse, PdbPatternsConfig, PdbSourceConfig, cartesian_config, configure,
        remaining_construction_time, require_only_component_sources, split_component_sources,
    };
    use crate::evaluation::abstractions::common::cegar::SplitSelection;

    #[test]
    fn separates_component_sources_from_combinator_options() {
        let args = vec![
            ConfigArg::new(
                None,
                ConfigValue::Call(ConfigCall::new("domain", Vec::new())),
            ),
            ConfigArg::new(
                None,
                ConfigValue::Call(ConfigCall::new("cartesian", Vec::new())),
            ),
            ConfigArg::new(
                Some("max_time".to_string()),
                ConfigValue::Atom("5".to_string()),
            ),
        ];
        let (sources, options) = split_component_sources("scp", &args).unwrap();
        assert_eq!(sources.len(), 2);
        assert_eq!(options.len(), 1);
    }

    #[test]
    fn flat_and_list_sources_split_identically() {
        let flat = crate::config::parse_heuristic_spec(
            "scp(domain_collection(max_collection_size=1000),cartesian(max_states=100),partitioning=region)",
        )
        .unwrap();
        let listed = crate::config::parse_heuristic_spec(
            "scp([domain_collection(max_collection_size=1000),cartesian(max_states=100)],partitioning=region)",
        )
        .unwrap();
        assert_eq!(
            split_component_sources("scp", &flat.args).unwrap(),
            split_component_sources("scp", &listed.args).unwrap()
        );
    }

    #[test]
    fn source_list_errors_name_the_invalid_structure() {
        fn split(raw: &str) -> Result<(Vec<ConfigCall>, Vec<ConfigArg>), String> {
            let spec = crate::config::parse_heuristic_spec(raw).unwrap();
            split_component_sources("scp", &spec.args)
        }

        assert_eq!(
            split("scp([domian()])").unwrap_err(),
            format!(
                "unknown abstraction source `domian`; expected one of: {}",
                super::component_source_help()
            )
        );
        assert_eq!(
            split("scp([domain_collection(), partitioning=region])").unwrap_err(),
            "`partitioning=region` is a collection option and cannot appear inside the source list; move it after `]`"
        );
        assert_eq!(
            split("scp([], partitioning=region)").unwrap_err(),
            "`scp` requires at least one abstraction source inside `[...]`"
        );
        assert_eq!(
            split("scp([domain_collection()], cartesian())").unwrap_err(),
            "abstraction sources must all appear inside the `[...]` list; found stray source `cartesian`"
        );
        assert_eq!(
            split("scp([domain_collection()], [cartesian()])").unwrap_err(),
            "duplicate source list for `scp`"
        );
    }

    #[test]
    fn all_collection_combinators_accept_a_leading_source_list() {
        for combinator in ["max", "canonical", "scp"] {
            let spec = crate::config::parse_heuristic_spec(&format!(
                "{combinator}([domain_collection(),cartesian()])"
            ))
            .unwrap();
            let (sources, options) = split_component_sources(combinator, &spec.args).unwrap();
            assert_eq!(
                sources.iter().map(ConfigCall::name).collect::<Vec<_>>(),
                ["domain_collection", "cartesian"]
            );
            assert!(options.is_empty());
        }
    }

    #[test]
    fn pdb_source_patterns_are_a_closed_strategy_enum() {
        let systematic = crate::config::parse_heuristic_spec(
            "pdb(patterns=systematic(max_pdb_states=321,max_pattern_size=3),frontier_heuristic=lmcut)",
        )
        .unwrap();
        let config = PdbSourceConfig::from_args(&systematic.args).unwrap();
        let PdbPatternsConfig::Systematic(patterns) = config.patterns else {
            panic!("expected systematic patterns");
        };
        assert_eq!(patterns.max_pdb_states, 321);
        assert_eq!(patterns.max_pattern_size, 3);
        assert_eq!(
            config.systematic_heuristic.frontier_heuristic,
            crate::evaluation::abstractions::pdb::pattern_database::PdbInternalHeuristic::Lmcut
        );

        let greedy = crate::config::parse_heuristic_spec(
            "pdb(patterns=greedy(max_pdb_states=654,numeric_first=false,random_seed=7))",
        )
        .unwrap();
        let config = PdbSourceConfig::from_args(&greedy.args).unwrap();
        let PdbPatternsConfig::Greedy(patterns) = config.patterns else {
            panic!("expected greedy patterns");
        };
        assert_eq!(patterns.max_pdb_states, 654);
        assert!(!patterns.numeric_first);
        assert_eq!(patterns.random_seed, 7);

        let cegar = crate::config::parse_heuristic_spec("pdb(patterns=cegar())").unwrap();
        let error = PdbSourceConfig::from_args(&cegar.args).unwrap_err();
        assert_eq!(
            error,
            "unknown PDB pattern strategy `cegar`; expected `systematic(...)` or `greedy(...)`"
        );
    }

    #[test]
    fn max_and_canonical_reject_non_source_options() {
        let args = vec![ConfigArg::new(
            Some("max_time".to_string()),
            ConfigValue::Atom("5".to_string()),
        )];
        let error = require_only_component_sources("canonical", &args).unwrap_err();
        assert!(error.contains("accepts only"));
    }

    #[test]
    fn expired_shared_deadline_is_a_typed_timeout() {
        let error = remaining_construction_time(Some(Instant::now())).unwrap_err();
        assert!(matches!(error, HeuristicBuildError::ConstructionTimeout));
        assert_eq!(error.into_io_error().kind(), std::io::ErrorKind::TimedOut);
    }

    #[test]
    fn parses_native_cartesian_split_selection() {
        for (name, expected) in [
            ("min_growth", SplitSelection::MinGrowth),
            ("random", SplitSelection::Random),
        ] {
            let args = vec![ConfigArg::new(
                Some("split_selection".to_string()),
                ConfigValue::Atom(name.to_string()),
            )];
            let config = configure(
                "cartesian",
                cartesian_config(ComponentUse::Standalone),
                &args,
            )
            .unwrap();
            assert_eq!(config.cegar.split_selection, expected);
        }
    }

    #[test]
    fn rejects_unknown_native_cartesian_split_selection() {
        let args = vec![ConfigArg::new(
            Some("split_selection".to_string()),
            ConfigValue::Atom("magic".to_string()),
        )];
        let error = configure(
            "cartesian",
            cartesian_config(ComponentUse::Standalone),
            &args,
        )
        .unwrap_err();
        assert!(
            error.starts_with("cartesian(...): invalid split_selection `magic`"),
            "{error}"
        );
    }
}
