use std::collections::HashSet;
use std::time::{Duration, Instant};

use planforge_search::config::{
    ApplyOptions, ConfigArg, ConfigCall, ConfigValue, FromOptionValue,
};
use planforge_search::evaluation::abstraction_collections::component::AbstractionComponent;
use planforge_search::evaluation::abstraction_collections::portfolio::CollectionStrategy;
use planforge_search::evaluation::cartesian_abstractions::{
    CartesianAbstractPlanSelection, CartesianAbstractionCollectionConfig,
    CartesianAbstractionCollectionGenerator, CartesianAbstractionConfig,
    CartesianAbstractionGenerator, CartesianFlawCandidateGeneration,
    CartesianRefinementDirection, CartesianSplitSelection,
};
use planforge_search::evaluation::cartesian_abstractions::icaps26::Icaps26SplitSelection;
use planforge_search::evaluation::cegar::FlawKind;
use planforge_search::evaluation::domain_abstractions::domain_abstraction_collection_generator_multiple_cegar::{
    DomainAbstractionCollectionGeneratorMultipleCegar,
    DomainAbstractionCollectionGeneratorMultipleCegarConfig,
};
use planforge_search::evaluation::pattern_databases::canonical_pdb_heuristic::CanonicalNumericPdbConfig;
use planforge_search::evaluation::pattern_databases::pattern_generator_systematic::{
    SystematicPatternGeneratorConfig, generate_systematic_patterns,
};
use planforge_search::evaluation::pattern_databases::pdb_collection::PdbCollection;
use planforge_search::evaluation::pattern_databases::validate_restricted_task;
use planforge_sas::numeric_task::AbstractNumericTask;
use tracing::info;

use crate::HeuristicBuildError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ComponentUse {
    Standalone,
    LabelCostPartitioning,
    RegionalCostPartitioning,
}

impl ComponentUse {
    fn needs_operator_footprints(self) -> bool {
        matches!(self, Self::RegionalCostPartitioning)
    }

    fn needs_transition_system(self) -> bool {
        !matches!(self, Self::Standalone)
    }
}

pub(crate) fn split_component_sources(
    args: &[ConfigArg],
) -> Result<(Vec<ConfigCall>, Vec<ConfigArg>), String> {
    let mut sources = Vec::new();
    let mut options = Vec::new();

    for arg in args {
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
    let (sources, options) = split_component_sources(args)?;
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

pub(crate) struct ScpSourceConfig {
    pub sources: Vec<ConfigCall>,
    pub options: Vec<ConfigArg>,
    pub construction_deadline: Option<Instant>,
}

pub(crate) fn scp_sources_options_and_deadline(
    args: &[ConfigArg],
) -> Result<ScpSourceConfig, String> {
    let (sources, options) = split_component_sources(args)?;
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
    let (sources, options) = split_component_sources(args)?;
    if let Some(option) = options.first() {
        let description = option.key().map_or_else(
            || format_config_value(option.value()),
            |key| format!("{key}={}", format_config_value(option.value())),
        );
        return Err(format!(
            "`{combinator}` accepts only domain(...), cartesian(...), cartesian_collection(...), icaps26_cartesian(...), and pdb(...) sources; got `{description}`"
        ));
    }
    if sources.is_empty() {
        return Err(format!(
            "`{combinator}` requires at least one domain(...), cartesian(...), cartesian_collection(...), icaps26_cartesian(...), or pdb(...) source"
        ));
    }
    Ok(sources)
}

pub(crate) fn validate_scp_combinator_options(args: &[ConfigArg]) -> Result<(), String> {
    const ALLOWED: &[&str] = &[
        "online",
        "max_time",
        "table_construction_max_time",
        "max_size",
        "diversify",
        "samples",
        "max_orders",
        "interval",
        "combine_labels",
        "scoring_function",
        "orders",
        "initial_order_generation_max_time",
        "order_optimization_max_time",
        "saturator",
        "residual_sweeps",
        "random_seed",
        "partitioning",
    ];

    let mut seen = HashSet::new();
    for arg in args {
        let key = arg.key().ok_or_else(|| {
            "options in hierarchical `scp(...)` must be named; abstraction sources are the only positional arguments"
                .to_string()
        })?;
        if !ALLOWED.contains(&key) {
            return Err(format!(
                "unknown `scp` combinator option `{key}`; abstraction-generation options belong inside domain(...), cartesian(...), cartesian_collection(...), or pdb(...)"
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
            "domain" | "domain_abstractions" => {
                let mut config = DomainAbstractionCollectionGeneratorMultipleCegarConfig::default();
                config.apply_options(source.args())?;
                if let Some(remaining) = remaining {
                    let seconds = remaining.as_secs_f64();
                    config.total_max_time = config.total_max_time.min(seconds);
                    config.abstraction_generation_max_time =
                        config.abstraction_generation_max_time.min(seconds);
                }
                // Footprints do not influence CEGAR. Build them after the collection so
                // canonical, label SCP, and region SCP receive the same generation budget.
                config.compute_operator_footprints = false;
                info!("Building domain abstraction source {source_index}...");
                let mut abstractions = DomainAbstractionCollectionGeneratorMultipleCegar::new(
                    config,
                )
                .generate_collection(task)
                .map_err(|error| {
                    format!("failed to build domain abstraction source {source_index}: {error:#}")
                })?;
                if component_use.needs_operator_footprints() {
                    for (component_index, abstraction) in abstractions.iter_mut().enumerate() {
                        abstraction
                            .ensure_abstract_operator_footprints(task)
                            .map_err(|error| {
                                format!(
                                    "failed to build regional footprints for domain abstraction source {source_index}, component {component_index}: {error:#}"
                                )
                            })?;
                    }
                }
                components.extend(abstractions.into_iter().enumerate().map(
                    |(component_index, abstraction)| {
                        AbstractionComponent::domain(
                            Some(format!("domain_{source_index}_{component_index}")),
                            abstraction,
                        )
                    },
                ));
            }
            "cartesian" | "cartesian_abstraction" => {
                let mut config = apply_cartesian_options(source.args(), component_use)?;
                cap_duration(&mut config.max_time, remaining);
                info!("Building Cartesian abstraction source {source_index}...");
                let abstraction = CartesianAbstractionGenerator::new(config)
                    .map_err(|error| {
                        format!(
                            "failed to construct Cartesian abstraction source {source_index}: {error:#}"
                        )
                    })?
                    .generate(task)
                    .map_err(|error| {
                        format!("failed to build Cartesian abstraction source {source_index}: {error:#}")
                    })?;
                components.push(AbstractionComponent::cartesian(
                    Some(format!("cartesian_{source_index}")),
                    abstraction,
                ));
            }
            "cartesian_collection" | "cartesian_abstraction_collection" => {
                let mut config = apply_cartesian_collection_options(source.args(), component_use)?;
                cap_duration(&mut config.total_max_time, remaining);
                cap_duration(&mut config.abstraction.max_time, remaining);
                info!("Building Cartesian abstraction collection source {source_index}...");
                let abstractions = CartesianAbstractionCollectionGenerator::new(config)
                    .map_err(|error| {
                        format!(
                            "failed to construct Cartesian abstraction collection source {source_index}: {error:#}"
                        )
                    })?
                    .generate(task)
                    .map_err(|error| {
                        format!(
                            "failed to build Cartesian abstraction collection source {source_index}: {error:#}"
                        )
                    })?;
                components.extend(abstractions.into_iter().enumerate().map(
                    |(goal_id, abstraction)| {
                        AbstractionComponent::cartesian(
                            Some(format!("cartesian_{source_index}_goal_{goal_id}")),
                            abstraction,
                        )
                    },
                ));
            }
            "icaps26_cartesian" => {
                validate_restricted_task(task)?;
                let mut config = apply_icaps26_cartesian_options(source.args(), component_use)?;
                cap_duration(&mut config.max_time, remaining);
                info!("Building ICAPS 2026 Cartesian abstraction source {source_index}...");
                let abstraction = CartesianAbstractionGenerator::new(config)
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
                components.push(AbstractionComponent::cartesian(
                    Some(format!("icaps26_cartesian_{source_index}")),
                    abstraction,
                ));
            }
            "pdb" | "numeric_pdb" => {
                validate_restricted_task(task)?;
                let mut config = CanonicalNumericPdbConfig::default();
                config.apply_options(source.args())?;
                info!("Building numeric PDB source {source_index}...");
                let patterns = generate_systematic_patterns(
                    task,
                    SystematicPatternGeneratorConfig {
                        max_pdb_states: config.max_pdb_states,
                        max_pattern_size: config.max_pattern_size,
                        only_interesting_patterns: config.only_interesting_patterns,
                    },
                );
                let pdbs = PdbCollection::with_heuristic_config(
                    task,
                    patterns,
                    config.max_pdb_states,
                    config.pdb_heuristic_config(),
                )
                .map_err(|error| {
                    format!("failed to build numeric PDB source {source_index}: {error}")
                })?
                .into_pdbs();
                components.extend(pdbs.into_iter().map(AbstractionComponent::pattern_database));
            }
            other => {
                return Err(format!(
                    "unknown abstraction source `{other}`; expected domain(...), cartesian(...), cartesian_collection(...), icaps26_cartesian(...), or pdb(...)"
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
        .map(|abstraction| abstraction.metadata.transition_count)
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
        _ => None,
    }
}

fn is_component_source_name(name: &str) -> bool {
    matches!(
        name,
        "domain"
            | "domain_abstractions"
            | "cartesian"
            | "cartesian_abstraction"
            | "cartesian_collection"
            | "cartesian_abstraction_collection"
            | "icaps26_cartesian"
            | "pdb"
            | "numeric_pdb"
    )
}

pub(crate) fn apply_icaps26_cartesian_options(
    args: &[ConfigArg],
    component_use: ComponentUse,
) -> Result<CartesianAbstractionConfig, String> {
    let mut config = CartesianAbstractionConfig {
        max_states: usize::MAX,
        max_time: Some(Duration::from_secs(900)),
        ..CartesianAbstractionConfig::default()
    };
    config.compute_operator_footprints = component_use.needs_operator_footprints();
    config.retain_transition_system = component_use.needs_transition_system();
    config.random_seed = Some(2011);
    config.refinement_direction = CartesianRefinementDirection::Regression;
    config.abstract_plan_selection = CartesianAbstractPlanSelection::StableAStar;
    config.flaw_candidate_generation = CartesianFlawCandidateGeneration::DesiredRegion;
    config.split_selection = CartesianSplitSelection::Icaps26(Icaps26SplitSelection::MaxUnwanted);

    let mut seen = HashSet::new();
    for arg in args {
        let key = arg
            .key()
            .ok_or_else(|| "all options for `icaps26_cartesian` must be named".to_string())?;
        if !seen.insert(key) {
            return Err(format!("duplicate option `{key}` for `icaps26_cartesian`"));
        }
        match key {
            "pick" => {
                let value = String::from_option_value(arg.value())?;
                let policy = match value.to_ascii_lowercase().as_str() {
                    "random" => Icaps26SplitSelection::Random,
                    "min_unwanted" => Icaps26SplitSelection::MinUnwanted,
                    "max_unwanted" => Icaps26SplitSelection::MaxUnwanted,
                    _ => {
                        return Err(format!(
                            "invalid ICAPS 2026 split selector `{value}`; expected random, min_unwanted, or max_unwanted"
                        ));
                    }
                };
                config.split_selection = CartesianSplitSelection::Icaps26(policy);
            }
            "max_states" => {
                config.max_states = usize::from_option_value(arg.value())?;
            }
            "max_time" => {
                let seconds = f64::from_option_value(arg.value())?;
                config.max_time = if seconds.is_infinite() {
                    None
                } else {
                    Some(Duration::try_from_secs_f64(seconds).map_err(|error| {
                        format!("invalid ICAPS 2026 max_time {seconds}: {error}")
                    })?)
                };
            }
            "random_seed" => {
                config.random_seed = Some(u64::from_option_value(arg.value())?);
            }
            "combine_labels" => {
                config.combine_labels = bool::from_option_value(arg.value())?;
            }
            "debug" => {
                config.debug = bool::from_option_value(arg.value())?;
            }
            other => {
                return Err(format!("unknown option `{other}` for `icaps26_cartesian`"));
            }
        }
    }
    Ok(config)
}

fn apply_cartesian_options(
    args: &[ConfigArg],
    component_use: ComponentUse,
) -> Result<CartesianAbstractionConfig, String> {
    Ok(apply_cartesian_source_options(args, component_use, false)?.abstraction)
}

pub(crate) fn apply_cartesian_collection_options(
    args: &[ConfigArg],
    component_use: ComponentUse,
) -> Result<CartesianAbstractionCollectionConfig, String> {
    apply_cartesian_source_options(args, component_use, true)
}

fn apply_cartesian_source_options(
    args: &[ConfigArg],
    component_use: ComponentUse,
    collection: bool,
) -> Result<CartesianAbstractionCollectionConfig, String> {
    const POSITIONAL_ORDER: &[&str] = &["max_states", "max_time", "combine_labels", "debug"];

    let source = if collection {
        "cartesian_collection"
    } else {
        "cartesian"
    };
    let mut config = CartesianAbstractionCollectionConfig::default();
    let mut seen = HashSet::new();
    let mut next_positional = 0;
    for arg in args {
        let raw_key = match arg.key() {
            Some(key) => key,
            None => {
                let key = POSITIONAL_ORDER.get(next_positional).ok_or_else(|| {
                    format!(
                        "too many positional arguments for `{source}` (maximum {})",
                        POSITIONAL_ORDER.len()
                    )
                })?;
                next_positional += 1;
                key
            }
        };
        let key = match raw_key {
            "max_abstraction_size" => "max_states",
            "abstraction_generation_max_time" => "max_time",
            other => other,
        };
        if !seen.insert(key.to_string()) {
            return Err(format!("duplicate option `{key}` for `{source}`"));
        }
        match key {
            "max_states" => {
                config.abstraction.max_states = usize::from_option_value(arg.value())?;
            }
            "max_time" => {
                let seconds = f64::from_option_value(arg.value())?;
                config.abstraction.max_time = if seconds.is_infinite() {
                    None
                } else {
                    Some(Duration::try_from_secs_f64(seconds).map_err(|error| {
                        format!("invalid Cartesian max_time {seconds}: {error}")
                    })?)
                };
            }
            "combine_labels" => {
                config.abstraction.combine_labels = bool::from_option_value(arg.value())?;
            }
            "debug" => {
                config.abstraction.debug = bool::from_option_value(arg.value())?;
            }
            "random_seed" => {
                config.abstraction.random_seed = Some(u64::from_option_value(arg.value())?);
            }
            "flaw_kind" => {
                let flaw_kind = FlawKind::from_option_value(arg.value())?;
                if !matches!(
                    flaw_kind,
                    FlawKind::Progression | FlawKind::ExecuteEntirePlan
                ) {
                    return Err(format!(
                        "Cartesian abstractions do not support flaw_kind={flaw_kind}; expected progression or execute_entire_plan"
                    ));
                }
                config.abstraction.flaw_kind = flaw_kind;
            }
            "refinement_direction" => {
                let value = String::from_option_value(arg.value())?;
                config.abstraction.refinement_direction = match value.as_str() {
                    "progression" => CartesianRefinementDirection::Progression,
                    "regression" | "target_centered" => CartesianRefinementDirection::Regression,
                    _ => {
                        return Err(format!(
                            "invalid Cartesian refinement_direction `{value}`; expected progression or regression"
                        ));
                    }
                };
            }
            "split_selection" => {
                let value = String::from_option_value(arg.value())?;
                config.abstraction.split_selection = match value.as_str() {
                    "min_transition_growth" | "min_growth" => {
                        CartesianSplitSelection::MinTransitionGrowth
                    }
                    "max_additive_steps" => CartesianSplitSelection::MaxAdditiveSteps,
                    "random" => CartesianSplitSelection::Random,
                    "least_refined" => CartesianSplitSelection::LeastRefined,
                    _ => {
                        return Err(format!(
                            "invalid Cartesian split_selection `{value}`; expected min_transition_growth, max_additive_steps, random, or least_refined"
                        ));
                    }
                };
            }
            "abstract_plan" => {
                let value = String::from_option_value(arg.value())?;
                config.abstraction.abstract_plan_selection = match value.as_str() {
                    "backward_shortest_path" => {
                        CartesianAbstractPlanSelection::BackwardShortestPath
                    }
                    "stable_astar" => CartesianAbstractPlanSelection::StableAStar,
                    _ => {
                        return Err(format!(
                            "invalid Cartesian abstract_plan `{value}`; expected backward_shortest_path or stable_astar"
                        ));
                    }
                };
            }
            "flaw_candidates" => {
                let value = String::from_option_value(arg.value())?;
                config.abstraction.flaw_candidate_generation = match value.as_str() {
                    "general" => CartesianFlawCandidateGeneration::General,
                    "desired_region" => CartesianFlawCandidateGeneration::DesiredRegion,
                    _ => {
                        return Err(format!(
                            "invalid Cartesian flaw_candidates `{value}`; expected general or desired_region"
                        ));
                    }
                };
            }
            "split_selection_rank" => {
                config.abstraction.split_selection_rank =
                    Some(usize::from_option_value(arg.value())?);
            }
            "variants_per_goal" if collection => {
                config.variants_per_goal = usize::from_option_value(arg.value())?;
            }
            "collection_strategy" if collection => {
                config.collection_strategy = CollectionStrategy::from_option_value(arg.value())?;
            }
            "progressive_goal_roots" if collection => {
                config.progressive_goal_roots = bool::from_option_value(arg.value())?;
            }
            "max_collection_size" if collection => {
                config.max_collection_states = usize::from_option_value(arg.value())?;
            }
            "total_max_time" if collection => {
                let seconds = f64::from_option_value(arg.value())?;
                config.total_max_time = if seconds.is_infinite() {
                    None
                } else {
                    Some(Duration::try_from_secs_f64(seconds).map_err(|error| {
                        format!("invalid Cartesian total_max_time {seconds}: {error}")
                    })?)
                };
            }
            other => {
                return Err(format!("unknown option `{other}` for `{source}`"));
            }
        }
    }
    config.abstraction.compute_operator_footprints = component_use.needs_operator_footprints();
    config.abstraction.retain_transition_system = component_use.needs_transition_system();
    Ok(config)
}

fn format_config_value(value: &ConfigValue) -> String {
    match value {
        ConfigValue::Atom(atom) => atom.clone(),
        ConfigValue::Call(call) => format!("{}(...)", call.name()),
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use planforge_search::config::{ConfigArg, ConfigCall, ConfigValue};

    use crate::HeuristicBuildError;

    use super::{
        ComponentUse, apply_cartesian_options, remaining_construction_time,
        require_only_component_sources, split_component_sources,
    };
    use planforge_search::evaluation::cartesian_abstractions::{
        CartesianAbstractPlanSelection, CartesianFlawCandidateGeneration, CartesianSplitSelection,
    };

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
        let (sources, options) = split_component_sources(&args).unwrap();
        assert_eq!(sources.len(), 2);
        assert_eq!(options.len(), 1);
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
            (
                "min_transition_growth",
                CartesianSplitSelection::MinTransitionGrowth,
            ),
            (
                "max_additive_steps",
                CartesianSplitSelection::MaxAdditiveSteps,
            ),
            ("random", CartesianSplitSelection::Random),
            ("least_refined", CartesianSplitSelection::LeastRefined),
        ] {
            let args = vec![ConfigArg::new(
                Some("split_selection".to_string()),
                ConfigValue::Atom(name.to_string()),
            )];
            let config = apply_cartesian_options(&args, ComponentUse::Standalone).unwrap();
            assert_eq!(config.split_selection, expected);
        }
    }

    #[test]
    fn rejects_unknown_native_cartesian_split_selection() {
        let args = vec![ConfigArg::new(
            Some("split_selection".to_string()),
            ConfigValue::Atom("magic".to_string()),
        )];
        let error = apply_cartesian_options(&args, ComponentUse::Standalone).unwrap_err();
        assert!(error.contains("invalid Cartesian split_selection"));
    }

    #[test]
    fn parses_native_cartesian_abstract_plan_selection() {
        for (name, expected) in [
            (
                "backward_shortest_path",
                CartesianAbstractPlanSelection::BackwardShortestPath,
            ),
            ("stable_astar", CartesianAbstractPlanSelection::StableAStar),
        ] {
            let args = vec![ConfigArg::new(
                Some("abstract_plan".to_string()),
                ConfigValue::Atom(name.to_string()),
            )];
            let config = apply_cartesian_options(&args, ComponentUse::Standalone).unwrap();
            assert_eq!(config.abstract_plan_selection, expected);
        }
    }

    #[test]
    fn parses_native_cartesian_flaw_candidate_generation() {
        for (name, expected) in [
            ("general", CartesianFlawCandidateGeneration::General),
            (
                "desired_region",
                CartesianFlawCandidateGeneration::DesiredRegion,
            ),
        ] {
            let args = vec![ConfigArg::new(
                Some("flaw_candidates".to_string()),
                ConfigValue::Atom(name.to_string()),
            )];
            let config = apply_cartesian_options(&args, ComponentUse::Standalone).unwrap();
            assert_eq!(config.flaw_candidate_generation, expected);
        }
    }
}
