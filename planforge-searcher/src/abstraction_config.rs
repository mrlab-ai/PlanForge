use std::collections::HashSet;
use std::time::Duration;

use planforge_search::config::{
    ApplyOptions, ConfigArg, ConfigCall, ConfigValue, FromOptionValue,
};
use planforge_search::evaluation::abstraction_collections::component::AbstractionComponent;
use planforge_search::evaluation::cartesian_abstractions::{
    CartesianAbstractionCollectionConfig, CartesianAbstractionCollectionGenerator,
    CartesianAbstractionConfig, CartesianAbstractionGenerator,
};
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ComponentUse {
    Standalone,
    LabelCostPartitioning,
    AbstractOperatorCostPartitioning,
}

impl ComponentUse {
    fn needs_operator_footprints(self) -> bool {
        matches!(self, Self::AbstractOperatorCostPartitioning)
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
            "`{combinator}` accepts only domain(...), cartesian(...), cartesian_collection(...), and pdb(...) sources; got `{description}`"
        ));
    }
    if sources.is_empty() {
        return Err(format!(
            "`{combinator}` requires at least one domain(...), cartesian(...), cartesian_collection(...), or pdb(...) source"
        ));
    }
    Ok(sources)
}

pub(crate) fn validate_scp_combinator_options(args: &[ConfigArg]) -> Result<(), String> {
    const ALLOWED: &[&str] = &[
        "max_time",
        "table_construction_max_time",
        "max_size",
        "interval",
        "combine_labels",
        "scoring_function",
        "orders",
        "order_optimization_max_time",
        "saturator",
        "random_seed",
        "use_abstract_operator_cost_partitioning",
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
) -> Result<Vec<AbstractionComponent<'task>>, String> {
    if sources.is_empty() {
        return Err("an abstraction collection requires at least one source".to_string());
    }

    let mut components = Vec::new();
    for (source_index, source) in sources.iter().enumerate() {
        let before = components.len();
        match source.name() {
            "domain" | "domain_abstractions" => {
                let mut config = DomainAbstractionCollectionGeneratorMultipleCegarConfig::default();
                config.apply_options(source.args())?;
                config.compute_operator_footprints = component_use.needs_operator_footprints();
                info!("Building domain abstraction source {source_index}...");
                let abstractions = DomainAbstractionCollectionGeneratorMultipleCegar::new(config)
                    .generate_collection(task)
                    .map_err(|error| {
                        format!(
                            "failed to build domain abstraction source {source_index}: {error:#}"
                        )
                    })?;
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
                let config = apply_cartesian_options(source.args(), component_use)?;
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
                let config = apply_cartesian_collection_options(source.args(), component_use)?;
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
                    "unknown abstraction source `{other}`; expected domain(...), cartesian(...), cartesian_collection(...), or pdb(...)"
                ));
            }
        }
        if components.len() == before {
            return Err(format!(
                "abstraction source `{}` produced no components",
                source.name()
            ));
        }
    }

    Ok(components)
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
            | "pdb"
            | "numeric_pdb"
    )
}

fn apply_cartesian_options(
    args: &[ConfigArg],
    component_use: ComponentUse,
) -> Result<CartesianAbstractionConfig, String> {
    Ok(apply_cartesian_source_options(args, component_use, false)?.abstraction)
}

fn apply_cartesian_collection_options(
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
            "variants_per_goal" if collection => {
                config.variants_per_goal = usize::from_option_value(arg.value())?;
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
    use planforge_search::config::{ConfigArg, ConfigCall, ConfigValue};

    use super::{require_only_component_sources, split_component_sources};

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
}
