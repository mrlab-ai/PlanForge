use clap::Parser;
use tracing::info;
use planforge_cli_utils::*;
use planforge_sas::numeric_task::{AbstractNumericTask, NumericRootTask, TaskRef};
use planforge_sas::state_registry::StateRegistry;
use std::sync::Arc;
use planforge_search::evaluation::domain_abstractions::cegar::CegarConfig;
use planforge_search::evaluation::abstraction_collections::canonical_heuristic::CanonicalAbstractionHeuristic;
use planforge_search::evaluation::abstraction_collections::component::AbstractionComponent;
use planforge_search::evaluation::abstraction_collections::max_heuristic::MaxAbstractionHeuristic;
use planforge_search::evaluation::abstraction_collections::saturated_cost_partitioning_online_heuristic::{
    FillScpHeuristic, SaturatedCostPartitioningOnlineHeuristic,
};
use planforge_search::evaluation::cartesian_abstractions::{
    CartesianAbstractionConfig, CartesianAbstractionGenerator, CartesianAbstractionHeuristic,
};
use planforge_search::evaluation::check_admissible::CheckAdmissibleHeuristic;
use planforge_search::evaluation::domain_abstractions::domain_abstraction_collection_generator_multiple_cegar::{
    DomainAbstractionCollectionGeneratorMultipleCegar,
};
use planforge_search::evaluation::domain_abstractions::domain_abstraction_generator::DomainAbstractionGenerator;
use planforge_search::evaluation::domain_abstractions::domain_abstraction_heuristic::DomainAbstractionHeuristic;
use planforge_search::task_restriction::{
    build_icaps26_restricted_task, build_restricted_task,
};
#[cfg(feature = "cplex")]
use planforge_search::evaluation::domain_abstractions::posthoc_optimization_heuristic::PostHocOptimizationHeuristic;
use planforge_search::evaluation::numeric_landmarks::lm_cut_numeric_heuristic::LandmarkCutNumericHeuristic;
use planforge_search::evaluation::pattern_databases::pattern_generator_systematic::{
    SystematicPatternGeneratorConfig, generate_systematic_patterns,
};
use planforge_search::evaluation::pattern_databases::pdb_collection::PdbCollection;
use planforge_search::evaluation::pattern_databases::validate_restricted_task;
use planforge_search::evaluation::pattern_databases::pdb_heuristic::GreedyNumericPdbHeuristic;
use planforge_search::search::{
    AStarSearch, SearchEngine, SearchResult, SearchStatus,
};
use std::ffi::OsString;
use std::fs;
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::process::Command;
use std::time::{Duration, Instant};
use std::num::NonZero;
use time::format_description::well_known::iso8601::{Config, TimePrecision};
use tracing_subscriber::fmt::time::UtcTime;
use tracing_subscriber::prelude::*;

mod abstraction_config;
pub mod recursive_config;
#[cfg(feature = "sgd")]
pub mod sgd;

pub use recursive_config::{HeuristicSpec, SearchSpec, parse_heuristic_spec, parse_search_spec};

pub fn preflight_required_backends(spec: &SearchSpec) -> std::io::Result<()> {
    if !spec.contains_call("numeric_potential")
        && !spec.contains_call("pot_da_ocp")
        && !spec.contains_call("posthoc_optimization")
        && !spec.contains_call("pho")
    {
        return Ok(());
    }
    #[cfg(feature = "cplex")]
    {
        planforge_search::evaluation::numeric_potentials::assert_cplex_ready()
            .map_err(std::io::Error::other)
    }
    #[cfg(not(feature = "cplex"))]
    {
        Err(std::io::Error::other(
            "the requested LP-backed heuristic requires unrestricted CPLEX, which is not compiled into this build; rebuild with `--features cplex` and set CPLEX_ROOT",
        ))
    }
}

use abstraction_config::{
    ComponentUse, build_components, remaining_construction_time, require_only_component_sources,
    split_component_sources, validate_scp_combinator_options,
};
use planforge_search::evaluation::Heuristic;

#[derive(Debug)]
pub enum HeuristicBuildError {
    ConstructionTimeout,
    Failure(String),
}

impl HeuristicBuildError {
    pub fn into_io_error(self) -> std::io::Error {
        match self {
            Self::ConstructionTimeout => std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "shared abstraction construction deadline exceeded",
            ),
            Self::Failure(message) => std::io::Error::other(message),
        }
    }
}

impl std::fmt::Display for HeuristicBuildError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConstructionTimeout => {
                formatter.write_str("shared abstraction construction deadline exceeded")
            }
            Self::Failure(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for HeuristicBuildError {}

impl From<String> for HeuristicBuildError {
    fn from(message: String) -> Self {
        Self::Failure(message)
    }
}

fn cartesian_config_from_collection(
    config: &planforge_search::evaluation::domain_abstractions::domain_abstraction_collection_generator_multiple_cegar::DomainAbstractionCollectionGeneratorMultipleCegarConfig,
    compute_operator_footprints: bool,
) -> Result<CartesianAbstractionConfig, String> {
    let max_time = if config.abstraction_generation_max_time.is_finite() {
        Some(
            Duration::try_from_secs_f64(config.abstraction_generation_max_time).map_err(
                |error| {
                    format!(
                        "invalid Cartesian abstraction_generation_max_time {}: {error}",
                        config.abstraction_generation_max_time
                    )
                },
            )?,
        )
    } else {
        None
    };
    Ok(CartesianAbstractionConfig {
        max_states: config.max_abstraction_size,
        max_time,
        combine_labels: config.combine_labels,
        compute_operator_footprints,
        random_seed: config.random_seed,
        debug: config.debug,
        ..Default::default()
    })
}

fn build_max_from_sources<'task>(
    task: &'task dyn AbstractNumericTask,
    sources: &[planforge_search::config::ConfigCall],
    name: &str,
) -> Result<Option<Box<dyn Heuristic + 'task>>, HeuristicBuildError> {
    let components = build_components(task, sources, ComponentUse::Standalone, None)?;
    let heuristic = MaxAbstractionHeuristic::new(Some(name.to_string()), components)
        .map_err(|error| format!("failed to construct `{name}`: {error}"))?;
    Ok(Some(Box::new(heuristic)))
}

fn build_canonical_from_sources<'task>(
    task: &'task dyn AbstractNumericTask,
    sources: &[planforge_search::config::ConfigCall],
    name: &str,
    construction_deadline: Option<Instant>,
) -> Result<Option<Box<dyn Heuristic + 'task>>, HeuristicBuildError> {
    let components = build_components(
        task,
        sources,
        ComponentUse::Standalone,
        construction_deadline,
    )?;
    remaining_construction_time(construction_deadline)?;
    let heuristic = CanonicalAbstractionHeuristic::new(Some(name.to_string()), task, components)
        .map_err(|error| format!("failed to construct `{name}`: {error}"))?;
    remaining_construction_time(construction_deadline)?;
    Ok(Some(Box::new(heuristic)))
}

fn build_scp_from_sources<'task>(
    task: &'task dyn AbstractNumericTask,
    sampling_task: Option<TaskRef<'task>>,
    sources: &[planforge_search::config::ConfigCall],
    options: &[planforge_search::config::ConfigArg],
    name: &str,
    construction_deadline: Option<Instant>,
) -> Result<Option<Box<dyn Heuristic + 'task>>, HeuristicBuildError> {
    if sources.is_empty() {
        return Err(format!(
            "`{name}` requires at least one domain(...), cartesian(...), cartesian_collection(...), or pdb(...) source"
        )
        .into());
    }
    validate_scp_combinator_options(options)?;
    let mut config = planforge_search::evaluation::abstraction_collections::saturated_cost_partitioning_online_heuristic::ScpOnlineConfig::default();
    recursive_config::ApplyOptions::apply_options(&mut config, options)?;
    let component_use = if config.partitioning.uses_regions() {
        ComponentUse::RegionalCostPartitioning
    } else {
        ComponentUse::LabelCostPartitioning
    };
    let components = build_components(task, sources, component_use, construction_deadline)?;
    if let Some(remaining) = remaining_construction_time(construction_deadline)? {
        let remaining = remaining.as_secs_f64();
        config.table_construction_max_time = config.table_construction_max_time.min(remaining);
        config.initial_order_generation_max_time =
            config.initial_order_generation_max_time.min(remaining);
        config.order_optimization_max_time = config.order_optimization_max_time.min(remaining);
    }
    let heuristic = if let Some(sampling_task) = sampling_task {
        SaturatedCostPartitioningOnlineHeuristic::from_components_with_sampling_task(
            Some(name.to_string()),
            components,
            config,
            task,
            sampling_task,
        )
    } else {
        SaturatedCostPartitioningOnlineHeuristic::from_components(
            Some(name.to_string()),
            components,
            config,
            task,
        )
    }
    .map_err(|error| format!("failed to construct `{name}`: {error}"))?;
    remaining_construction_time(construction_deadline)?;
    Ok(Some(Box::new(heuristic)))
}

/// Build a heuristic from a parsed `HeuristicSpec`. Used by both this crate's
/// `run()` and by the top-level `planforge` binary.
pub fn build_heuristic_from_spec<'a>(
    spec: &HeuristicSpec,
    task: &'a dyn AbstractNumericTask,
) -> Result<Option<Box<dyn Heuristic + 'a>>, HeuristicBuildError> {
    build_heuristic_from_spec_internal(spec, task, None)
}

pub fn build_heuristic_from_spec_with_task_ref<'a>(
    spec: &HeuristicSpec,
    task: &'a dyn AbstractNumericTask,
    sampling_task: TaskRef<'a>,
) -> Result<Option<Box<dyn Heuristic + 'a>>, HeuristicBuildError> {
    build_heuristic_from_spec_internal(spec, task, Some(sampling_task))
}

fn build_heuristic_from_spec_internal<'a>(
    spec: &HeuristicSpec,
    task: &'a dyn AbstractNumericTask,
    sampling_task: Option<TaskRef<'a>>,
) -> Result<Option<Box<dyn Heuristic + 'a>>, HeuristicBuildError> {
    match spec.name.as_str() {
        "blind" => {
            if !spec.args.is_empty() {
                return Err("`blind` does not accept arguments".to_string().into());
            }
            Ok(None)
        }
        "check_admissible" => {
            let inner_spec = single_wrapped_heuristic_spec("check_admissible", &spec.args)?;
            // The oracle solves the remaining task from scratch, which needs a
            // registry of its own and therefore a shared handle on the task.
            let sampling_task = sampling_task.ok_or_else(|| {
                "`check_admissible` needs a shared task handle for its goal-distance oracle; \
                 build it through `build_heuristic_from_spec_with_task_ref`"
                    .to_string()
            })?;
            let inner =
                build_heuristic_from_spec_internal(&inner_spec, task, Some(sampling_task.clone()))?;
            let h = CheckAdmissibleHeuristic::new(inner, sampling_task)
                .map_err(|error| format!("failed to construct `check_admissible`: {error}"))?;
            Ok(Some(Box::new(h) as Box<dyn Heuristic + 'a>))
        }
        "ff" => {
            if !spec.args.is_empty() {
                return Err("`ff` does not accept arguments".to_string().into());
            }
            let h = planforge_search::evaluation::ff_heuristic::FfHeuristic::new(task)
                .map_err(|e| format!("failed to construct ff heuristic: {e}"))?;
            Ok(Some(Box::new(h) as Box<dyn Heuristic + 'a>))
        }
        "max" => {
            let sources = require_only_component_sources("max", &spec.args)?;
            build_max_from_sources(task, &sources, "max")
        }
        "canonical" => {
            let (sources, construction_deadline) =
                abstraction_config::canonical_sources_and_deadline(&spec.args)?;
            build_canonical_from_sources(task, &sources, "canonical", construction_deadline)
        }
        "scp" | "cost_partitioning" => {
            let source_config = abstraction_config::scp_sources_options_and_deadline(&spec.args)?;
            build_scp_from_sources(
                task,
                sampling_task.clone(),
                &source_config.sources,
                &source_config.options,
                spec.name.as_str(),
                source_config.construction_deadline,
            )
        }
        "domain_abstraction" => {
            info!("Building domain abstraction (CEGAR)...");
            let mut cfg = CegarConfig::default();
            recursive_config::apply_da_options(&mut cfg, &spec.args)?;
            // Single DA reads only the distance table; footprints are
            // SCP-specific. Skip the per-concrete-op StateRegion cost.
            cfg.compute_operator_footprints = false;
            let generator = DomainAbstractionGenerator::new(cfg)
                .map_err(|e| format!("failed to construct DomainAbstractionGenerator: {e:#}"))?;
            let abstraction = generator
                .generate(task)
                .map_err(|e| format!("failed to build domain abstraction: {e:#}"))?;
            Ok(Some(
                Box::new(DomainAbstractionHeuristic::new(None, abstraction))
                    as Box<dyn Heuristic + 'a>,
            ))
        }
        "cartesian_abstraction" => {
            info!("Building Cartesian abstraction (CEGAR)...");
            let mut cegar_cfg = CegarConfig::default();
            recursive_config::apply_da_options(&mut cegar_cfg, &spec.args)?;
            let cfg = CartesianAbstractionConfig {
                max_states: cegar_cfg.max_abstraction_size,
                max_time: cegar_cfg.max_time,
                combine_labels: cegar_cfg.combine_labels,
                compute_operator_footprints: false,
                random_seed: cegar_cfg.random_seed,
                debug: cegar_cfg.debug,
                ..Default::default()
            };
            let generator = CartesianAbstractionGenerator::new(cfg)
                .map_err(|error| format!("failed to construct Cartesian generator: {error:#}"))?;
            let abstraction = generator
                .generate(task)
                .map_err(|error| format!("failed to build Cartesian abstraction: {error:#}"))?;
            Ok(Some(
                Box::new(CartesianAbstractionHeuristic::new(None, abstraction))
                    as Box<dyn Heuristic + 'a>,
            ))
        }
        "max_cartesian_abstraction" | "canonical_cartesian_abstraction" => {
            let mut cegar_cfg = CegarConfig::default();
            recursive_config::apply_da_options(&mut cegar_cfg, &spec.args)?;
            let generator = CartesianAbstractionGenerator::new(CartesianAbstractionConfig {
                max_states: cegar_cfg.max_abstraction_size,
                max_time: cegar_cfg.max_time,
                combine_labels: cegar_cfg.combine_labels,
                compute_operator_footprints: false,
                random_seed: cegar_cfg.random_seed,
                debug: cegar_cfg.debug,
                ..Default::default()
            })
            .map_err(|error| format!("failed to construct Cartesian generator: {error:#}"))?;
            let abstraction = generator
                .generate(task)
                .map_err(|error| format!("failed to build Cartesian abstraction: {error:#}"))?;
            let components = vec![AbstractionComponent::cartesian(None, abstraction)];
            if spec.name == "max_cartesian_abstraction" {
                let heuristic = MaxAbstractionHeuristic::new(
                    Some("max_cartesian_abstraction".to_string()),
                    components,
                )?;
                Ok(Some(Box::new(heuristic) as Box<dyn Heuristic + 'a>))
            } else {
                let heuristic = CanonicalAbstractionHeuristic::new(
                    Some("canonical_cartesian_abstraction".to_string()),
                    task,
                    components,
                )?;
                Ok(Some(Box::new(heuristic) as Box<dyn Heuristic + 'a>))
            }
        }
        "canonical_domain_abstractions" => {
            use planforge_search::evaluation::domain_abstractions::domain_abstraction_collection_generator_multiple_cegar::DomainAbstractionCollectionGeneratorMultipleCegarConfig;
            let mut cfg = DomainAbstractionCollectionGeneratorMultipleCegarConfig::default();
            recursive_config::ApplyOptions::apply_options(&mut cfg, &spec.args)?;
            // Canonical never consumes operator footprints — skip ~12 GB of
            // per-concrete-op StateRegion storage on big tasks.
            cfg.compute_operator_footprints = false;
            let generator = DomainAbstractionCollectionGeneratorMultipleCegar::new(cfg);
            info!("Building canonical domain abstractions (CEGAR)...");
            let abstractions = generator
                .generate_collection(task)
                .map_err(|e| format!("failed to build canonical domain abstractions: {e:#}"))?;
            let components = abstractions
                .into_iter()
                .enumerate()
                .map(|(index, abstraction)| {
                    AbstractionComponent::domain(
                        Some(format!("canonical_domain_abstraction_{index}")),
                        abstraction,
                    )
                })
                .collect();
            let h = CanonicalAbstractionHeuristic::new(
                Some("canonical_domain_abstractions".to_string()),
                task,
                components,
            )
            .map_err(|e| format!("failed to construct canonical abstraction heuristic: {e}"))?;
            Ok(Some(Box::new(h) as Box<dyn Heuristic + 'a>))
        }
        "multi_domain_abstractions" => {
            use planforge_search::evaluation::domain_abstractions::domain_abstraction_collection_generator_multiple_cegar::DomainAbstractionCollectionGeneratorMultipleCegarConfig;
            let mut cfg = DomainAbstractionCollectionGeneratorMultipleCegarConfig::default();
            recursive_config::ApplyOptions::apply_options(&mut cfg, &spec.args)?;
            cfg.compute_operator_footprints = false;
            let generator = DomainAbstractionCollectionGeneratorMultipleCegar::new(cfg);
            info!("Building multiple domain abstractions (CEGAR)...");
            let abstractions = generator
                .generate_collection(task)
                .map_err(|e| format!("failed to build multi domain abstractions: {e:#}"))?;
            let components = abstractions
                .into_iter()
                .enumerate()
                .map(|(index, abstraction)| {
                    AbstractionComponent::domain(
                        Some(format!("multi_domain_abstraction_{index}")),
                        abstraction,
                    )
                })
                .collect();
            let h = MaxAbstractionHeuristic::new(
                Some("multi_domain_abstractions".to_string()),
                components,
            )
            .map_err(|e| format!("failed to construct max abstraction heuristic: {e}"))?;
            Ok(Some(Box::new(h) as Box<dyn Heuristic + 'a>))
        }
        #[cfg(feature = "cplex")]
        "posthoc_optimization" | "pho" => {
            use planforge_search::evaluation::domain_abstractions::domain_abstraction_collection_generator_multiple_cegar::DomainAbstractionCollectionGeneratorMultipleCegarConfig;
            let mut cfg = DomainAbstractionCollectionGeneratorMultipleCegarConfig::default();
            recursive_config::ApplyOptions::apply_options(&mut cfg, &spec.args)?;
            cfg.compute_operator_footprints = false;
            let generator = DomainAbstractionCollectionGeneratorMultipleCegar::new(cfg);
            info!("Building posthoc_optimization domain abstractions (CEGAR)...");
            let abstractions = generator.generate_collection(task).map_err(|e| {
                format!("failed to build posthoc_optimization domain abstractions: {e:#}")
            })?;
            let h = PostHocOptimizationHeuristic::new(None, task, abstractions)
                .map_err(|e| format!("failed to construct posthoc_optimization heuristic: {e}"))?;
            Ok(Some(Box::new(h) as Box<dyn Heuristic + 'a>))
        }
        #[cfg(not(feature = "cplex"))]
        "posthoc_optimization" | "pho" => Err(
            "posthoc_optimization requires CPLEX, which is not compiled into this build. \
             Rebuild with `--features cplex` and set CPLEX_ROOT to an unrestricted CPLEX \
             installation."
                .to_string()
                .into(),
        ),
        #[cfg(feature = "cplex")]
        "pot_da_ocp" => {
            use planforge_search::config::{
                ApplyOptions, ConfigArg, FromOptionValue, for_each_option,
            };
            use planforge_search::evaluation::numeric_potentials::{
                NumericPotentialConfig, PotentialAbstractionOcpHeuristic,
            };

            const ORDER: &[&str] = &[
                "abstraction",
                "nonnegative",
                "max_potential",
                "ignore_numeric_variables",
                "bounds",
                "precision",
                "epsilon",
                "dump_lp",
            ];
            let mut abstraction_call = None;
            let mut nonnegative = false;
            let mut potential_args = Vec::new();
            for_each_option(&spec.args, ORDER, |key, value| {
                match key {
                    "abstraction" => abstraction_call = Some(value.as_call()?.clone()),
                    "nonnegative" => nonnegative = bool::from_option_value(value)?,
                    other => potential_args.push(ConfigArg::new(
                        Some(other.to_string()),
                        value.clone(),
                    )),
                }
                Ok(())
            })?;
            let abstraction_call = abstraction_call.ok_or_else(|| {
                "pot_da_ocp requires abstraction=domain_abstraction_cegar(...)".to_string()
            })?;
            if !matches!(
                abstraction_call.name(),
                "domain_abstraction_cegar" | "domain_abstraction"
            ) {
                return Err(format!(
                    "pot_da_ocp requires a domain_abstraction_cegar generator, got `{}`",
                    abstraction_call.name()
                )
                .into());
            }
            let mut record_transition_system = false;
            let mut max_recorded_transitions = 100_000_usize;
            let mut da_args = Vec::new();
            for arg in abstraction_call.args() {
                match arg.key() {
                    Some("record_transition_system") => {
                        record_transition_system =
                            bool::from_option_value(arg.value())?;
                    }
                    Some("record_transition_system_max_transitions") => {
                        max_recorded_transitions =
                            usize::from_option_value(arg.value())?;
                    }
                    _ => da_args.push(arg.clone()),
                }
            }
            if !record_transition_system {
                return Err(
                    "pot_da_ocp requires record_transition_system=true in its abstraction generator"
                        .to_string()
                        .into(),
                );
            }
            let mut da_config = CegarConfig::default();
            recursive_config::apply_da_options(&mut da_config, &da_args)?;
            da_config.compute_operator_footprints = false;
            info!("Building recorded domain abstraction for pot_da_ocp...");
            let abstraction = DomainAbstractionGenerator::new(da_config)
                .map_err(|error| format!("failed to construct pot_da_ocp abstraction: {error:#}"))?
                .generate(task)
                .map_err(|error| format!("failed to build pot_da_ocp abstraction: {error:#}"))?;
            let mut potential_config = NumericPotentialConfig::default();
            potential_config.apply_options(&potential_args)?;
            let task_ref = sampling_task.ok_or_else(|| {
                "pot_da_ocp requires an owned task reference".to_string()
            })?;
            let heuristic = PotentialAbstractionOcpHeuristic::new(
                task,
                task_ref,
                abstraction,
                potential_config,
                nonnegative,
                max_recorded_transitions,
            )
            .map_err(|error| format!("failed to construct pot_da_ocp: {error}"))?;
            Ok(Some(Box::new(heuristic) as Box<dyn Heuristic + 'a>))
        }
        #[cfg(not(feature = "cplex"))]
        "pot_da_ocp" => Err(
            "pot_da_ocp requires unrestricted CPLEX, which is not compiled into this build. \
             Rebuild with `--features cplex` and set CPLEX_ROOT to an unrestricted CPLEX \
             installation."
                .to_string()
                .into(),
        ),
        #[cfg(feature = "cplex")]
        "numeric_potential" => {
            use planforge_search::config::ApplyOptions;
            use planforge_search::evaluation::numeric_potentials::{
                NumericPotentialConfig, NumericPotentialHeuristic,
            };

            let mut config = NumericPotentialConfig::default();
            config.apply_options(&spec.args)?;
            let task_ref = sampling_task.ok_or_else(|| {
                "numeric_potential requires an owned task reference".to_string()
            })?;
            let heuristic = NumericPotentialHeuristic::from_config(task, task_ref, config)
                .map_err(|error| {
                    format!("failed to construct numeric_potential heuristic: {error}")
                })?;
            Ok(Some(Box::new(heuristic) as Box<dyn Heuristic + 'a>))
        }
        #[cfg(not(feature = "cplex"))]
        "numeric_potential" => Err(
            "numeric_potential requires unrestricted CPLEX, which is not compiled into this build. \
             Rebuild with `--features cplex` and set CPLEX_ROOT to an unrestricted CPLEX \
             installation."
                .to_string()
                .into(),
        ),
        "scp_online" | "scp_online_cartesian" => {
            let (component_sources, _) = split_component_sources(&spec.args)?;
            if !component_sources.is_empty() {
                let source_config =
                    abstraction_config::scp_sources_options_and_deadline(&spec.args)?;
                return build_scp_from_sources(
                    task,
                    sampling_task.clone(),
                    &source_config.sources,
                    &source_config.options,
                    spec.name.as_str(),
                    source_config.construction_deadline,
                );
            }
            let mut cfg = planforge_search::evaluation::abstraction_collections::saturated_cost_partitioning_online_heuristic::ScpOnlineConfig::default();
            recursive_config::ApplyOptions::apply_options(&mut cfg, &spec.args)?;
            let use_cartesian = spec.name == "scp_online_cartesian";
            let abstractions = if use_cartesian {
                Vec::new()
            } else {
                let generator = DomainAbstractionCollectionGeneratorMultipleCegar::new(
                    cfg.collection_config.clone(),
                );
                info!("Building scp_online domain abstractions (CEGAR)...");
                generator
                    .generate_collection(task)
                    .map_err(|e| format!("failed to build scp_online domain abstractions: {e:#}"))?
            };
            let pdbs = if cfg.use_numeric_pdbs {
                validate_restricted_task(task)?;
                info!("Building scp_online systematic numeric PDBs...");
                let patterns = generate_systematic_patterns(
                    task,
                    SystematicPatternGeneratorConfig {
                        max_pdb_states: cfg.max_pdb_states,
                        max_pattern_size: cfg.max_pattern_size,
                        only_interesting_patterns: cfg.only_interesting_patterns,
                    },
                );
                PdbCollection::with_heuristic_config(
                    task,
                    patterns,
                    cfg.max_pdb_states,
                    cfg.pdb_heuristic_config(),
                )
                .map_err(|e| format!("failed to build scp_online numeric PDBs: {e}"))?
                .into_pdbs()
            } else {
                Vec::new()
            };
            let mut components: Vec<AbstractionComponent<'a>> = abstractions
                .into_iter()
                .enumerate()
                .map(|(index, abstraction)| {
                    AbstractionComponent::domain(
                        Some(format!("scp_online_domain_{index}")),
                        abstraction,
                    )
                })
                .collect();
            if use_cartesian {
                info!("Building scp_online Cartesian abstraction (CEGAR)...");
                let cartesian_config = cartesian_config_from_collection(
                    &cfg.collection_config,
                    cfg.partitioning.uses_regions(),
                )?;
                let abstraction = CartesianAbstractionGenerator::new(cartesian_config)
                    .map_err(|error| format!("failed to construct Cartesian generator: {error:#}"))?
                    .generate(task)
                    .map_err(|error| {
                        format!("failed to build scp_online Cartesian abstraction: {error:#}")
                    })?;
                components.push(AbstractionComponent::cartesian(None, abstraction));
            }
            components.extend(pdbs.into_iter().map(AbstractionComponent::pattern_database));
            let h = if let Some(sampling_task) = sampling_task.clone() {
                SaturatedCostPartitioningOnlineHeuristic::from_components_with_sampling_task(
                    None,
                    components,
                    cfg,
                    task,
                    sampling_task,
                )
            } else {
                SaturatedCostPartitioningOnlineHeuristic::from_components(
                    None, components, cfg, task,
                )
            }
            .map_err(|e| format!("failed to construct scp_online heuristic: {e}"))?;
            Ok(Some(Box::new(h) as Box<dyn Heuristic + 'a>))
        }
        "fillscp" | "fill_scp" | "fillscp_cartesian" | "fill_scp_cartesian" => {
            let mut cfg = planforge_search::evaluation::abstraction_collections::saturated_cost_partitioning_online_heuristic::FillScpConfig::default();
            recursive_config::ApplyOptions::apply_options(&mut cfg, &spec.args)?;
            cfg.force_full_goal_tasks();
            let use_cartesian = matches!(
                spec.name.as_str(),
                "fillscp_cartesian" | "fill_scp_cartesian"
            );
            let (abstractions, cartesian_abstractions) = if use_cartesian {
                info!("Building fillSCP Cartesian abstraction (CEGAR)...");
                let cartesian_config = cartesian_config_from_collection(
                    &cfg.collection_config,
                    cfg.partitioning.uses_regions(),
                )?;
                let abstraction = CartesianAbstractionGenerator::new(cartesian_config)
                    .map_err(|error| format!("failed to construct Cartesian generator: {error:#}"))?
                    .generate(task)
                    .map_err(|error| {
                        format!("failed to build fillSCP Cartesian abstraction: {error:#}")
                    })?;
                (Vec::new(), vec![abstraction])
            } else {
                let generator = DomainAbstractionCollectionGeneratorMultipleCegar::new(
                    cfg.collection_config.clone(),
                );
                info!("Building fillSCP domain abstractions (CEGAR)...");
                let abstractions = generator
                    .generate_collection(task)
                    .map_err(|e| format!("failed to build fillSCP domain abstractions: {e:#}"))?;
                (abstractions, Vec::new())
            };
            let h = FillScpHeuristic::new_with_cartesian(
                None,
                abstractions,
                cartesian_abstractions,
                cfg,
                task,
            )
            .map_err(|e| format!("failed to construct fillSCP heuristic: {e}"))?;
            Ok(Some(Box::new(h) as Box<dyn Heuristic + 'a>))
        }
        "greedy_numeric_pdb" => {
            let mut cfg = planforge_search::evaluation::pattern_databases::pattern_generator_greedy::GreedyPatternGeneratorConfig::default();
            recursive_config::ApplyOptions::apply_options(&mut cfg, &spec.args)?;
            let h = GreedyNumericPdbHeuristic::new(task, cfg)
                .map_err(|e| format!("failed to build greedy numeric pdb heuristic: {e}"))?;
            Ok(Some(Box::new(h) as Box<dyn Heuristic + 'a>))
        }
        "canonical_numeric_pdb" => {
            validate_restricted_task(task)?;
            let mut cfg = planforge_search::evaluation::pattern_databases::canonical_pdb_heuristic::CanonicalNumericPdbConfig::default();
            recursive_config::ApplyOptions::apply_options(&mut cfg, &spec.args)?;
            let patterns = generate_systematic_patterns(
                task,
                SystematicPatternGeneratorConfig {
                    max_pdb_states: cfg.max_pdb_states,
                    max_pattern_size: cfg.max_pattern_size,
                    only_interesting_patterns: cfg.only_interesting_patterns,
                },
            );
            let components = PdbCollection::with_heuristic_config(
                task,
                patterns,
                cfg.max_pdb_states,
                cfg.pdb_heuristic_config(),
            )
            .map_err(|e| format!("failed to build canonical numeric PDBs: {e}"))?
            .into_pdbs()
            .into_iter()
            .map(AbstractionComponent::pattern_database)
            .collect();
            let h = CanonicalAbstractionHeuristic::new(
                Some("canonical_numeric_pdb".to_string()),
                task,
                components,
            )
            .map_err(|e| format!("failed to build canonical numeric PDB heuristic: {e}"))?;
            Ok(Some(Box::new(h) as Box<dyn Heuristic + 'a>))
        }
        "max_numeric_pdb" => {
            validate_restricted_task(task)?;
            let mut cfg = planforge_search::evaluation::pattern_databases::canonical_pdb_heuristic::CanonicalNumericPdbConfig::default();
            recursive_config::ApplyOptions::apply_options(&mut cfg, &spec.args)?;
            let patterns = generate_systematic_patterns(
                task,
                SystematicPatternGeneratorConfig {
                    max_pdb_states: cfg.max_pdb_states,
                    max_pattern_size: cfg.max_pattern_size,
                    only_interesting_patterns: cfg.only_interesting_patterns,
                },
            );
            let components = PdbCollection::with_heuristic_config(
                task,
                patterns,
                cfg.max_pdb_states,
                cfg.pdb_heuristic_config(),
            )
            .map_err(|e| format!("failed to build max numeric PDBs: {e}"))?
            .into_pdbs()
            .into_iter()
            .map(AbstractionComponent::pattern_database)
            .collect();
            let h = MaxAbstractionHeuristic::new(Some("max_numeric_pdb".to_string()), components)
                .map_err(|e| format!("failed to build max numeric PDB heuristic: {e}"))?;
            Ok(Some(Box::new(h) as Box<dyn Heuristic + 'a>))
        }
        "lmcutnumeric" => {
            let mut cfg = planforge_search::evaluation::numeric_landmarks::lm_cut_numeric_heuristic::LmCutNumericConfig::default();
            recursive_config::ApplyOptions::apply_options(&mut cfg, &spec.args)?;
            let h = LandmarkCutNumericHeuristic::from_config(task, cfg)
                .map_err(|e| format!("failed to build lmcutnumeric heuristic: {e}"))?;
            Ok(Some(Box::new(h) as Box<dyn Heuristic + 'a>))
        }
        other => Err(format!("unknown heuristic `{other}`").into()),
    }
}

/// Read the single positional heuristic argument of a wrapping heuristic such
/// as `check_admissible(<inner>)`.
fn single_wrapped_heuristic_spec(
    wrapper: &str,
    args: &[recursive_config::ConfigArg],
) -> Result<HeuristicSpec, String> {
    let [arg] = args else {
        return Err(format!(
            "`{wrapper}` expects exactly one heuristic, e.g. `{wrapper}(domain_abstraction())`, \
             but got {} arguments",
            args.len()
        ));
    };
    if let Some(key) = arg.key() {
        return Err(format!(
            "`{wrapper}` takes its heuristic positionally, not as `{key}=...`"
        ));
    }
    recursive_config::heuristic_spec_from_value(arg.value())
}

use tracing_subscriber::filter::LevelFilter;

pub fn init_logger(level: LevelFilter) {
    let timer = UtcTime::new(
        time::format_description::well_known::Iso8601::<
            {
                Config::DEFAULT
                    .set_time_precision(TimePrecision::Second {
                        decimal_digits: NonZero::new(3),
                    })
                    .encode()
            },
        >,
    );
    // Layer for stdout (info + deubg + trace)
    let stdout_layer = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stdout)
        .with_target(false)
        .with_timer(timer)
        .with_filter(level);

    // Layer for stderr (error + warn only)
    let stderr_layer = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stderr)
        .with_target(false)
        .with_filter(LevelFilter::WARN);

    tracing_subscriber::registry()
        .with(stdout_layer)
        .with(stderr_layer)
        .init();
}

#[derive(Parser, Debug, Clone)]
#[command(author, version, about = "Numeric planner")]
pub struct PlannersSearcherCli {
    #[arg(long = "max-memory", value_name = "SIZE", value_parser = parse_memory_limit)]
    pub max_memory: Option<u64>,

    #[arg(long = "max-time", value_name = "DURATION", value_parser = parse_time_limit)]
    pub max_time: Option<Duration>,

    #[arg(long = "log-level")]
    pub log_level: Option<LevelFilter>,

    #[arg(long, hide = true)]
    pub internal_run: bool,

    #[arg(long = "restrict-task")]
    pub restrict_task: bool,

    /// Recursive search configuration.
    /// Examples: `astar(blind())`, `astar(domain_abstraction())`,
    /// `astar(check_admissible(domain_abstraction()))`.
    #[arg(
        long,
        value_name = "SPEC",
        default_value = "astar(blind())",
        value_parser = crate::recursive_config::parse_search_spec
    )]
    pub search: crate::recursive_config::SearchSpec,

    #[arg(value_name = "SAS_FILE", required = true)]
    pub sas_file: String,
}

#[cfg(unix)]
pub fn run_wrapped_process(cli: &PlannersSearcherCli) -> std::io::Result<()> {
    let current_executable = std::env::current_exe()?;
    let mut child_args = vec![OsString::from("--internal-run")];
    let memory_limit = cli
        .max_memory
        .map(planforge_cli_utils::effective_rss_limit)
        .transpose()?;
    if let Some(max_memory) = memory_limit {
        child_args.push(OsString::from("--max-memory"));
        child_args.push(OsString::from(max_memory.to_string()));
    }
    if let Some(max_time) = cli.max_time {
        child_args.push(OsString::from("--max-time"));
        child_args.push(OsString::from(format_time_limit(max_time)));
    }
    // Preserve the selected search configuration when re-executing ourselves.
    if let Some(level) = cli.log_level {
        child_args.push(OsString::from("--log-level"));
        child_args.push(OsString::from(level.to_string()));
    }
    if cli.restrict_task {
        child_args.push(OsString::from("--restrict-task"));
    }
    child_args.push(OsString::from("--search"));
    child_args.push(OsString::from(cli.search.to_string()));
    child_args.extend([cli.sas_file.clone()].iter().map(OsString::from));

    let time_limit = cli.max_time;
    let mut command = Command::new(current_executable);
    command.args(child_args);
    command.stdin(std::process::Stdio::inherit());
    command.stdout(std::process::Stdio::inherit());
    command.stderr(std::process::Stdio::inherit());

    unsafe {
        command.pre_exec(move || apply_process_limits(time_limit, memory_limit));
    }

    let mut child = command.spawn()?;
    #[cfg(target_os = "linux")]
    let status = match memory_limit {
        Some(memory_limit) => {
            planforge_cli_utils::wait_with_memory_limit(&mut child, memory_limit)?
        }
        None => child.wait()?,
    };
    #[cfg(not(target_os = "linux"))]
    let status = child.wait()?;
    let exit_code = normalize_wrapped_exit(status, time_limit, memory_limit);

    std::process::exit(exit_code)
}

#[allow(clippy::field_reassign_with_default)]
pub fn run_internal(cli: &PlannersSearcherCli) -> std::io::Result<SearchResult> {
    register_event_handlers();
    preflight_required_backends(&cli.search)?;

    let sas_file = &cli.sas_file;

    let start_time = std::time::Instant::now();
    let mut task = NumericRootTask::try_from_file(sas_file).map_err(std::io::Error::other)?;
    if cli.restrict_task {
        let original_numeric_count = task.numeric_variables().len();
        let restricted_task = if cli.search.contains_call("icaps26_cartesian") {
            build_icaps26_restricted_task(&task)
        } else {
            build_restricted_task(&task)
        };
        if let Some(restricted_task) = restricted_task.map_err(|err| {
            std::io::Error::other(format!("failed to build restricted task: {err:#}"))
        })? {
            task = restricted_task.into_task();
            info!(
                "restricted task: numeric variables {} -> {}",
                original_numeric_count,
                task.numeric_variables().len()
            );
        }
    }
    let task: TaskRef<'static> = Arc::new(task);
    let parse_time = start_time.elapsed();
    info!("Parsed numeric SAS output in: {:?}", parse_time);

    info!("=== Search Engine ===");
    info!("File: {}", sas_file);
    info!(
        "Variables: {} regular, {} numeric",
        task.variables().len(),
        task.numeric_variables().len()
    );

    let state_registry = StateRegistry::for_task(task.clone());

    // Both A* and GBFS go through identical heuristic construction; only the
    // open-list priority differs. Project the search spec onto (heuristic,
    // priority kind) and let the shared block below build the heuristic.
    let (heuristic_spec, gbfs_priority, mpd) = match &cli.search {
        crate::recursive_config::SearchSpec::Astar(h, mpd) => (h, false, *mpd),
        crate::recursive_config::SearchSpec::Gbfs(h) => (h, true, false),
        crate::recursive_config::SearchSpec::AstarFs(_, _) => {
            return Err(std::io::Error::other(
                "`astar_fs(...)` is implemented in the `planforge` binary path, not `planforge-searcher`",
            ));
        }
        crate::recursive_config::SearchSpec::Sgd(_) => {
            return Err(std::io::Error::other(
                "`sgd(...)` is implemented in the `planforge` binary path, not `planforge-searcher`",
            ));
        }
    };
    let result = {
        {
            let heuristic_override =
                build_heuristic_from_spec_with_task_ref(heuristic_spec, &*task, task.clone())
                    .map_err(HeuristicBuildError::into_io_error)?;

            let time_limit = cli.max_time;
            let memory_limit = cli.max_memory;
            let mut search = if gbfs_priority {
                AStarSearch::new_gbfs(
                    task.clone(),
                    state_registry,
                    heuristic_override,
                    time_limit,
                    memory_limit,
                )
            } else {
                AStarSearch::new_with_mpd(
                    task.clone(),
                    state_registry,
                    heuristic_override,
                    time_limit,
                    memory_limit,
                    mpd,
                )
            };

            info!(
                "Starting {} search with {:?}...",
                if gbfs_priority { "GBFS" } else { "A*" },
                heuristic_spec,
            );
            search.search().map_err(std::io::Error::other)?
        }
    };

    write_plan_file(&result)?;
    print_search_result(&result);

    Ok(result)
}

pub fn exit_code_for_search_status(status: &SearchStatus) -> i32 {
    match status {
        SearchStatus::Timeout => EXIT_TIMEOUT,
        SearchStatus::MemoryLimitReached => EXIT_OUT_OF_MEMORY,
        SearchStatus::InProgress | SearchStatus::Solved(_) | SearchStatus::Failed => EXIT_SUCCESS,
    }
}

/// Write the plan to `sas_plan`, if the search found one.
///
/// Kept separate from [`print_search_result`] because this can fail and the
/// failure has to reach the caller: exiting successfully with no plan on disk
/// is worse than exiting with an error.
pub fn write_plan_file(result: &SearchResult) -> std::io::Result<()> {
    let Some(plan) = result.plan.as_ref() else {
        return Ok(());
    };
    if !matches!(result.status, SearchStatus::Solved(_)) {
        return Ok(());
    }

    let mut plan_content = String::new();
    for op in plan.iter() {
        plan_content.push_str(&format!("({})\n", op.name()));
    }
    fs::write("sas_plan", plan_content)
}

pub fn print_search_result(result: &SearchResult) {
    print_plan_result(result);

    // Fast Downward-style statistics block.
    info!("Expanded {} state(s).", result.nodes_expanded);
    info!("Reopened {} state(s).", result.nodes_reopened);
    info!("Evaluated {} state(s).", result.nodes_evaluated);
    info!("Evaluations: {}", result.evaluations);
    info!("Generated {} state(s).", result.nodes_generated);
    info!("Dead ends: {} state(s).", result.dead_ends);
    info!(
        "Expanded until last jump: {} state(s).",
        result.nodes_expanded_until_last_jump
    );
    info!(
        "Reopened until last jump: {} state(s).",
        result.nodes_reopened_until_last_jump
    );
    info!(
        "Evaluated until last jump: {} state(s).",
        result.nodes_evaluated_until_last_jump
    );
    info!(
        "Generated until last jump: {} state(s).",
        result.nodes_generated_until_last_jump
    );
    info!("Number of registered states: {}", result.registered_states);
    info!("Search time: {:.6}s", result.search_time.as_secs_f64());
}

/// Print the outcome and write a found plan without claiming search statistics.
///
/// Gradient plan synthesis uses this directly because it has no search nodes,
/// frontier, generated states, or expansions.
pub fn print_plan_result(result: &SearchResult) {
    match result.status {
        SearchStatus::Solved(_) => {
            info!("Solution found!");
            if let Some(plan) = result.plan.as_ref() {
                let plan_cost = result
                    .solution_cost
                    .unwrap_or_else(|| plan.iter().map(|op| op.cost() as f64).sum());

                for (i, op) in plan.iter().enumerate() {
                    info!("  {}: {}", i + 1, op.name());
                }

                info!("Plan length: {} step(s).", plan.len());
                info!("Plan cost: {:.6}", plan_cost);
            }
        }
        SearchStatus::Failed => {
            info!("No solution found");
        }
        SearchStatus::Timeout => {
            info!("Search timed out");
        }
        SearchStatus::MemoryLimitReached => {
            info!("Search stopped after reaching the memory limit");
        }
        SearchStatus::InProgress => {
            info!("Search ended in progress");
        }
    }
}
