//! `--search` heuristic construction: a parsed [`HeuristicSpec`] in, a
//! `Box<dyn Heuristic>` out.
//!
//! This lives next to the heuristics it builds, so adding one is a change to
//! this crate alone: the heuristic's own module, and one arm of
//! [`build_heuristic_from_spec_with_task_ref`] below.

use planforge_sas::numeric_task::{AbstractNumericTask, TaskRef};
use tracing::info;
use crate::evaluation::domain_abstractions::cegar::CegarConfig;
use crate::evaluation::abstraction_collections::canonical_heuristic::CanonicalAbstractionHeuristic;
use crate::evaluation::abstraction_collections::component::AbstractionComponent;
use crate::evaluation::abstraction_collections::max_heuristic::MaxAbstractionHeuristic;
use crate::evaluation::abstraction_collections::saturated_cost_partitioning_online_heuristic::{
    FillScpHeuristic, SaturatedCostPartitioningOnlineHeuristic,
};
use crate::evaluation::cartesian_abstractions::{
    CartesianAbstractPlanSelection, CartesianAbstractionConfig, CartesianAbstractionGenerator,
    CartesianAbstractionHeuristic, CartesianFlawCandidateGeneration,
    CartesianRefinementDirection, CartesianSplitSelection,
};
use crate::evaluation::check_admissible::CheckAdmissibleHeuristic;
use crate::evaluation::domain_abstractions::domain_abstraction_collection_generator_multiple_cegar::{
    DomainAbstractionCollectionGeneratorMultipleCegar,
};
use crate::evaluation::domain_abstractions::domain_abstraction_generator::DomainAbstractionGenerator;
use crate::evaluation::domain_abstractions::domain_abstraction_heuristic::DomainAbstractionHeuristic;
#[cfg(feature = "cplex")]
use crate::evaluation::domain_abstractions::posthoc_optimization_heuristic::PostHocOptimizationHeuristic;
use crate::evaluation::numeric_landmarks::lm_cut_numeric_heuristic::LandmarkCutNumericHeuristic;
use crate::evaluation::pattern_databases::pattern_generator_systematic::{
    SystematicPatternGeneratorConfig, generate_systematic_patterns,
};
use crate::evaluation::pattern_databases::pdb_collection::PdbCollection;
use crate::evaluation::pattern_databases::validate_restricted_task;
use crate::evaluation::pattern_databases::pdb_heuristic::GreedyNumericPdbHeuristic;
use std::time::{Duration, Instant};

mod abstraction_config;
#[cfg(test)]
mod tests;

use crate::config::{ApplyOptions, ConfigArg, HeuristicSpec};

/// Fail before construction starts if `spec` names a heuristic this build
/// cannot supply a solver backend for. Nothing else in the pipeline can tell
/// the difference between "no CPLEX compiled in" and "CPLEX said no", and a
/// half-hour translation before that error is a waste.
///
/// The list of LP-backed names lives here, with the heuristics themselves, so
/// that adding one is still a change to this crate alone.
pub fn preflight_required_backends(spec: &HeuristicSpec) -> std::io::Result<()> {
    if !spec.contains_call("numeric_potential")
        && !spec.contains_call("pot_da_ocp")
        && !spec.contains_call("posthoc_optimization")
        && !spec.contains_call("pho")
    {
        return Ok(());
    }
    #[cfg(feature = "cplex")]
    {
        crate::evaluation::numeric_potentials::assert_cplex_ready().map_err(std::io::Error::other)
    }
    #[cfg(not(feature = "cplex"))]
    {
        Err(std::io::Error::other(
            "the requested LP-backed heuristic requires unrestricted CPLEX, which is not compiled into this build; rebuild with `--features cplex` and set CPLEX_ROOT",
        ))
    }
}

use crate::evaluation::Heuristic;
use abstraction_config::{
    ComponentUse, build_components, remaining_construction_time, require_only_component_sources,
    split_component_sources, validate_scp_combinator_options,
};

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
    config: &crate::evaluation::domain_abstractions::domain_abstraction_collection_generator_multiple_cegar::DomainAbstractionCollectionGeneratorMultipleCegarConfig,
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
    if !matches!(
        config.flaw_kind,
        crate::evaluation::cegar::FlawKind::Progression
            | crate::evaluation::cegar::FlawKind::ExecuteEntirePlan
    ) {
        return Err(format!(
            "Cartesian abstractions do not support flaw_kind={}; expected progression or execute_entire_plan",
            config.flaw_kind
        ));
    }
    Ok(CartesianAbstractionConfig {
        max_states: config.max_abstraction_size,
        max_time,
        combine_labels: config.combine_labels,
        compute_operator_footprints,
        retain_transition_system: true,
        random_seed: config.random_seed,
        flaw_kind: config.flaw_kind,
        refinement_direction: CartesianRefinementDirection::Progression,
        abstract_plan_selection: CartesianAbstractPlanSelection::BackwardShortestPath,
        flaw_candidate_generation: CartesianFlawCandidateGeneration::General,
        split_selection_rank: None,
        split_selection: CartesianSplitSelection::MinTransitionGrowth,
        debug: config.debug,
    })
}

fn cartesian_config_from_cegar(config: &CegarConfig) -> CartesianAbstractionConfig {
    CartesianAbstractionConfig {
        max_states: config.max_abstraction_size,
        max_time: config.max_time,
        combine_labels: config.combine_labels,
        compute_operator_footprints: false,
        retain_transition_system: true,
        random_seed: config.random_seed,
        flaw_kind: config.flaw_kind,
        refinement_direction: CartesianRefinementDirection::Progression,
        abstract_plan_selection: CartesianAbstractPlanSelection::BackwardShortestPath,
        flaw_candidate_generation: CartesianFlawCandidateGeneration::General,
        split_selection_rank: None,
        split_selection: CartesianSplitSelection::MinTransitionGrowth,
        debug: config.debug,
    }
}

fn validate_cartesian_cegar_options(args: &[ConfigArg]) -> Result<(), String> {
    const ORDER: &[&str] = &[
        "max_abstraction_size",
        "max_iterations",
        "max_time",
        "use_wildcard_plans",
        "combine_labels",
        "random_seed",
        "flaw_treatment",
        "flaw_kind",
        "init_split_method",
    ];
    crate::config::for_each_option(args, ORDER, |key, _| match key {
        "max_abstraction_size" | "max_time" | "combine_labels" | "random_seed" | "flaw_kind" => {
            Ok(())
        }
        "max_iterations" | "use_wildcard_plans" | "flaw_treatment" | "init_split_method" => Err(
            format!("option `{key}` is not supported for Cartesian abstractions"),
        ),
        other => Err(format!(
            "unknown option `{other}` for `cartesian_abstraction`"
        )),
    })
}

fn validate_legacy_cartesian_collection_options(args: &[ConfigArg]) -> Result<(), String> {
    const DOMAIN_ONLY: &[&str] = &[
        "max_collection_size",
        "total_max_time",
        "stagnation_limit",
        "blacklist_trigger_percentage",
        "enable_blacklist_on_stagnation",
        "blacklist_option",
        "init_split_candidates",
        "init_split_quantity",
        "use_wildcard_plans",
        "flaw_treatment",
        "init_split_method",
        "numeric_split_strategy",
        "collection_strategy",
        "interleave_split_directions",
        "split_direction",
    ];
    for arg in args {
        if let Some(key) = arg.key()
            && DOMAIN_ONLY.contains(&key)
        {
            return Err(format!(
                "option `{key}` is not supported for a Cartesian abstraction collection"
            ));
        }
    }
    Ok(())
}

fn build_max_from_sources<'task>(
    task: &'task dyn AbstractNumericTask,
    sources: &[crate::config::ConfigCall],
    name: &str,
) -> Result<Option<Box<dyn Heuristic + 'task>>, HeuristicBuildError> {
    let components = build_components(task, sources, ComponentUse::Standalone, None)?;
    let heuristic = MaxAbstractionHeuristic::new(Some(name.to_string()), components)
        .map_err(|error| format!("failed to construct `{name}`: {error}"))?;
    Ok(Some(Box::new(heuristic)))
}

fn build_canonical_from_sources<'task>(
    task: &'task dyn AbstractNumericTask,
    sources: &[crate::config::ConfigCall],
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
    sampling_task: TaskRef<'task>,
    sources: &[crate::config::ConfigCall],
    options: &[crate::config::ConfigArg],
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
    let mut config = crate::evaluation::abstraction_collections::saturated_cost_partitioning_online_heuristic::ScpOnlineConfig::default();
    ApplyOptions::apply_options(&mut config, options)?;
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
    let heuristic = SaturatedCostPartitioningOnlineHeuristic::from_components_with_sampling_task(
        Some(name.to_string()),
        components,
        config,
        task,
        sampling_task,
    )
    .map_err(|error| format!("failed to construct `{name}`: {error}"))?;
    remaining_construction_time(construction_deadline)?;
    Ok(Some(Box::new(heuristic)))
}

/// Build the heuristic `spec` names. `Ok(None)` is `blind`, which is the
/// absence of a heuristic rather than a heuristic that returns zero.
///
/// `task` is what the heuristic abstracts; `sampling_task` is the same task as
/// a shared handle, which the heuristics that run their own search or sample
/// their own states need.
pub fn build_heuristic_from_spec<'a>(
    spec: &HeuristicSpec,
    task: &'a dyn AbstractNumericTask,
    sampling_task: TaskRef<'a>,
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
            let inner = build_heuristic_from_spec(&inner_spec, task, sampling_task.clone())?;
            let h = CheckAdmissibleHeuristic::new(inner, sampling_task)
                .map_err(|error| format!("failed to construct `check_admissible`: {error}"))?;
            Ok(Some(Box::new(h) as Box<dyn Heuristic + 'a>))
        }
        "ff" => {
            if !spec.args.is_empty() {
                return Err("`ff` does not accept arguments".to_string().into());
            }
            let h = crate::evaluation::ff_heuristic::FfHeuristic::new(task)
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
                sampling_task,
                &source_config.sources,
                &source_config.options,
                spec.name.as_str(),
                source_config.construction_deadline,
            )
        }
        "domain_abstraction" => {
            info!("Building domain abstraction (CEGAR)...");
            let mut cfg = CegarConfig::default();
            cfg.apply_options(&spec.args)?;
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
            validate_cartesian_cegar_options(&spec.args)?;
            let mut cegar_cfg = CegarConfig::default();
            cegar_cfg.apply_options(&spec.args)?;
            let cfg = cartesian_config_from_cegar(&cegar_cfg);
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
            validate_cartesian_cegar_options(&spec.args)?;
            let mut cegar_cfg = CegarConfig::default();
            cegar_cfg.apply_options(&spec.args)?;
            let generator = CartesianAbstractionGenerator::new(cartesian_config_from_cegar(
                &cegar_cfg,
            ))
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
            use crate::evaluation::domain_abstractions::domain_abstraction_collection_generator_multiple_cegar::DomainAbstractionCollectionGeneratorMultipleCegarConfig;
            let mut cfg = DomainAbstractionCollectionGeneratorMultipleCegarConfig::default();
            ApplyOptions::apply_options(&mut cfg, &spec.args)?;
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
            use crate::evaluation::domain_abstractions::domain_abstraction_collection_generator_multiple_cegar::DomainAbstractionCollectionGeneratorMultipleCegarConfig;
            let mut cfg = DomainAbstractionCollectionGeneratorMultipleCegarConfig::default();
            ApplyOptions::apply_options(&mut cfg, &spec.args)?;
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
            use crate::evaluation::domain_abstractions::domain_abstraction_collection_generator_multiple_cegar::DomainAbstractionCollectionGeneratorMultipleCegarConfig;
            let mut cfg = DomainAbstractionCollectionGeneratorMultipleCegarConfig::default();
            ApplyOptions::apply_options(&mut cfg, &spec.args)?;
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
            use crate::config::for_each_option;
            use crate::evaluation::numeric_potentials::{
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
            da_config.apply_options(&da_args)?;
            da_config.compute_operator_footprints = false;
            info!("Building recorded domain abstraction for pot_da_ocp...");
            let abstraction = DomainAbstractionGenerator::new(da_config)
                .map_err(|error| format!("failed to construct pot_da_ocp abstraction: {error:#}"))?
                .generate(task)
                .map_err(|error| format!("failed to build pot_da_ocp abstraction: {error:#}"))?;
            let mut potential_config = NumericPotentialConfig::default();
            potential_config.apply_options(&potential_args)?;
            let heuristic = PotentialAbstractionOcpHeuristic::new(
                task,
                sampling_task,
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
            use crate::evaluation::numeric_potentials::{
                NumericPotentialConfig, NumericPotentialHeuristic,
            };

            let mut config = NumericPotentialConfig::default();
            config.apply_options(&spec.args)?;
            let heuristic = NumericPotentialHeuristic::from_config(task, sampling_task, config)
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
                    sampling_task,
                    &source_config.sources,
                    &source_config.options,
                    spec.name.as_str(),
                    source_config.construction_deadline,
                );
            }
            let use_cartesian = spec.name == "scp_online_cartesian";
            if use_cartesian {
                validate_legacy_cartesian_collection_options(&spec.args)?;
            }
            let mut cfg = crate::evaluation::abstraction_collections::saturated_cost_partitioning_online_heuristic::ScpOnlineConfig::default();
            ApplyOptions::apply_options(&mut cfg, &spec.args)?;
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
            let h = SaturatedCostPartitioningOnlineHeuristic::from_components_with_sampling_task(
                None,
                components,
                cfg,
                task,
                sampling_task,
            )
            .map_err(|e| format!("failed to construct scp_online heuristic: {e}"))?;
            Ok(Some(Box::new(h) as Box<dyn Heuristic + 'a>))
        }
        "fillscp" | "fill_scp" | "fillscp_cartesian" | "fill_scp_cartesian" => {
            let use_cartesian = matches!(
                spec.name.as_str(),
                "fillscp_cartesian" | "fill_scp_cartesian"
            );
            if use_cartesian {
                validate_legacy_cartesian_collection_options(&spec.args)?;
            }
            let mut cfg = crate::evaluation::abstraction_collections::saturated_cost_partitioning_online_heuristic::FillScpConfig::default();
            ApplyOptions::apply_options(&mut cfg, &spec.args)?;
            cfg.force_full_goal_tasks();
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
            let mut cfg = crate::evaluation::pattern_databases::pattern_generator_greedy::GreedyPatternGeneratorConfig::default();
            ApplyOptions::apply_options(&mut cfg, &spec.args)?;
            let h = GreedyNumericPdbHeuristic::new(task, cfg)
                .map_err(|e| format!("failed to build greedy numeric pdb heuristic: {e}"))?;
            Ok(Some(Box::new(h) as Box<dyn Heuristic + 'a>))
        }
        "canonical_numeric_pdb" => {
            validate_restricted_task(task)?;
            let mut cfg = crate::evaluation::pattern_databases::canonical_pdb_heuristic::CanonicalNumericPdbConfig::default();
            ApplyOptions::apply_options(&mut cfg, &spec.args)?;
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
            let mut cfg = crate::evaluation::pattern_databases::canonical_pdb_heuristic::CanonicalNumericPdbConfig::default();
            ApplyOptions::apply_options(&mut cfg, &spec.args)?;
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
            let mut cfg = crate::evaluation::numeric_landmarks::lm_cut_numeric_heuristic::LmCutNumericConfig::default();
            ApplyOptions::apply_options(&mut cfg, &spec.args)?;
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
    args: &[ConfigArg],
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
    Ok(HeuristicSpec::from_value(arg.value()))
}
