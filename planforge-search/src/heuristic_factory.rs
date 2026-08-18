//! `--search` heuristic construction: a parsed [`HeuristicSpec`] in, a
//! `Box<dyn Heuristic>` out.
//!
//! The registry at the bottom of this file declares every built-in exactly
//! once. Dispatch, backend preflight, task-shape validation, legal names and
//! CLI help are generated from that declaration.

use planforge_sas::numeric_task::{AbstractNumericTask, TaskRef};
use std::sync::OnceLock;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequiredBackend {
    None,
    Cplex,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TaskRequirements {
    pub restricted: bool,
    pub abstractable_goal: bool,
}

impl TaskRequirements {
    pub const ANY: Self = Self {
        restricted: false,
        abstractable_goal: false,
    };
    pub const ABSTRACTABLE_GOAL: Self = Self {
        restricted: false,
        abstractable_goal: true,
    };
    pub const RESTRICTED_ABSTRACTABLE_GOAL: Self = Self {
        restricted: true,
        abstractable_goal: true,
    };

    fn validate(self, task: &dyn AbstractNumericTask) -> Result<(), String> {
        if self.restricted {
            validate_restricted_task(task)?;
        }
        if self.abstractable_goal {
            crate::evaluation::validate_abstractable_goal(task)?;
        }
        Ok(())
    }
}

pub type RequirementsFn = fn(&HeuristicSpec) -> Result<TaskRequirements, String>;
pub type NestedHeuristicsFn = fn(&HeuristicSpec) -> Result<Vec<HeuristicSpec>, String>;

pub type ExternalHeuristicBuilder =
    for<'a> fn(
        &HeuristicSpec,
        &'a dyn AbstractNumericTask,
        TaskRef<'a>,
    ) -> Result<Option<Box<dyn Heuristic + 'a>>, HeuristicBuildError>;

pub struct ExternalHeuristic {
    pub name: &'static str,
    pub backend: RequiredBackend,
    pub requirements: RequirementsFn,
    pub nested_heuristics: NestedHeuristicsFn,
    pub build: ExternalHeuristicBuilder,
}

static EXTERNAL_HEURISTICS: OnceLock<Vec<ExternalHeuristic>> = OnceLock::new();

fn external_heuristics() -> &'static [ExternalHeuristic] {
    EXTERNAL_HEURISTICS.get_or_init(Vec::new)
}

fn external_heuristic(name: &str) -> Option<&'static ExternalHeuristic> {
    external_heuristics()
        .iter()
        .find(|heuristic| heuristic.name == name)
}

/// Register heuristics that are not compiled into PlanForge. Call once, before
/// any search is constructed. Returns an error if called twice or if a name
/// collides with a built-in.
pub fn register_external_heuristics(entries: Vec<ExternalHeuristic>) -> Result<(), String> {
    for (index, entry) in entries.iter().enumerate() {
        if entry.name.is_empty() {
            return Err("external heuristic names must not be empty".to_string());
        }
        if heuristic_plugin(entry.name).is_some() {
            return Err(format!(
                "external heuristic `{}` collides with a built-in heuristic",
                entry.name
            ));
        }
        if entries[..index]
            .iter()
            .any(|registered| registered.name == entry.name)
        {
            return Err(format!(
                "external heuristic `{}` is registered more than once",
                entry.name
            ));
        }
    }
    EXTERNAL_HEURISTICS
        .set(entries)
        .map_err(|_| "external heuristics have already been registered or used".to_string())
}

struct HeuristicPlugin {
    backend: RequiredBackend,
    requirements: RequirementsFn,
    nested_heuristics: NestedHeuristicsFn,
}

fn any_task(_: &HeuristicSpec) -> Result<TaskRequirements, String> {
    Ok(TaskRequirements::ANY)
}

fn abstractable_task(_: &HeuristicSpec) -> Result<TaskRequirements, String> {
    Ok(TaskRequirements::ABSTRACTABLE_GOAL)
}

fn restricted_abstractable_task(_: &HeuristicSpec) -> Result<TaskRequirements, String> {
    Ok(TaskRequirements::RESTRICTED_ABSTRACTABLE_GOAL)
}

fn component_task_requirements(spec: &HeuristicSpec) -> Result<TaskRequirements, String> {
    let restricted = spec.contains_call("pdb") || spec.contains_call("numeric_pdb");
    Ok(TaskRequirements {
        restricted,
        abstractable_goal: true,
    })
}

fn scp_task_requirements(spec: &HeuristicSpec) -> Result<TaskRequirements, String> {
    let (sources, _) = split_component_sources(spec.name.as_str(), &spec.args)?;
    if !sources.is_empty() {
        return component_task_requirements(spec);
    }
    let mut config = crate::evaluation::abstraction_collections::saturated_cost_partitioning_online_heuristic::ScpOnlineConfig::default();
    config.apply_options(&spec.args)?;
    Ok(TaskRequirements {
        restricted: config.use_numeric_pdbs,
        abstractable_goal: true,
    })
}

fn no_nested_heuristics(_: &HeuristicSpec) -> Result<Vec<HeuristicSpec>, String> {
    Ok(Vec::new())
}

fn wrapped_heuristic(spec: &HeuristicSpec) -> Result<Vec<HeuristicSpec>, String> {
    Ok(vec![single_wrapped_heuristic_spec(&spec.name, &spec.args)?])
}

macro_rules! heuristic_registry {
    (
        build($spec:ident, $task:ident, $sampling_task:ident);
        $(
            $entry:ident {
                names: [$($name:literal),+ $(,)?],
                backend: $backend:expr,
                requirements: $requirements:expr,
                nested: $nested:expr,
                build: $body:block
            }
        )+
    ) => {
        crate::plugin_registry! {
            static HEURISTIC_PLUGINS: HeuristicPlugin;
            fn heuristic_plugin;
            entries {
                $(
                    $(
                        $name => HeuristicPlugin {
                            backend: $backend,
                            requirements: $requirements,
                            nested_heuristics: $nested,
                        }
                    ),+
                ),+
            }
        }

        pub const HEURISTIC_HELP: &str = concat!(
            "Built-in heuristics: ",
            $($( $name, ", ", )+)+
            "Externally registered heuristic names are also accepted. ",
            "Custom Rust heuristics can still be passed directly as Box<dyn Heuristic>."
        );

        pub fn heuristic_names() -> impl Iterator<Item = &'static str> {
            HEURISTIC_PLUGINS.iter().map(|(name, _)| *name)
                .chain(external_heuristics().iter().map(|heuristic| heuristic.name))
        }

        pub fn build_heuristic_from_spec<'a>(
            $spec: &HeuristicSpec,
            $task: &'a dyn AbstractNumericTask,
            $sampling_task: TaskRef<'a>,
        ) -> Result<Option<Box<dyn Heuristic + 'a>>, HeuristicBuildError> {
            // Resolving the external registry here also seals an empty
            // registry, so registration after construction is an error.
            let external_heuristics = external_heuristics();
            if let Some(plugin) = heuristic_plugin(&$spec.name) {
                (plugin.requirements)($spec)?.validate($task)?;
                return match $spec.name.as_str() {
                    $(
                        $($name)|+ => $body,
                    )+
                    _ => unreachable!("the registry lookup and generated dispatch have the same names"),
                };
            }
            let external = external_heuristics
                .iter()
                .find(|heuristic| heuristic.name == $spec.name)
                .ok_or_else(|| {
                    format!(
                        "unknown heuristic `{}`; expected one of {}",
                        $spec.name,
                        heuristic_names().collect::<Vec<_>>().join(", ")
                    )
                })?;
            (external.requirements)($spec)?.validate($task)?;
            (external.build)($spec, $task, $sampling_task)
        }
    };
}

/// Validate a heuristic name and every nested heuristic against the generated
/// registry. Parsers call this before translation, while direct Rust callers
/// receive the same check at construction.
pub fn validate_heuristic_spec(spec: &HeuristicSpec) -> Result<(), String> {
    let nested = if let Some(plugin) = heuristic_plugin(&spec.name) {
        (plugin.nested_heuristics)(spec)?
    } else if let Some(external) = external_heuristic(&spec.name) {
        (external.nested_heuristics)(spec)?
    } else {
        return Err(format!(
            "unknown heuristic `{}`; expected one of {}",
            spec.name,
            heuristic_names().collect::<Vec<_>>().join(", ")
        ));
    };
    for nested in nested {
        validate_heuristic_spec(&nested)?;
    }
    Ok(())
}

/// Fail before construction starts if `spec` or one of its nested heuristics
/// needs a solver backend this build cannot supply.
pub fn preflight_required_backends(spec: &HeuristicSpec) -> std::io::Result<()> {
    fn needs_cplex(spec: &HeuristicSpec) -> Result<bool, String> {
        let (backend, nested) = if let Some(plugin) = heuristic_plugin(&spec.name) {
            (plugin.backend, (plugin.nested_heuristics)(spec)?)
        } else if let Some(external) = external_heuristic(&spec.name) {
            (external.backend, (external.nested_heuristics)(spec)?)
        } else {
            return Err(format!(
                "unknown heuristic `{}`; expected one of {}",
                spec.name,
                heuristic_names().collect::<Vec<_>>().join(", ")
            ));
        };
        if backend == RequiredBackend::Cplex {
            return Ok(true);
        }
        for nested in nested {
            if needs_cplex(&nested)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    if !needs_cplex(spec).map_err(std::io::Error::other)? {
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
    ComponentUse, build_components, component_source_help, remaining_construction_time,
    require_only_component_sources, split_component_sources, validate_scp_combinator_options,
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
    compute_operator_regions: bool,
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
        compute_operator_regions,
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
        compute_operator_regions: false,
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
            "`{name}` requires at least one abstraction source: {}",
            component_source_help(),
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
        config.cap_construction_time(remaining.as_secs_f64());
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

heuristic_registry! {
    build(spec, task, sampling_task);

    Blind {
        names: ["blind"],
        backend: RequiredBackend::None,
        requirements: any_task,
        nested: no_nested_heuristics,
        build: {
            if !spec.args.is_empty() {
                return Err("`blind` does not accept arguments".to_string().into());
            }
            Ok(None)
        }
    }

    CheckAdmissible {
        names: ["check_admissible"],
        backend: RequiredBackend::None,
        requirements: any_task,
        nested: wrapped_heuristic,
        build: {
            let inner_spec = single_wrapped_heuristic_spec("check_admissible", &spec.args)?;
            // The oracle solves the remaining task from scratch, which needs a
            // registry of its own and therefore a shared handle on the task.
            let inner = build_heuristic_from_spec(&inner_spec, task, sampling_task.clone())?;
            let h = CheckAdmissibleHeuristic::new(inner, sampling_task)
                .map_err(|error| format!("failed to construct `check_admissible`: {error}"))?;
            Ok(Some(Box::new(h) as Box<dyn Heuristic + 'a>))
        }
    }

    Ff {
        names: ["ff"],
        backend: RequiredBackend::None,
        requirements: any_task,
        nested: no_nested_heuristics,
        build: {
            if !spec.args.is_empty() {
                return Err("`ff` does not accept arguments".to_string().into());
            }
            let h = crate::evaluation::ff_heuristic::FfHeuristic::new(task)
                .map_err(|e| format!("failed to construct ff heuristic: {e}"))?;
            Ok(Some(Box::new(h) as Box<dyn Heuristic + 'a>))
        }
    }

    Max {
        names: ["max"],
        backend: RequiredBackend::None,
        requirements: component_task_requirements,
        nested: no_nested_heuristics,
        build: {
            let sources = require_only_component_sources("max", &spec.args)?;
            build_max_from_sources(task, &sources, "max")
        }
    }

    Canonical {
        names: ["canonical"],
        backend: RequiredBackend::None,
        requirements: component_task_requirements,
        nested: no_nested_heuristics,
        build: {
            let (sources, construction_deadline) =
                abstraction_config::canonical_sources_and_deadline(&spec.args)?;
            build_canonical_from_sources(task, &sources, "canonical", construction_deadline)
        }
    }

    Scp {
        names: ["scp", "cost_partitioning"],
        backend: RequiredBackend::None,
        requirements: component_task_requirements,
        nested: no_nested_heuristics,
        build: {
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
    }

    DomainAbstraction {
        names: ["domain_abstraction"],
        backend: RequiredBackend::None,
        requirements: abstractable_task,
        nested: no_nested_heuristics,
        build: {
            info!("Building domain abstraction (CEGAR)...");
            let mut cfg = CegarConfig::default();
            cfg.apply_options(&spec.args)?;
            // Single DA reads only the distance table; operator regions are
            // SCP-specific. Skip the per-concrete-op StateRegion cost.
            cfg.compute_operator_regions = false;
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
    }

    CartesianAbstraction {
        names: ["cartesian_abstraction"],
        backend: RequiredBackend::None,
        requirements: abstractable_task,
        nested: no_nested_heuristics,
        build: {
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
    }

    SingleCartesianCollection {
        names: ["max_cartesian_abstraction", "canonical_cartesian_abstraction"],
        backend: RequiredBackend::None,
        requirements: abstractable_task,
        nested: no_nested_heuristics,
        build: {
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
    }

    CanonicalDomainAbstractions {
        names: ["canonical_domain_abstractions"],
        backend: RequiredBackend::None,
        requirements: abstractable_task,
        nested: no_nested_heuristics,
        build: {
            use crate::evaluation::domain_abstractions::domain_abstraction_collection_generator_multiple_cegar::DomainAbstractionCollectionGeneratorMultipleCegarConfig;
            let mut cfg = DomainAbstractionCollectionGeneratorMultipleCegarConfig::default();
            ApplyOptions::apply_options(&mut cfg, &spec.args)?;
            // Canonical never consumes operator regions — skip ~12 GB of
            // per-concrete-op StateRegion storage on big tasks.
            cfg.set_compute_operator_regions(false);
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
    }

    MultiDomainAbstractions {
        names: ["multi_domain_abstractions"],
        backend: RequiredBackend::None,
        requirements: abstractable_task,
        nested: no_nested_heuristics,
        build: {
            use crate::evaluation::domain_abstractions::domain_abstraction_collection_generator_multiple_cegar::DomainAbstractionCollectionGeneratorMultipleCegarConfig;
            let mut cfg = DomainAbstractionCollectionGeneratorMultipleCegarConfig::default();
            ApplyOptions::apply_options(&mut cfg, &spec.args)?;
            cfg.set_compute_operator_regions(false);
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
    }

    PosthocOptimization {
        names: ["posthoc_optimization", "pho"],
        backend: RequiredBackend::Cplex,
        requirements: abstractable_task,
        nested: no_nested_heuristics,
        build: {
            #[cfg(feature = "cplex")]
            {
                use crate::evaluation::domain_abstractions::domain_abstraction_collection_generator_multiple_cegar::DomainAbstractionCollectionGeneratorMultipleCegarConfig;
                let mut cfg = DomainAbstractionCollectionGeneratorMultipleCegarConfig::default();
                ApplyOptions::apply_options(&mut cfg, &spec.args)?;
                cfg.set_compute_operator_regions(false);
                let generator = DomainAbstractionCollectionGeneratorMultipleCegar::new(cfg);
                info!("Building posthoc_optimization domain abstractions (CEGAR)...");
                let abstractions = generator.generate_collection(task).map_err(|e| {
                    format!("failed to build posthoc_optimization domain abstractions: {e:#}")
                })?;
                let h = PostHocOptimizationHeuristic::new(None, task, abstractions).map_err(|e| {
                    format!("failed to construct posthoc_optimization heuristic: {e}")
                })?;
                Ok(Some(Box::new(h) as Box<dyn Heuristic + 'a>))
            }
            #[cfg(not(feature = "cplex"))]
            {
                Err("posthoc_optimization requires CPLEX, which is not compiled into this build. \
                     Rebuild with `--features cplex` and set CPLEX_ROOT to an unrestricted CPLEX \
                     installation."
                    .to_string()
                    .into())
            }
        }
    }

    PotentialDomainAbstractionOcp {
        names: ["pot_da_ocp"],
        backend: RequiredBackend::Cplex,
        requirements: abstractable_task,
        nested: no_nested_heuristics,
        build: {
            #[cfg(feature = "cplex")]
            {
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
            da_config.compute_operator_regions = false;
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
            {
                Err("pot_da_ocp requires unrestricted CPLEX, which is not compiled into this build. \
                     Rebuild with `--features cplex` and set CPLEX_ROOT to an unrestricted CPLEX \
                     installation."
                    .to_string()
                    .into())
            }
        }
    }

    NumericPotential {
        names: ["numeric_potential"],
        backend: RequiredBackend::Cplex,
        requirements: abstractable_task,
        nested: no_nested_heuristics,
        build: {
            #[cfg(feature = "cplex")]
            {
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
            {
                Err("numeric_potential requires unrestricted CPLEX, which is not compiled into this build. \
                     Rebuild with `--features cplex` and set CPLEX_ROOT to an unrestricted CPLEX \
                     installation."
                    .to_string()
                    .into())
            }
        }
    }

    ScpOnline {
        names: ["scp_online", "scp_online_cartesian"],
        backend: RequiredBackend::None,
        requirements: scp_task_requirements,
        nested: no_nested_heuristics,
        build: {
            let (component_sources, _) =
                split_component_sources(spec.name.as_str(), &spec.args)?;
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
    }

    FillScp {
        names: ["fillscp", "fill_scp", "fillscp_cartesian", "fill_scp_cartesian"],
        backend: RequiredBackend::None,
        requirements: abstractable_task,
        nested: no_nested_heuristics,
        build: {
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
    }

    GreedyNumericPdb {
        names: ["greedy_numeric_pdb"],
        backend: RequiredBackend::None,
        requirements: restricted_abstractable_task,
        nested: no_nested_heuristics,
        build: {
            let mut cfg = crate::evaluation::pattern_databases::pattern_generator_greedy::GreedyPatternGeneratorConfig::default();
            ApplyOptions::apply_options(&mut cfg, &spec.args)?;
            let h = GreedyNumericPdbHeuristic::new(task, cfg)
                .map_err(|e| format!("failed to build greedy numeric pdb heuristic: {e}"))?;
            Ok(Some(Box::new(h) as Box<dyn Heuristic + 'a>))
        }
    }

    CanonicalNumericPdb {
        names: ["canonical_numeric_pdb"],
        backend: RequiredBackend::None,
        requirements: restricted_abstractable_task,
        nested: no_nested_heuristics,
        build: {
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
    }

    MaxNumericPdb {
        names: ["max_numeric_pdb"],
        backend: RequiredBackend::None,
        requirements: restricted_abstractable_task,
        nested: no_nested_heuristics,
        build: {
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
    }

    LmCutNumeric {
        names: ["lmcutnumeric"],
        backend: RequiredBackend::None,
        requirements: any_task,
        nested: no_nested_heuristics,
        build: {
            let mut cfg = crate::evaluation::numeric_landmarks::lm_cut_numeric_heuristic::LmCutNumericConfig::default();
            ApplyOptions::apply_options(&mut cfg, &spec.args)?;
            let h = LandmarkCutNumericHeuristic::from_config(task, cfg)
                .map_err(|e| format!("failed to build lmcutnumeric heuristic: {e}"))?;
            Ok(Some(Box::new(h) as Box<dyn Heuristic + 'a>))
        }
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
    HeuristicSpec::from_value(arg.value())
}
