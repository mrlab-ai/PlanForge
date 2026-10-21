//! `--search` heuristic construction: a parsed [`HeuristicSpec`] in, a
//! `Box<dyn Heuristic>` out.
//!
//! The registry at the bottom of this file declares every built-in exactly
//! once. Dispatch, backend preflight, task-shape validation, legal names and
//! CLI help are generated from that declaration.

use crate::evaluation::abstractions::cartesian::CartesianAbstractionGenerator;
use crate::evaluation::abstractions::collections::config::MemberConfig;
use crate::evaluation::abstractions::common::abstraction::AbstractionHeuristic;
use crate::evaluation::abstractions::cost_partitioning::canonical::CanonicalAbstractionHeuristic;
use crate::evaluation::abstractions::cost_partitioning::max::MaxAbstractionHeuristic;
#[cfg(feature = "cplex")]
use crate::evaluation::abstractions::cost_partitioning::posthoc_optimization::PostHocOptimizationHeuristic;
use crate::evaluation::abstractions::cost_partitioning::saturated::{
    SaturatedCostPartitioningHeuristic, ScpBuildOptions,
};
use crate::evaluation::abstractions::domain::generator::DomainAbstractionGenerator;
use crate::evaluation::abstractions::pdb::validate_restricted_task;
use crate::evaluation::check_admissible::CheckAdmissibleHeuristic;
use crate::evaluation::numeric_landmarks::lm_cut_numeric_heuristic::LandmarkCutNumericHeuristic;
use crate::resource_limits::Deadline;
use planforge_sas::numeric_task::{AbstractNumericTask, TaskRef};
use std::sync::OnceLock;
use tracing::info;

mod abstraction_config;
#[cfg(test)]
mod tests;

use crate::config::{ConfigArg, HeuristicSpec, configure};
use crate::evaluation::abstractions::collections::generator::AbstractionGenerator;

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
    ComponentUse, ScpSourceConfig, build_components, component_source_help,
    remaining_construction_time, require_only_component_sources,
};

#[derive(Debug)]
pub enum HeuristicBuildError {
    ConstructionTimeout,
    /// The task uses a feature the heuristic does not support.
    Unsupported(String),
    Failure(String),
}

impl HeuristicBuildError {
    /// The error building an abstraction source ended in, keeping whether
    /// the task was unsupported.
    pub(crate) fn from_source(error: anyhow::Error, context: String) -> Self {
        let message = format!("{context}: {error:#}");
        if crate::unsupported::is_unsupported_task(&error) {
            Self::Unsupported(message)
        } else {
            Self::Failure(message)
        }
    }
}

impl HeuristicBuildError {
    pub fn into_io_error(self) -> std::io::Error {
        match self {
            Self::ConstructionTimeout => std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "shared abstraction construction deadline exceeded",
            ),
            Self::Unsupported(message) => {
                std::io::Error::new(std::io::ErrorKind::Unsupported, message)
            }
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
            Self::Unsupported(message) | Self::Failure(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for HeuristicBuildError {}

impl From<String> for HeuristicBuildError {
    fn from(message: String) -> Self {
        Self::Failure(message)
    }
}

fn build_max_from_sources<'task>(
    task: &'task dyn AbstractNumericTask,
    sources: &[crate::config::ConfigCall],
    name: &str,
) -> Result<Option<Box<dyn Heuristic + 'task>>, HeuristicBuildError> {
    let components = build_components(task, sources, ComponentUse::Standalone, Deadline::NONE)?;
    let heuristic = MaxAbstractionHeuristic::new(Some(name.to_string()), components)
        .map_err(|error| format!("failed to construct `{name}`: {error}"))?;
    Ok(Some(Box::new(heuristic)))
}

fn build_canonical_from_sources<'task>(
    task: &'task dyn AbstractNumericTask,
    sources: &[crate::config::ConfigCall],
    name: &str,
    construction_deadline: Deadline,
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

/// Builds `name(...)` from its sources and the combinator options that
/// [`abstraction_config::scp_sources_options_and_deadline`] validated.
fn build_scp_from_sources<'task>(
    task: &'task dyn AbstractNumericTask,
    sampling_task: TaskRef<'task>,
    source_config: &ScpSourceConfig,
    name: &str,
) -> Result<Option<Box<dyn Heuristic + 'task>>, HeuristicBuildError> {
    let ScpSourceConfig {
        sources,
        options,
        construction_deadline,
    } = source_config;
    let construction_deadline = *construction_deadline;
    if sources.is_empty() {
        return Err(format!(
            "`{name}` requires at least one abstraction source: {}",
            component_source_help(),
        )
        .into());
    }
    let mut config = configure(
        name,
        crate::evaluation::abstractions::cost_partitioning::saturated::ScpConfig::default(),
        options,
    )?;
    let component_use = if config.partitioning.uses_regions() {
        ComponentUse::RegionalCostPartitioning
    } else {
        ComponentUse::LabelCostPartitioning
    };
    let components = build_components(task, sources, component_use, construction_deadline)?;
    if let Some(remaining) = remaining_construction_time(construction_deadline)? {
        config.cap_construction_time(remaining.as_secs_f64());
    }
    let heuristic = SaturatedCostPartitioningHeuristic::new(
        Some(name.to_string()),
        components,
        config,
        task,
        ScpBuildOptions {
            sampling_task: Some(sampling_task),
        },
    )
    .map_err(|error| format!("failed to construct `{name}`: {error}"))?;
    remaining_construction_time(construction_deadline)?;
    Ok(Some(Box::new(heuristic)))
}

/// The heuristic of the one abstraction `spec` configures `G` to build.
fn standalone_abstraction_heuristic<'a, G>(
    spec: &HeuristicSpec,
    task: &dyn AbstractNumericTask,
) -> Result<Option<Box<dyn Heuristic + 'a>>, HeuristicBuildError>
where
    G: AbstractionGenerator,
    G::Config: Default,
    G::Abstraction: 'a,
{
    info!("Building {} abstraction (CEGAR)...", G::NAME);
    let mut config = G::Config::default();
    config.cegar_mut().keep_transition_system = false;
    let config = configure(&spec.name, config, &spec.args)?;
    let abstraction = G::new(config)
        .and_then(|generator| generator.generate(task))
        .map_err(|error| format!("failed to build {} abstraction: {error:#}", G::NAME))?;
    Ok(Some(Box::new(AbstractionHeuristic::new(
        &spec.name,
        abstraction,
    ))))
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
            build_scp_from_sources(task, sampling_task, &source_config, spec.name.as_str())
        }
    }

    DomainAbstraction {
        names: ["domain_abstraction"],
        backend: RequiredBackend::None,
        requirements: abstractable_task,
        nested: no_nested_heuristics,
        build: { standalone_abstraction_heuristic::<DomainAbstractionGenerator>(spec, task) }
    }

    CartesianAbstraction {
        names: ["cartesian_abstraction"],
        backend: RequiredBackend::None,
        requirements: abstractable_task,
        nested: no_nested_heuristics,
        build: { standalone_abstraction_heuristic::<CartesianAbstractionGenerator>(spec, task) }
    }

    PosthocOptimization {
        names: ["posthoc_optimization", "pho"],
        backend: RequiredBackend::Cplex,
        requirements: component_task_requirements,
        nested: no_nested_heuristics,
        build: {
            #[cfg(feature = "cplex")]
            {
                let (sources, construction_deadline) =
                    abstraction_config::canonical_sources_and_deadline(&spec.args)?;
                let components = build_components(
                    task,
                    &sources,
                    ComponentUse::Standalone,
                    construction_deadline,
                )?;
                remaining_construction_time(construction_deadline)?;
                let h = PostHocOptimizationHeuristic::new(Some(spec.name.clone()), task, components)
                    .map_err(|e| format!("failed to construct `{}`: {e}", spec.name))?;
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

            let config = configure(&spec.name, NumericPotentialConfig::default(), &spec.args)?;
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

    GreedyNumericPdb {
        names: ["greedy_numeric_pdb"],
        backend: RequiredBackend::None,
        requirements: restricted_abstractable_task,
        nested: no_nested_heuristics,
        build: {
            let source = crate::config::ConfigCall::new(
                "pdb",
                vec![crate::config::ConfigArg::new(
                    Some("patterns".to_string()),
                    crate::config::ConfigValue::Call(crate::config::ConfigCall::new(
                        "greedy",
                        spec.args.clone(),
                    )),
                )],
            );
            build_max_from_sources(task, &[source], "greedy_numeric_pdb")
        }
    }

    CanonicalNumericPdb {
        names: ["canonical_numeric_pdb"],
        backend: RequiredBackend::None,
        requirements: restricted_abstractable_task,
        nested: no_nested_heuristics,
        build: {
            let source = crate::config::ConfigCall::new("pdb", spec.args.clone());
            build_canonical_from_sources(task, &[source], "canonical_numeric_pdb", Deadline::NONE)
        }
    }

    MaxNumericPdb {
        names: ["max_numeric_pdb"],
        backend: RequiredBackend::None,
        requirements: restricted_abstractable_task,
        nested: no_nested_heuristics,
        build: {
            let source = crate::config::ConfigCall::new("pdb", spec.args.clone());
            build_max_from_sources(task, &[source], "max_numeric_pdb")
        }
    }

    LmCutNumeric {
        names: ["lmcutnumeric"],
        backend: RequiredBackend::None,
        requirements: any_task,
        nested: no_nested_heuristics,
        build: {
            let cfg = configure(
                &spec.name,
                crate::evaluation::numeric_landmarks::lm_cut_numeric_heuristic::LmCutNumericConfig::default(),
                &spec.args,
            )?;
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
