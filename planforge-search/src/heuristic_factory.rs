//! Typed heuristic features and the explicit, deterministic registry.
use crate::config::{ApplyOptions, ConfigArg, ConfigError, ConfigValue, HeuristicSpec, configure};
use crate::evaluation::Heuristic;
use crate::evaluation::abstractions::collections::config::MemberConfig;
use crate::evaluation::abstractions::collections::generator::AbstractionGenerator;
use crate::evaluation::abstractions::common::abstraction::AbstractionHeuristic;
use crate::evaluation::abstractions::cost_partitioning::canonical::CanonicalAbstractionHeuristic;
use crate::evaluation::abstractions::cost_partitioning::max::MaxAbstractionHeuristic;
use crate::evaluation::abstractions::cost_partitioning::saturated::SaturatedCostPartitioningHeuristic;
use crate::evaluation::abstractions::pdb::validate_restricted_task;
use crate::resource_limits::Deadline;
use planforge_sas::numeric_task::{AbstractNumericTask, TaskRef};
use std::fmt::{self, Debug};
use std::sync::{Arc, OnceLock};
mod abstraction_config;
pub(crate) mod features;
pub use features::{EmptyConfig, MaxSourcesConfig};
#[cfg(test)]
mod tests;
pub use abstraction_config::{
    AbstractionPurpose, AbstractionSource, PdbSourceConfig, ScpHeuristicConfig, SourcesConfig,
};
pub(crate) use abstraction_config::{build_components, remaining_construction_time};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequiredBackend {
    None,
    Cplex,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TaskRequirements {
    pub restricted: bool,
    pub abstractable_goal: bool,
    pub icaps26_restriction: bool,
}

impl TaskRequirements {
    pub const ANY: Self = Self {
        restricted: false,
        abstractable_goal: false,
        icaps26_restriction: false,
    };
    pub const ABSTRACTABLE_GOAL: Self = Self {
        restricted: false,
        abstractable_goal: true,
        icaps26_restriction: false,
    };
    pub const RESTRICTED_ABSTRACTABLE_GOAL: Self = Self {
        restricted: true,
        abstractable_goal: true,
        icaps26_restriction: false,
    };

    pub fn union(self, other: Self) -> Self {
        Self {
            restricted: self.restricted || other.restricted,
            abstractable_goal: self.abstractable_goal || other.abstractable_goal,
            icaps26_restriction: self.icaps26_restriction || other.icaps26_restriction,
        }
    }

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

/// Shared task and construction deadline. Each feature receives the task once.
pub struct BuildContext<'a> {
    pub task: &'a TaskRef<'a>,
    pub deadline: Deadline,
    pub name: String,
}
impl<'a> BuildContext<'a> {
    pub fn new(task: &'a TaskRef<'a>) -> Self {
        Self {
            task,
            deadline: Deadline::NONE,
            name: String::new(),
        }
    }
}

pub trait HeuristicFeature: 'static {
    const NAME: &'static str;
    type Config: ApplyOptions + Default + Clone + Debug + Send + Sync + 'static;
    const BACKEND: RequiredBackend = RequiredBackend::None;
    fn requirements(_: &Self::Config) -> TaskRequirements {
        TaskRequirements::ANY
    }
    fn validate(_: &Self::Config) -> Result<(), ConfigError> {
        Ok(())
    }
    fn nested(_: &Self::Config) -> Vec<&HeuristicConfig> {
        Vec::new()
    }
    fn nested_specs(_: &HeuristicSpec) -> Result<Vec<HeuristicSpec>, String> {
        Ok(Vec::new())
    }
    fn source(_: &Self::Config) -> Option<AbstractionSource> {
        None
    }
    fn build<'a>(
        config: &Self::Config,
        context: &BuildContext<'a>,
    ) -> Result<Box<dyn Heuristic + 'a>, HeuristicBuildError>;
}

trait ErasedHeuristicConfig: Debug + Send + Sync {
    fn spec(&self) -> &HeuristicSpec;
    fn requirements(&self) -> TaskRequirements;
    fn backend(&self) -> RequiredBackend;
    fn source(&self) -> Option<AbstractionSource>;
    fn build<'a>(
        &self,
        context: &BuildContext<'a>,
    ) -> Result<Box<dyn Heuristic + 'a>, HeuristicBuildError>;
}
struct Configured<F: HeuristicFeature> {
    config: F::Config,
    spec: HeuristicSpec,
}
impl<F: HeuristicFeature> Debug for Configured<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct(F::NAME)
            .field("config", &self.config)
            .finish()
    }
}
impl<F: HeuristicFeature> ErasedHeuristicConfig for Configured<F> {
    fn spec(&self) -> &HeuristicSpec {
        &self.spec
    }
    fn requirements(&self) -> TaskRequirements {
        F::nested(&self.config)
            .into_iter()
            .fold(F::requirements(&self.config), |requirements, child| {
                requirements.union(child.requirements())
            })
    }
    fn backend(&self) -> RequiredBackend {
        if F::BACKEND == RequiredBackend::Cplex
            || F::nested(&self.config)
                .iter()
                .any(|child| child.required_backend() == RequiredBackend::Cplex)
        {
            RequiredBackend::Cplex
        } else {
            RequiredBackend::None
        }
    }
    fn source(&self) -> Option<AbstractionSource> {
        F::source(&self.config)
    }
    fn build<'a>(
        &self,
        context: &BuildContext<'a>,
    ) -> Result<Box<dyn Heuristic + 'a>, HeuristicBuildError> {
        let context = BuildContext {
            task: context.task,
            deadline: context.deadline,
            name: self.spec.name.clone(),
        };
        F::build(&self.config, &context)
    }
}

#[derive(Debug, Clone)]
pub struct HeuristicConfig(Arc<dyn ErasedHeuristicConfig>);
impl HeuristicConfig {
    pub fn from_spec(spec: &HeuristicSpec) -> Result<Self, ConfigError> {
        let entry = lookup(&spec.name).ok_or_else(|| {
            ConfigError::from(format!(
                "unknown heuristic `{}`; expected one of {}",
                spec.name,
                heuristic_names().collect::<Vec<_>>().join(", ")
            ))
        })?;
        (entry.parse)(spec)
    }
    pub fn requirements(&self) -> TaskRequirements {
        self.0.requirements()
    }
    pub fn required_backend(&self) -> RequiredBackend {
        self.0.backend()
    }
    pub fn as_abstraction_source(&self) -> Option<AbstractionSource> {
        self.0.source()
    }
    pub fn build<'a>(
        &self,
        context: &BuildContext<'a>,
    ) -> Result<Box<dyn Heuristic + 'a>, HeuristicBuildError> {
        let _ = external_heuristics();
        check_backend(self.required_backend())
            .map_err(|error| HeuristicBuildError::Failure(error.to_string()))?;
        self.requirements().validate(&**context.task)?;
        self.0.build(context)
    }
}
impl std::ops::Deref for HeuristicConfig {
    type Target = HeuristicSpec;
    fn deref(&self) -> &HeuristicSpec {
        self.0.spec()
    }
}
impl Default for HeuristicConfig {
    fn default() -> Self {
        Self::from_spec(&HeuristicSpec::blind()).expect("the built-in blind configuration is valid")
    }
}
impl PartialEq for HeuristicConfig {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}
impl fmt::Display for HeuristicConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.0.spec(), f)
    }
}
impl crate::config::FromOptionValue for HeuristicConfig {
    fn from_option_value(value: &ConfigValue) -> Result<Self, ConfigError> {
        Self::from_spec(&HeuristicSpec::from_value(value).map_err(ConfigError::from)?)
    }
}

#[derive(Clone, Copy)]
pub struct HeuristicEntry {
    name: &'static str,
    aliases: &'static [&'static str],
    backend: RequiredBackend,
    parse: fn(&HeuristicSpec) -> Result<HeuristicConfig, ConfigError>,
    nested_specs: fn(&HeuristicSpec) -> Result<Vec<HeuristicSpec>, String>,
}
impl HeuristicEntry {
    pub fn name(&self) -> &'static str {
        self.name
    }
    pub const fn with_aliases(mut self, aliases: &'static [&'static str]) -> Self {
        self.aliases = aliases;
        self
    }
    pub const fn with_name(mut self, name: &'static str) -> Self {
        self.name = name;
        self
    }
    fn names(&self) -> impl Iterator<Item = &'static str> {
        std::iter::once(self.name).chain(self.aliases.iter().copied())
    }
    fn matches(&self, name: &str) -> bool {
        self.names()
            .any(|registered| registered.eq_ignore_ascii_case(name))
    }
}
pub const fn entry<F: HeuristicFeature>() -> HeuristicEntry {
    HeuristicEntry {
        name: F::NAME,
        aliases: &[],
        backend: F::BACKEND,
        parse: parse_feature::<F>,
        nested_specs: F::nested_specs,
    }
}
fn parse_feature<F: HeuristicFeature>(
    spec: &HeuristicSpec,
) -> Result<HeuristicConfig, ConfigError> {
    let config = configure(&spec.name, F::Config::default(), &spec.args)?;
    F::validate(&config).map_err(|error| error.at(&spec.name))?;
    Ok(HeuristicConfig(Arc::new(Configured::<F> {
        config,
        spec: spec.clone(),
    })))
}

static BUILTIN: &[HeuristicEntry] = &[
    entry::<crate::evaluation::heuristic::BlindFeature>(),
    entry::<crate::evaluation::check_admissible::CheckAdmissibleFeature>(),
    entry::<crate::evaluation::ff_heuristic::FfFeature>(),
    entry::<features::MaxFeature>(),
    entry::<features::CanonicalFeature>(),
    entry::<features::ScpFeature>().with_aliases(&["cost_partitioning"]),
    entry::<features::DomainFeature>(),
    entry::<features::CartesianFeature>(),
    entry::<features::PostHocFeature>().with_aliases(&["pho"]),
    entry::<crate::evaluation::numeric_potential_config::NumericPotentialFeature>(),
    entry::<features::GreedyPdbFeature>(),
    entry::<features::CanonicalPdbFeature>(),
    entry::<features::MaxPdbFeature>(),
    entry::<crate::evaluation::numeric_landmarks::lm_cut_numeric_heuristic::LmCutNumericFeature>(),
];

/// External features use exactly the same entries as the built-ins.
pub type ExternalHeuristic = HeuristicEntry;
static EXTERNAL: OnceLock<Vec<HeuristicEntry>> = OnceLock::new();
fn external_heuristics() -> &'static [HeuristicEntry] {
    EXTERNAL.get_or_init(Vec::new)
}
fn lookup(name: &str) -> Option<&'static HeuristicEntry> {
    BUILTIN
        .iter()
        .find(|entry| entry.matches(name))
        .or_else(|| {
            external_heuristics()
                .iter()
                .find(|entry| entry.matches(name))
        })
}
pub fn register_external_heuristics(entries: Vec<HeuristicEntry>) -> Result<(), String> {
    let mut names = std::collections::HashSet::new();
    for entry in &entries {
        for name in entry.names() {
            if name.is_empty() {
                return Err("external heuristic names must not be empty".into());
            }
            if BUILTIN.iter().any(|builtin| builtin.matches(name)) {
                return Err(format!(
                    "external heuristic `{name}` collides with a built-in heuristic"
                ));
            }
            if !names.insert(name.to_ascii_lowercase()) {
                return Err(format!(
                    "external heuristic `{name}` is registered more than once"
                ));
            }
        }
    }
    EXTERNAL
        .set(entries)
        .map_err(|_| "external heuristics have already been registered or used".into())
}
pub fn heuristic_names() -> impl Iterator<Item = &'static str> {
    BUILTIN
        .iter()
        .chain(external_heuristics())
        .flat_map(HeuristicEntry::names)
}
pub fn heuristic_help() -> String {
    format!(
        "Built-in heuristics: {}. Externally registered heuristic names are also accepted. Custom Rust heuristics can be passed directly as Box<dyn Heuristic>.",
        heuristic_names().collect::<Vec<_>>().join(", ")
    )
}
pub fn validate_heuristic_spec(spec: &HeuristicSpec) -> Result<(), String> {
    let entry = lookup(&spec.name).ok_or_else(|| {
        format!(
            "unknown heuristic `{}`; expected one of {}",
            spec.name,
            heuristic_names().collect::<Vec<_>>().join(", ")
        )
    })?;
    for child in (entry.nested_specs)(spec)? {
        validate_heuristic_spec(&child)?;
    }
    Ok(())
}
pub fn heuristic_requirements(spec: &HeuristicSpec) -> Result<TaskRequirements, String> {
    Ok(HeuristicConfig::from_spec(spec)?.requirements())
}
fn check_backend(backend: RequiredBackend) -> std::io::Result<()> {
    if backend == RequiredBackend::None {
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
pub fn preflight_required_backends(spec: &HeuristicSpec) -> std::io::Result<()> {
    fn backend(spec: &HeuristicSpec) -> Result<RequiredBackend, String> {
        let entry =
            lookup(&spec.name).ok_or_else(|| format!("unknown heuristic `{}`", spec.name))?;
        if entry.backend == RequiredBackend::Cplex {
            return Ok(RequiredBackend::Cplex);
        }
        for child in (entry.nested_specs)(spec)? {
            if backend(&child)? == RequiredBackend::Cplex {
                return Ok(RequiredBackend::Cplex);
            }
        }
        Ok(RequiredBackend::None)
    }
    check_backend(backend(spec).map_err(std::io::Error::other)?)
}
#[derive(Debug)]
pub enum HeuristicBuildError {
    Config(crate::config::ConfigError),
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
            Self::Config(error) => std::io::Error::new(std::io::ErrorKind::InvalidInput, error),
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
            Self::Config(error) => std::fmt::Display::fmt(error, formatter),
        }
    }
}

impl std::error::Error for HeuristicBuildError {}

impl From<String> for HeuristicBuildError {
    fn from(message: String) -> Self {
        Self::Failure(message)
    }
}

pub(crate) fn build_max_from_sources<'task>(
    task: &'task dyn AbstractNumericTask,
    sources: &[AbstractionSource],
    name: &str,
) -> Result<Box<dyn Heuristic + 'task>, HeuristicBuildError> {
    let components = build_components(
        task,
        sources,
        AbstractionPurpose::Standalone,
        Deadline::NONE,
    )?;
    let heuristic = MaxAbstractionHeuristic::new(Some(name.to_string()), components)
        .map_err(|error| format!("failed to construct `{name}`: {error}"))?;
    Ok(Box::new(heuristic))
}

pub(crate) fn build_canonical_from_sources<'task>(
    task: &'task dyn AbstractNumericTask,
    sources: &[AbstractionSource],
    name: &str,
    construction_deadline: Deadline,
) -> Result<Box<dyn Heuristic + 'task>, HeuristicBuildError> {
    let components = build_components(
        task,
        sources,
        AbstractionPurpose::Standalone,
        construction_deadline,
    )?;
    remaining_construction_time(construction_deadline)?;
    let heuristic = CanonicalAbstractionHeuristic::new(Some(name.to_string()), task, components)
        .map_err(|error| format!("failed to construct `{name}`: {error}"))?;
    remaining_construction_time(construction_deadline)?;
    Ok(Box::new(heuristic))
}

/// Build SCP from its already configured sources and cost-partitioning options.
pub(crate) fn build_scp_from_sources<'a>(
    context: &BuildContext<'a>,
    config: &ScpHeuristicConfig,
) -> Result<Box<dyn Heuristic + 'a>, HeuristicBuildError> {
    let deadline = context.deadline.earliest(config.sources.deadline()?);
    let purpose = if config.scp.partitioning.uses_regions() {
        AbstractionPurpose::RegionalCostPartitioning
    } else {
        AbstractionPurpose::LabelCostPartitioning
    };
    let components = build_components(&**context.task, &config.sources.sources, purpose, deadline)?;
    remaining_construction_time(deadline)?;
    let heuristic = SaturatedCostPartitioningHeuristic::new(
        Some(context.name.clone()),
        components,
        config.scp.clone(),
        context.task.clone(),
        deadline,
    )
    .map_err(|error| format!("failed to construct `{}`: {error}", context.name))?;
    remaining_construction_time(deadline)?;
    Ok(Box::new(heuristic))
}

pub(crate) fn standalone_abstraction_heuristic<'a, G>(
    mut config: G::Config,
    context: &BuildContext<'a>,
) -> Result<Box<dyn Heuristic + 'a>, HeuristicBuildError>
where
    G: AbstractionGenerator,
    G::Abstraction: 'a,
{
    config.cegar_mut().keep_transitions = false;
    let abstraction = G::new(config)
        .and_then(|generator| generator.generate(&**context.task))
        .map_err(|error| format!("failed to build {}: {error:#}", G::KIND))?;
    Ok(Box::new(AbstractionHeuristic::new(
        &context.name,
        abstraction,
    )))
}
pub(crate) fn single_wrapped_heuristic_spec(
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

impl From<ConfigError> for HeuristicBuildError {
    fn from(error: ConfigError) -> Self {
        Self::Config(error)
    }
}

pub fn build_heuristic_from_spec<'a>(
    spec: &HeuristicSpec,
    context: &BuildContext<'a>,
) -> Result<Box<dyn Heuristic + 'a>, HeuristicBuildError> {
    HeuristicConfig::from_spec(spec)?.build(context)
}
