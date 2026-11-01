//! The heuristic registry: every heuristic the configuration grammar names
//! is a [`HeuristicFeature`], a typed configuration and how to build the
//! heuristic from it. Only blind search is built in; a binary registers every
//! other heuristic once, with [`register_heuristics`], before it parses a
//! configuration.

use crate::config::{ApplyOptions, ConfigCall, ConfigError, ConfigValue, OptionDoc, configure};
use crate::evaluation::Heuristic;
use crate::task_transformations::restriction::validate_restricted_task;
use planforge_sas::numeric_task::Task;
use std::any::Any;
use std::fmt::{self, Debug};
use std::sync::{Arc, OnceLock};
#[cfg(test)]
mod tests;

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

    /// A task that is not restricted is a usage error, since `--restrict-task`
    /// would restrict it; a goal no abstraction can express is a task the
    /// heuristic does not support.
    pub fn validate(self, task: &Task) -> Result<(), HeuristicBuildError> {
        if self.restricted {
            validate_restricted_task(task).map_err(HeuristicBuildError::Failure)?;
        }
        if self.abstractable_goal {
            crate::evaluation::validate_abstractable_goal(task)
                .map_err(HeuristicBuildError::Unsupported)?;
        }
        Ok(())
    }
}

/// What a feature builds its heuristic for: the task, and the name the
/// heuristic reports itself under.
pub struct BuildContext<'a> {
    pub task: &'a Arc<Task>,
    pub name: String,
}
impl<'a> BuildContext<'a> {
    pub fn new(task: &'a Arc<Task>) -> Self {
        Self {
            task,
            name: String::new(),
        }
    }
}

/// The configuration of no options.
#[derive(Debug, Clone, Default, crate::config::ApplyOptions)]
pub struct EmptyConfig {}

/// A heuristic the configuration grammar names: its typed configuration, the
/// task properties and solver backend it needs, and its construction.
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
    /// The heuristics the configuration contains, whose requirements and
    /// backend it inherits.
    fn nested(_: &Self::Config) -> Vec<&HeuristicConfig> {
        Vec::new()
    }
    fn build<'a>(
        config: &Self::Config,
        context: &BuildContext<'a>,
    ) -> Result<Box<dyn Heuristic + 'a>, HeuristicBuildError>;
}

trait ErasedHeuristicConfig: Debug + Send + Sync {
    fn spec(&self) -> &ConfigCall;
    fn requirements(&self) -> TaskRequirements;
    fn backend(&self) -> RequiredBackend;
    fn config(&self) -> &dyn Any;
    fn build<'a>(
        &self,
        context: &BuildContext<'a>,
    ) -> Result<Box<dyn Heuristic + 'a>, HeuristicBuildError>;
}
struct Configured<F: HeuristicFeature> {
    config: F::Config,
    spec: ConfigCall,
}
impl<F: HeuristicFeature> Debug for Configured<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct(F::NAME)
            .field("config", &self.config)
            .finish()
    }
}
impl<F: HeuristicFeature> ErasedHeuristicConfig for Configured<F> {
    fn spec(&self) -> &ConfigCall {
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
    fn config(&self) -> &dyn Any {
        &self.config
    }
    fn build<'a>(
        &self,
        context: &BuildContext<'a>,
    ) -> Result<Box<dyn Heuristic + 'a>, HeuristicBuildError> {
        let context = BuildContext {
            task: context.task,
            name: self.spec.name.clone(),
        };
        F::build(&self.config, &context)
    }
}

#[derive(Debug, Clone)]
pub struct HeuristicConfig(Arc<dyn ErasedHeuristicConfig>);
impl HeuristicConfig {
    pub fn preflight(&self) -> std::io::Result<()> {
        check_backend(self.required_backend())
    }

    pub fn from_spec(spec: &ConfigCall) -> Result<Self, ConfigError> {
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
    /// The typed configuration, if the heuristic's configuration is a `T`.
    pub fn config<T: 'static>(&self) -> Option<&T> {
        self.0.config().downcast_ref()
    }
    pub fn build<'a>(
        &self,
        context: &BuildContext<'a>,
    ) -> Result<Box<dyn Heuristic + 'a>, HeuristicBuildError> {
        let _ = registered_heuristics();
        check_backend(self.required_backend())
            .map_err(|error| HeuristicBuildError::Failure(error.to_string()))?;
        self.requirements().validate(context.task)?;
        self.0.build(context)
    }
}
impl std::ops::Deref for HeuristicConfig {
    type Target = ConfigCall;
    fn deref(&self) -> &ConfigCall {
        self.0.spec()
    }
}
impl Default for HeuristicConfig {
    fn default() -> Self {
        Self::from_spec(&ConfigCall::blind()).expect("the built-in blind configuration is valid")
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
        Self::from_spec(&ConfigCall::from_value(value).map_err(ConfigError::from)?)
    }

    fn option_text(&self) -> String {
        self.to_string()
    }
}

/// A registered heuristic feature. External features use the same entries as
/// the built-ins.
#[derive(Clone, Copy)]
pub struct HeuristicEntry {
    name: &'static str,
    parse: fn(&ConfigCall) -> Result<HeuristicConfig, ConfigError>,
    options: fn() -> Vec<OptionDoc>,
}
impl HeuristicEntry {
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// The options the heuristic's configuration accepts, with its defaults.
    pub fn options(&self) -> Vec<OptionDoc> {
        (self.options)()
    }
    fn matches(&self, name: &str) -> bool {
        self.name.eq_ignore_ascii_case(name)
    }
}
pub const fn entry<F: HeuristicFeature>() -> HeuristicEntry {
    HeuristicEntry {
        name: F::NAME,
        parse: parse_feature::<F>,
        options: feature_options::<F>,
    }
}
fn feature_options<F: HeuristicFeature>() -> Vec<OptionDoc> {
    let mut options = Vec::new();
    F::Config::default().describe_options(&mut options);
    options
}
fn parse_feature<F: HeuristicFeature>(spec: &ConfigCall) -> Result<HeuristicConfig, ConfigError> {
    let mut spec = spec.clone();
    spec.name.make_ascii_lowercase();
    let config = configure(&spec.name, F::Config::default(), &spec.args)?;
    F::validate(&config).map_err(|error| error.at(&spec.name))?;
    Ok(HeuristicConfig(Arc::new(Configured::<F> {
        config,
        spec: spec.clone(),
    })))
}

/// The heuristics every build knows.
static BUILTIN: &[HeuristicEntry] = &[entry::<crate::evaluation::heuristic::BlindFeature>()];

static REGISTERED: OnceLock<Vec<HeuristicEntry>> = OnceLock::new();
fn registered_heuristics() -> &'static [HeuristicEntry] {
    REGISTERED.get_or_init(Vec::new)
}
fn lookup(name: &str) -> Option<&'static HeuristicEntry> {
    BUILTIN
        .iter()
        .find(|entry| entry.matches(name))
        .or_else(|| {
            registered_heuristics()
                .iter()
                .find(|entry| entry.matches(name))
        })
}

/// Registers the heuristics besides blind search, once per process and
/// before any configuration is parsed.
pub fn register_heuristics(entries: Vec<HeuristicEntry>) -> Result<(), String> {
    let mut names = std::collections::HashSet::new();
    for HeuristicEntry { name, .. } in &entries {
        if name.is_empty() {
            return Err("heuristic names must not be empty".into());
        }
        if BUILTIN.iter().any(|builtin| builtin.matches(name)) {
            return Err(format!(
                "heuristic `{name}` collides with a built-in heuristic"
            ));
        }
        if !names.insert(name.to_ascii_lowercase()) {
            return Err(format!("heuristic `{name}` is registered more than once"));
        }
    }
    REGISTERED
        .set(entries)
        .map_err(|_| "heuristics have already been registered or used".into())
}

/// Registers `entries` unless heuristics were registered already: for test
/// processes, which all register the same entries.
pub fn ensure_heuristics_registered(entries: impl FnOnce() -> Vec<HeuristicEntry>) {
    REGISTERED.get_or_init(entries);
}
/// Every known heuristic, built-in first, in registration order.
pub fn heuristic_entries() -> impl Iterator<Item = &'static HeuristicEntry> {
    BUILTIN.iter().chain(registered_heuristics())
}

/// A reference of every known heuristic and its options with their
/// defaults, generated from the configuration types themselves.
pub fn heuristic_reference() -> String {
    let mut text = String::new();
    for entry in heuristic_entries() {
        let options = entry.options();
        let positional: Vec<&str> = options
            .iter()
            .filter(|option| option.positional)
            .map(|option| option.name.as_str())
            .collect();
        text.push_str(entry.name());
        text.push('(');
        text.push_str(&positional.join(", "));
        text.push_str(")\n");
        for option in &options {
            text.push_str(&format!("    {} = {}", option.name, option.default));
            if !option.doc.is_empty() {
                text.push_str(&format!("  -- {}", option.doc));
            }
            text.push('\n');
        }
    }
    text
}

pub fn heuristic_names() -> impl Iterator<Item = &'static str> {
    BUILTIN
        .iter()
        .chain(registered_heuristics())
        .map(HeuristicEntry::name)
}
pub fn heuristic_help() -> String {
    format!(
        "Heuristics: {}. `planforge heuristics` lists their options and defaults. Custom Rust heuristics can be passed directly as Box<dyn Heuristic>.",
        heuristic_names().collect::<Vec<_>>().join(", ")
    )
}
fn check_backend(backend: RequiredBackend) -> std::io::Result<()> {
    if backend == RequiredBackend::None {
        return Ok(());
    }
    #[cfg(feature = "cplex")]
    {
        planforge_cplex::assert_unrestricted_license().map_err(|error| {
            std::io::Error::other(format!(
                "an unrestricted CPLEX installation is required: {error}"
            ))
        })
    }
    #[cfg(not(feature = "cplex"))]
    {
        Err(std::io::Error::other(
            "the requested LP-backed heuristic requires unrestricted CPLEX, which is not compiled into this build; rebuild with `--features cplex` and set CPLEX_ROOT",
        ))
    }
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
    pub fn from_source(error: anyhow::Error, context: String) -> Self {
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

impl From<ConfigError> for HeuristicBuildError {
    fn from(error: ConfigError) -> Self {
        Self::Config(error)
    }
}
