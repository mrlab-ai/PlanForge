//! The heuristic registry: every heuristic the configuration grammar names
//! is a [`HeuristicFeature`], a typed configuration and how to build the
//! heuristic from it. Only blind search is built in; a binary registers every
//! other heuristic once, with [`HEURISTICS`]`.register`, before it parses a
//! configuration.

use crate::config::{ApplyOptions, ConfigCall, ConfigError, ConfigValue, configure};
use crate::evaluation::Heuristic;
use crate::registry::{Entry, Registry, options_of};
use planforge_sas::numeric_task::Task;
use std::any::Any;
use std::fmt::{self, Debug};
use std::sync::Arc;
#[cfg(test)]
mod tests;

/// A property a heuristic needs its task to have, such as a normal form
/// or a goal it can express. Checks of the same name are one check.
#[derive(Debug, Clone, Copy)]
pub struct TaskCheck {
    pub name: &'static str,
    pub check: fn(&Task) -> Result<(), HeuristicBuildError>,
}

/// What a heuristic needs of its task: the task transformations to apply
/// first, by name, and the checks the transformed task must pass.
#[derive(Debug, Clone, Default)]
pub struct TaskRequirements {
    pub transformations: Vec<&'static str>,
    pub checks: Vec<TaskCheck>,
}

impl TaskRequirements {
    /// What both `self` and `other` need.
    pub fn union(mut self, other: Self) -> Self {
        for transformation in other.transformations {
            if !self.transformations.contains(&transformation) {
                self.transformations.push(transformation);
            }
        }
        for check in other.checks {
            if !self.requires(check.name) {
                self.checks.push(check);
            }
        }
        self
    }

    /// Whether the check `name` is required.
    pub fn requires(&self, name: &str) -> bool {
        self.checks.iter().any(|check| check.name == name)
    }

    /// Whether `task` passes every check.
    pub fn validate(&self, task: &Task) -> Result<(), HeuristicBuildError> {
        self.checks.iter().try_for_each(|check| (check.check)(task))
    }
}

/// What a feature builds its heuristic for: the task, and the name the
/// heuristic reports itself under.
pub struct BuildContext {
    pub task: Arc<Task>,
    pub name: String,
}
impl BuildContext {
    pub fn new(task: Arc<Task>) -> Self {
        Self {
            task,
            name: String::new(),
        }
    }
}

/// The configuration of no options.
#[derive(Debug, Clone, Default, crate::config::ApplyOptions)]
pub struct EmptyConfig {}

/// A heuristic the configuration grammar names: its typed configuration,
/// what it needs of the build and of the task, and its construction.
pub trait HeuristicFeature: 'static {
    const NAME: &'static str;
    type Config: ApplyOptions + Default + Clone + Debug + Send + Sync + 'static;
    /// Fails if this build cannot construct the heuristic, for example
    /// without the LP solver it needs; checked before the task is
    /// translated.
    fn preflight(_: &Self::Config) -> std::io::Result<()> {
        Ok(())
    }
    fn requirements(_: &Self::Config) -> TaskRequirements {
        TaskRequirements::default()
    }
    fn validate(_: &Self::Config) -> Result<(), ConfigError> {
        Ok(())
    }
    /// The heuristics the configuration contains, whose preflight checks and
    /// requirements it inherits.
    fn nested(_: &Self::Config) -> Vec<&HeuristicConfig> {
        Vec::new()
    }
    fn build(
        config: &Self::Config,
        context: &BuildContext,
    ) -> Result<Box<dyn Heuristic>, HeuristicBuildError>;
}

trait ErasedHeuristicConfig: Debug + Send + Sync {
    fn spec(&self) -> &ConfigCall;
    fn preflight(&self) -> std::io::Result<()>;
    fn requirements(&self) -> TaskRequirements;
    fn config(&self) -> &dyn Any;
    fn build(&self, context: &BuildContext) -> Result<Box<dyn Heuristic>, HeuristicBuildError>;
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
    fn preflight(&self) -> std::io::Result<()> {
        F::preflight(&self.config)?;
        F::nested(&self.config)
            .into_iter()
            .try_for_each(HeuristicConfig::preflight)
    }
    fn config(&self) -> &dyn Any {
        &self.config
    }
    fn build(&self, context: &BuildContext) -> Result<Box<dyn Heuristic>, HeuristicBuildError> {
        let context = BuildContext {
            task: context.task.clone(),
            name: self.spec.name.clone(),
        };
        F::build(&self.config, &context)
    }
}

#[derive(Debug, Clone)]
pub struct HeuristicConfig(Arc<dyn ErasedHeuristicConfig>);
impl HeuristicConfig {
    /// Fails if this build cannot construct the heuristic or one it
    /// contains.
    pub fn preflight(&self) -> std::io::Result<()> {
        self.0.preflight()
    }

    pub fn from_spec(spec: &ConfigCall) -> Result<Self, ConfigError> {
        HEURISTICS.parse(spec)
    }
    pub fn requirements(&self) -> TaskRequirements {
        self.0.requirements()
    }
    /// The typed configuration, if the heuristic's configuration is a `T`.
    pub fn config<T: 'static>(&self) -> Option<&T> {
        self.0.config().downcast_ref()
    }
    pub fn build(&self, context: &BuildContext) -> Result<Box<dyn Heuristic>, HeuristicBuildError> {
        self.preflight()
            .map_err(|error| HeuristicBuildError::Failure(error.to_string()))?;
        self.requirements().validate(&context.task)?;
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
pub type HeuristicEntry = Entry<HeuristicConfig>;

pub const fn entry<F: HeuristicFeature>() -> HeuristicEntry {
    Entry::new(F::NAME, parse_feature::<F>, options_of::<F::Config>)
}
fn parse_feature<F: HeuristicFeature>(spec: &ConfigCall) -> Result<HeuristicConfig, ConfigError> {
    let mut spec = spec.clone();
    spec.name.make_ascii_lowercase();
    let config = configure(&spec.name, F::Config::default(), &spec.args)?;
    F::validate(&config).map_err(|error| error.at(&spec.name))?;
    Ok(HeuristicConfig(Arc::new(Configured::<F> { config, spec })))
}

/// Every heuristic the configuration grammar names: blind search is built
/// in, a binary registers the others.
pub static HEURISTICS: Registry<HeuristicConfig> =
    Registry::new("heuristic", &[entry::<crate::evaluation::BlindFeature>()]);

pub(crate) fn heuristic_help() -> String {
    format!(
        "Heuristics: {}. `planforge heuristics` lists their options and defaults. Custom Rust heuristics can be passed directly as Box<dyn Heuristic>.",
        HEURISTICS.names().collect::<Vec<_>>().join(", ")
    )
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
