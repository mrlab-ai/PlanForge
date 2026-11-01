//! The heuristic registry: every heuristic the configuration grammar names
//! is a [`HeuristicFeature`], a typed configuration and how to build the
//! heuristic from it. Only blind search is built in; a binary registers every
//! other heuristic once, with [`HEURISTICS`]`.register`, before it parses a
//! configuration.

use crate::config::{ApplyOptions, ConfigCall, ConfigError, ConfigValue, configure};
use crate::evaluation::Heuristic;
use crate::registry::{Entry, Registry, options_of};
use crate::task_transformations::restriction::validate_restricted_task;
use planforge_sas::numeric_task::Task;
use std::any::Any;
use std::fmt::{self, Debug};
use std::sync::Arc;
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
    /// The task transformation the heuristic needs applied, by name, when it
    /// needs another normal form than the standard restriction.
    pub transformation: Option<&'static str>,
}

impl TaskRequirements {
    pub const ANY: Self = Self {
        restricted: false,
        abstractable_goal: false,
        transformation: None,
    };
    pub const ABSTRACTABLE_GOAL: Self = Self {
        restricted: false,
        abstractable_goal: true,
        transformation: None,
    };
    pub const RESTRICTED_ABSTRACTABLE_GOAL: Self = Self {
        restricted: true,
        abstractable_goal: true,
        transformation: None,
    };

    pub fn union(self, other: Self) -> Self {
        Self {
            restricted: self.restricted || other.restricted,
            abstractable_goal: self.abstractable_goal || other.abstractable_goal,
            transformation: match (self.transformation, other.transformation) {
                (Some(left), Some(right)) => {
                    assert_eq!(
                        left, right,
                        "one search cannot need two task transformations"
                    );
                    Some(left)
                }
                (left, right) => left.or(right),
            },
        }
    }

    /// A task that is not restricted is a usage error, since `--transform
    /// restrict` would restrict it; a goal no abstraction can express is a task the
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
        HEURISTICS.parse(spec)
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
pub static HEURISTICS: Registry<HeuristicConfig> = Registry::new(
    "heuristic",
    &[entry::<crate::evaluation::heuristic::BlindFeature>()],
);

pub fn heuristic_help() -> String {
    format!(
        "Heuristics: {}. `planforge heuristics` lists their options and defaults. Custom Rust heuristics can be passed directly as Box<dyn Heuristic>.",
        HEURISTICS.names().collect::<Vec<_>>().join(", ")
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
