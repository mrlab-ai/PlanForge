//! Transformations of a planning task into another task that heuristics
//! read instead of the original.
//!
//! - [`restriction`]: the restricted task (`--transform restrict`), in which
//!   every numeric comparison and metric reads state variables directly:
//!   derived arithmetic is folded into the comparisons, which the PDB and
//!   ICAPS 2026 abstractions require.
//! - [`projection`]: the task projected to a pattern of variables, primarily
//!   the abstract task of a pattern database.
//!
//! The `--transform` grammar names the registered transformations,
//! [`TRANSFORMATIONS`]: the restriction is built in; a binary registers
//! others like heuristics.
//!
//! A task with another goal, which goal-focused abstraction collections
//! refine for, needs no transformation: [`Task::with_goals`] shares
//! everything else.
//!
//! [`Task::with_goals`]: planforge_sas::numeric_task::Task::with_goals
pub mod projection;
pub mod restriction;

use std::fmt::{self, Debug};
use std::sync::Arc;

use planforge_sas::numeric_task::Task;
use tracing::info;

use crate::config::{ApplyOptions, ConfigCall, ConfigError, canonical_call, configure, parse_call};
use crate::heuristic_factory::TaskRequirements;
use crate::registry::{Entry, Registry, options_of};
use crate::unsupported::UnsupportedTask;

/// Why a task could not be transformed.
#[derive(Debug)]
pub enum TransformError {
    /// The search needs a transformation the configuration does not apply.
    Missing(&'static str),
    /// The task uses what the transformation does not support; why.
    Unsupported(String),
    /// The transformation failed on the task.
    Failed(String),
}

impl fmt::Display for TransformError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing(name) => write!(
                f,
                "the search needs the task transformation `{name}`; add `--transform {name}`"
            ),
            Self::Unsupported(message) | Self::Failed(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for TransformError {}

impl From<UnsupportedTask> for TransformError {
    fn from(error: UnsupportedTask) -> Self {
        Self::Unsupported(error.to_string())
    }
}

/// A task transformation the `--transform` grammar names: its typed
/// configuration and how it transforms a task.
pub trait TransformationFeature: 'static {
    const NAME: &'static str;
    type Config: ApplyOptions + Default + Debug + Send + Sync + 'static;
    /// The transformed task; `task` itself when it already has the form.
    fn apply(config: &Self::Config, task: Task) -> Result<Task, TransformError>;
}

trait ErasedTransformation: Debug + Send + Sync {
    fn name(&self) -> &'static str;
    fn call(&self) -> String;
    fn apply(&self, task: Task) -> Result<Task, TransformError>;
}

struct Configured<F: TransformationFeature>(F::Config);

impl<F: TransformationFeature> Debug for Configured<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple(F::NAME).field(&self.0).finish()
    }
}

impl<F: TransformationFeature> ErasedTransformation for Configured<F> {
    fn name(&self) -> &'static str {
        F::NAME
    }
    fn call(&self) -> String {
        canonical_call(F::NAME, &self.0)
    }
    fn apply(&self, task: Task) -> Result<Task, TransformError> {
        F::apply(&self.0, task)
    }
}

/// A configured task transformation.
#[derive(Debug, Clone)]
pub struct TransformationConfig(Arc<dyn ErasedTransformation>);

impl TransformationConfig {
    pub fn name(&self) -> &'static str {
        self.0.name()
    }

    pub fn apply(&self, task: Task) -> Result<Task, TransformError> {
        self.0.apply(task)
    }
}

/// The canonical call, which parses back to an equal configuration.
impl fmt::Display for TransformationConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0.call())
    }
}

impl PartialEq for TransformationConfig {
    fn eq(&self, other: &Self) -> bool {
        self.0.call() == other.0.call()
    }
}

/// A registered task transformation.
pub type TransformationEntry = Entry<TransformationConfig>;

pub const fn entry<F: TransformationFeature>() -> TransformationEntry {
    Entry::new(F::NAME, parse_feature::<F>, options_of::<F::Config>)
}

fn parse_feature<F: TransformationFeature>(
    spec: &ConfigCall,
) -> Result<TransformationConfig, ConfigError> {
    let config = configure(F::NAME, F::Config::default(), &spec.args)?;
    Ok(TransformationConfig(Arc::new(Configured::<F>(config))))
}

/// Every task transformation the `--transform` grammar names.
pub static TRANSFORMATIONS: Registry<TransformationConfig> = Registry::new(
    "task transformation",
    &[entry::<restriction::RestrictFeature>()],
);

pub fn parse_transformation_spec(raw: &str) -> Result<TransformationConfig, ConfigError> {
    TRANSFORMATIONS.parse(&parse_call(raw)?)
}

/// The `--transform` help text.
pub fn transformation_help() -> String {
    format!(
        "Task transformation, applied before the search in the order given. Available transformations: {}",
        TRANSFORMATIONS.names().collect::<Vec<_>>().join(", ")
    )
}

/// `task` transformed by `transformations` in order, which must include the
/// transformations `requirements` names.
pub fn transform_task(
    task: Task,
    transformations: &[TransformationConfig],
    requirements: &TaskRequirements,
) -> Result<Task, TransformError> {
    for required in &requirements.transformations {
        if !transformations
            .iter()
            .any(|transformation| transformation.name() == *required)
        {
            return Err(TransformError::Missing(required));
        }
    }
    transformations
        .iter()
        .try_fold(task, |task, transformation| {
            let numeric_variables = task.numeric_variables().len();
            let task = transformation.apply(task)?;
            info!(
                "task transformation {transformation}: numeric variables {numeric_variables} -> {}",
                task.numeric_variables().len()
            );
            Ok(task)
        })
}
