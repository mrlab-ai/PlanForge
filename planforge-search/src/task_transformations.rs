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
//! The `--transform` language names the transformations in
//! [`Features::transformations`](crate::registry::Features): the restriction
//! is built in; a program adds others like heuristics.
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

use crate::config::{ApplyOptions, ConfigCall, ConfigError};
use crate::heuristic_features::TaskRequirements;
use crate::registry::{Configured, Entry, Features, Typed, options_of, parse_configured};
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

/// The interface of a configured task transformation, whatever its
/// configuration type: what [`TransformationConfig`] dereferences to.
pub trait ConfiguredTransformation: Send + Sync {
    fn name(&self) -> &'static str;
    /// The transformed task; `task` itself when it already has the form.
    fn apply(&self, task: Task) -> Result<Task, TransformError>;
}

impl<F: TransformationFeature> ConfiguredTransformation for Typed<F, F::Config> {
    fn name(&self) -> &'static str {
        F::NAME
    }
    fn apply(&self, task: Task) -> Result<Task, TransformError> {
        F::apply(&self.config, task)
    }
}

/// A configured task transformation. It displays as its canonical call.
pub type TransformationConfig = Configured<dyn ConfiguredTransformation>;

/// A task transformation's registry entry.
pub type TransformationEntry = Entry<TransformationConfig>;

pub const fn entry<F: TransformationFeature>() -> TransformationEntry {
    Entry::new(F::NAME, parse_feature::<F>, options_of::<F::Config>)
}

fn parse_feature<F: TransformationFeature>(
    spec: &ConfigCall,
    features: &Features,
) -> Result<TransformationConfig, ConfigError> {
    parse_configured(F::NAME, spec, features, |typed: Typed<F, F::Config>| {
        Arc::new(typed) as Arc<dyn ConfiguredTransformation>
    })
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
