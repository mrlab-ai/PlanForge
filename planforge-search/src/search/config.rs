//! The `--search` language: every search algorithm it names is a
//! [`SearchFeature`], a typed configuration and how to run the search.
//! A* and greedy best-first search are built in; a program adds others to
//! [`Features::search_algorithms`](crate::registry::Features).

use std::any::Any;
use std::fmt::Debug;
use std::sync::Arc;
use std::time::Duration;

use planforge_sas::numeric_task::Task;

use super::{SearchError, SearchResult};
use crate::config::{ApplyOptions, ConfigCall, ConfigError};
use crate::heuristic_features::{HeuristicBuildError, HeuristicConfig, TaskRequirements};
use crate::registry::{Configured, Entry, Features, Typed, options_of, parse_configured};

/// The limits a search runs within.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SearchLimits {
    pub time: Option<Duration>,
    pub memory_bytes: Option<u64>,
}

/// A search algorithm the configuration grammar names: its typed
/// configuration, the heuristics that configuration contains, and the search.
pub trait SearchFeature: 'static {
    const NAME: &'static str;
    type Config: ApplyOptions + Default + Debug + Send + Sync + 'static;
    /// The heuristics the search evaluates; it inherits their task
    /// requirements and solver backends.
    fn heuristics(config: &Self::Config) -> Vec<&HeuristicConfig>;
    /// What the search itself needs of the task, besides what its heuristics
    /// need.
    fn requirements(_: &Self::Config) -> TaskRequirements {
        TaskRequirements::default()
    }
    fn run(
        config: &Self::Config,
        task: Arc<Task>,
        limits: SearchLimits,
    ) -> Result<SearchResult, SearchError>;
}

/// The interface of a configured search algorithm, whatever its
/// configuration type: what [`SearchConfig`] dereferences to.
pub trait ConfiguredSearch: Send + Sync {
    /// Every heuristic the search will build, in configuration order.
    fn heuristics(&self) -> Vec<&HeuristicConfig>;
    /// What the search itself needs of the task.
    fn own_requirements(&self) -> TaskRequirements;
    /// Searches `task` within `limits`.
    fn solve(&self, task: Arc<Task>, limits: SearchLimits) -> Result<SearchResult, SearchError>;
    fn config_any(&self) -> &dyn Any;
}

impl<F: SearchFeature> ConfiguredSearch for Typed<F, F::Config> {
    fn heuristics(&self) -> Vec<&HeuristicConfig> {
        F::heuristics(&self.config)
    }
    fn own_requirements(&self) -> TaskRequirements {
        F::requirements(&self.config)
    }
    fn solve(&self, task: Arc<Task>, limits: SearchLimits) -> Result<SearchResult, SearchError> {
        F::run(&self.config, task, limits)
    }
    fn config_any(&self) -> &dyn Any {
        &self.config
    }
}

/// A configured search algorithm and the heuristics it runs. It displays as
/// its canonical call.
pub type SearchConfig = Configured<dyn ConfiguredSearch>;

impl SearchConfig {
    /// What the search and its heuristics need of the task.
    pub fn requirements(&self) -> TaskRequirements {
        self.heuristics()
            .into_iter()
            .fold(self.own_requirements(), |requirements, heuristic| {
                requirements.union(heuristic.requirements())
            })
    }

    /// Fail before translation if this build cannot construct a heuristic,
    /// for example without the LP solver it needs.
    pub fn preflight(&self) -> Result<(), HeuristicBuildError> {
        self.heuristics()
            .into_iter()
            .try_for_each(|heuristic| heuristic.preflight())
    }

    /// The typed configuration, if the algorithm's configuration is a `T`.
    pub fn config<T: 'static>(&self) -> Option<&T> {
        self.config_any().downcast_ref()
    }
}

/// A search algorithm's registry entry.
pub type SearchEntry = Entry<SearchConfig>;

pub const fn entry<F: SearchFeature>() -> SearchEntry {
    Entry::new(F::NAME, parse_feature::<F>, options_of::<F::Config>)
}

fn parse_feature<F: SearchFeature>(
    spec: &ConfigCall,
    features: &Features,
) -> Result<SearchConfig, ConfigError> {
    parse_configured(F::NAME, spec, features, |typed: Typed<F, F::Config>| {
        Arc::new(typed) as Arc<dyn ConfiguredSearch>
    })
}
