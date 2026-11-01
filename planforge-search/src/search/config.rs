//! The `--search` grammar: every search algorithm it names is a
//! [`SearchFeature`], a typed configuration and how to run the search.
//! A* and greedy best-first search are built in; a binary registers others
//! with [`SEARCH_ALGORITHMS`]`.register` before it parses a configuration.

use std::any::Any;
use std::fmt::{self, Debug};
use std::sync::Arc;
use std::time::Duration;

use planforge_sas::numeric_task::Task;

use super::SearchResult;
use super::algorithms::{AStarFeature, GreedyBestFirstFeature};
use crate::config::{ApplyOptions, ConfigCall, ConfigError, canonical_call, configure, parse_call};
use crate::heuristic_factory::{HeuristicConfig, TaskRequirements};
use crate::registry::{Entry, Registry, options_of};

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
    fn run(
        config: &Self::Config,
        task: Arc<Task>,
        limits: SearchLimits,
    ) -> std::io::Result<SearchResult>;
}

trait ErasedSearchConfig: Debug + Send + Sync {
    fn heuristics(&self) -> Vec<&HeuristicConfig>;
    fn config(&self) -> &dyn Any;
    fn call(&self) -> String;
    fn run(&self, task: Arc<Task>, limits: SearchLimits) -> std::io::Result<SearchResult>;
}

struct Configured<F: SearchFeature>(F::Config);

impl<F: SearchFeature> Debug for Configured<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple(F::NAME).field(&self.0).finish()
    }
}

impl<F: SearchFeature> ErasedSearchConfig for Configured<F> {
    fn heuristics(&self) -> Vec<&HeuristicConfig> {
        F::heuristics(&self.0)
    }
    fn config(&self) -> &dyn Any {
        &self.0
    }
    fn call(&self) -> String {
        canonical_call(F::NAME, &self.0)
    }
    fn run(&self, task: Arc<Task>, limits: SearchLimits) -> std::io::Result<SearchResult> {
        F::run(&self.0, task, limits)
    }
}

/// A configured search algorithm and the heuristics it runs.
#[derive(Debug, Clone)]
pub struct SearchConfig(Arc<dyn ErasedSearchConfig>);

impl SearchConfig {
    /// Every heuristic the search will build, in configuration order.
    pub fn heuristics(&self) -> impl Iterator<Item = &HeuristicConfig> {
        self.0.heuristics().into_iter()
    }

    pub fn requirements(&self) -> TaskRequirements {
        self.heuristics()
            .fold(TaskRequirements::ANY, |requirements, heuristic| {
                requirements.union(heuristic.requirements())
            })
    }

    /// Fail before translation if a heuristic needs a solver backend this
    /// build does not have.
    pub fn preflight(&self) -> std::io::Result<()> {
        self.heuristics().try_for_each(HeuristicConfig::preflight)
    }

    /// The typed configuration, if the algorithm's configuration is a `T`.
    pub fn config<T: 'static>(&self) -> Option<&T> {
        self.0.config().downcast_ref()
    }

    /// Searches `task` within `limits`.
    pub fn solve(&self, task: Arc<Task>, limits: SearchLimits) -> std::io::Result<SearchResult> {
        self.0.run(task, limits)
    }
}

/// The canonical call, which parses back to an equal configuration.
impl fmt::Display for SearchConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0.call())
    }
}

impl PartialEq for SearchConfig {
    fn eq(&self, other: &Self) -> bool {
        self.0.call() == other.0.call()
    }
}

/// A registered search algorithm.
pub type SearchEntry = Entry<SearchConfig>;

pub const fn entry<F: SearchFeature>() -> SearchEntry {
    Entry::new(F::NAME, parse_feature::<F>, options_of::<F::Config>)
}

fn parse_feature<F: SearchFeature>(spec: &ConfigCall) -> Result<SearchConfig, ConfigError> {
    let config = configure(F::NAME, F::Config::default(), &spec.args)?;
    Ok(SearchConfig(Arc::new(Configured::<F>(config))))
}

/// Every search algorithm the `--search` grammar names.
pub static SEARCH_ALGORITHMS: Registry<SearchConfig> = Registry::new(
    "search algorithm",
    &[entry::<AStarFeature>(), entry::<GreedyBestFirstFeature>()],
);

/// The `--search` help text: the algorithms and every registered heuristic.
pub fn search_algorithm_help() -> String {
    let algorithms: Vec<String> = SEARCH_ALGORITHMS
        .entries()
        .map(|entry| {
            let options: Vec<String> = entry
                .options()
                .iter()
                .map(|option| {
                    if option.positional {
                        format!("[{}]", option.name)
                    } else {
                        format!("{}={}", option.name, option.default)
                    }
                })
                .collect();
            format!("{}({})", entry.name(), options.join(", "))
        })
        .collect();
    format!(
        "Search specification. Available algorithms: {}\n\n{}",
        algorithms.join(", "),
        crate::heuristic_factory::heuristic_help()
    )
}

pub fn parse_search_spec(raw: &str) -> Result<SearchConfig, String> {
    Ok(SEARCH_ALGORITHMS.parse(&parse_call(raw)?)?)
}
