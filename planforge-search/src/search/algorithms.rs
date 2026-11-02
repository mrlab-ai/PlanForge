//! The built-in search algorithms: A* and greedy best-first search.

use std::sync::Arc;

use planforge_sas::numeric_task::Task;
use tracing::info;

use super::config::{SearchFeature, SearchLimits};
use super::{AStarSearch, GreedyBestFirstSearch, SearchParams, SearchResult};
use crate::heuristic_factory::{BuildContext, HeuristicBuildError, HeuristicConfig};

/// `astar(heuristic, lazy=none, mpd=false)`; the heuristic defaults to blind.
#[derive(Debug, Default, crate::config::ApplyOptions)]
pub struct AStarConfig {
    #[option(positional)]
    pub heuristic: HeuristicConfig,
    /// A second heuristic, evaluated when a node is first popped; the node
    /// is requeued under the larger of both estimates.
    pub lazy: Option<HeuristicConfig>,
    /// Multi-path dependence: re-evaluate a popped node whose estimate
    /// predates the heuristic's current revision.
    pub mpd: bool,
}

pub struct AStarFeature;

impl SearchFeature for AStarFeature {
    const NAME: &'static str = "astar";
    type Config = AStarConfig;

    fn heuristics(config: &AStarConfig) -> Vec<&HeuristicConfig> {
        std::iter::once(&config.heuristic)
            .chain(&config.lazy)
            .collect()
    }

    fn run(
        config: &AStarConfig,
        task: Arc<Task>,
        limits: SearchLimits,
    ) -> std::io::Result<SearchResult> {
        let context = BuildContext::new(&task);
        let params = SearchParams {
            heuristic: build(&config.heuristic, &context)?,
            lazy: config
                .lazy
                .as_ref()
                .map(|lazy| build(lazy, &context))
                .transpose()?,
            mpd: config.mpd,
            limits,
        };
        info!(
            "Starting A* search, mpd={}, lazy={}",
            config.mpd,
            params.lazy.is_some()
        );
        searched(AStarSearch::new(task.clone(), params).run())
    }
}

/// `gbfs(heuristic)`; the heuristic defaults to blind.
#[derive(Debug, Default, crate::config::ApplyOptions)]
pub struct GreedyBestFirstConfig {
    #[option(positional)]
    pub heuristic: HeuristicConfig,
}

pub struct GreedyBestFirstFeature;

impl SearchFeature for GreedyBestFirstFeature {
    const NAME: &'static str = "gbfs";
    type Config = GreedyBestFirstConfig;

    fn heuristics(config: &GreedyBestFirstConfig) -> Vec<&HeuristicConfig> {
        vec![&config.heuristic]
    }

    fn run(
        config: &GreedyBestFirstConfig,
        task: Arc<Task>,
        limits: SearchLimits,
    ) -> std::io::Result<SearchResult> {
        let context = BuildContext::new(&task);
        let params = SearchParams {
            limits,
            ..SearchParams::new(build(&config.heuristic, &context)?)
        };
        info!("Starting GBFS search...");
        searched(GreedyBestFirstSearch::new(task.clone(), params).run())
    }
}

fn build<'a>(
    heuristic: &HeuristicConfig,
    context: &BuildContext<'a>,
) -> std::io::Result<Box<dyn crate::evaluation::Heuristic + 'a>> {
    heuristic
        .build(context)
        .map_err(HeuristicBuildError::into_io_error)
}

fn searched(result: anyhow::Result<SearchResult>) -> std::io::Result<SearchResult> {
    result.map_err(|error| std::io::Error::other(format!("search failed: {error:#}")))
}
