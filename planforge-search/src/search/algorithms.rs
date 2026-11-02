//! The built-in search algorithms: A* and greedy best-first search.

use std::sync::Arc;

use planforge_sas::numeric_task::Task;
use tracing::info;

use super::best_first::{
    AStar, BestFirstSearch, Greedy, NodeOrdering, OpenList, OpenListConfig, PreferredFirstOpenList,
    TieBreakingOpenList,
};
use super::config::{SearchFeature, SearchLimits};
use super::{SearchParams, SearchResult};
use crate::heuristic_factory::{BuildContext, HeuristicBuildError, HeuristicConfig};
use crate::successor_generator::SuccessorGeneratorConfig;

/// `astar(heuristic, lazy=none, mpd=false, open_list=tiebreaking(),
/// successor_generator=standard_grounded_tree())`; the heuristic defaults to blind.
#[derive(Debug, crate::config::ApplyOptions)]
pub struct AStarConfig {
    #[option(positional)]
    pub heuristic: HeuristicConfig,
    /// A second heuristic, evaluated when a node is first popped; the node
    /// is requeued under the larger of both estimates.
    pub lazy: Option<HeuristicConfig>,
    /// Multi-path dependence: re-evaluate a popped node whose estimate
    /// predates the heuristic's current revision.
    pub mpd: bool,
    pub open_list: OpenListConfig,
    pub successor_generator: SuccessorGeneratorConfig,
}

impl Default for AStarConfig {
    fn default() -> Self {
        Self {
            heuristic: HeuristicConfig::default(),
            lazy: None,
            mpd: false,
            open_list: OpenListConfig::TieBreaking,
            successor_generator: SuccessorGeneratorConfig::default(),
        }
    }
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
            successor_generator: config.successor_generator.clone(),
            limits,
        };
        info!(
            "Starting A* search, mpd={}, lazy={}",
            config.mpd,
            params.lazy.is_some()
        );
        run_best_first::<AStar>(task.clone(), params, &config.open_list)
    }
}

/// `gbfs(heuristic, open_list=preferred_first(),
/// successor_generator=standard_grounded_tree())`; the heuristic defaults to blind.
#[derive(Debug, crate::config::ApplyOptions)]
pub struct GreedyBestFirstConfig {
    #[option(positional)]
    pub heuristic: HeuristicConfig,
    pub open_list: OpenListConfig,
    pub successor_generator: SuccessorGeneratorConfig,
}

impl Default for GreedyBestFirstConfig {
    fn default() -> Self {
        Self {
            heuristic: HeuristicConfig::default(),
            open_list: OpenListConfig::PreferredFirst,
            successor_generator: SuccessorGeneratorConfig::default(),
        }
    }
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
            successor_generator: config.successor_generator.clone(),
            limits,
            ..SearchParams::new(build(&config.heuristic, &context)?)
        };
        info!("Starting GBFS search...");
        run_best_first::<Greedy>(task.clone(), params, &config.open_list)
    }
}

/// Best-first search ordered by `O` on the open list `open_list` names: a
/// built-in list statically dispatched, a registered one through
/// `Box<dyn OpenList>`.
fn run_best_first<O: NodeOrdering>(
    task: Arc<Task>,
    params: SearchParams<'_>,
    open_list: &OpenListConfig,
) -> std::io::Result<SearchResult> {
    let result = match open_list {
        OpenListConfig::TieBreaking => {
            BestFirstSearch::<O, TieBreakingOpenList<O>>::new(task, params).run()
        }
        OpenListConfig::PreferredFirst => {
            BestFirstSearch::<O, PreferredFirstOpenList<O>>::new(task, params).run()
        }
        OpenListConfig::Registered(list) => {
            BestFirstSearch::<O, Box<dyn OpenList>>::with_open_list(task, params, list.build())
                .run()
        }
    };
    result.map_err(|error| std::io::Error::other(format!("search failed: {error:#}")))
}

fn build<'a>(
    heuristic: &HeuristicConfig,
    context: &BuildContext<'a>,
) -> std::io::Result<Box<dyn crate::evaluation::Heuristic + 'a>> {
    heuristic
        .build(context)
        .map_err(HeuristicBuildError::into_io_error)
}
