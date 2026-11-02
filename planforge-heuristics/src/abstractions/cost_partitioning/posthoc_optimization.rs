//! Post-hoc optimization (Pommerening et al. 2013) over abstraction
//! heuristics: the best additive weighting of the abstractions that no
//! operator's cost exceeds, an LP solved per state.
//!
//! The feature is registered in every build, so that requesting the heuristic
//! where it is not compiled in is a configuration error before translation;
//! the heuristic needs CPLEX.

#[cfg(feature = "cplex")]
mod heuristic;

use crate::abstractions::sources::CombinationConfig;
use planforge_search::config::ConfigError;
use planforge_search::evaluation::Heuristic;
use planforge_search::heuristic_factory::{
    BuildContext, HeuristicBuildError, HeuristicFeature, TaskRequirements,
};

/// `posthoc_optimization(sources)`.
pub struct PostHocFeature;
impl HeuristicFeature for PostHocFeature {
    const NAME: &'static str = "posthoc_optimization";
    type Config = CombinationConfig;
    fn preflight(_: &Self::Config) -> std::io::Result<()> {
        crate::requirements::require_cplex()
    }
    fn requirements(config: &Self::Config) -> TaskRequirements {
        config.requirements()
    }
    fn validate(config: &Self::Config) -> Result<(), ConfigError> {
        config.validate()
    }
    #[cfg(feature = "cplex")]
    fn build<'a>(
        config: &Self::Config,
        context: &BuildContext<'a>,
    ) -> Result<Box<dyn Heuristic + 'a>, HeuristicBuildError> {
        use crate::abstractions::sources::{
            AbstractionPurpose, build_components, remaining_construction_time,
        };
        let deadline = config.deadline()?;
        let task = &**context.task;
        let components = build_components(
            task,
            &config.sources,
            AbstractionPurpose::Standalone,
            deadline,
        )?;
        remaining_construction_time(deadline)?;
        let heuristic = heuristic::PostHocOptimizationHeuristic::new(
            Some(context.name.clone()),
            task,
            components,
        )?;
        remaining_construction_time(deadline)?;
        Ok(Box::new(heuristic))
    }
    #[cfg(not(feature = "cplex"))]
    fn build<'a>(
        _: &Self::Config,
        _: &BuildContext<'a>,
    ) -> Result<Box<dyn Heuristic + 'a>, HeuristicBuildError> {
        unreachable!("the registry rejects a CPLEX heuristic in a build without CPLEX")
    }
}
