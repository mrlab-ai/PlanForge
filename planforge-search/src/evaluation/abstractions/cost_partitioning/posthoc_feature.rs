//! Registration is available even when the CPLEX implementation is unavailable.
use crate::config::ConfigError;
use crate::evaluation::Heuristic;
use crate::heuristic_factory::features::{source_requirements, validate_sources};
use crate::heuristic_factory::*;
pub struct PostHocFeature;
impl HeuristicFeature for PostHocFeature {
    const NAME: &'static str = "posthoc_optimization";
    const BACKEND: RequiredBackend = RequiredBackend::Cplex;
    type Config = SourcesConfig;
    fn requirements(config: &Self::Config) -> TaskRequirements {
        source_requirements(config)
    }
    fn validate(config: &Self::Config) -> Result<(), ConfigError> {
        validate_sources(config)
    }
    fn build<'a>(
        config: &Self::Config,
        context: &BuildContext<'a>,
    ) -> Result<Box<dyn Heuristic + 'a>, HeuristicBuildError> {
        #[cfg(feature = "cplex")]
        {
            let deadline = context.deadline.earliest(config.deadline()?);
            let components = build_components(
                &**context.task,
                &config.sources,
                AbstractionPurpose::Standalone,
                deadline,
            )?;
            remaining_construction_time(deadline)?;
            let heuristic = crate::evaluation::abstractions::cost_partitioning::posthoc_optimization::PostHocOptimizationHeuristic::new(Some(context.name.clone()), &**context.task, components)?;
            remaining_construction_time(deadline)?;
            Ok(Box::new(heuristic))
        }
        #[cfg(not(feature = "cplex"))]
        {
            let _ = (config, context);
            unreachable!("the registry checks backend availability before construction")
        }
    }
}
