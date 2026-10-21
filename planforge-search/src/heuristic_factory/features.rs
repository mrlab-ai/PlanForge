use super::*;
use crate::evaluation::abstractions::pdb::pattern_generator_greedy::GreedyPatternGeneratorConfig;

#[derive(Debug, Clone, Default)]
pub struct EmptyConfig;
impl ApplyOptions for EmptyConfig {
    fn apply_options(&mut self, args: &[ConfigArg]) -> Result<(), ConfigError> {
        if args.is_empty() {
            Ok(())
        } else {
            Err("does not accept arguments".into())
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct MaxSourcesConfig(pub SourcesConfig);
impl ApplyOptions for MaxSourcesConfig {
    fn apply_options(&mut self, args: &[ConfigArg]) -> Result<(), ConfigError> {
        if args
            .iter()
            .any(|arg| arg.key() == Some("construction_max_time"))
        {
            return Err("max does not accept construction_max_time".into());
        }
        self.0.apply_options(args)
    }
}

pub(crate) fn validate_sources(config: &SourcesConfig) -> Result<(), ConfigError> {
    for (index, source) in config.sources.iter().enumerate() {
        source
            .validate()
            .map_err(|error| error.at(format!("[{index}]")).at_option("sources"))?;
    }
    Ok(())
}
pub(crate) fn source_requirements(config: &SourcesConfig) -> TaskRequirements {
    config
        .sources
        .iter()
        .fold(TaskRequirements::ANY, |requirements, source| {
            requirements.union(source.requirements())
        })
}

pub use crate::evaluation::abstractions::cost_partitioning::max::MaxFeature;

pub use crate::evaluation::abstractions::cost_partitioning::canonical::CanonicalFeature;

pub use crate::evaluation::abstractions::cost_partitioning::saturated::ScpFeature;

pub use crate::evaluation::abstractions::cartesian::CartesianFeature;
pub use crate::evaluation::abstractions::domain::config::DomainFeature;

pub use crate::evaluation::abstractions::cost_partitioning::posthoc_feature::PostHocFeature;

pub struct GreedyPdbFeature;
impl HeuristicFeature for GreedyPdbFeature {
    const NAME: &'static str = "greedy_numeric_pdb";
    type Config = GreedyPatternGeneratorConfig;
    fn requirements(_: &Self::Config) -> TaskRequirements {
        TaskRequirements::RESTRICTED_ABSTRACTABLE_GOAL
    }
    fn source(config: &Self::Config) -> Option<AbstractionSource> {
        Some(AbstractionSource::Pdb(PdbSourceConfig::greedy(*config)))
    }
    fn build<'a>(
        config: &Self::Config,
        context: &BuildContext<'a>,
    ) -> Result<Box<dyn Heuristic + 'a>, HeuristicBuildError> {
        build_max_from_sources(
            &**context.task,
            &[Self::source(config).expect("PDB feature has a source")],
            &context.name,
        )
    }
}
pub struct CanonicalPdbFeature;
impl HeuristicFeature for CanonicalPdbFeature {
    const NAME: &'static str = "canonical_numeric_pdb";
    type Config = PdbSourceConfig;
    fn requirements(_: &Self::Config) -> TaskRequirements {
        TaskRequirements::RESTRICTED_ABSTRACTABLE_GOAL
    }
    fn build<'a>(
        config: &Self::Config,
        context: &BuildContext<'a>,
    ) -> Result<Box<dyn Heuristic + 'a>, HeuristicBuildError> {
        build_canonical_from_sources(
            &**context.task,
            &[AbstractionSource::Pdb(config.clone())],
            &context.name,
            context.deadline,
        )
    }
}
pub struct MaxPdbFeature;
impl HeuristicFeature for MaxPdbFeature {
    const NAME: &'static str = "max_numeric_pdb";
    type Config = PdbSourceConfig;
    fn requirements(_: &Self::Config) -> TaskRequirements {
        TaskRequirements::RESTRICTED_ABSTRACTABLE_GOAL
    }
    fn source(config: &Self::Config) -> Option<AbstractionSource> {
        Some(AbstractionSource::Pdb(config.clone()))
    }
    fn build<'a>(
        config: &Self::Config,
        context: &BuildContext<'a>,
    ) -> Result<Box<dyn Heuristic + 'a>, HeuristicBuildError> {
        build_max_from_sources(
            &**context.task,
            &[AbstractionSource::Pdb(config.clone())],
            &context.name,
        )
    }
}
