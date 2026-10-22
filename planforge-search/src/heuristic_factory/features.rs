use super::*;

/// The configuration of a heuristic without options.
#[derive(Debug, Clone, Default)]
pub struct EmptyConfig {}
impl ApplyOptions for EmptyConfig {
    fn apply_options(&mut self, args: &[ConfigArg]) -> Result<(), ConfigError> {
        crate::config::for_each_option(args, &[], |key, _| {
            Err(format!("unknown option `{key}`").into())
        })
    }
}

/// The heuristics a maximum ranges over: one list, or a single heuristic.
#[derive(Debug, Clone, Default, crate::config::ApplyOptions)]
pub struct MaxConfig {
    #[option(positional)]
    pub heuristics: Vec<HeuristicConfig>,
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

// A collection or PDB source used as a heuristic is the maximum over its
// abstractions, with the same typed configuration as inside a combinator.
pub trait SourceConfig: ApplyOptions + Default + Clone + Debug + Send + Sync + 'static {
    const NAME: &'static str;
    fn source(&self) -> AbstractionSource;
}
pub struct SourceFeature<C>(std::marker::PhantomData<C>);
impl<C: SourceConfig> HeuristicFeature for SourceFeature<C> {
    const NAME: &'static str = C::NAME;
    type Config = C;
    fn requirements(config: &C) -> TaskRequirements {
        config.source().requirements()
    }
    fn validate(config: &C) -> Result<(), ConfigError> {
        config.source().validate()
    }
    fn source(config: &C) -> Option<AbstractionSource> {
        Some(config.source())
    }
    fn build<'a>(
        config: &C,
        context: &BuildContext<'a>,
    ) -> Result<Box<dyn Heuristic + 'a>, HeuristicBuildError> {
        build_max_from_sources(&**context.task, &[config.source()], &context.name)
    }
}
impl SourceConfig
    for crate::evaluation::abstractions::domain::config::DomainAbstractionCollectionConfig
{
    const NAME: &'static str = "domain_collection";
    fn source(&self) -> AbstractionSource {
        AbstractionSource::DomainCollection(self.clone())
    }
}
impl SourceConfig
    for crate::evaluation::abstractions::cartesian::CartesianAbstractionCollectionConfig
{
    const NAME: &'static str = "cartesian_collection";
    fn source(&self) -> AbstractionSource {
        AbstractionSource::CartesianCollection(self.clone())
    }
}
impl SourceConfig for PdbSourceConfig {
    const NAME: &'static str = "pdb";
    fn source(&self) -> AbstractionSource {
        AbstractionSource::Pdb(self.clone())
    }
}
impl SourceConfig for crate::evaluation::abstractions::icaps26::Icaps26CartesianConfig {
    const NAME: &'static str = "icaps26_cartesian";
    fn source(&self) -> AbstractionSource {
        AbstractionSource::Icaps26Cartesian(self.clone())
    }
}
