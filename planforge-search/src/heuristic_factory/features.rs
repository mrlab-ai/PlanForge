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

/// Children of a maximum, accepting both the historical flat form and a list.
#[derive(Debug, Clone, Default)]
pub struct MaxConfig {
    pub heuristics: Vec<HeuristicConfig>,
}
impl ApplyOptions for MaxConfig {
    fn apply_options(&mut self, args: &[ConfigArg]) -> Result<(), ConfigError> {
        let mut listed = false;
        for (index, arg) in args.iter().enumerate() {
            if let Some(key) = arg.key() {
                return Err(format!("unknown option `{key}` for max").into());
            }
            match arg.value() {
                ConfigValue::List(_) => {
                    if index != 0 {
                        return Err("the heuristic list must be the first argument".into());
                    }
                    listed = true;
                    self.heuristics =
                        crate::config::FromOptionValue::from_option_value(arg.value())
                            .map_err(|error: ConfigError| error.at_option("heuristics"))?;
                }
                value => {
                    if listed {
                        return Err("all heuristics must appear inside the list".into());
                    }
                    self.heuristics
                        .push(HeuristicConfig::from_value(value).map_err(|error| {
                            error
                                .at(format!("[{}]", self.heuristics.len()))
                                .at_option("heuristics")
                        })?);
                }
            }
        }
        if self.heuristics.is_empty() {
            return Err("max requires at least one heuristic".into());
        }
        Ok(())
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

// Sources that did not previously have a standalone spelling use the same typed
// configuration as their occurrences inside abstraction combinators.
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
