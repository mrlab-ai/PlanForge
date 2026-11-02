//! The registry features of abstraction heuristics: one abstraction, and one
//! abstraction source -- a collection or a pattern collection -- used as the
//! maximum over its abstractions. The combinations of sources register next
//! to their heuristics in [`super::cost_partitioning`].

use planforge_sas::numeric_task::Task;
use std::marker::PhantomData;

use super::cartesian::{
    CartesianAbstractionCollectionConfig, CartesianAbstractionConfig, CartesianAbstractionGenerator,
};
use super::collections::config::MemberConfig;
use super::collections::generator::AbstractionGenerator;
use super::common::abstraction::AbstractionHeuristic;
use super::cost_partitioning::max::MaxAbstractionHeuristic;
use super::domain::config::{DomainAbstractionCollectionConfig, DomainAbstractionConfig};
use super::domain::generator::DomainAbstractionGenerator;
use super::icaps26::Icaps26CartesianConfig;
use super::sources::{AbstractionPurpose, AbstractionSource, PdbSourceConfig, build_components};
use planforge_search::config::{ApplyOptions, ConfigError};
use planforge_search::evaluation::Heuristic;
use planforge_search::heuristic_factory::{
    BuildContext, HeuristicBuildError, HeuristicConfig, HeuristicFeature, TaskRequirements,
};
use planforge_search::resource_limits::Deadline;

/// The configuration of an abstraction source, which names it in the
/// configuration grammar.
pub trait SourceConfig:
    ApplyOptions + Default + Clone + std::fmt::Debug + Send + Sync + 'static
{
    const NAME: &'static str;
    fn source(&self) -> AbstractionSource;
}

impl SourceConfig for DomainAbstractionConfig {
    const NAME: &'static str = "domain";
    fn source(&self) -> AbstractionSource {
        AbstractionSource::Domain(self.clone())
    }
}

impl SourceConfig for CartesianAbstractionConfig {
    const NAME: &'static str = "cartesian";
    fn source(&self) -> AbstractionSource {
        AbstractionSource::Cartesian(self.clone())
    }
}

impl SourceConfig for DomainAbstractionCollectionConfig {
    const NAME: &'static str = "domain_collection";
    fn source(&self) -> AbstractionSource {
        AbstractionSource::DomainCollection(self.clone())
    }
}

impl SourceConfig for CartesianAbstractionCollectionConfig {
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

impl SourceConfig for Icaps26CartesianConfig {
    const NAME: &'static str = "icaps26_cartesian";
    fn source(&self) -> AbstractionSource {
        AbstractionSource::Icaps26Cartesian(self.clone())
    }
}

/// The abstraction source a configured heuristic is the maximum over, for a
/// combination of abstraction heuristics to build from directly; `None` for
/// any other heuristic.
pub(crate) fn abstraction_source(heuristic: &HeuristicConfig) -> Option<AbstractionSource> {
    fn source_of<C: SourceConfig>(heuristic: &HeuristicConfig) -> Option<AbstractionSource> {
        heuristic.config::<C>().map(SourceConfig::source)
    }
    source_of::<DomainAbstractionConfig>(heuristic)
        .or_else(|| source_of::<CartesianAbstractionConfig>(heuristic))
        .or_else(|| source_of::<DomainAbstractionCollectionConfig>(heuristic))
        .or_else(|| source_of::<CartesianAbstractionCollectionConfig>(heuristic))
        .or_else(|| source_of::<PdbSourceConfig>(heuristic))
        .or_else(|| source_of::<Icaps26CartesianConfig>(heuristic))
}

/// One abstraction as a heuristic: its goal distances, without the
/// transition system a cost partitioning needs.
pub struct SingleAbstractionFeature<G>(PhantomData<G>);

impl<G> HeuristicFeature for SingleAbstractionFeature<G>
where
    G: AbstractionGenerator + 'static,
    G::Config: SourceConfig,
{
    const NAME: &'static str = <G::Config as SourceConfig>::NAME;
    type Config = G::Config;
    fn requirements(config: &Self::Config) -> TaskRequirements {
        config.source().requirements()
    }
    fn validate(config: &Self::Config) -> Result<(), ConfigError> {
        config.source().validate()
    }
    fn build(
        config: &Self::Config,
        context: &BuildContext,
    ) -> Result<Box<dyn Heuristic>, HeuristicBuildError> {
        let mut config = config.clone();
        config.cegar_mut().keep_transitions = false;
        let abstraction = G::new(config)
            .and_then(|generator| generator.generate(&context.task))
            .map_err(|error| error.context(format!("failed to build {}", G::KIND)))?;
        Ok(Box::new(AbstractionHeuristic::new(
            &context.name,
            abstraction,
        )))
    }
}

pub type DomainFeature = SingleAbstractionFeature<DomainAbstractionGenerator>;
pub type CartesianFeature = SingleAbstractionFeature<CartesianAbstractionGenerator>;

/// A collection or pattern-collection source as a heuristic: the maximum over
/// its abstractions.
pub struct SourceFeature<C>(PhantomData<C>);

impl<C: SourceConfig> HeuristicFeature for SourceFeature<C> {
    const NAME: &'static str = C::NAME;
    type Config = C;
    fn requirements(config: &C) -> TaskRequirements {
        config.source().requirements()
    }
    fn validate(config: &C) -> Result<(), ConfigError> {
        config.source().validate()
    }
    fn build(
        config: &C,
        context: &BuildContext,
    ) -> Result<Box<dyn Heuristic>, HeuristicBuildError> {
        max_over_sources(&context.task, &[config.source()], &context.name)
    }
}

/// The maximum over the abstractions of `sources`.
pub(crate) fn max_over_sources(
    task: &Task,
    sources: &[AbstractionSource],
    name: &str,
) -> Result<Box<dyn Heuristic>, HeuristicBuildError> {
    let components = build_components(
        task,
        sources,
        AbstractionPurpose::Standalone,
        Deadline::NONE,
    )?;
    let heuristic = MaxAbstractionHeuristic::new(Some(name.to_string()), components)
        .map_err(|error| format!("failed to construct `{name}`: {error}"))?;
    Ok(Box::new(heuristic))
}
