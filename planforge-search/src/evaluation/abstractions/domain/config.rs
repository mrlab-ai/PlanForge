//! Configuration of domain-abstraction CEGAR.

use crate::evaluation::abstractions::collections::config::{CollectionConfig, MemberConfig};
use crate::evaluation::abstractions::common::cegar::options::CegarOptions;

#[derive(Debug, Clone, Default, PartialEq, planforge_search::config::ApplyOptions)]
pub struct DomainAbstractionConfig {
    #[option(flatten)]
    pub cegar: CegarOptions,
    /// Extract abstract plans whose steps leave the variables they do not
    /// touch open.
    pub use_wildcard_plans: bool,
}

impl MemberConfig for DomainAbstractionConfig {
    fn cegar(&self) -> &CegarOptions {
        &self.cegar
    }

    fn cegar_mut(&mut self) -> &mut CegarOptions {
        &mut self.cegar
    }
}

/// A collection of domain abstractions.
pub type DomainAbstractionCollectionConfig = CollectionConfig<DomainAbstractionConfig>;

pub use registry_feature::DomainFeature;
mod registry_feature {
    use super::DomainAbstractionConfig;
    use crate::config::ConfigError;
    use crate::evaluation::Heuristic;
    use crate::evaluation::abstractions::domain::generator::DomainAbstractionGenerator;
    use crate::heuristic_factory::*;
    pub struct DomainFeature;
    impl HeuristicFeature for DomainFeature {
        const NAME: &'static str = "domain";
        type Config = DomainAbstractionConfig;
        fn requirements(_: &Self::Config) -> TaskRequirements {
            TaskRequirements::ABSTRACTABLE_GOAL
        }
        fn validate(config: &Self::Config) -> Result<(), ConfigError> {
            AbstractionSource::Domain(config.clone()).validate()
        }
        fn source(config: &Self::Config) -> Option<AbstractionSource> {
            Some(AbstractionSource::Domain(config.clone()))
        }
        fn build<'a>(
            config: &Self::Config,
            context: &BuildContext<'a>,
        ) -> Result<Box<dyn Heuristic + 'a>, HeuristicBuildError> {
            standalone_abstraction_heuristic::<DomainAbstractionGenerator>(config.clone(), context)
        }
    }
}
