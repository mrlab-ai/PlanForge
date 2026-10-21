//! Configuration of Cartesian-abstraction CEGAR.

use crate::evaluation::abstractions::collections::config::{CollectionConfig, MemberConfig};
use crate::evaluation::abstractions::common::cegar::options::CegarOptions;

#[derive(Debug, Clone, Default, PartialEq, planforge_search::config::ApplyOptions)]
pub struct CartesianAbstractionConfig {
    #[option(flatten)]
    pub cegar: CegarOptions,
}

impl MemberConfig for CartesianAbstractionConfig {
    fn cegar(&self) -> &CegarOptions {
        &self.cegar
    }

    fn cegar_mut(&mut self) -> &mut CegarOptions {
        &mut self.cegar
    }
}

/// A collection of Cartesian abstractions.
pub type CartesianAbstractionCollectionConfig = CollectionConfig<CartesianAbstractionConfig>;

pub use registry_feature::CartesianFeature;
mod registry_feature {
    use super::CartesianAbstractionConfig;
    use crate::config::ConfigError;
    use crate::evaluation::Heuristic;
    use crate::evaluation::abstractions::cartesian::CartesianAbstractionGenerator;
    use crate::heuristic_factory::*;
    pub struct CartesianFeature;
    impl HeuristicFeature for CartesianFeature {
        const NAME: &'static str = "cartesian_abstraction";
        type Config = CartesianAbstractionConfig;
        fn requirements(_: &Self::Config) -> TaskRequirements {
            TaskRequirements::ABSTRACTABLE_GOAL
        }
        fn validate(config: &Self::Config) -> Result<(), ConfigError> {
            AbstractionSource::Cartesian(config.clone()).validate()
        }
        fn source(config: &Self::Config) -> Option<AbstractionSource> {
            Some(AbstractionSource::Cartesian(config.clone()))
        }
        fn build<'a>(
            config: &Self::Config,
            context: &BuildContext<'a>,
        ) -> Result<Box<dyn Heuristic + 'a>, HeuristicBuildError> {
            standalone_abstraction_heuristic::<CartesianAbstractionGenerator>(
                config.clone(),
                context,
            )
        }
    }
}
