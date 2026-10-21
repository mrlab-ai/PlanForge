//! Configuration of Cartesian-abstraction CEGAR.

use crate::evaluation::abstractions::collections::config::{CollectionConfig, MemberConfig};
use crate::evaluation::abstractions::common::cegar::initial_splits::RefinementScope;
use crate::evaluation::abstractions::common::cegar::options::CegarOptions;

#[derive(Debug, Clone, Default, PartialEq, planforge_search::config::ApplyOptions)]
pub struct CartesianAbstractionConfig {
    #[option(flatten)]
    pub cegar: CegarOptions,
    #[option(skip)]
    pub scope: RefinementScope,
}

impl MemberConfig for CartesianAbstractionConfig {
    fn cegar(&self) -> &CegarOptions {
        &self.cegar
    }

    fn cegar_mut(&mut self) -> &mut CegarOptions {
        &mut self.cegar
    }

    fn scope_mut(&mut self) -> &mut RefinementScope {
        &mut self.scope
    }
}

/// A collection of Cartesian abstractions.
pub type CartesianAbstractionCollectionConfig = CollectionConfig<CartesianAbstractionConfig>;
