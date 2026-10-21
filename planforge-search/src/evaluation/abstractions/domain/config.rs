//! Configuration of domain-abstraction CEGAR.

use crate::evaluation::abstractions::collections::config::{CollectionConfig, MemberConfig};
use crate::evaluation::abstractions::common::cegar::initial_splits::RefinementScope;
use crate::evaluation::abstractions::common::cegar::options::CegarOptions;

#[derive(Debug, Clone, Default, PartialEq, planforge_search::config::ApplyOptions)]
pub struct DomainAbstractionConfig {
    #[option(flatten)]
    pub cegar: CegarOptions,
    /// Extract abstract plans whose steps leave the variables they do not
    /// touch open.
    pub use_wildcard_plans: bool,
    #[option(skip)]
    pub scope: RefinementScope,
}

impl MemberConfig for DomainAbstractionConfig {
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

/// A collection of domain abstractions.
pub type DomainAbstractionCollectionConfig = CollectionConfig<DomainAbstractionConfig>;
