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
