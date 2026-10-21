//! Collections of domain abstractions: the generic collection generator with
//! domain-abstraction members, and the key standard collections drop
//! duplicate members by.

#[cfg(test)]
mod tests;

use super::abstract_operators::DomainMapping;
use super::abstraction::DomainAbstraction;
use super::generator::DomainAbstractionGenerator;
use super::partitions::NumericPartitions;
use crate::evaluation::abstractions::collections::generator::CollectionGenerator;

pub type DomainAbstractionCollectionGenerator = CollectionGenerator<DomainAbstractionGenerator>;

/// Equal for domain abstractions with the same domain mapping and numeric
/// partitions.
pub type AbstractionKey = (DomainMapping, NumericPartitions);

/// The key standard collections drop duplicate members by.
pub(super) fn abstraction_key(abstraction: &DomainAbstraction) -> AbstractionKey {
    let function = abstraction.function();
    (
        function.domain_mapping().clone(),
        function.partitions().clone(),
    )
}
