//! Collections of domain abstractions: the generic collection generator with
//! domain-abstraction members, and the key standard collections drop
//! duplicate members by.

#[cfg(test)]
mod tests;

use planforge_sas::numeric_task::VariableIndex;
use planforge_sas::utils::float_tolerance;
use planforge_sas::utils::interval::Interval;

use super::abstract_operators::DomainMapping;
use super::abstraction::DomainAbstraction;
use super::generator::DomainAbstractionGenerator;
use crate::evaluation::abstractions::collections::generator::CollectionGenerator;

pub type DomainAbstractionCollectionGenerator = CollectionGenerator<DomainAbstractionGenerator>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct IntervalFingerprint {
    lower: u64,
    upper: u64,
    lower_closed: bool,
    upper_closed: bool,
}

impl IntervalFingerprint {
    fn from_interval(interval: Interval) -> Self {
        Self {
            lower: float_tolerance::canonical_bits(interval.lower),
            upper: float_tolerance::canonical_bits(interval.upper),
            lower_closed: interval.lower_closed,
            upper_closed: interval.upper_closed,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AbstractionKey {
    domain_mapping: DomainMapping,
    numeric_fingerprint: Vec<Vec<IntervalFingerprint>>,
}

impl AbstractionKey {
    pub(super) fn from_abstraction(abstraction: &DomainAbstraction) -> Self {
        let factory = &abstraction.factory;
        let numeric_fingerprint = (0..factory.numeric_domain_sizes().len())
            .map(|numeric_var_id| {
                factory
                    .partitions()
                    .partitions(VariableIndex::from_usize(numeric_var_id))
                    .iter()
                    .copied()
                    .map(IntervalFingerprint::from_interval)
                    .collect::<Vec<_>>()
            })
            .collect();

        Self {
            domain_mapping: factory.domain_mapping().clone(),
            numeric_fingerprint,
        }
    }
}
