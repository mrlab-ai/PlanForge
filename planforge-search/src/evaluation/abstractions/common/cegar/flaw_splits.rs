//! The splits that refine a flaw: separate a concrete witness from the facts
//! it violates, or from the abstract successor it misses.

mod comparison;
mod deviation;
mod fact;

use planforge_sas::numeric_task::{ExplicitFact, ExplicitValueIndex, NumericValue};

pub(crate) use deviation::{DeviationWitness, split_deviation_candidates};
pub(crate) use fact::split_failed_fact;

/// A concrete state that witnesses a flaw.
#[derive(Clone, Copy)]
pub(crate) struct ConcreteWitness<'a> {
    pub(crate) propositions: &'a [ExplicitValueIndex],
    pub(crate) numeric: &'a [NumericValue],
}

impl ConcreteWitness<'_> {
    /// The witness's value of the variable of `fact`.
    fn value(&self, fact: &ExplicitFact) -> ExplicitValueIndex {
        self.propositions[fact.var()]
    }

    /// Whether the witness satisfies every one of `facts`.
    fn satisfies_all(&self, facts: &[ExplicitFact]) -> bool {
        facts
            .iter()
            .all(|fact| self.value(fact) == fact.value_index())
    }
}
