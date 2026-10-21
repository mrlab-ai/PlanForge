//! A built domain abstraction: the factory that maps concrete states to
//! abstract ones, its goal distances, and what was recorded about its
//! construction.

use std::cell::{Ref, RefCell};

use anyhow::{Context, Result, ensure};

use planforge_sas::numeric_task::{
    AbstractNumericTask, ExplicitValueIndex, NumericValue, OperatorIndex,
};

use super::abstract_operators::AbstractOperator;
use super::factory::{DistanceTableOptions, DomainAbstractionFactory};
use crate::evaluation::abstractions::common::abstraction::{AbstractionMetadata, CegarAbstraction};
use crate::evaluation::abstractions::common::transition_system::AbstractDistanceTable;
use crate::evaluation::abstractions::common::transition_system::{
    AbstractOperatorSourceSets, AbstractTransitionSystem,
};
use crate::resource_limits::Deadline;

/// Fully built abstraction artifact that can be used to evaluate concrete states.
#[derive(Debug, Clone)]
pub struct DomainAbstraction {
    pub factory: DomainAbstractionFactory,
    pub distance_table: AbstractDistanceTable,
    pub combine_labels: bool,
    pub relevant_operator_ids: Vec<OperatorIndex>,
    pub abstract_operators: Vec<AbstractOperator>,
    /// `abstract_operators` holds the abstraction's operators; false once
    /// they were discarded or for a lookup-only abstraction. An abstraction
    /// may have no operators at all, so emptiness cannot tell.
    pub transition_data_retained: bool,
    pub abstract_source_sets: Vec<AbstractOperatorSourceSets>,
    pub(crate) interval_transition_system: RefCell<Option<AbstractTransitionSystem>>,
    pub metadata: AbstractionMetadata,
}

impl CegarAbstraction for DomainAbstraction {
    fn abstract_state(
        &self,
        propositional: &[ExplicitValueIndex],
        numeric: &[NumericValue],
    ) -> Result<usize> {
        self.factory.abstract_state_hash(
            propositional,
            numeric,
            &self.distance_table.numeric_domain_sizes,
            &self.distance_table.hash_multipliers,
        )
    }

    fn distance_table(&self) -> &AbstractDistanceTable {
        &self.distance_table
    }

    fn metadata(&self) -> &AbstractionMetadata {
        &self.metadata
    }

    fn metadata_mut(&mut self) -> &mut AbstractionMetadata {
        &mut self.metadata
    }

    fn discard_transition_data(&mut self) {
        self.abstract_operators.clear();
        self.transition_data_retained = false;
        self.abstract_source_sets.clear();
        self.interval_transition_system.get_mut().take();
    }
}

impl DomainAbstraction {
    pub fn ensure_abstract_source_sets(&mut self, task: &dyn AbstractNumericTask) -> Result<()> {
        if !self.abstract_source_sets.is_empty() {
            ensure!(
                self.abstract_source_sets.len() == self.abstract_operators.len(),
                "domain abstraction has {} operator regions for {} abstract operators",
                self.abstract_source_sets.len(),
                self.abstract_operators.len()
            );
            return Ok(());
        }
        ensure!(
            self.transition_data_retained,
            "cannot construct regional operator regions after abstract operators were discarded"
        );
        self.abstract_source_sets = self
            .factory
            .build_abstract_source_sets(task, &self.abstract_operators)
            .context("failed to build abstract-operator regions")?;
        ensure!(
            self.abstract_source_sets.len() == self.abstract_operators.len(),
            "domain operator-region construction produced {} entries for {} abstract operators",
            self.abstract_source_sets.len(),
            self.abstract_operators.len()
        );
        Ok(())
    }

    pub fn interval_transition_system<'a>(
        &'a self,
        task: &dyn AbstractNumericTask,
        deadline: Deadline,
    ) -> Result<Ref<'a, AbstractTransitionSystem>> {
        if self.interval_transition_system.borrow().is_none() {
            let mut transition_system = self
                .factory
                .build_abstract_transition_system_from_operators(
                    task,
                    self.combine_labels,
                    &self.abstract_operators,
                    DistanceTableOptions::default()
                        .without_cartesian_sets()
                        .with_deadline(deadline),
                )?;
            transition_system.forward.clear();
            *self.interval_transition_system.borrow_mut() = Some(transition_system);
        }
        Ok(Ref::map(
            self.interval_transition_system.borrow(),
            |transition_system| {
                transition_system
                    .as_ref()
                    .expect("regional transition system was initialized")
            },
        ))
    }
}

/// The mixed-radix multipliers of abstract state hashes: one digit per
/// variable, propositional variables first, the first digit least
/// significant.
pub(crate) fn hash_multipliers(
    domain_sizes: &[usize],
    numeric_domain_sizes: &[usize],
) -> Result<Vec<usize>> {
    let mut multipliers = Vec::with_capacity(domain_sizes.len() + numeric_domain_sizes.len());
    let mut states: usize = 1;
    for &size in domain_sizes.iter().chain(numeric_domain_sizes) {
        multipliers.push(states);
        states = states
            .checked_mul(size)
            .context("the abstract state space does not fit usize")?;
    }
    Ok(multipliers)
}

/// The number of abstract states of an abstraction with these domain sizes,
/// or `None` if it does not fit `u128`.
pub(crate) fn abstract_state_count(
    domain_sizes: &[usize],
    numeric_domain_sizes: &[usize],
) -> Option<u128> {
    domain_sizes
        .iter()
        .chain(numeric_domain_sizes)
        .try_fold(1_u128, |states, &size| {
            assert!(size > 0, "abstract domain sizes are positive");
            states.checked_mul(size as u128)
        })
}
