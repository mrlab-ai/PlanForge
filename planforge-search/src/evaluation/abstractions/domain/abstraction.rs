//! A built domain abstraction: the factory that maps concrete states to
//! abstract ones, its goal distances, and what was recorded about its
//! construction.

use std::cell::{Ref, RefCell};
use std::time::Instant;

use anyhow::{Context, Result, ensure};

use planforge_sas::numeric_task::{AbstractNumericTask, OperatorIndex};

use super::abstract_operators::AbstractOperator;
use super::config::SplitDirection;
use super::factory::{DistanceTableOptions, DomainAbstractionFactory};
use crate::evaluation::abstractions::collections::goal_focused::CollectionMembership;
use crate::evaluation::abstractions::common::cegar::CegarStopReason;
use crate::evaluation::abstractions::common::task::AbstractionUse;
use crate::evaluation::abstractions::common::transition_system::AbstractDistanceTable;
use crate::evaluation::abstractions::common::transition_system::{
    AbstractOperatorSourceSets, AbstractTransitionSystem,
};

/// Fully built abstraction artifact that can be used to evaluate concrete states.
#[derive(Debug, Clone)]
pub struct DomainAbstraction {
    pub factory: DomainAbstractionFactory,
    pub distance_table: AbstractDistanceTable,
    pub hash_multipliers: Vec<usize>,
    pub combine_labels: bool,
    pub relevant_operator_ids: Vec<OperatorIndex>,
    pub abstract_operators: Vec<AbstractOperator>,
    /// `abstract_operators` holds the abstraction's operators; false once
    /// they were discarded or for a lookup-only abstraction. An abstraction
    /// may have no operators at all, so emptiness cannot tell.
    pub transition_data_retained: bool,
    pub abstract_source_sets: Vec<AbstractOperatorSourceSets>,
    pub(crate) interval_transition_system: RefCell<Option<AbstractTransitionSystem>>,
    pub metadata: DomainAbstractionMetadata,
}

impl DomainAbstraction {
    pub fn lookup_only(
        factory: DomainAbstractionFactory,
        distance_table: AbstractDistanceTable,
        hash_multipliers: Vec<usize>,
        combine_labels: bool,
        metadata: DomainAbstractionMetadata,
    ) -> Self {
        Self {
            factory,
            distance_table,
            hash_multipliers,
            combine_labels,
            relevant_operator_ids: Vec::new(),
            abstract_operators: Vec::new(),
            transition_data_retained: false,
            abstract_source_sets: Vec::new(),
            interval_transition_system: RefCell::new(None),
            metadata,
        }
    }

    pub fn task_for_factory<'task>(
        &'task self,
        fallback: &'task dyn AbstractNumericTask,
    ) -> &'task dyn AbstractNumericTask {
        fallback
    }

    pub fn discard_transition_data(&mut self) {
        self.abstract_operators.clear();
        self.transition_data_retained = false;
        self.abstract_source_sets.clear();
        self.interval_transition_system.get_mut().take();
    }

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
        deadline: Option<Instant>,
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

    pub fn lookup_clone(&self) -> Self {
        let mut abstraction = self.clone();
        abstraction.discard_transition_data();
        abstraction
    }
}

#[derive(Debug, Clone, Default)]
pub struct DomainAbstractionMetadata {
    pub collection_iteration: Option<usize>,
    pub collection_strategy: Option<String>,
    pub flaw_kind: Option<String>,
    pub full_goal_task: Option<bool>,
    pub abstraction_use: AbstractionUse,
    pub stop_reason: Option<CegarStopReason>,
    pub max_abstraction_size: Option<usize>,
    /// CEGAR exited because the wildcard plan has no flaws. This proves
    /// `h(init)` optimal only when `abstraction_use` is `Standalone`;
    /// collection combinators deliberately do not expose that search shortcut.
    pub solved_by_self: bool,
    /// The split direction CEGAR refined with.
    pub split_direction: SplitDirection,
    /// The rank CEGAR used among tied best flaws, if any.
    pub flaw_selection_rank: Option<usize>,
    /// Operator ids of the concrete plan CEGAR executed when it solved the task
    /// from its refinement root.
    pub concrete_plan_operator_ids: Option<Vec<OperatorIndex>>,
    /// Set when the abstraction is a member of a goal-focused collection.
    pub collection_member: Option<CollectionMembership>,
}

/// Computes mixed-radix hash multipliers for propositional and numeric variables.
///
/// This mirrors the hashing scheme used by `AbstractOperatorGenerator`.
pub fn compute_hash_multipliers(
    domain_sizes: &[usize],
    numeric_domain_sizes: &[usize],
) -> Result<Vec<usize>> {
    let total = domain_sizes
        .len()
        .checked_add(numeric_domain_sizes.len())
        .context("variable count overflow")?;
    ensure!(total > 0, "cannot compute hash multipliers for 0 variables");

    let mut multipliers: Vec<usize> = vec![0; total];
    let mut mult: usize = 1;
    for idx in 0..total {
        multipliers[idx] = mult;

        let radix: usize = if idx < domain_sizes.len() {
            domain_sizes[idx]
        } else {
            let n = idx - domain_sizes.len();
            *numeric_domain_sizes
                .get(n)
                .context("numeric domain size out of range")?
        };

        mult = mult
            .checked_mul(radix)
            .context("hash multiplier overflow")?;
    }

    Ok(multipliers)
}

/// The number of abstract states of an abstraction with these domain
/// sizes, or `None` if it does not fit `u128`.
pub(crate) fn compute_abstraction_size_u128(
    domain_sizes: &[usize],
    numeric_domain_sizes: &[usize],
) -> Option<u128> {
    let mut size: u128 = 1;
    for &d in domain_sizes.iter() {
        let du = u128::try_from(d).ok()?;
        if du == 0 {
            return Some(0);
        }
        size = size.checked_mul(du)?;
    }
    for &p in numeric_domain_sizes.iter() {
        let pu = u128::try_from(p).ok()?;
        if pu == 0 {
            return Some(0);
        }
        size = size.checked_mul(pu)?;
    }
    Some(size)
}
