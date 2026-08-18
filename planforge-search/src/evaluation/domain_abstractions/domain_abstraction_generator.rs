use std::cell::{Ref, RefCell};
use std::time::Instant;

use anyhow::{Context, Result, ensure};

use planforge_sas::numeric_task::AbstractNumericTask;

use super::abstract_operator_generator::AbstractOperator;
use super::cegar::{Cegar, CegarConfig, CegarStopReason};
use super::domain_abstraction_factory::{
    AbstractDistanceTable, DistanceTableOptions, DomainAbstractionFactory,
};
use crate::evaluation::abstraction_collections::cost_partitioning::{
    AbstractOperatorRegions, AbstractTransitionSystem,
};
use crate::evaluation::abstraction_task::AbstractionUse;

/// Fully built abstraction artifact that can be used to evaluate concrete states.
#[derive(Debug, Clone)]
pub struct DomainAbstraction {
    pub factory: DomainAbstractionFactory,
    pub distance_table: AbstractDistanceTable,
    pub hash_multipliers: Vec<usize>,
    pub combine_labels: bool,
    pub relevant_operator_ids: Vec<usize>,
    pub abstract_operators: Vec<AbstractOperator>,
    pub abstract_operator_regions: Vec<AbstractOperatorRegions>,
    pub(crate) regional_transition_system: RefCell<Option<AbstractTransitionSystem>>,
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
            abstract_operator_regions: Vec::new(),
            regional_transition_system: RefCell::new(None),
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
        self.abstract_operator_regions.clear();
        self.regional_transition_system.get_mut().take();
    }

    pub fn ensure_abstract_operator_regions(
        &mut self,
        task: &dyn AbstractNumericTask,
    ) -> Result<()> {
        if !self.abstract_operator_regions.is_empty() {
            ensure!(
                self.abstract_operator_regions.len() == self.abstract_operators.len(),
                "domain abstraction has {} operator regions for {} abstract operators",
                self.abstract_operator_regions.len(),
                self.abstract_operators.len()
            );
            return Ok(());
        }
        ensure!(
            !self.abstract_operators.is_empty(),
            "cannot construct regional operator regions after abstract operators were discarded"
        );
        self.abstract_operator_regions = self
            .factory
            .build_abstract_operator_regions(task, &self.abstract_operators)
            .context("failed to build abstract-operator regions")?;
        ensure!(
            self.abstract_operator_regions.len() == self.abstract_operators.len(),
            "domain operator-region construction produced {} entries for {} abstract operators",
            self.abstract_operator_regions.len(),
            self.abstract_operators.len()
        );
        Ok(())
    }

    pub fn regional_transition_system<'a>(
        &'a self,
        task: &dyn AbstractNumericTask,
        deadline: Option<Instant>,
    ) -> Result<Ref<'a, AbstractTransitionSystem>> {
        if self.regional_transition_system.borrow().is_none() {
            let mut transition_system = self
                .factory
                .build_abstract_transition_system_from_operators(
                    task,
                    self.combine_labels,
                    &self.abstract_operators,
                    DistanceTableOptions::default()
                        .without_state_regions()
                        .with_deadline(deadline),
                )?;
            transition_system.forward.clear();
            *self.regional_transition_system.borrow_mut() = Some(transition_system);
        }
        Ok(Ref::map(
            self.regional_transition_system.borrow(),
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
    pub initial_seed_splits: Vec<String>,
    pub max_abstraction_size: Option<usize>,
    /// CEGAR exited because the wildcard plan has no flaws. This proves
    /// `h(init)` optimal only when `abstraction_use` is `Standalone`;
    /// collection combinators deliberately do not expose that search shortcut.
    pub solved_by_self: bool,
}

/// Numeric-fd style generator that constructs a domain abstraction via CEGAR.
#[derive(Debug, Clone)]
pub struct DomainAbstractionGenerator {
    cegar: Cegar,
    config: CegarConfig,
}

impl DomainAbstractionGenerator {
    pub fn new(config: CegarConfig) -> Result<Self> {
        let cegar = Cegar::new(config.clone()).context("failed to construct CEGAR")?;
        Ok(Self { cegar, config })
    }

    pub fn config(&self) -> &CegarConfig {
        &self.config
    }

    /// Builds a domain abstraction and its abstract distance table.
    pub fn generate(&self, task: &dyn AbstractNumericTask) -> Result<DomainAbstraction> {
        let outcome = self
            .cegar
            .build_abstraction(task)
            .context("CEGAR failed to build abstraction")?;
        let solved_by_self = outcome.solved_by_self;
        let stop_reason = outcome.stop_reason;
        let factory = outcome.final_state.factory;
        let mut operator_generator =
            factory.make_operator_generator(task, self.config.combine_labels)?;
        let abstract_operators = operator_generator
            .build_abstract_operators_with_deadline(task, None)
            .context("failed to build abstract operators")?;
        let abstract_operator_regions = if self.config.compute_operator_regions {
            factory
                .build_abstract_operator_regions(task, &abstract_operators)
                .context("failed to build abstract-operator regions")?
        } else {
            // Heuristics that read only the distance table (canonical, max,
            // single domain abstraction) do not consume operator regions; skipping
            // saves ~12 GB on minecraft-sword-advanced/prob_30x30_5. SCP /
            // Callers that need regional SCP can construct operator regions from
            // the finalized abstraction after collection generation.
            Vec::new()
        };
        let distance_table = factory
            .build_distance_table_with_operators(
                task,
                &operator_generator,
                &abstract_operators,
                false,
            )
            .context("failed to build abstract distance table")?;
        let initial_h = distance_table
            .distances
            .get(distance_table.initial_state_hash)
            .copied()
            .with_context(|| {
                format!(
                    "abstract initial state hash {} out of bounds for distance table of length {}",
                    distance_table.initial_state_hash,
                    distance_table.distances.len()
                )
            })?;
        ensure!(
            initial_h.is_finite(),
            "domain abstraction initial state is abstract-dead after CEGAR; initial_hash={}, states={}, abstract_ops={}, prop_domains={:?}, numeric_domains={:?}",
            distance_table.initial_state_hash,
            distance_table.distances.len(),
            abstract_operators.len(),
            factory.domain_sizes(),
            factory.numeric_domain_sizes()
        );
        let hash_multipliers =
            compute_hash_multipliers(factory.domain_sizes(), factory.numeric_domain_sizes())
                .context("failed to compute hash multipliers")?;
        let relevant_operator_ids = factory
            .relevant_operator_ids_from_operators(
                task,
                self.config.combine_labels,
                &abstract_operators,
                DistanceTableOptions::default(),
            )
            .context("failed to compute relevant operator ids")?;

        Ok(DomainAbstraction {
            factory,
            distance_table,
            hash_multipliers,
            combine_labels: self.config.combine_labels,
            relevant_operator_ids,
            abstract_operators,
            abstract_operator_regions,
            regional_transition_system: RefCell::new(None),
            metadata: DomainAbstractionMetadata {
                solved_by_self,
                abstraction_use: AbstractionUse::Standalone,
                stop_reason: Some(stop_reason),
                ..DomainAbstractionMetadata::default()
            },
        })
    }
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
