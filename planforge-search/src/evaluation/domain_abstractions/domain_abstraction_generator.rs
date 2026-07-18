use anyhow::{Context, Result, ensure};

use planforge_sas::numeric_task::AbstractNumericTask;

use super::abstract_operator_generator::AbstractOperator;
use super::cegar::{Cegar, CegarConfig, CegarStopReason};
use super::domain_abstraction_factory::{AbstractDistanceTable, DomainAbstractionFactory};
use crate::evaluation::abstraction_collections::transition_cost_partitioning::AbstractOperatorFootprint;
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
    pub abstract_operator_footprints: Vec<AbstractOperatorFootprint>,
    pub metadata: DomainAbstractionMetadata,
}

impl DomainAbstraction {
    pub fn task_for_factory<'task>(
        &'task self,
        fallback: &'task dyn AbstractNumericTask,
    ) -> &'task dyn AbstractNumericTask {
        fallback
    }

    pub fn discard_transition_data(&mut self) {
        self.abstract_operators.clear();
        self.abstract_operator_footprints.clear();
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
    pub portfolio_strategy: Option<String>,
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
        let abstract_operator_footprints = if self.config.compute_operator_footprints {
            factory
                .build_abstract_operator_footprints(task, &abstract_operators)
                .context("failed to build abstract-operator footprints")?
        } else {
            // Heuristics that read only the distance table (canonical, max,
            // single domain abstraction) do not consume footprints; skipping
            // saves ~12 GB on minecraft-sword-advanced/prob_30x30_5. SCP /
            // fillSCP / abstract-operator cost partitioning leave the flag
            // on and pay the cost.
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
            .relevant_operator_ids_from_operators_with_deadline(
                task,
                self.config.combine_labels,
                &abstract_operators,
                None,
            )
            .context("failed to compute relevant operator ids")?;

        Ok(DomainAbstraction {
            factory,
            distance_table,
            hash_multipliers,
            combine_labels: self.config.combine_labels,
            relevant_operator_ids,
            abstract_operators,
            abstract_operator_footprints,
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
