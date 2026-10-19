//! Builds a domain abstraction with CEGAR.

use std::cell::RefCell;

use anyhow::{Context, Result, ensure};

use planforge_sas::numeric_task::AbstractNumericTask;

use super::cegar::Cegar;
use super::config::CegarConfig;
use super::factory::DistanceTableOptions;
use crate::evaluation::abstractions::common::concrete_state::ConcreteState;
use crate::evaluation::abstractions::common::task::{AbstractionUse, RootedTask};

use super::abstraction::{DomainAbstraction, DomainAbstractionMetadata, compute_hash_multipliers};

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
        self.generate_from_root(task, None)
    }

    /// Builds a domain abstraction whose CEGAR loop searches for flaws from
    /// `refinement_root` instead of the task initial state.
    ///
    /// The root only chooses counterexamples. The abstraction still partitions
    /// the full state space of `task`, operator costs stay those of `task`, and
    /// the distance table's initial state is the initial state of `task`.
    pub(crate) fn generate_from_root(
        &self,
        task: &dyn AbstractNumericTask,
        refinement_root: Option<&ConcreteState>,
    ) -> Result<DomainAbstraction> {
        let outcome = match refinement_root {
            Some(root) => {
                let rooted_task = RootedTask::new(task, root)?;
                self.cegar.build_abstraction(&rooted_task)
            }
            None => self.cegar.build_abstraction(task),
        }
        .context("CEGAR failed to build abstraction")?;
        let solved_by_self = outcome.solved_by_self;
        let stop_reason = outcome.stop_reason;
        let concrete_plan_operator_ids = outcome.concrete_plan_operator_ids;
        let factory = outcome.final_state.factory;
        let mut operator_generator =
            factory.make_operator_generator(task, self.config.combine_labels)?;
        let abstract_operators = operator_generator
            .build_abstract_operators_with_deadline(task, None)
            .context("failed to build abstract operators")?;
        let abstract_source_sets = if self.config.compute_source_sets {
            factory
                .build_abstract_source_sets(task, &abstract_operators)
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
            abstract_source_sets,
            interval_transition_system: RefCell::new(None),
            metadata: DomainAbstractionMetadata {
                solved_by_self,
                abstraction_use: AbstractionUse::Standalone,
                stop_reason: Some(stop_reason),
                split_direction: self.config.effective_split_direction(),
                flaw_selection_rank: self.config.flaw_selection_rank,
                concrete_plan_operator_ids,
                ..DomainAbstractionMetadata::default()
            },
        })
    }
}
