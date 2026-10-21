//! Builds a domain abstraction with CEGAR.

#[cfg(test)]
mod tests;

use std::cell::RefCell;
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use planforge_sas::explicit_state::ExplicitState;
use planforge_sas::numeric_task::AbstractNumericTask;
use rand::{SeedableRng, rngs::SmallRng};

use super::config::DomainAbstractionConfig;
use super::factory::{DistanceTableOptions, DomainAbstractionFactory};
use super::initial_splits::{apply_initial_goal_splits, trivial_domain_mapping_and_sizes};
use super::partitions::NumericPartitions;
use super::view::DomainRefinement;
use crate::evaluation::abstractions::common::abstraction::AbstractionMetadata;
use crate::evaluation::abstractions::common::cegar::refine::{CegarOutcome, CegarRun, refine};
use crate::evaluation::abstractions::common::cegar::regression::ensure_regressable;
use crate::evaluation::abstractions::common::cegar::semantics::{CegarSemantics, SplitOptions};
use crate::evaluation::abstractions::common::cegar::view::RefinementLimits;
use crate::evaluation::abstractions::common::cegar::{CegarDriver, CegarStopReason, FlawKind};
use crate::evaluation::abstractions::common::task::RootedTask;

use super::abstraction::DomainAbstraction;
use super::collection::AbstractionKey;
use super::debug_dump::log_final_abstraction;
use crate::evaluation::abstractions::collections::generator::AbstractionGenerator;

/// The domain abstraction CEGAR refined, before its distances are finalized.
#[derive(Debug)]
pub(super) struct DomainRefinementOutcome {
    pub(super) factory: DomainAbstractionFactory,
    pub(super) outcome: CegarOutcome,
}

/// Refines a domain abstraction of `task` by CEGAR from the initial state of
/// `task`: the trivial abstraction split on the goal variables, refined until
/// the abstract plan is concrete or a limit is reached.
pub(super) fn refine_domain_abstraction(
    config: &DomainAbstractionConfig,
    task: &dyn AbstractNumericTask,
) -> Result<DomainRefinementOutcome> {
    let semantics = CegarSemantics::new(
        task,
        SplitOptions {
            kind: config.cegar.split_kind,
            selection: config.cegar.split_selection,
            rank: config.cegar.split_selection_rank,
            random_seed: config.cegar.random_seed,
        },
    )?;
    if config.cegar.flaw_kind == FlawKind::Regression {
        ensure_regressable(task, semantics.coordinates())?;
    }

    let start = Instant::now();
    // The run is a function of its configuration alone: collection
    // generators hand every member its own seed.
    let mut rng = SmallRng::seed_from_u64(
        config
            .cegar
            .random_seed
            .unwrap_or(crate::evaluation::DEFAULT_RANDOM_SEED),
    );
    let (mut domain_mapping, mut domain_sizes) =
        trivial_domain_mapping_and_sizes(task).context("failed to build trivial domain mapping")?;
    let mut partitions = NumericPartitions::trivial(task);
    apply_initial_goal_splits(
        task,
        config,
        semantics.numeric_integer_lattice(),
        &mut rng,
        &mut domain_mapping,
        &mut domain_sizes,
        &mut partitions,
    )?;
    let factory = DomainAbstractionFactory::new(task, domain_mapping, domain_sizes, partitions)
        .context("failed to construct the initial domain abstraction")?;

    let root = semantics.states().initial_state()?;
    let limits = RefinementLimits {
        max_states: config.cegar.max_states,
        blacklisted_prop_var_ids: &config.blacklisted_prop_var_ids,
        blacklisted_numeric_var_ids: &config.blacklisted_numeric_var_ids,
    };
    let mut refinement = DomainRefinement::new(task, factory, limits, config);
    let outcome = refine(
        &mut refinement,
        &CegarRun {
            semantics: &semantics,
            flaw_kind: config.cegar.flaw_kind,
            flaw_scope: config.cegar.flaw_scope,
            refinement_root: &root,
            driver: CegarDriver::new(config.cegar.max_iterations, config.cegar.max_time),
        },
        start,
    )?;
    let factory = refinement.into_factory();
    if config.cegar.debug {
        log_final_abstraction(task, &factory);
    }
    Ok(DomainRefinementOutcome { factory, outcome })
}

/// Numeric-fd style generator that constructs a domain abstraction via CEGAR.
#[derive(Debug, Clone)]
pub struct DomainAbstractionGenerator {
    config: DomainAbstractionConfig,
}

impl AbstractionGenerator for DomainAbstractionGenerator {
    type Config = DomainAbstractionConfig;
    type Abstraction = DomainAbstraction;
    type Key = AbstractionKey;

    const NAME: &'static str = "domain";

    fn new(config: DomainAbstractionConfig) -> Result<Self> {
        config.cegar.validate()?;
        Ok(Self { config })
    }

    fn key(abstraction: &DomainAbstraction) -> AbstractionKey {
        AbstractionKey::from_abstraction(abstraction)
    }

    /// The root only chooses counterexamples. The abstraction still partitions
    /// the full state space of `task`, operator costs stay those of `task`, and
    /// the distance table's initial state is the initial state of `task`.
    fn generate_from_root(
        &self,
        task: &dyn AbstractNumericTask,
        refinement_root: Option<&ExplicitState>,
    ) -> Result<DomainAbstraction> {
        let DomainRefinementOutcome { factory, outcome } = match refinement_root {
            Some(root) => {
                let rooted_task = RootedTask::new(task, root)?;
                refine_domain_abstraction(&self.config, &rooted_task)
            }
            None => refine_domain_abstraction(&self.config, task),
        }
        .context("CEGAR failed to build abstraction")?;
        let stop_reason = outcome.stop_reason;
        let combine_labels = self.config.cegar.combine_labels;
        let abstract_operators = factory
            .abstract_operators(task, combine_labels, None)
            .context("failed to build abstract operators")?;
        let abstract_source_sets = if self.config.compute_source_sets {
            factory
                .build_abstract_source_sets(task, &abstract_operators)
                .context("failed to build abstract-operator regions")?
        } else {
            // Heuristics that read only the distance table (canonical, max,
            // single domain abstraction) do not consume operator regions;
            // skipping them saves ~12 GB on minecraft-sword-advanced/prob_30x30_5.
            // Regional SCP builds them from the finished abstraction.
            Vec::new()
        };
        let distance_table = factory
            .build_distance_table(task, &abstract_operators, DistanceTableOptions::default())
            .context("failed to build abstract distance table")?;
        let initial_h = distance_table.distances[distance_table.initial_state_hash];
        // The refinement root is reachable from the initial state, so a dead
        // initial state means a dead root, which is where CEGAR stops.
        ensure!(
            initial_h.is_finite() || stop_reason == CegarStopReason::DeadEnd,
            "domain abstraction initial state is abstract-dead, but CEGAR stopped with {stop_reason:?}; initial_hash={}, states={}, abstract_ops={}, prop_domains={:?}, numeric_domains={:?}",
            distance_table.initial_state_hash,
            distance_table.distances.len(),
            abstract_operators.len(),
            factory.domain_sizes(),
            factory.numeric_domain_sizes()
        );
        let relevant_operator_ids = factory.relevant_operator_ids(task, &abstract_operators);

        Ok(DomainAbstraction {
            factory,
            distance_table,
            combine_labels,
            relevant_operator_ids,
            abstract_operators,
            transition_data_retained: true,
            abstract_source_sets,
            interval_transition_system: RefCell::new(None),
            metadata: AbstractionMetadata::standalone(
                outcome,
                self.config.cegar.split_kind,
                self.config.cegar.split_selection_rank,
            ),
        })
    }
}
