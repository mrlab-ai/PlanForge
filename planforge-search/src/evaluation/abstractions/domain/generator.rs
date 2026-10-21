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
use super::factory::DomainAbstractionFactory;
use super::initial_splits::{apply_initial_goal_splits, trivial_domain_mapping_and_sizes};
use super::partitions::NumericPartitions;
use super::view::DomainRefinement;
use crate::evaluation::abstractions::common::abstraction::AbstractionMetadata;
use crate::evaluation::abstractions::common::cegar::refine::{CegarOutcome, CegarRun, refine};
use crate::evaluation::abstractions::common::cegar::semantics::CegarSemantics;
use crate::evaluation::abstractions::common::cegar::view::RefinementLimits;
use crate::evaluation::abstractions::common::cegar::{CegarDriver, CegarStopReason};
use crate::evaluation::abstractions::common::transition_system::AbstractDistanceTable;
use crate::resource_limits::Deadline;

use super::abstract_operators::AbstractOperator;
use super::abstraction::DomainAbstraction;
use super::collection::AbstractionKey;
use super::debug_dump::log_final_abstraction;
use crate::evaluation::abstractions::collections::generator::AbstractionGenerator;

/// The domain abstraction CEGAR refined, with the operators and goal
/// distances of its last flaw search if no split followed it.
#[derive(Debug)]
pub(super) struct DomainRefinementOutcome {
    pub(super) factory: DomainAbstractionFactory,
    pub(super) outcome: CegarOutcome,
    pub(super) built: Option<(Vec<AbstractOperator>, AbstractDistanceTable)>,
}

/// Refines a domain abstraction of `task` by CEGAR from `refinement_root`,
/// or from the initial state of `task` when `None`: the trivial abstraction
/// split on the goal variables, refined until the abstract plan is concrete
/// or a limit is reached.
pub(super) fn refine_domain_abstraction(
    config: &DomainAbstractionConfig,
    task: &dyn AbstractNumericTask,
    refinement_root: Option<&ExplicitState>,
) -> Result<DomainRefinementOutcome> {
    let semantics = CegarSemantics::for_options(task, &config.cegar)?;

    let start = Instant::now();
    // The run is a function of its configuration alone: collection
    // generators hand every member its own seed.
    let mut rng = SmallRng::seed_from_u64(
        config
            .cegar
            .random_seed
            .unwrap_or(crate::evaluation::seed::DEFAULT_RANDOM_SEED),
    );
    let (mut domain_mapping, mut domain_sizes) =
        trivial_domain_mapping_and_sizes(task).context("failed to build trivial domain mapping")?;
    let mut partitions = NumericPartitions::trivial(task);
    apply_initial_goal_splits(
        &semantics,
        config,
        &mut rng,
        &mut domain_mapping,
        &mut domain_sizes,
        &mut partitions,
    )?;
    let factory = DomainAbstractionFactory::new(task, domain_mapping, domain_sizes, partitions)
        .context("failed to construct the initial domain abstraction")?;

    let initial_state;
    let root = match refinement_root {
        Some(root) => root,
        None => {
            initial_state = semantics.states().initial_state()?;
            &initial_state
        }
    };
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
            refinement_root: root,
            driver: CegarDriver::new(config.cegar.max_iterations, config.cegar.max_time),
        },
        start,
    )?;
    let (factory, built) = refinement.into_parts();
    if config.cegar.debug {
        log_final_abstraction(task, &factory);
    }
    Ok(DomainRefinementOutcome {
        factory,
        outcome,
        built,
    })
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
    /// the full state space of `task`, and the distance table's initial state
    /// is the initial state of `task`.
    fn generate_from_root(
        &self,
        task: &dyn AbstractNumericTask,
        refinement_root: Option<&ExplicitState>,
    ) -> Result<DomainAbstraction> {
        let DomainRefinementOutcome {
            factory,
            outcome,
            built,
        } = refine_domain_abstraction(&self.config, task, refinement_root)
            .context("CEGAR failed to build abstraction")?;
        let stop_reason = outcome.stop_reason;
        let combine_labels = self.config.cegar.combine_labels;
        let (abstract_operators, distance_table) = match built {
            Some(built) => built,
            None => {
                let operators = factory
                    .abstract_operators(task, combine_labels, Deadline::NONE)
                    .context("failed to build abstract operators")?;
                let table = factory
                    .build_distance_table(task, &operators, Deadline::NONE)
                    .context("failed to build abstract distance table")?;
                (operators, table)
            }
        };
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
        let initial_h = distance_table.distances[distance_table.initial_state];
        // The refinement root is reachable from the initial state, so a dead
        // initial state means a dead root, which is where CEGAR stops.
        ensure!(
            initial_h.is_finite() || stop_reason == CegarStopReason::DeadEnd,
            "domain abstraction initial state is abstract-dead, but CEGAR stopped with {stop_reason:?}; initial_state={}, states={}, abstract_ops={}, prop_domains={:?}, numeric_domains={:?}",
            distance_table.initial_state,
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
