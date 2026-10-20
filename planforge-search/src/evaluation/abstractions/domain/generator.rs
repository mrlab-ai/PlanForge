//! Builds a domain abstraction with CEGAR.

#[cfg(test)]
mod tests;

use std::cell::RefCell;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use planforge_sas::axioms::AxiomEvaluator;
use planforge_sas::numeric_task::AbstractNumericTask;
use rand::{SeedableRng, rngs::SmallRng};

use super::config::CegarConfig;
use super::factory::{DistanceTableOptions, DomainAbstractionFactory};
use super::initial_splits::{
    RefinementState, apply_initial_goal_splits, trivial_domain_mapping_and_sizes,
};
use super::partitions::NumericPartitions;
use super::view::{DomainRefinement, RefinementLimits};
use crate::evaluation::abstractions::common::cegar::progression::ConcretePlan;
use crate::evaluation::abstractions::common::cegar::refine::{CegarRun, refine};
use crate::evaluation::abstractions::common::cegar::regression::ensure_regressable;
use crate::evaluation::abstractions::common::cegar::semantics::{CegarSemantics, SplitOptions};
use crate::evaluation::abstractions::common::cegar::{CegarDriver, CegarStopReason};
use crate::evaluation::abstractions::common::concrete_state::{
    ConcreteState, make_prop_state_packer,
};
use crate::evaluation::abstractions::common::task::{AbstractionUse, RootedTask};

use super::abstraction::{DomainAbstraction, DomainAbstractionMetadata, compute_hash_multipliers};
use super::debug_dump::log_final_abstraction;

/// The domain abstraction CEGAR refined, before its distances are finalized.
#[derive(Debug)]
pub(super) struct DomainRefinementOutcome {
    pub(super) factory: DomainAbstractionFactory,
    pub(super) stop_reason: CegarStopReason,
    /// The concrete plan from the task initial state, exactly when CEGAR
    /// stopped with [`CegarStopReason::ConcretePlan`].
    pub(super) concrete_plan: Option<ConcretePlan>,
}

/// Refines a domain abstraction of `task` by CEGAR from the initial state of
/// `task`: the trivial abstraction split on the goal variables, refined until
/// the abstract plan is concrete or a limit is reached.
pub(super) fn refine_domain_abstraction(
    config: &CegarConfig,
    task: &dyn AbstractNumericTask,
) -> Result<DomainRefinementOutcome> {
    let semantics = CegarSemantics::new(
        task,
        SplitOptions {
            kind: config.split_kind(),
            selection: config.flaw_treatment.split_selection(),
            rank: config.flaw_selection_rank,
            random_seed: config.random_seed,
        },
    )?;
    if config.flaw_kind.regresses() {
        ensure_regressable(task, semantics.coordinates())?;
    }

    let start = Instant::now();
    // The run is a function of its configuration alone: collection
    // generators hand every member its own seed.
    let mut rng = SmallRng::seed_from_u64(
        config
            .random_seed
            .unwrap_or(crate::evaluation::DEFAULT_RANDOM_SEED),
    );
    let (mut domain_mapping, mut domain_sizes) =
        trivial_domain_mapping_and_sizes(task).context("failed to build trivial domain mapping")?;
    let mut partitions = NumericPartitions::trivial(task);
    let mut numeric_domain_sizes: Vec<usize> = vec![1; task.numeric_variables().len()];
    let mut blacklisted_prop_var_ids = config.blacklisted_prop_var_ids.clone();
    let mut blacklisted_numeric_var_ids = config.blacklisted_numeric_var_ids.clone();
    apply_initial_goal_splits(
        task,
        config,
        semantics.numeric_integer_lattice(),
        &mut rng,
        RefinementState {
            domain_mapping: &mut domain_mapping,
            domain_sizes: &mut domain_sizes,
            partitions: &mut partitions,
            numeric_domain_sizes: &mut numeric_domain_sizes,
            blacklisted_prop_var_ids: &mut blacklisted_prop_var_ids,
            blacklisted_numeric_var_ids: &mut blacklisted_numeric_var_ids,
        },
    )?;
    let factory = DomainAbstractionFactory::new(
        task,
        domain_mapping,
        domain_sizes,
        partitions,
        numeric_domain_sizes,
    )
    .context("failed to construct the initial domain abstraction")?;

    let state_packer = Arc::new(make_prop_state_packer(task));
    let axiom_evaluator = AxiomEvaluator::new(Arc::new(task), state_packer.clone());
    let root = ConcreteState::initial(task)?;
    let limits = RefinementLimits {
        max_states: config.max_abstraction_size,
        blacklisted_prop_var_ids: &blacklisted_prop_var_ids,
        blacklisted_numeric_var_ids: &blacklisted_numeric_var_ids,
    };
    let mut refinement = DomainRefinement::new(task, factory, limits, config);
    let outcome = refine(
        &mut refinement,
        &CegarRun {
            semantics: &semantics,
            flaw_kind: config.flaw_kind,
            state_packer: &state_packer,
            axiom_evaluator: &axiom_evaluator,
            refinement_root: &root,
            driver: CegarDriver::new(config.max_iterations, config.max_time),
        },
        start,
    )?;
    let factory = refinement.into_factory();
    if config.debug {
        log_final_abstraction(task, &factory);
    }
    Ok(DomainRefinementOutcome {
        factory,
        stop_reason: outcome.stop_reason,
        concrete_plan: outcome.concrete_plan,
    })
}

/// Numeric-fd style generator that constructs a domain abstraction via CEGAR.
#[derive(Debug, Clone)]
pub struct DomainAbstractionGenerator {
    config: CegarConfig,
}

impl DomainAbstractionGenerator {
    pub fn new(config: CegarConfig) -> Result<Self> {
        ensure!(
            config.max_abstraction_size > 0,
            "max_abstraction_size must be > 0"
        );
        ensure!(config.max_iterations > 0, "max_iterations must be > 0");
        Ok(Self { config })
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
                refine_domain_abstraction(&self.config, &rooted_task)
            }
            None => refine_domain_abstraction(&self.config, task),
        }
        .context("CEGAR failed to build abstraction")?;
        let solved_by_self = outcome.concrete_plan.is_some();
        let stop_reason = outcome.stop_reason;
        let concrete_plan_operator_ids = outcome.concrete_plan.map(ConcretePlan::into_operator_ids);
        let factory = outcome.factory;
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
            transition_data_retained: true,
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
