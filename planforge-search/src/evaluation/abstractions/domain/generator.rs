//! Builds a domain abstraction with CEGAR.

#[cfg(test)]
mod tests;

use std::time::Instant;

use anyhow::{Context, Result};
use planforge_sas::explicit_state::ExplicitState;
use planforge_sas::numeric_task::AbstractNumericTask;

use super::config::DomainAbstractionConfig;
use super::function::DomainAbstractionFunction;
use super::view::{DomainBuild, DomainRefinement};
use crate::evaluation::abstractions::common::abstraction::{
    AbstractionCore, AbstractionMetadata, finish_abstraction,
};
use crate::evaluation::abstractions::common::cegar::initial_splits::{
    RefinementScope, initial_splits,
};
use crate::evaluation::abstractions::common::cegar::refine::{CegarOutcome, refine_from_root};
use crate::evaluation::abstractions::common::cegar::semantics::CegarSemantics;
use crate::evaluation::abstractions::common::cegar::view::RefinementLimits;
use crate::resource_limits::Deadline;

use super::abstraction::DomainAbstraction;
use super::collection::{AbstractionKey, abstraction_key};
use super::debug_dump::log_final_abstraction;
use crate::evaluation::abstractions::collections::generator::AbstractionGenerator;
use crate::evaluation::abstractions::component::AbstractionKind;

/// The domain abstraction CEGAR refined, with the transitions and goal
/// distances of its last flaw search if no split followed it.
#[derive(Debug)]
pub(super) struct DomainRefinementOutcome {
    pub(super) function: DomainAbstractionFunction,
    pub(super) outcome: CegarOutcome,
    pub(super) built: Option<DomainBuild>,
}

/// Refines a domain abstraction of `task` by CEGAR from `refinement_root`,
/// or from the initial state of `task` when `None`: the trivial abstraction
/// with the initial splits of the configured scope, refined until the
/// abstract plan is concrete or a limit is reached.
pub(super) fn refine_domain_abstraction(
    config: &DomainAbstractionConfig,
    task: &dyn AbstractNumericTask,
    refinement_root: Option<&ExplicitState>,
    scope: &RefinementScope,
) -> Result<DomainRefinementOutcome> {
    let start = Instant::now();
    let semantics = CegarSemantics::for_options(task, &config.cegar)?;

    let mut function = DomainAbstractionFunction::trivial(&semantics)
        .context("failed to construct the trivial domain abstraction")?;
    // Every initial split divides a different variable of the trivial
    // abstraction, so it divides every abstract state if it divides one. As
    // for Cartesian abstractions, a split that would exceed `max_states` is
    // left out.
    for initial_split in initial_splits(&semantics, scope, config.cegar.random_seed) {
        if let Some(split) = initial_split.in_state(0, &function.state_region(0))
            && function.states_after_split(&split) <= config.cegar.max_states as u128
        {
            function.apply_split(task, &split)?;
        }
    }

    let mut refinement = DomainRefinement::new(
        task,
        function,
        RefinementLimits::new(&config.cegar, scope),
        config,
    );
    let outcome = refine_from_root(
        &mut refinement,
        &semantics,
        &config.cegar,
        refinement_root,
        start,
    )?;
    let (function, built) = refinement.into_parts();
    if config.cegar.debug {
        log_final_abstraction(task, &function);
    }
    Ok(DomainRefinementOutcome {
        function,
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

    const KIND: AbstractionKind = AbstractionKind::Domain;

    fn new(config: DomainAbstractionConfig) -> Result<Self> {
        config.cegar.validate()?;
        Ok(Self { config })
    }

    fn key(abstraction: &DomainAbstraction) -> AbstractionKey {
        abstraction_key(abstraction)
    }

    /// The root only chooses counterexamples. The abstraction still partitions
    /// the full state space of `task`, and the distance table's initial state
    /// is the initial state of `task`.
    fn generate_from_root(
        &self,
        task: &dyn AbstractNumericTask,
        refinement_root: Option<&ExplicitState>,
        scope: &RefinementScope,
    ) -> Result<DomainAbstraction> {
        let start = Instant::now();
        let DomainRefinementOutcome {
            function,
            outcome,
            built,
        } = refine_domain_abstraction(&self.config, task, refinement_root, scope)
            .context("CEGAR failed to build abstraction")?;
        let DomainBuild {
            transitions,
            distances,
            ..
        } = match built {
            Some(built) => built,
            None => DomainBuild::new(
                function
                    .transitions(task, self.config.cegar.combine_labels, Deadline::NONE)
                    .context("failed to build abstract operators")?,
                Deadline::NONE,
            )
            .context("failed to build the goal distances")?,
        };
        let core = AbstractionCore {
            distances,
            initial_state: function.initial_state(task)?,
            relevant_operator_ids: transitions.relevant_operator_ids(),
            num_abstract_operators: transitions.operators.len(),
            metadata: AbstractionMetadata::standalone(
                outcome,
                self.config.cegar.split_kind,
                self.config.cegar.split_selection_rank,
            ),
        };
        let abstraction = DomainAbstraction::new(
            function,
            core,
            self.config.cegar.keep_transitions.then_some(transitions),
        );
        finish_abstraction(&abstraction, start.elapsed())?;
        Ok(abstraction)
    }
}
