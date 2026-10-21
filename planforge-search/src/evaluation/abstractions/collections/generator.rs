//! The collection generator of every abstraction representation: a standard
//! or goal-focused collection of the abstractions one generator builds.

use std::hash::Hash;

use anyhow::Result;
use planforge_sas::explicit_state::ExplicitState;
use planforge_sas::numeric_task::AbstractNumericTask;
use tracing::info;

use super::config::{CollectionConfig, MemberConfig};
use super::goal_focused::{
    GoalFocusedBackend, GoalVariant, MemberBudget, build_goal_focused_collection,
};
use super::standard::{StandardBackend, StandardMember, build_standard_collection};
use super::strategy::CollectionStrategy;
use crate::evaluation::abstractions::common::abstraction::CegarAbstraction;

/// A CEGAR generator of single abstractions, which collections build their
/// members with.
pub trait AbstractionGenerator: Sized {
    type Config: MemberConfig;
    type Abstraction: CegarAbstraction;
    /// Equal exactly for abstractions that are the same abstraction.
    type Key: Eq + Hash;

    /// The representation's name in log and error messages.
    const NAME: &'static str;

    /// Rejects a configuration the generator cannot follow.
    fn new(config: Self::Config) -> Result<Self>;

    /// Builds an abstraction of `task` whose CEGAR loop searches for flaws
    /// from `refinement_root`, or from the task's initial state when `None`.
    fn generate_from_root(
        &self,
        task: &dyn AbstractNumericTask,
        refinement_root: Option<&ExplicitState>,
    ) -> Result<Self::Abstraction>;

    /// Builds an abstraction of `task` refined from its initial state.
    fn generate(&self, task: &dyn AbstractNumericTask) -> Result<Self::Abstraction> {
        self.generate_from_root(task, None)
    }

    fn key(abstraction: &Self::Abstraction) -> Self::Key;
}

/// Builds the collection [`CollectionConfig`] describes, with members from
/// `G`.
pub struct CollectionGenerator<G: AbstractionGenerator> {
    config: CollectionConfig<G::Config>,
}

impl<G: AbstractionGenerator> CollectionGenerator<G> {
    pub fn new(config: CollectionConfig<G::Config>) -> Result<Self> {
        config.validate()?;
        G::new(config.member.clone())?;
        Ok(Self { config })
    }

    pub fn config(&self) -> &CollectionConfig<G::Config> {
        &self.config
    }

    /// Builds the collection of `task`: by random restarts for the standard
    /// strategy (see [`standard`](super::standard)), otherwise by the
    /// goal-focused schedule (see [`goal_focused`](super::goal_focused)).
    ///
    /// Each member changes only the goal view and the refinement root.
    /// Operators, state mappings, and concrete operator ids stay those of the
    /// task, and every member still partitions the task's full state space,
    /// so the members remain admissible components for canonical and
    /// cost-partitioned collection heuristics.
    pub fn generate(&self, task: &dyn AbstractNumericTask) -> Result<Vec<G::Abstraction>> {
        let members = match self.config.collection_strategy {
            CollectionStrategy::Standard => {
                build_standard_collection(task, &self.config.standard_schedule(), self)?
            }
            CollectionStrategy::PerGoal | CollectionStrategy::GoalFocused => {
                build_goal_focused_collection(task, &self.config.goal_focused_schedule(), self)?
            }
        };
        if self.config.member.cegar().debug {
            for member in &members {
                let metadata = member.metadata();
                info!(
                    "{} collection member: {:?}, states={}, h(init)={}, stop={:?}, split_kind={}",
                    G::NAME,
                    metadata.abstraction_use,
                    member.num_states(),
                    member.initial_h(),
                    metadata.stop_reason,
                    metadata.split_kind,
                );
            }
        }
        Ok(members)
    }
}

impl<G: AbstractionGenerator> StandardBackend for CollectionGenerator<G> {
    type Member = G::Abstraction;
    type Key = G::Key;

    const NAME: &'static str = G::NAME;

    fn build(
        &self,
        task: &dyn AbstractNumericTask,
        member: &StandardMember,
    ) -> Result<Self::Member> {
        G::new(self.config.member.for_restart(member))?.generate_from_root(task, None)
    }

    fn key(&self, member: &Self::Member) -> Self::Key {
        G::key(member)
    }
}

impl<G: AbstractionGenerator> GoalFocusedBackend for CollectionGenerator<G> {
    type Member = G::Abstraction;

    const NAME: &'static str = G::NAME;

    fn build(
        &self,
        task: &dyn AbstractNumericTask,
        root: Option<&ExplicitState>,
        variant: &GoalVariant,
        budget: MemberBudget,
    ) -> Result<Self::Member> {
        let config = self
            .config
            .member
            .for_variant(variant, self.config.split_kind, budget);
        G::new(config)?.generate_from_root(task, root)
    }
}
