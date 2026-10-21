//! What every abstraction CEGAR builds offers and records about its
//! construction and use, for domain and Cartesian abstractions alike.

use std::cell::RefCell;

use anyhow::Result;
use planforge_sas::explicit_state::ExplicitState;
use planforge_sas::numeric_task::{ExplicitValueIndex, NumericValue, OperatorIndex};

use super::cegar::progression::ConcretePlan;
use super::cegar::refine::CegarOutcome;
use super::cegar::{CegarStopReason, SplitKind};
use super::transition_system::{AbstractDistanceTable, AbstractOperatorSourceSets};
use crate::evaluation::abstractions::collections::goal_focused::CollectionMembership;
use crate::evaluation::evaluator::{EvaluationError, EvaluationState};
use crate::evaluation::heuristic::Heuristic;

/// Whether an abstraction stands alone or is a member of a collection.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AbstractionUse {
    Standalone,
    #[default]
    CollectionMember,
}

impl AbstractionUse {
    pub fn permits_initial_optimality_proof(self) -> bool {
        matches!(self, Self::Standalone)
    }
}

/// How CEGAR built an abstraction, and where the abstraction is used.
#[derive(Debug, Clone)]
pub struct AbstractionMetadata {
    pub abstraction_use: AbstractionUse,
    pub stop_reason: CegarStopReason,
    /// The concrete plan from the refinement root, exactly when CEGAR stopped
    /// with [`CegarStopReason::ConcretePlan`].
    pub concrete_plan_operator_ids: Option<Vec<OperatorIndex>>,
    /// The flaw a limit kept CEGAR from refining.
    pub pending_flaw: Option<String>,
    pub refinements: usize,
    /// The split kind CEGAR refined with.
    pub split_kind: SplitKind,
    /// The rank CEGAR took among tied best splits, if any.
    pub split_selection_rank: Option<usize>,
    /// Set for a member of a collection.
    pub collection: Option<CollectionRecord>,
    /// Set for a member of a goal-focused collection.
    pub collection_member: Option<CollectionMembership>,
}

/// Where a member sits in its collection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CollectionRecord {
    /// The one-based number of the build that produced the member; a
    /// standard collection counts the restarts it dropped as duplicates too.
    pub iteration: usize,
    /// The member abstracts the full task rather than one goal's task.
    pub full_task: bool,
}

impl AbstractionMetadata {
    /// The metadata of a standalone abstraction CEGAR built with `outcome`.
    pub(crate) fn standalone(
        outcome: CegarOutcome,
        split_kind: SplitKind,
        split_selection_rank: Option<usize>,
    ) -> Self {
        Self {
            abstraction_use: AbstractionUse::Standalone,
            stop_reason: outcome.stop_reason,
            concrete_plan_operator_ids: outcome.concrete_plan.map(ConcretePlan::into_operator_ids),
            pending_flaw: outcome.pending_flaw,
            refinements: outcome.refinements,
            split_kind,
            split_selection_rank,
            collection: None,
            collection_member: None,
        }
    }

    /// CEGAR's abstract plan is a concrete plan.
    pub fn solved_by_self(&self) -> bool {
        self.stop_reason == CegarStopReason::ConcretePlan
    }
}

/// An abstraction CEGAR built: an abstraction function with the goal
/// distances of its abstract states.
pub trait CegarAbstraction {
    /// The abstract state containing the concrete state with these values.
    fn abstract_state(
        &self,
        propositional: &[ExplicitValueIndex],
        numeric: &[NumericValue],
    ) -> Result<usize>;

    fn distance_table(&self) -> &AbstractDistanceTable;

    fn metadata(&self) -> &AbstractionMetadata;

    fn metadata_mut(&mut self) -> &mut AbstractionMetadata;

    /// Drops what only cost partitioning reads: the transitions and the
    /// operators' source sets. Lookups stay intact.
    fn discard_transition_data(&mut self);

    /// The number of abstract operators; a Cartesian abstraction has one
    /// per transition.
    fn num_abstract_operators(&self) -> usize;

    /// The source set of every abstract operator's labels, which regional
    /// cost partitioning attaches costs to; empty when not built.
    fn abstract_source_sets(&self) -> &[AbstractOperatorSourceSets];

    fn num_states(&self) -> usize {
        self.distance_table().distances.len()
    }

    /// The goal distance of the task's initial state.
    fn initial_h(&self) -> f64 {
        let table = self.distance_table();
        table.distances[table.initial_state_hash]
    }
}

/// The heuristic of one abstraction: the goal distance of the abstract state
/// a state maps to.
#[derive(Debug, Clone)]
pub struct AbstractionHeuristic<A> {
    name: String,
    abstraction: A,
    values: RefCell<ExplicitState>,
}

impl<A: CegarAbstraction> AbstractionHeuristic<A> {
    pub fn new(name: impl Into<String>, abstraction: A) -> Self {
        Self {
            name: name.into(),
            abstraction,
            values: RefCell::default(),
        }
    }

    pub fn abstraction(&self) -> &A {
        &self.abstraction
    }
}

impl<A: CegarAbstraction> Heuristic for AbstractionHeuristic<A> {
    fn compute_heuristic(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError> {
        let mut values = self.values.borrow_mut();
        eval_state.fill_values(&mut values)?;
        let state_id = self
            .abstraction
            .abstract_state(&values.propositional, &values.numeric)
            .map_err(|error| EvaluationError::ComputationFailed(format!("{error:#}")))?;
        Ok(self.abstraction.distance_table().distances[state_id])
    }

    fn proves_initial_state_optimal(&self) -> bool {
        let metadata = self.abstraction.metadata();
        metadata.solved_by_self() && metadata.abstraction_use.permits_initial_optimality_proof()
    }

    fn heuristic_name(&self) -> &str {
        &self.name
    }
}
