//! What every abstraction CEGAR builds offers and records about its
//! construction and use, for domain and Cartesian abstractions alike.

use anyhow::Result;
use planforge_sas::explicit_state::StateValues;
use planforge_sas::numeric_task::{NumericValue, OperatorIndex};
use planforge_search::heuristic_features::HeuristicBuildError;

use super::cegar::progression::ConcretePlan;
use super::cegar::refine::CegarOutcome;
use super::cegar::{CegarStopReason, SplitKind};
use super::regions::RegionSemantics;
use super::transition_system::AbstractOperatorSourceSets;
use crate::abstractions::component::AbstractionKind;
use crate::abstractions::cost_partitioning::costs::{RemainingCosts, SourceSetCostAllocation};
use planforge_search::evaluation::Heuristic;
use planforge_search::evaluation::{Estimate, EvaluationError, EvaluationState};
use planforge_search::resource_limits::Deadline;

/// How CEGAR built an abstraction.
#[derive(Debug, Clone)]
pub struct AbstractionMetadata {
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
}

impl AbstractionMetadata {
    /// The metadata of an abstraction CEGAR built with `outcome`.
    pub(crate) fn new(
        outcome: CegarOutcome,
        split_kind: SplitKind,
        split_selection_rank: Option<usize>,
    ) -> Self {
        Self {
            stop_reason: outcome.stop_reason,
            concrete_plan_operator_ids: outcome.concrete_plan.map(ConcretePlan::into_operator_ids),
            pending_flaw: outcome.pending_flaw,
            refinements: outcome.refinements,
            split_kind,
            split_selection_rank,
        }
    }

    /// CEGAR's abstract plan is a concrete plan.
    pub fn solved_by_self(&self) -> bool {
        self.stop_reason == CegarStopReason::ConcretePlan
    }
}

/// What every abstraction CEGAR built records, whatever its representation.
#[derive(Debug, Clone)]
pub struct AbstractionCore {
    /// The goal distance of every abstract state.
    pub(crate) distances: Vec<f64>,
    /// The abstract state of the task's initial state.
    pub(crate) initial_state: usize,
    /// The concrete operators that label a transition between two distinct
    /// abstract states, ascending.
    pub(crate) relevant_operator_ids: Vec<OperatorIndex>,
    /// The abstract operators CEGAR built; a Cartesian abstraction has one
    /// per transition.
    pub(crate) num_abstract_operators: usize,
    pub(crate) metadata: AbstractionMetadata,
}

/// An abstraction CEGAR built: an abstraction function with the goal
/// distances of its abstract states, and, until discarded, the transitions
/// cost partitioning saturates.
pub trait CegarAbstraction: std::fmt::Debug + Send {
    fn kind(&self) -> AbstractionKind;

    fn core(&self) -> &AbstractionCore;

    fn core_mut(&mut self) -> &mut AbstractionCore;

    /// The abstract state containing the concrete state with these values.
    fn abstract_state(&self, state: StateValues<'_>) -> usize;

    /// Drops the transitions, which only cost partitioning reads. Lookups
    /// stay intact; saturation fails afterwards.
    fn discard_transitions(&mut self);

    /// The transitions were not discarded.
    fn retains_transitions(&self) -> bool;

    /// Builds what regional saturation reads: the source set of every
    /// abstract operator's labels, unless built already. Fails once the
    /// transitions were discarded.
    fn prepare_regional_saturation(
        &mut self,
        regions: &RegionSemantics<'_>,
        deadline: Deadline,
    ) -> Result<(), HeuristicBuildError>;

    /// The source sets regional cost partitioning attaches costs to; `None`
    /// until [`Self::prepare_regional_saturation`] built them.
    fn abstract_source_sets(&self) -> Option<&[AbstractOperatorSourceSets]>;

    /// The goal distances under `operator_costs` and the saturated cost of
    /// every operator; see `cost_partitioning::costs::saturate_labels`.
    fn saturate_labels(
        &self,
        operator_costs: &[f64],
        perimeter_state: Option<usize>,
        deadline: Deadline,
    ) -> Result<(Vec<f64>, Vec<f64>), HeuristicBuildError>;

    /// The goal distances under `remaining` and what every transition
    /// needs of its labels' source sets; see
    /// `cost_partitioning::costs::saturate_regions`.
    fn saturate_regions(
        &self,
        remaining: &RemainingCosts,
        perimeter_state: Option<usize>,
        deadline: Deadline,
    ) -> Result<(Vec<f64>, SourceSetCostAllocation), HeuristicBuildError>;

    fn metadata(&self) -> &AbstractionMetadata {
        &self.core().metadata
    }

    fn distances(&self) -> &[f64] {
        &self.core().distances
    }

    fn num_states(&self) -> usize {
        self.core().distances.len()
    }

    fn num_abstract_operators(&self) -> usize {
        self.core().num_abstract_operators
    }

    fn relevant_operator_ids(&self) -> &[OperatorIndex] {
        &self.core().relevant_operator_ids
    }

    /// The goal distance of the task's initial state.
    fn initial_h(&self) -> f64 {
        let core = self.core();
        core.distances[core.initial_state]
    }

    /// The goal distance of the abstract state of the concrete state with
    /// these values.
    fn goal_distance(&self, state: StateValues<'_>) -> f64 {
        self.core().distances[self.abstract_state(state)]
    }
}

/// Checks and logs an abstraction built in `elapsed`: CEGAR
/// reaches a refinement root from the initial state, so an initial state
/// without finite goal distance means CEGAR stopped at a dead end.
pub(crate) fn finish_abstraction(
    abstraction: &impl CegarAbstraction,
    elapsed: std::time::Duration,
) -> Result<()> {
    let metadata = abstraction.metadata();
    let initial_h = abstraction.initial_h();
    anyhow::ensure!(
        initial_h.is_finite() || metadata.stop_reason == CegarStopReason::DeadEnd,
        "the initial state of a {} has no finite goal distance, but CEGAR stopped with {:?}",
        abstraction.kind(),
        metadata.stop_reason
    );
    tracing::debug!(
        "{}: states={}, abstract_operators={}, refinements={}, h(init)={initial_h}, stop={:?}, elapsed={:.3}s",
        abstraction.kind(),
        abstraction.num_states(),
        abstraction.num_abstract_operators(),
        metadata.refinements,
        metadata.stop_reason,
        elapsed.as_secs_f64()
    );
    Ok(())
}

/// The heuristic of one abstraction: the goal distance of the abstract state
/// a state maps to.
#[derive(Debug, Clone)]
pub struct AbstractionHeuristic<A> {
    name: String,
    abstraction: A,
    numeric: Vec<NumericValue>,
}

impl<A: CegarAbstraction> AbstractionHeuristic<A> {
    pub fn new(name: impl Into<String>, abstraction: A) -> Self {
        Self {
            name: name.into(),
            abstraction,
            numeric: Vec::new(),
        }
    }

    pub fn abstraction(&self) -> &A {
        &self.abstraction
    }
}

impl<A: CegarAbstraction> Heuristic for AbstractionHeuristic<A> {
    fn evaluate(
        &mut self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<Estimate, EvaluationError> {
        let state = eval_state.values(&mut self.numeric)?;
        Ok(Estimate::new(self.abstraction.goal_distance(state)))
    }

    /// The abstraction is of the full task and refined from its initial
    /// state, so a concrete plan CEGAR found there is optimal.
    fn proves_initial_state_optimal(&self) -> bool {
        self.abstraction.metadata().solved_by_self()
    }

    fn name(&self) -> &str {
        &self.name
    }
}
