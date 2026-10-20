//! The CEGAR loop shared by every abstraction: find the flaws of an optimal
//! abstract plan, refine one, and repeat until the plan is concrete, the
//! refinement root is an abstract dead end, or a limit is reached.

use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use planforge_sas::axioms::AxiomEvaluator;
use planforge_sas::utils::state_packer::StatePacker;

use super::progression::{
    ConcretePlan, PlanCheck, replay_entire_optimal_abstract_trace, replay_optimal_abstract_trace,
};
use super::regression::{FlawScope, regression_plan_check};
use super::semantics::CegarSemantics;
use super::split::Split;
use super::view::AbstractionView;
use super::{CegarDriver, CegarIterationResult, CegarStopReason, FlawKind};
use crate::evaluation::abstractions::common::concrete_state::ConcreteState;

/// An abstraction that CEGAR refines.
pub(crate) trait Refinable {
    type View<'a>: AbstractionView
    where
        Self: 'a;

    /// The abstraction as the flaw search sees it now. A representation that
    /// rebuilds its transitions after a split does so here and fails with a
    /// deadline error once `deadline` passes.
    fn view(&mut self, deadline: Option<Instant>) -> Result<Self::View<'_>>;

    /// Refines the abstraction by `split`, which its view found refinable.
    fn apply_split(&mut self, split: Split) -> Result<()>;
}

/// Where a CEGAR run searches for flaws and what it may spend.
pub(crate) struct CegarRun<'a> {
    pub(crate) semantics: &'a CegarSemantics<'a>,
    pub(crate) flaw_kind: FlawKind,
    pub(crate) state_packer: &'a Arc<StatePacker>,
    pub(crate) axiom_evaluator: &'a AxiomEvaluator<'a>,
    pub(crate) refinement_root: &'a ConcreteState,
    pub(crate) driver: CegarDriver,
}

/// How a CEGAR run ended.
#[derive(Debug)]
pub(crate) struct CegarOutcome {
    pub(crate) stop_reason: CegarStopReason,
    /// The concrete plan from the refinement root, exactly when the run
    /// stopped with [`CegarStopReason::ConcretePlan`].
    pub(crate) concrete_plan: Option<ConcretePlan>,
    /// The flaw a size limit kept CEGAR from refining.
    pub(crate) pending_flaw: Option<String>,
    pub(crate) refinements: usize,
}

/// The flaws of the optimal abstract plan from the abstract state of
/// `refinement_root` that `flaw_kind` asks for.
pub(crate) fn check_abstract_plan(
    view: &impl AbstractionView,
    run: &CegarRun<'_>,
) -> Result<PlanCheck> {
    let CegarRun {
        semantics,
        state_packer,
        axiom_evaluator,
        refinement_root,
        ..
    } = *run;
    match run.flaw_kind {
        FlawKind::Progression => replay_optimal_abstract_trace(
            view,
            semantics,
            state_packer,
            axiom_evaluator,
            refinement_root,
        ),
        FlawKind::ExecuteEntirePlan => replay_entire_optimal_abstract_trace(
            view,
            semantics,
            state_packer,
            axiom_evaluator,
            refinement_root,
        ),
        FlawKind::Regression => regression_plan_check(
            view,
            semantics,
            state_packer,
            refinement_root,
            FlawScope::First,
        ),
        FlawKind::RegressEntirePlan => regression_plan_check(
            view,
            semantics,
            state_packer,
            refinement_root,
            FlawScope::EntirePlan,
        ),
    }
}

/// Refines `abstraction` by CEGAR from `start` on.
pub(crate) fn refine<R: Refinable>(
    abstraction: &mut R,
    run: &CegarRun<'_>,
    start: Instant,
) -> Result<CegarOutcome> {
    let mut concrete_plan = None;
    let mut pending_flaw = None;
    let mut refinements = 0;
    let result = run.driver.run_from(start, |_iteration, deadline| {
        let split = {
            let view = match abstraction.view(deadline) {
                Ok(view) => view,
                Err(error) if crate::resource_limits::is_deadline_exceeded(&error) => {
                    return Ok(CegarIterationResult::Stop(CegarStopReason::TimeLimit));
                }
                Err(error) => return Err(error),
            };
            let split = match check_abstract_plan(&view, run)? {
                PlanCheck::ConcretePlan(plan) => {
                    concrete_plan = Some(plan);
                    return Ok(CegarIterationResult::Stop(CegarStopReason::ConcretePlan));
                }
                PlanCheck::AbstractDeadEnd => {
                    return Ok(CegarIterationResult::Stop(CegarStopReason::DeadEnd));
                }
                PlanCheck::Refine(split) => split,
            };
            if let Some(limit) = view.refinement_limit(&split) {
                pending_flaw = Some(split.description().to_string());
                return Ok(CegarIterationResult::Stop(limit));
            }
            split
        };
        abstraction.apply_split(split)?;
        refinements += 1;
        Ok(CegarIterationResult::Continue)
    })?;
    Ok(CegarOutcome {
        stop_reason: result.stop_reason,
        concrete_plan,
        pending_flaw,
        refinements,
    })
}
