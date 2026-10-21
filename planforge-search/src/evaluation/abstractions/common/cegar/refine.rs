//! The CEGAR loop shared by every abstraction: find the flaws of an optimal
//! abstract plan, refine one, and repeat until the plan is concrete, the
//! refinement root is an abstract dead end, or a limit is reached.

use std::time::Instant;

use anyhow::{Result, ensure};
use planforge_sas::explicit_state::ExplicitState;
use planforge_sas::utils::float_tolerance;

use super::progression::{ConcretePlan, PlanCheck, progression_plan_check, validate_concrete_plan};
use super::regression::regression_plan_check;
use super::semantics::CegarSemantics;
use super::split::Split;
use super::view::AbstractionView;
use super::{CegarDriver, CegarIterationResult, CegarStopReason, FlawKind, FlawScope};
use crate::resource_limits::Deadline;

/// An abstraction that CEGAR refines.
pub(crate) trait Refinable {
    type View<'a>: AbstractionView
    where
        Self: 'a;

    /// The abstraction as the flaw search sees it now. A representation that
    /// rebuilds its transitions after a split does so here and fails with a
    /// deadline error once `deadline` passes.
    fn view(&mut self, deadline: Deadline) -> Result<Self::View<'_>>;

    /// Refines the abstraction by `split`, which its view found refinable.
    fn apply_split(&mut self, split: Split) -> Result<()>;
}

/// Where a CEGAR run searches for flaws and what it may spend.
pub(crate) struct CegarRun<'a> {
    pub(crate) semantics: &'a CegarSemantics<'a>,
    pub(crate) flaw_kind: FlawKind,
    pub(crate) flaw_scope: FlawScope,
    /// The concrete state flaws are searched from.
    pub(crate) refinement_root: &'a ExplicitState,
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
/// `refinement_root` that the run's flaw kind and scope ask for.
pub(crate) fn check_abstract_plan(
    view: &impl AbstractionView,
    run: &CegarRun<'_>,
) -> Result<PlanCheck> {
    match run.flaw_kind {
        FlawKind::Progression => {
            progression_plan_check(view, run.semantics, run.refinement_root, run.flaw_scope)
        }
        FlawKind::Regression => {
            regression_plan_check(view, run.semantics, run.refinement_root, run.flaw_scope)
        }
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
                    validate_concrete_plan(run.semantics, run.refinement_root, &plan)?;
                    let root = view.abstract_state(
                        &run.refinement_root.propositional,
                        &run.refinement_root.numeric,
                    )?;
                    let root_distance = view.goal_distance(root);
                    ensure!(
                        (plan.cost() - root_distance).abs() <= float_tolerance::AGREEMENT_EPSILON,
                        "the concrete plan costs {}, but the refinement root's abstract goal distance is {root_distance}",
                        plan.cost()
                    );
                    concrete_plan = Some(plan);
                    return Ok(CegarIterationResult::Stop(CegarStopReason::ConcretePlan));
                }
                PlanCheck::AbstractDeadEnd => {
                    return Ok(CegarIterationResult::Stop(CegarStopReason::DeadEnd));
                }
                PlanCheck::Refine(split) => split,
            };
            if let Some(limit) = view.refinement_limit(&split) {
                pending_flaw = Some(format!("{split:?}"));
                return Ok(CegarIterationResult::Stop(limit));
            }
            tracing::debug!(
                "CEGAR refinement {} at {} abstract states: {split:?}",
                refinements + 1,
                view.num_states()
            );
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
