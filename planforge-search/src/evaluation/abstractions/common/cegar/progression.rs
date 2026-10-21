//! Progression flaws: replay a concrete state along an optimal abstract plan.
//!
//! The first-flaw search follows the plan until the concrete state violates
//! an operator precondition, leaves the planned abstract successor, or ends
//! in an abstract goal state that is not a concrete one. The whole-plan search
//! collects every flaw of one replay: after a deviation it continues from the
//! abstract state the concrete successor actually lies in.

use std::collections::HashSet;

use anyhow::{Context, Result, ensure};
use planforge_sas::explicit_state::ExplicitState;
use planforge_sas::numeric_task::OperatorIndex;

use super::FlawScope;
use super::fact_splits::{DeviationWitness, split_deviation_candidates, split_failed_fact};
use super::selection::{
    DEVIATION_TAG, GOAL_TAG, PRECONDITION_TAG, push_unique_split, select_refinement,
    select_refinement_split,
};
use super::semantics::CegarSemantics;
use super::split::{Split, approximately_equal};
use super::view::AbstractionView;

#[derive(Debug)]
pub(crate) struct ConcretePlan {
    operator_ids: Vec<OperatorIndex>,
    cost: f64,
}

impl ConcretePlan {
    pub(crate) fn new(operator_ids: Vec<OperatorIndex>, cost: f64) -> Self {
        Self { operator_ids, cost }
    }

    pub(crate) fn cost(&self) -> f64 {
        self.cost
    }

    pub(crate) fn into_operator_ids(self) -> Vec<OperatorIndex> {
        self.operator_ids
    }
}

pub(crate) enum PlanCheck {
    ConcretePlan(ConcretePlan),
    /// The refinement root is an abstract dead end.
    AbstractDeadEnd,
    Refine(Split),
}

pub(crate) fn replay_plan(
    view: &impl AbstractionView,
    semantics: &CegarSemantics<'_>,
    refinement_root: &ExplicitState,
) -> Result<PlanCheck> {
    let mut state = refinement_root.clone();
    let mut successor = refinement_root.clone();
    let initial_abstract_state = view.abstract_state(&state.propositional, &state.numeric)?;
    if !view.goal_distance(initial_abstract_state).is_finite() {
        return Ok(PlanCheck::AbstractDeadEnd);
    }
    let abstract_plan_cost = view.goal_distance(initial_abstract_state);
    let mut operator_ids: Vec<OperatorIndex> = Vec::new();
    let mut concrete_cost = 0.0;

    loop {
        let abstract_state = view.abstract_state(&state.propositional, &state.numeric)?;
        let abstract_distance = view.goal_distance(abstract_state);
        ensure!(
            approximately_equal(concrete_cost + abstract_distance, abstract_plan_cost),
            "concrete trace left optimal abstract path: g={concrete_cost} h={abstract_distance} initial_h={abstract_plan_cost}"
        );

        if view.is_goal(abstract_state)? {
            if semantics.states().is_goal(&state) {
                return Ok(PlanCheck::ConcretePlan(ConcretePlan {
                    operator_ids,
                    cost: concrete_cost,
                }));
            }
            let failed_goals = (0..semantics.task().get_num_goals())
                .map(|goal_id| semantics.task().get_goal_fact(goal_id))
                .filter(|goal| !state.holds(goal))
                .collect::<Vec<_>>();
            ensure!(
                !failed_goals.is_empty(),
                "abstract goal contains a concrete non-goal without a failed goal fact"
            );
            let candidates = failed_goals
                .iter()
                .map(|goal| {
                    split_failed_fact(
                        view,
                        semantics,
                        abstract_state,
                        goal,
                        &state.propositional,
                        &state.numeric,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            return Ok(PlanCheck::Refine(select_refinement_split(
                view, semantics, candidates, GOAL_TAG,
            )?));
        }

        ensure!(
            operator_ids.len() <= view.num_states(),
            "optimal abstract transitions contain a cycle"
        );
        let transition = view.optimal_transition(abstract_state)?;
        // Every operator of the step is tried in turn: the first one that
        // applies and lands in the planned successor is executed, and only if
        // none does are the flaws the step's operators witnessed refined.
        let mut precondition_splits = Vec::new();
        let mut deviation_splits = Vec::new();
        let mut executed = None;
        for op_id in view.step_operators(transition) {
            let op = &semantics.task().get_operators()[op_id.index()];
            let failed_preconditions = op
                .preconditions()
                .iter()
                .filter(|fact| !state.holds(fact))
                .collect::<Vec<_>>();
            if !failed_preconditions.is_empty() {
                for failed in failed_preconditions {
                    precondition_splits.push(split_failed_fact(
                        view,
                        semantics,
                        abstract_state,
                        failed,
                        &state.propositional,
                        &state.numeric,
                    )?);
                }
                continue;
            }
            semantics
                .states()
                .successor_into(&state, op, &mut successor)?;
            let concrete_target =
                view.abstract_state(&successor.propositional, &successor.numeric)?;
            if concrete_target != transition.target {
                deviation_splits.extend(split_deviation_candidates(
                    view,
                    semantics,
                    DeviationWitness::new(
                        abstract_state,
                        transition.target,
                        op_id,
                        &successor.propositional,
                        &state.numeric,
                        &successor.numeric,
                    ),
                )?);
                continue;
            }
            executed = Some(op_id);
            break;
        }
        let Some(op_id) = executed else {
            let (candidates, tag) = if precondition_splits.is_empty() {
                (deviation_splits, DEVIATION_TAG)
            } else {
                precondition_splits.extend(deviation_splits);
                (precondition_splits, PRECONDITION_TAG)
            };
            return Ok(PlanCheck::Refine(select_refinement_split(
                view, semantics, candidates, tag,
            )?));
        };
        // `successor` holds the executed operator's successor.
        std::mem::swap(&mut state, &mut successor);

        let op_cost = semantics.operator_costs()[op_id.index()];
        ensure!(
            approximately_equal(
                op_cost.value() + view.goal_distance(transition.target),
                abstract_distance
            ),
            "optimal abstract transition does not preserve the goal distance"
        );
        concrete_cost += op_cost.value();
        operator_ids.push(op_id);
    }
}

/// Progression flaw search in `scope` on the optimal abstract plan from the
/// abstract state of `refinement_root`. Without a flaw, the executed operators
/// are a concrete plan; with several, the split is selected among them.
pub(crate) fn progression_plan_check(
    view: &impl AbstractionView,
    semantics: &CegarSemantics<'_>,
    refinement_root: &ExplicitState,
    scope: FlawScope,
) -> Result<PlanCheck> {
    match scope {
        FlawScope::First => replay_plan(view, semantics, refinement_root),
        FlawScope::WholePlan => replay_whole_plan(view, semantics, refinement_root),
    }
}

/// Replays the current optimal abstract policy against one concrete execution
/// and ranks every witnessed flaw together. After a deviation, replay resumes
/// from the abstract state containing the real concrete successor. This keeps
/// every split tied to a concrete witness and avoids inventing projected states.
fn replay_whole_plan(
    view: &impl AbstractionView,
    semantics: &CegarSemantics<'_>,
    refinement_root: &ExplicitState,
) -> Result<PlanCheck> {
    let mut state = refinement_root.clone();
    let mut successor = refinement_root.clone();
    let mut operator_ids = Vec::new();
    let mut concrete_cost = 0.0;
    let mut candidates = Vec::new();
    let mut candidate_identities = HashSet::new();
    let mut visited_abstract_states = HashSet::new();

    loop {
        let abstract_state = view.abstract_state(&state.propositional, &state.numeric)?;
        if !visited_abstract_states.insert(abstract_state) {
            ensure!(
                !candidates.is_empty(),
                "whole-plan replay cycled without witnessing a flaw"
            );
            return select_refinement(view, semantics, candidates);
        }
        let abstract_distance = view.goal_distance(abstract_state);
        if !abstract_distance.is_finite() {
            return if candidates.is_empty() {
                // Without a flaw the replay is still on the optimal abstract
                // plan, whose states all reach the goal: this is the root.
                assert!(
                    operator_ids.is_empty(),
                    "flawless whole-plan replay reached an abstract dead end"
                );
                Ok(PlanCheck::AbstractDeadEnd)
            } else {
                select_refinement(view, semantics, candidates)
            };
        }

        if view.is_goal(abstract_state)? {
            if semantics.states().is_goal(&state) {
                return if candidates.is_empty() {
                    Ok(PlanCheck::ConcretePlan(ConcretePlan {
                        operator_ids,
                        cost: concrete_cost,
                    }))
                } else {
                    select_refinement(view, semantics, candidates)
                };
            }
            for goal_id in 0..semantics.task().get_num_goals() {
                let goal = semantics.task().get_goal_fact(goal_id);
                if !state.holds(goal) {
                    let split = split_failed_fact(
                        view,
                        semantics,
                        abstract_state,
                        goal,
                        &state.propositional,
                        &state.numeric,
                    )?;
                    push_unique_split(&mut candidates, &mut candidate_identities, split);
                }
            }
            ensure!(
                !candidates.is_empty(),
                "abstract goal contains a concrete non-goal without a refinable failed goal fact"
            );
            return select_refinement(view, semantics, candidates);
        }

        let transition = view.optimal_transition(abstract_state)?;
        ensure!(
            approximately_equal(
                semantics.operator_costs()[transition.concrete_op_id.index()].value()
                    + view.goal_distance(transition.target),
                abstract_distance
            ),
            "optimal abstract transition does not preserve the goal distance"
        );

        // The first operator of the step whose preconditions hold is executed;
        // the precondition flaws of the operators before it are not flaws of
        // the plan. If none applies, the first operator is executed anyway and
        // the step's precondition flaws are kept.
        let step_operators = view.step_operators(transition);
        let mut step_splits = Vec::new();
        let mut applicable = None;
        for &op_id in &step_operators {
            let op = &semantics.task().get_operators()[op_id.index()];
            let failed_preconditions = op
                .preconditions()
                .iter()
                .filter(|fact| !state.holds(fact))
                .collect::<Vec<_>>();
            if failed_preconditions.is_empty() {
                applicable = Some(op_id);
                step_splits.clear();
                break;
            }
            for failed in failed_preconditions {
                step_splits.push(split_failed_fact(
                    view,
                    semantics,
                    abstract_state,
                    failed,
                    &state.propositional,
                    &state.numeric,
                )?);
            }
        }
        for split in step_splits {
            push_unique_split(&mut candidates, &mut candidate_identities, split);
        }
        let op_id = applicable.unwrap_or(step_operators[0]);
        let op = &semantics.task().get_operators()[op_id.index()];

        // Past a failed precondition the replay continues counterfactually.
        semantics
            .states()
            .apply_effects_into(&state, op, &mut successor)?;
        let concrete_target = view.abstract_state(&successor.propositional, &successor.numeric)?;
        if concrete_target != transition.target {
            for split in split_deviation_candidates(
                view,
                semantics,
                DeviationWitness::new(
                    abstract_state,
                    transition.target,
                    op_id,
                    &successor.propositional,
                    &state.numeric,
                    &successor.numeric,
                ),
            )? {
                push_unique_split(&mut candidates, &mut candidate_identities, split);
            }
        }
        std::mem::swap(&mut state, &mut successor);

        concrete_cost += semantics.operator_costs()[op_id.index()].value();
        operator_ids.push(op_id);
    }
}

/// Checks that `plan` reaches the goal from `refinement_root` at its recorded
/// cost.
pub(crate) fn validate_concrete_plan(
    semantics: &CegarSemantics<'_>,
    refinement_root: &ExplicitState,
    plan: &ConcretePlan,
) -> Result<()> {
    let mut state = refinement_root.clone();
    let mut successor = refinement_root.clone();
    let mut cost = 0.0;
    for (step, &op_id) in plan.operator_ids.iter().enumerate() {
        let op = semantics
            .task()
            .get_operators()
            .get(op_id.index())
            .with_context(|| format!("concrete plan step {step} has invalid operator {op_id}"))?;
        for precondition in op.preconditions() {
            ensure!(
                state.holds(precondition),
                "concrete plan operator {op_id} ({}) has false precondition {precondition:?} at step {step}",
                op.name()
            );
        }
        semantics
            .states()
            .successor_into(&state, op, &mut successor)?;
        std::mem::swap(&mut state, &mut successor);
        cost += semantics.operator_costs()[op_id.index()].value();
    }
    ensure!(
        semantics.states().is_goal(&state),
        "replayed concrete plan does not satisfy the full goal"
    );
    ensure!(
        approximately_equal(cost, plan.cost),
        "replayed concrete plan cost {cost} differs from recorded cost {}",
        plan.cost
    );
    Ok(())
}
