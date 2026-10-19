//! Progression flaws: replay a concrete state along an optimal abstract plan.
//!
//! The first-flaw search follows the plan until the concrete state violates
//! an operator precondition, leaves the planned abstract successor, or ends
//! in an abstract goal state that is not a concrete one. The whole-plan search
//! collects every flaw of one replay: after a deviation it continues from the
//! abstract state the concrete successor actually lies in.

use std::collections::HashSet;
use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use planforge_sas::axioms::AxiomEvaluator;
use planforge_sas::numeric_task::OperatorIndex;
use planforge_sas::utils::state_packer::StatePacker;

use super::fact_splits::{
    DeviationWitness, split_deviation, split_deviation_candidates, split_failed_fact,
};
use super::progress_concrete_state;
use super::selection::{push_unique_split, select_refinement, select_refinement_split};
use super::semantics::CegarSemantics;
use super::split::{Split, approximately_equal};
use super::view::AbstractionView;
use crate::evaluation::abstractions::common::concrete_state::{
    ConcreteState, concrete_prop_values, fact_is_held,
};

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
    AbstractDeadEnd(usize),
    Refine(Split),
}

fn concrete_is_goal(
    semantics: &CegarSemantics<'_>,
    state_packer: &StatePacker,
    propositions: &[u64],
) -> bool {
    (0..semantics.task().get_num_goals()).all(|goal_id| {
        fact_is_held(
            semantics.task().get_goal_fact(goal_id),
            state_packer,
            propositions,
        )
    })
}

pub(crate) fn replay_optimal_abstract_trace(
    view: &impl AbstractionView,
    semantics: &CegarSemantics<'_>,
    state_packer: &Arc<StatePacker>,
    axiom_evaluator: &AxiomEvaluator<'_>,
    refinement_root: &ConcreteState,
) -> Result<PlanCheck> {
    let mut propositions = refinement_root.propositions.clone();
    let mut numeric = refinement_root.numeric.clone();
    let mut prop_values = Vec::new();
    let mut successor_prop_values = Vec::new();
    concrete_prop_values(
        semantics.task(),
        state_packer,
        &propositions,
        &mut prop_values,
    );
    let initial_abstract_state = view.abstract_state(&prop_values, &numeric)?;
    if !view.goal_distance(initial_abstract_state).is_finite() {
        return Ok(PlanCheck::AbstractDeadEnd(initial_abstract_state));
    }
    let abstract_plan_cost = view.goal_distance(initial_abstract_state);
    let mut operator_ids: Vec<OperatorIndex> = Vec::new();
    let mut concrete_cost = 0.0;

    loop {
        concrete_prop_values(
            semantics.task(),
            state_packer,
            &propositions,
            &mut prop_values,
        );
        let abstract_state = view.abstract_state(&prop_values, &numeric)?;
        let abstract_distance = view.goal_distance(abstract_state);
        ensure!(
            approximately_equal(concrete_cost + abstract_distance, abstract_plan_cost),
            "concrete trace left optimal abstract path: g={concrete_cost} h={abstract_distance} initial_h={abstract_plan_cost}"
        );

        if view.is_goal(abstract_state) {
            if concrete_is_goal(semantics, state_packer, &propositions) {
                return Ok(PlanCheck::ConcretePlan(ConcretePlan {
                    operator_ids,
                    cost: concrete_cost,
                }));
            }
            let failed_goals = (0..semantics.task().get_num_goals())
                .map(|goal_id| semantics.task().get_goal_fact(goal_id))
                .filter(|goal| !fact_is_held(goal, state_packer, &propositions))
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
                        &prop_values,
                        &numeric,
                        format!("goal {goal:?}"),
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            return Ok(PlanCheck::Refine(select_refinement_split(
                view,
                semantics,
                candidates,
                0x474F_414C,
            )?));
        }

        ensure!(
            operator_ids.len() <= view.num_states(),
            "optimal abstract transitions contain a cycle"
        );
        let transition = view.optimal_transition(abstract_state)?;
        let op_id = transition.concrete_op_id;
        let op = &semantics.task().get_operators()[op_id.index()];
        let failed_preconditions = op
            .preconditions()
            .iter()
            .filter(|fact| !fact_is_held(fact, state_packer, &propositions))
            .collect::<Vec<_>>();
        if !failed_preconditions.is_empty() {
            let candidates = failed_preconditions
                .iter()
                .map(|failed| {
                    split_failed_fact(
                        view,
                        semantics,
                        abstract_state,
                        failed,
                        &prop_values,
                        &numeric,
                        format!("operator {op_id} ({}) precondition {failed:?}", op.name()),
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            return Ok(PlanCheck::Refine(select_refinement_split(
                view,
                semantics,
                candidates,
                0x5052_4543,
            )?));
        }

        let source_numeric = numeric.clone();
        progress_concrete_state(
            op,
            axiom_evaluator,
            state_packer,
            &mut propositions,
            &mut numeric,
        )?;
        concrete_prop_values(
            semantics.task(),
            state_packer,
            &propositions,
            &mut successor_prop_values,
        );
        let concrete_target = view.abstract_state(&successor_prop_values, &numeric)?;
        if concrete_target != transition.target {
            return Ok(PlanCheck::Refine(split_deviation(
                view,
                semantics,
                DeviationWitness::new(
                    abstract_state,
                    transition.target,
                    op_id,
                    &successor_prop_values,
                    &source_numeric,
                    &numeric,
                ),
            )?));
        }

        let op_cost = semantics.operator_costs()[op_id.index()];
        ensure!(
            approximately_equal(
                op_cost.value() + view.goal_distance(transition.target),
                abstract_distance
            ),
            "Cartesian generating transition is not distance preserving"
        );
        concrete_cost += op_cost.value();
        operator_ids.push(op_id);
    }
}

/// Replays the current optimal abstract policy against one concrete execution
/// and ranks every witnessed flaw together. After a deviation, replay resumes
/// from the abstract state containing the real concrete successor. This keeps
/// every split tied to a concrete witness and avoids inventing projected states.
pub(crate) fn replay_entire_optimal_abstract_trace(
    view: &impl AbstractionView,
    semantics: &CegarSemantics<'_>,
    state_packer: &Arc<StatePacker>,
    axiom_evaluator: &AxiomEvaluator<'_>,
    refinement_root: &ConcreteState,
) -> Result<PlanCheck> {
    let mut propositions = refinement_root.propositions.clone();
    let mut numeric = refinement_root.numeric.clone();
    let mut prop_values = Vec::new();
    let mut successor_prop_values = Vec::new();
    let mut operator_ids = Vec::new();
    let mut concrete_cost = 0.0;
    let mut candidates = Vec::new();
    let mut candidate_identities = HashSet::new();
    let mut visited_abstract_states = HashSet::new();

    loop {
        concrete_prop_values(
            semantics.task(),
            state_packer,
            &propositions,
            &mut prop_values,
        );
        let abstract_state = view.abstract_state(&prop_values, &numeric)?;
        if !visited_abstract_states.insert(abstract_state) {
            ensure!(
                !candidates.is_empty(),
                "Cartesian whole-plan replay cycled without witnessing a flaw"
            );
            return select_refinement(view, semantics, candidates);
        }
        let abstract_distance = view.goal_distance(abstract_state);
        if !abstract_distance.is_finite() {
            return if candidates.is_empty() {
                Ok(PlanCheck::AbstractDeadEnd(abstract_state))
            } else {
                select_refinement(view, semantics, candidates)
            };
        }

        if view.is_goal(abstract_state) {
            if concrete_is_goal(semantics, state_packer, &propositions) {
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
                if !fact_is_held(goal, state_packer, &propositions) {
                    let split = split_failed_fact(
                        view,
                        semantics,
                        abstract_state,
                        goal,
                        &prop_values,
                        &numeric,
                        format!("goal {goal:?}"),
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
            "Cartesian generating transition is not distance preserving"
        );

        let op_id = transition.concrete_op_id;
        let op = &semantics.task().get_operators()[op_id.index()];
        for failed in op
            .preconditions()
            .iter()
            .filter(|fact| !fact_is_held(fact, state_packer, &propositions))
        {
            let split = split_failed_fact(
                view,
                semantics,
                abstract_state,
                failed,
                &prop_values,
                &numeric,
                format!("operator {op_id} ({}) precondition {failed:?}", op.name()),
            )?;
            push_unique_split(&mut candidates, &mut candidate_identities, split);
        }

        let source_numeric = numeric.clone();
        progress_concrete_state(
            op,
            axiom_evaluator,
            state_packer,
            &mut propositions,
            &mut numeric,
        )?;
        concrete_prop_values(
            semantics.task(),
            state_packer,
            &propositions,
            &mut successor_prop_values,
        );
        let concrete_target = view.abstract_state(&successor_prop_values, &numeric)?;
        if concrete_target != transition.target {
            for split in split_deviation_candidates(
                view,
                semantics,
                DeviationWitness::new(
                    abstract_state,
                    transition.target,
                    op_id,
                    &successor_prop_values,
                    &source_numeric,
                    &numeric,
                ),
            )? {
                push_unique_split(&mut candidates, &mut candidate_identities, split);
            }
        }

        concrete_cost += semantics.operator_costs()[op_id.index()].value();
        operator_ids.push(op_id);
    }
}

pub(crate) fn validate_concrete_plan(
    semantics: &CegarSemantics<'_>,
    state_packer: &Arc<StatePacker>,
    axiom_evaluator: &AxiomEvaluator<'_>,
    refinement_root: &ConcreteState,
    plan: &ConcretePlan,
) -> Result<()> {
    let mut propositions = refinement_root.propositions.clone();
    let mut numeric = refinement_root.numeric.clone();
    let mut cost = 0.0;
    for (step, &op_id) in plan.operator_ids.iter().enumerate() {
        let op = semantics
            .task()
            .get_operators()
            .get(op_id.index())
            .with_context(|| format!("concrete plan step {step} has invalid operator {op_id}"))?;
        for precondition in op.preconditions() {
            ensure!(
                fact_is_held(precondition, state_packer, &propositions),
                "concrete plan operator {op_id} ({}) has false precondition {precondition:?} at step {step}",
                op.name()
            );
        }
        progress_concrete_state(
            op,
            axiom_evaluator,
            state_packer,
            &mut propositions,
            &mut numeric,
        )?;
        cost += semantics.operator_costs()[op_id.index()].value();
    }
    ensure!(
        concrete_is_goal(semantics, state_packer, &propositions),
        "replayed Cartesian concrete plan does not satisfy the full goal"
    );
    ensure!(
        approximately_equal(cost, plan.cost),
        "replayed Cartesian concrete plan cost {cost} differs from recorded cost {}",
        plan.cost
    );
    Ok(())
}
