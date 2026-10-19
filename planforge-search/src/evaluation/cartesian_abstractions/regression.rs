//! Regression flaw search for Cartesian abstractions; see
//! [`crate::evaluation::cegar::regression`] for the flaw definition.

use super::*;
use crate::evaluation::cegar::regression::{
    FailedConditions, failed_conditions_on_relaxed_suffix, regression_candidates,
    writes_propositional_variable,
};

/// Regression flaw search: the failed conditions of the optimal abstract
/// plan's relaxed suffix, regressed to the latest abstract state of the
/// concrete prefix whose split decides them. The pair `(k, k)` -- the
/// conditions `s_k` itself violates -- is the progression flaw, so the search
/// ends there when no later condition regresses to a split.
#[allow(clippy::too_many_arguments)]
pub(super) fn find_regression_flaw(
    working: &WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    shortest_paths: &ShortestPaths,
    state_packer: &Arc<StatePacker>,
    axiom_evaluator: &AxiomEvaluator<'_>,
    refinement_root: &ConcreteState,
    selected_plan: Option<&[TransitionKey]>,
) -> Result<PlanCheck> {
    let mut prefix = Vec::new();
    let failed = match trace_abstract_plan(
        working,
        semantics,
        shortest_paths,
        state_packer,
        axiom_evaluator,
        refinement_root,
        selected_plan,
        Some(&mut prefix),
    )? {
        Trace::ConcretePlan(plan) => return Ok(PlanCheck::ConcretePlan(plan)),
        Trace::AbstractDeadEnd(state_id) => return Ok(PlanCheck::AbstractDeadEnd(state_id)),
        Trace::Failed(failed) => failed,
    };
    let last_anchor = failed.operator_ids.len();
    ensure!(
        prefix.len() == last_anchor,
        "a trace failing at step {last_anchor} has {} prefix states",
        prefix.len()
    );
    let mut plan_operator_ids = failed.operator_ids.clone();
    plan_operator_ids.extend(abstract_plan_suffix(
        working,
        shortest_paths,
        selected_plan,
        &failed,
    )?);
    let failed_conditions = failed_conditions_on_relaxed_suffix(
        semantics.task(),
        axiom_evaluator,
        state_packer,
        failed.last.propositions.clone(),
        failed.last.numeric.clone(),
        last_anchor,
        &plan_operator_ids[last_anchor..],
    )?;
    for (anchor, conditions) in regression_candidates(&failed_conditions, last_anchor) {
        if anchor == last_anchor && conditions.step == last_anchor {
            break;
        }
        let anchor_state = if anchor == last_anchor {
            &failed.last
        } else {
            &prefix[anchor]
        };
        if let Some(split) = regression_split(
            working,
            semantics,
            anchor,
            anchor_state,
            conditions,
            &plan_operator_ids[anchor..conditions.step],
        )? {
            return Ok(PlanCheck::Refine(split));
        }
    }
    Ok(PlanCheck::Refine(progression_split(
        working, semantics, &failed,
    )?))
}

/// The operators `o_{k+1}, ..., o_n` of the abstract plan after a trace's
/// failing state `s_k`.
fn abstract_plan_suffix(
    working: &WorkingAbstraction,
    shortest_paths: &ShortestPaths,
    selected_plan: Option<&[TransitionKey]>,
    failed: &FailedTrace<'_>,
) -> Result<Vec<OperatorIndex>> {
    let transition = match &failed.flaw {
        TraceFlaw::Goal(_) => return Ok(Vec::new()),
        TraceFlaw::Precondition { transition, .. } | TraceFlaw::Deviation { transition, .. } => {
            *transition
        }
    };
    if let Some(plan) = selected_plan {
        let step = failed.operator_ids.len();
        ensure!(
            plan.get(step) == Some(&transition),
            "selected Cartesian plan does not take the failing transition at step {step}"
        );
        return Ok(plan[step..]
            .iter()
            .map(|transition| transition.concrete_op_id)
            .collect());
    }
    let mut suffix = vec![transition.concrete_op_id];
    let mut state = transition.target;
    while !shortest_paths.is_goal(state) {
        ensure!(
            suffix.len() <= working.states().len(),
            "Cartesian generating transitions contain a cycle"
        );
        let next = shortest_paths.generating_transition(state).context(
            "non-goal Cartesian state with finite distance has no generating transition",
        )?;
        suffix.push(next.concrete_op_id);
        state = next.target;
    }
    Ok(suffix)
}

/// The split of the anchor's abstract state that decides `conditions`,
/// regressed through `ops`, as violated, or `None` when no split of that
/// state does.
fn regression_split(
    working: &WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    anchor: usize,
    anchor_state: &TraceState,
    conditions: &FailedConditions,
    ops: &[OperatorIndex],
) -> Result<Option<Split>> {
    let description = format!(
        "conditions {:?} of step {} regressed to step {anchor}",
        conditions.facts, conditions.step
    );
    let candidates = match semantics.flaw_candidate_generation() {
        CartesianFlawCandidateGeneration::General => general_regression_splits(
            working,
            semantics,
            anchor_state,
            conditions,
            ops,
            &description,
        )?,
        CartesianFlawCandidateGeneration::DesiredRegion => desired_region_regression_splits(
            working,
            semantics,
            anchor_state,
            conditions,
            ops,
            &description,
        )?,
    };
    if candidates.is_empty() {
        return Ok(None);
    }
    select_refinement_split(working, semantics, candidates, 0x5245_4752).map(Some)
}

/// One split per regressed condition that a split of the anchor's abstract
/// state can decide. A propositional condition on a variable that `ops`
/// writes regresses to a constant, which no split decides.
fn general_regression_splits(
    working: &WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    anchor_state: &TraceState,
    conditions: &FailedConditions,
    ops: &[OperatorIndex],
    description: &str,
) -> Result<Vec<Split>> {
    let state_id = anchor_state.abstract_state;
    let mut splits = Vec::new();
    for fact in &conditions.facts {
        if let Some(tree_id) = semantics
            .task()
            .numeric_conditions()
            .id_for_var(fact.var_index())
        {
            splits.extend(regressed_comparison_refinement(
                working,
                semantics,
                state_id,
                tree_id,
                &anchor_state.numeric,
                ops,
                fact.value_index(),
                description,
            )?);
            continue;
        }
        if writes_propositional_variable(semantics.task(), ops, fact.var_index()) {
            continue;
        }
        let witness_value = anchor_state.prop_values[fact.var()];
        ensure!(
            witness_value != fact.value_index(),
            "regressed fact {fact:?} holds in the anchor state although no operator writes it"
        );
        if working.states()[state_id].propositions()[fact.var()]
            .binary_search(&fact.value_index())
            .is_ok()
        {
            splits.push(Split::Propositional {
                state_id,
                var_id: fact.var_index(),
                wanted: vec![fact.value_index()],
                witness_value,
                description: description.to_string(),
            });
        }
    }
    Ok(splits)
}

/// The splits separating the anchor state from the desired region of
/// `conditions`, regressed through `ops`; empty when the regressed region is
/// empty or disjoint from the anchor's abstract state.
fn desired_region_regression_splits(
    working: &WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    anchor_state: &TraceState,
    conditions: &FailedConditions,
    ops: &[OperatorIndex],
    description: &str,
) -> Result<Vec<Split>> {
    let mut desired = semantics.trivial_region()?;
    for fact in &conditions.facts {
        let is_comparison = semantics
            .task()
            .numeric_conditions()
            .is_condition_var(fact.var_index());
        if !is_comparison && writes_propositional_variable(semantics.task(), ops, fact.var_index())
        {
            // The last write decides the fact: it fails on the relaxed
            // suffix, so the write violates it for every anchor.
            return Ok(Vec::new());
        }
        constrain_desired_region(semantics, &mut desired, fact)
            .with_context(|| format!("failed to construct desired region for {description}"))?;
    }
    let mut numeric = desired.numeric.to_vec();
    for &op_id in ops.iter().rev() {
        for (var_id, interval) in numeric.iter_mut().enumerate() {
            match semantics.numeric_effect_preimage(
                *interval,
                op_id,
                VariableIndex::from_usize(var_id),
            )? {
                Some(preimage) if !preimage.is_empty() => *interval = preimage,
                _ => return Ok(Vec::new()),
            }
        }
    }
    let state = &working.states()[anchor_state.abstract_state];
    let overlaps = state
        .numeric
        .iter()
        .zip(&numeric)
        .all(|(current, regressed)| current.intersects(regressed))
        && state
            .propositions()
            .iter()
            .zip(desired.propositions())
            .all(|(current, wanted)| sorted_value_sets_overlap(current, wanted));
    if !overlaps {
        return Ok(Vec::new());
    }
    Arc::make_mut(&mut desired.numeric).copy_from_slice(&numeric);
    splits_for_desired_region(
        working,
        semantics,
        anchor_state.abstract_state,
        &desired,
        &anchor_state.prop_values,
        &anchor_state.numeric,
        description,
    )
}
