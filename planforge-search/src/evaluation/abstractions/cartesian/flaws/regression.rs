//! Regression flaws for Cartesian abstractions; see
//! [`crate::evaluation::abstractions::common::cegar::regression`] for the definition.

use super::super::*;
use super::*;
use crate::evaluation::abstractions::common::cegar::regression::{
    Cut, InitialCut, InitialState, find_regression_flaw as find_trace_flaw,
};
use planforge_sas::numeric_task::ONE_VALUE;

/// Regression flaw search on the optimal abstract plan from the abstract
/// state of `refinement_root`. Without a flaw, the plan's operators are a
/// concrete plan.
pub(in super::super) fn find_regression_flaw(
    working: &WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    shortest_paths: &ShortestPaths,
    state_packer: &Arc<StatePacker>,
    refinement_root: &ConcreteState,
    selected_plan: Option<&[TransitionKey]>,
) -> Result<PlanCheck> {
    let mut prop_values = Vec::new();
    semantics.concrete_prop_values(
        state_packer,
        &refinement_root.propositions,
        &mut prop_values,
    );
    let initial_state = working
        .hierarchy()
        .map_state(&prop_values, &refinement_root.numeric)?;
    if !shortest_paths.distance(initial_state).is_finite() {
        return Ok(PlanCheck::AbstractDeadEnd(initial_state));
    }
    let plan = abstract_plan(working, shortest_paths, initial_state, selected_plan)?;
    let state_ids = std::iter::once(initial_state)
        .chain(plan.iter().map(|transition| transition.target))
        .collect::<Vec<_>>();
    let states = state_ids
        .iter()
        .map(|&state_id| working.states()[state_id].clone())
        .collect::<Vec<_>>();
    let operator_ids = plan
        .iter()
        .map(|transition| transition.concrete_op_id)
        .collect::<Vec<_>>();
    let initial_cut = match semantics.flaw_candidate_generation() {
        CartesianFlawCandidateGeneration::General => InitialCut::Concrete,
        CartesianFlawCandidateGeneration::DesiredRegion => InitialCut::Wanted,
    };
    let Some(flaw) = find_trace_flaw(
        semantics.task(),
        semantics.coordinates(),
        &states,
        &operator_ids,
        InitialState {
            propositions: &prop_values,
            numeric: &refinement_root.numeric,
        },
        initial_cut,
    )?
    else {
        let cost = operator_ids
            .iter()
            .map(|op_id| semantics.operator_costs()[op_id.index()].value())
            .sum();
        return Ok(PlanCheck::ConcretePlan(ConcretePlan::new(
            operator_ids,
            cost,
        )));
    };
    let state_id = state_ids[flaw.state];
    let description = format!(
        "regression flaw in abstract state {} of the plan",
        flaw.state
    );
    Ok(PlanCheck::Refine(match flaw.cut {
        Cut::Numeric {
            var,
            boundary,
            lower_includes_boundary,
            wanted_is_lower,
        } => {
            let integer_lattice = semantics.numeric_integer_lattice()[var.index()];
            let (lower, upper) = numeric_split_intervals(
                working.states()[state_id].numeric[var.index()],
                boundary,
                lower_includes_boundary,
                integer_lattice,
            )?;
            Split::Numeric {
                state_id,
                var_id: var,
                boundary,
                lower_includes_boundary,
                witness_value: interior_value(if wanted_is_lower { upper } else { lower }),
                desired_contains_witness: false,
                integer_lattice,
                description,
            }
        }
        Cut::Propositional { var, wanted } => {
            let witness_value = *working.states()[state_id].propositions()[var.index()]
                .iter()
                .find(|&&value| value != wanted)
                .with_context(|| {
                    format!("propositional cut of {var} at {wanted} in abstract state {state_id} is not strict")
                })?;
            Split::Propositional {
                state_id,
                var_id: var,
                wanted: vec![wanted],
                witness_value,
                description,
            }
        }
    }))
}

/// The transitions of the optimal abstract plan from `initial_state`: the
/// selected plan, or the generating transitions of the shortest paths.
fn abstract_plan(
    working: &WorkingAbstraction,
    shortest_paths: &ShortestPaths,
    initial_state: usize,
    selected_plan: Option<&[TransitionKey]>,
) -> Result<Vec<TransitionKey>> {
    if let Some(plan) = selected_plan {
        return Ok(plan.to_vec());
    }
    let mut plan = Vec::new();
    let mut state = initial_state;
    while !shortest_paths.is_goal(state) {
        ensure!(
            plan.len() <= working.states().len(),
            "Cartesian generating transitions contain a cycle"
        );
        let transition = shortest_paths.generating_transition(state).context(
            "non-goal Cartesian state with finite distance has no generating transition",
        )?;
        ensure!(
            working.contains_transition(transition),
            "Cartesian shortest path references missing transition {transition:?}"
        );
        plan.push(transition);
        state = transition.target;
    }
    Ok(plan)
}

/// A value of the non-empty interval `interval`.
fn interior_value(interval: Interval) -> NumericValue {
    match (interval.lower.is_finite(), interval.upper.is_finite()) {
        (true, _) if interval.lower_closed => interval.lower,
        (_, true) if interval.upper_closed => interval.upper,
        (true, true) => NumericValue::new((interval.lower.value() + interval.upper.value()) / 2.0),
        (true, false) => interval.lower + ONE_VALUE,
        (false, true) => interval.upper - ONE_VALUE,
        (false, false) => ZERO_VALUE,
    }
}
