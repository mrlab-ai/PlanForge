#[cfg(test)]
mod tests;

use std::collections::BTreeSet;

use anyhow::{Result, ensure};
use planners_sas::numeric::{
    axioms::AxiomEvaluator,
    numeric_task::{AbstractNumericTask, ExplicitFact, Operator},
    utils::int_packer::IntDoublePacker,
};

use super::target_centered::{
    dependent_numeric_flaws_backward, numeric_effect_deltas, preimage_split_for_expected_successor,
};
use super::{
    Flaw, NumericFlaw, PropFlaw, SplitDirection, can_split_numeric_var,
    dependent_numeric_flaws_for_comparison_prop_var, state::progress,
};
use crate::numeric::evaluation::domain_abstractions::{
    domain_abstraction::{ComparisonAxiomIndex, NumericPartitions},
    domain_abstraction_factory::WildcardPlanResult,
    utils::{fact_is_hold, get_initial_state, make_prop_state_packer, partition_for_value},
};

/// Walk the wildcard plan and emit flaws using the chosen split direction.
///
/// `direction` selects how the *value* of each numeric flaw is chosen:
/// [`SplitDirection::Forward`] keeps the numeric-FD progression behavior:
/// direct deviation flaws split at the current concrete value with the side
/// determined from the operator delta. [`SplitDirection::Backward`] places
/// splits at boundaries derived from the regressed-target / required interval.
#[allow(unused_assignments)]
pub fn get_progression_flaws(
    task: &dyn AbstractNumericTask,
    partitions: &NumericPartitions,
    wildcard_plan: &WildcardPlanResult,
    direction: SplitDirection,
) -> Result<Vec<Flaw>> {
    let comparison_index = ComparisonAxiomIndex::from_task(task)
        .map_err(|e| anyhow::anyhow!("failed to build ComparisonAxiomIndex: {e}"))?;

    let state_packer = make_prop_state_packer(task);
    let axiom_evaluator = AxiomEvaluator::new(task, &state_packer);

    // `target_centered_shell_flaws` (in the `Backward` direction) needs the
    // per-numeric-var stack of operator effect deltas. The legacy code
    // recomputed the entire `task.get_operators() × assignment_effects` scan
    // on every call — 36% of total CPU on minecraft. Compute once here and
    // thread through the flaw helpers below.
    let deltas = numeric_effect_deltas(task);

    let (mut prop_state, mut numeric_state) =
        get_initial_state(task, &state_packer, &axiom_evaluator)?;
    let mut next_prop_state = None;
    let mut next_numeric_state = None;

    let mut collected_flaws: Vec<Flaw> = Vec::new();
    let mut flawed = false;
    let mut step: usize = 1;

    for equivalent_ops in wildcard_plan.wildcard_plan.iter() {
        let expected_abs_numeric_state = &wildcard_plan.abstract_numeric_states[step];
        ensure!(
            step < wildcard_plan.abstract_numeric_states.len(),
            "WildcardPlanResult abstract_numeric_states too short for step {step}"
        );

        for &op_id in equivalent_ops.iter() {
            let Some(op) = task.get_operators().get(op_id) else {
                continue;
            };
            let operator_flaws = get_progression_precondition_flaws(
                task,
                &deltas,
                partitions,
                &comparison_index,
                op,
                &state_packer,
                &prop_state,
                &numeric_state,
                step,
                direction,
            );
            if operator_flaws.is_empty() {
                (next_prop_state, next_numeric_state, flawed) = progress_and_get_deviation_flaws(
                    &prop_state,
                    &numeric_state,
                    expected_abs_numeric_state,
                    &state_packer,
                    &axiom_evaluator,
                    op,
                    partitions,
                    &deltas,
                    &mut collected_flaws,
                    step,
                    direction,
                )?;
                if !flawed {
                    collected_flaws.clear();
                    prop_state = next_prop_state.take().unwrap();
                    numeric_state = next_numeric_state.take().unwrap();
                    break;
                }
            } else {
                collected_flaws.extend(operator_flaws);
            }
        }

        if !collected_flaws.is_empty() {
            break;
        }

        step += 1;
    }

    if !collected_flaws.is_empty() {
        return Ok(collected_flaws);
    }

    let goal_flaws = get_goal_flaws(
        task,
        &deltas,
        partitions,
        &comparison_index,
        &state_packer,
        &prop_state,
        &numeric_state,
        step,
        direction,
    );

    Ok(goal_flaws)
}

#[allow(unused_assignments)]
pub fn get_execute_entire_plan_flaws(
    task: &dyn AbstractNumericTask,
    partitions: &NumericPartitions,
    wildcard_plan: &WildcardPlanResult,
    direction: SplitDirection,
) -> Result<Vec<Flaw>> {
    let comparison_index = ComparisonAxiomIndex::from_task(task)
        .map_err(|e| anyhow::anyhow!("failed to build ComparisonAxiomIndex: {e}"))?;

    let state_packer = make_prop_state_packer(task);
    let axiom_evaluator = AxiomEvaluator::new(task, &state_packer);
    let deltas = numeric_effect_deltas(task);

    let (mut prop_state, mut numeric_state) =
        get_initial_state(task, &state_packer, &axiom_evaluator)?;
    let mut next_prop_state = None;
    let mut next_numeric_state = None;

    let mut collected_flaws: Vec<Flaw> = Vec::new();
    let mut step: usize = 1;

    for equivalent_ops in wildcard_plan.wildcard_plan.iter() {
        let expected_abs_numeric_state = &wildcard_plan.abstract_numeric_states[step];
        ensure!(
            step < wildcard_plan.abstract_numeric_states.len(),
            "WildcardPlanResult abstract_numeric_states too short for step {step}"
        );

        let mut step_flaws = Vec::new();
        let mut chosen_op: Option<&Operator> = None;
        let mut fallback_op: Option<&Operator> = None;
        for &op_id in equivalent_ops.iter() {
            let Some(op) = task.get_operators().get(op_id) else {
                continue;
            };
            if fallback_op.is_none() {
                fallback_op = Some(op);
            }

            let operator_flaws = get_progression_precondition_flaws(
                task,
                &deltas,
                partitions,
                &comparison_index,
                op,
                &state_packer,
                &prop_state,
                &numeric_state,
                step,
                direction,
            );
            if operator_flaws.is_empty() {
                chosen_op = Some(op);
                step_flaws.clear();
                break;
            }
            step_flaws.extend(operator_flaws);
        }

        collected_flaws.extend(step_flaws);

        if let Some(op) = chosen_op.or(fallback_op) {
            (next_prop_state, next_numeric_state, _) = progress_and_get_deviation_flaws(
                &prop_state,
                &numeric_state,
                expected_abs_numeric_state,
                &state_packer,
                &axiom_evaluator,
                op,
                partitions,
                &deltas,
                &mut collected_flaws,
                step,
                direction,
            )?;
            prop_state = next_prop_state.take().unwrap();
            numeric_state = next_numeric_state.take().unwrap();
        }

        step += 1;
    }

    collected_flaws.extend(get_goal_flaws(
        task,
        &deltas,
        partitions,
        &comparison_index,
        &state_packer,
        &prop_state,
        &numeric_state,
        step,
        direction,
    ));

    Ok(collected_flaws)
}

type OptionalPropAndNumStateAndFlawed = (Option<Vec<u64>>, Option<Vec<f64>>, bool);
#[allow(clippy::too_many_arguments)]
pub(crate) fn progress_and_get_deviation_flaws(
    prop_state: &[u64],
    numeric_state: &[f64],
    expected_abs_numeric_state: &[usize],
    state_packer: &IntDoublePacker,
    axiom_evaluator: &AxiomEvaluator<'_>,
    op: &Operator,
    partitions: &NumericPartitions,
    _deltas: &std::collections::HashMap<usize, Vec<f64>>,
    collected_flaws: &mut Vec<Flaw>,
    step: usize,
    direction: SplitDirection,
) -> Result<OptionalPropAndNumStateAndFlawed> {
    let mut next_prop_state = prop_state.to_vec();
    let mut next_numeric_state = numeric_state.to_vec();
    let mut flawed = false;
    progress(
        op,
        axiom_evaluator,
        state_packer,
        &mut next_prop_state,
        &mut next_numeric_state,
    )?;

    let deviation_flaws = get_progression_numeric_deviation_flaws(
        op,
        numeric_state,
        &next_numeric_state,
        expected_abs_numeric_state,
        partitions,
        step,
        direction,
    );
    if !deviation_flaws.is_empty() {
        collected_flaws.extend(deviation_flaws);
        flawed = true;
    }

    Ok((Some(next_prop_state), Some(next_numeric_state), flawed))
}

/// Emit numeric deviation flaws for an operator whose abstract successor
/// differs from the concrete one.
///
/// In `Forward` direction the flaw is split at the *concrete current* value
/// using direction-of-change to pick `include_in_lower`. In `Backward`
/// direction the flaw is split at the boundary of the regressed target
/// interval — the split that separates the cell containing the regressed
/// preimage support from the rest of the source cell.
pub fn get_progression_numeric_deviation_flaws(
    op: &Operator,
    numeric_current_state: &[f64],
    numeric_successor_state: &[f64],
    abstract_numeric_successor_state: &[usize],
    partitions: &NumericPartitions,
    step: usize,
    direction: SplitDirection,
) -> Vec<Flaw> {
    let mut flaws: Vec<Flaw> = Vec::new();

    let num_vars = numeric_successor_state
        .len()
        .min(abstract_numeric_successor_state.len());
    for var_id in 0..num_vars {
        // Forward direction only emits a flaw if the operator actually
        // modifies this variable. Backward inspects every variable whose
        // concrete successor disagrees with the abstract one — this matches
        // the legacy target-centered behavior, which also covered effects
        // routed through derived/axiom variables.
        if matches!(
            direction,
            SplitDirection::Forward | SplitDirection::ForwardPartitionDeviation
        ) {
            let operator_modified_var = op
                .assignment_effects()
                .iter()
                .any(|eff| eff.affected_var_id() == var_id);
            if !operator_modified_var {
                continue;
            }
        }

        let abstract_value = abstract_numeric_successor_state[var_id];
        let Some(parts) = partitions.partitions(var_id) else {
            continue;
        };
        let Some(correct_abstract_value) =
            partition_for_value(parts, numeric_successor_state[var_id])
        else {
            continue;
        };
        if std::env::var_os("DA_DEV_PROBE").is_some() && (var_id == 24 || var_id == 25) {
            tracing::info!(
                "DEV_PROBE step={step} var={var_id} cur={} succ={} abs={} correct={} parts.len={}",
                numeric_current_state.get(var_id).copied().unwrap_or(f64::NAN),
                numeric_successor_state[var_id],
                abstract_value,
                correct_abstract_value,
                parts.len()
            );
        }
        if abstract_value == correct_abstract_value {
            continue;
        }

        let concrete_next_value = numeric_successor_state[var_id];
        let concrete_current_value = numeric_current_state
            .get(var_id)
            .copied()
            .unwrap_or(concrete_next_value);
        if concrete_next_value == concrete_current_value {
            continue;
        }

        match direction {
            SplitDirection::Forward | SplitDirection::ForwardPartitionDeviation => {
                let mut include_in_lower = if direction == SplitDirection::Forward {
                    let operator_increased_value = concrete_next_value > concrete_current_value;
                    !operator_increased_value
                } else {
                    abstract_value > correct_abstract_value
                };

                if can_split_numeric_var(
                    partitions,
                    var_id,
                    concrete_current_value,
                    include_in_lower,
                ) {
                    flaws.push(Flaw::Numeric(NumericFlaw {
                        numeric_var_id: var_id,
                        value: concrete_current_value,
                        include_in_lower,
                        step,
                    }));
                } else {
                    // The principal side is on an existing partition boundary
                    // and cannot be split (would yield an empty cell). Try
                    // the opposite side as a fallback. If neither side
                    // produces a valid split, the flaw is permanently
                    // unresolvable at this point — emit nothing so the same
                    // flaw cannot recur infinitely in CEGAR's loop.
                    include_in_lower = !include_in_lower;
                    if can_split_numeric_var(
                        partitions,
                        var_id,
                        concrete_current_value,
                        include_in_lower,
                    ) {
                        flaws.push(Flaw::Numeric(NumericFlaw {
                            numeric_var_id: var_id,
                            value: concrete_current_value,
                            include_in_lower,
                            step,
                        }));
                    }
                }
            }
            SplitDirection::Backward => {
                let Some(expected_interval) = parts.get(abstract_value).copied() else {
                    continue;
                };
                let delta = concrete_next_value - concrete_current_value;
                let Some((value, include_in_lower)) = preimage_split_for_expected_successor(
                    expected_interval,
                    concrete_next_value,
                    delta,
                ) else {
                    continue;
                };
                if can_split_numeric_var(partitions, var_id, value, include_in_lower) {
                    flaws.push(Flaw::Numeric(NumericFlaw {
                        numeric_var_id: var_id,
                        value,
                        include_in_lower,
                        step,
                    }));
                }
            }
        }
    }

    flaws
}

#[allow(clippy::too_many_arguments)]
pub fn get_progression_precondition_flaws(
    task: &dyn AbstractNumericTask,
    deltas: &std::collections::HashMap<usize, Vec<f64>>,
    partitions: &NumericPartitions,
    comparison_index: &ComparisonAxiomIndex,
    op: &Operator,
    packer: &IntDoublePacker,
    buffer: &[u64],
    numeric_state: &[f64],
    step: usize,
    direction: SplitDirection,
) -> Vec<Flaw> {
    let mut out: Vec<Flaw> = Vec::new();
    for pre in op.preconditions().iter() {
        if !fact_is_hold(pre, packer, buffer) {
            out.push(build_prop_flaw_for_fact(
                task,
                deltas,
                partitions,
                comparison_index,
                pre,
                numeric_state,
                step,
                direction,
            ));
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
pub fn get_goal_flaws(
    task: &dyn AbstractNumericTask,
    deltas: &std::collections::HashMap<usize, Vec<f64>>,
    partitions: &NumericPartitions,
    comparison_index: &ComparisonAxiomIndex,
    packer: &IntDoublePacker,
    buffer: &[u64],
    numeric_state: &[f64],
    step: usize,
    direction: SplitDirection,
) -> Vec<Flaw> {
    let num_goals = task.get_num_goals();
    let mut out: Vec<Flaw> = Vec::new();
    let mut seen: BTreeSet<ExplicitFact> = BTreeSet::new();
    let mut derived_goal_vars: BTreeSet<usize> = BTreeSet::new();
    for goal_id in 0..num_goals {
        let goal_fact = task.get_goal_fact(goal_id);
        let goal_var = goal_fact.var;
        let goal_is_derived = task.axioms().iter().any(|ax| ax.var_id() == goal_var);
        if goal_is_derived {
            derived_goal_vars.insert(goal_var);
            continue;
        }
        if !fact_is_hold(goal_fact, packer, buffer) && seen.insert(goal_fact.clone()) {
            out.push(build_prop_flaw_for_fact(
                task,
                deltas,
                partitions,
                comparison_index,
                goal_fact,
                numeric_state,
                step,
                direction,
            ));
        }
    }

    // Reconstruct (potentially hidden) goal conditions from propositional goal axioms.
    for ax in task.axioms().iter() {
        if ax.conditions().is_empty() {
            continue;
        }
        if !derived_goal_vars.is_empty() && !derived_goal_vars.contains(&ax.var_id()) {
            continue;
        }
        for pre in ax.conditions().iter() {
            if !fact_is_hold(pre, packer, buffer) && seen.insert(pre.clone()) {
                out.push(build_prop_flaw_for_fact(
                    task,
                    deltas,
                    partitions,
                    comparison_index,
                    pre,
                    numeric_state,
                    step,
                    direction,
                ));
            }
        }
    }
    out
}

/// Build a propositional flaw for `fact`, attaching dependent numeric flaws
/// when `fact` references a comparison-axiom propositional variable. The
/// dependent flaws are computed forward (concrete-value split per variable)
/// or backward (boundary-aligned shell splits) according to `direction`.
fn build_prop_flaw_for_fact(
    task: &dyn AbstractNumericTask,
    deltas: &std::collections::HashMap<usize, Vec<f64>>,
    partitions: &NumericPartitions,
    comparison_index: &ComparisonAxiomIndex,
    fact: &ExplicitFact,
    numeric_state: &[f64],
    step: usize,
    direction: SplitDirection,
) -> Flaw {
    let dependent_numeric_flaws = if comparison_index.is_comparison_axiom_variable(fact.var) {
        match direction {
            SplitDirection::Forward | SplitDirection::ForwardPartitionDeviation => {
                dependent_numeric_flaws_for_comparison_prop_var(
                    task,
                    partitions,
                    comparison_index,
                    fact.var,
                    numeric_state,
                    step,
                )
            }
            SplitDirection::Backward => dependent_numeric_flaws_backward(
                task,
                deltas,
                partitions,
                comparison_index,
                fact,
                numeric_state,
                step,
            ),
        }
    } else {
        Vec::new()
    };
    Flaw::Propositional(PropFlaw {
        fact: fact.clone(),
        dependent_numeric_flaws,
        step,
    })
}
