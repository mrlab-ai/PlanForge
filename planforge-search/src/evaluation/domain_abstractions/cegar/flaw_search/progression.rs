#[cfg(test)]
mod tests;

use std::collections::HashMap;

use anyhow::{Result, ensure};
use planforge_sas::{
    axioms::AxiomEvaluator,
    numeric_task::{AbstractNumericTask, ExplicitFact, Operator},
    utils::int_packer::IntDoublePacker,
};

use super::target_centered::{
    dependent_numeric_flaws_backward, numeric_effect_deltas, preimage_split_for_expected_successor,
};
use super::{
    Flaw, NumericFlaw, PropFlaw, SplitDirection, can_split_numeric_var,
    dependent_numeric_flaws_for_comparison_prop_var, goal_requirements,
};
use crate::evaluation::domain_abstractions::{
    additive_numeric_views::numeric_dimension_delta_for_operator,
    domain_abstraction::NumericPartitions,
    domain_abstraction_factory::WildcardPlanResult,
    utils::{fact_is_hold, get_initial_state, make_prop_state_packer, partition_for_value},
};

/// The task a flaw is measured against: the task itself, the numeric partitions
/// that abstract it, and the per-numeric-variable operator effect deltas cached
/// from it. The deltas are only read in the `Backward` direction, but they are
/// cached rather than recomputed per flaw because the `operators x
/// assignment_effects` scan they need was 36% of total CPU on minecraft.
#[derive(Clone, Copy)]
pub struct PartitionedTask<'a> {
    pub task: &'a dyn AbstractNumericTask,
    pub partitions: &'a NumericPartitions,
    pub deltas: &'a HashMap<usize, Vec<f64>>,
}

/// One concrete state as the flaw walk carries it: the packed propositional half
/// with the packer that reads facts out of it, and the numeric values.
#[derive(Clone, Copy)]
pub struct ConcreteStateView<'a> {
    pub packer: &'a IntDoublePacker,
    pub prop: &'a [u64],
    pub numeric: &'a [f64],
}

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
    let state_packer = std::sync::Arc::new(make_prop_state_packer(task));
    let axiom_evaluator = AxiomEvaluator::new(std::sync::Arc::new(task), state_packer.clone());

    // `target_centered_shell_flaws` (in the `Backward` direction) needs the
    // per-numeric-var stack of operator effect deltas. The legacy code
    // recomputed the entire `task.get_operators() × assignment_effects` scan
    // on every call — 36% of total CPU on minecraft. Compute once here and
    // thread through the flaw helpers below.
    let deltas = numeric_effect_deltas(task);

    let partitioned = PartitionedTask {
        task,
        partitions,
        deltas: &deltas,
    };
    let (mut prop_state, mut numeric_state) =
        get_initial_state(task, &state_packer, &axiom_evaluator)?;

    let mut collected_flaws: Vec<Flaw> = Vec::new();
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
            let state = ConcreteStateView {
                packer: &state_packer,
                prop: &prop_state,
                numeric: &numeric_state,
            };
            let operator_flaws =
                get_progression_precondition_flaws(partitioned, op, state, step, direction);
            if !operator_flaws.is_empty() {
                collected_flaws.extend(operator_flaws);
                continue;
            }
            let (next_prop_state, next_numeric_state, deviation_flaws) =
                progress_and_get_deviation_flaws(
                    partitioned,
                    state,
                    expected_abs_numeric_state,
                    &axiom_evaluator,
                    op,
                    step,
                    direction,
                )?;
            if deviation_flaws.is_empty() {
                // This operator of the wildcard step is executable and lands
                // where the abstract plan said, so the flaws the alternatives
                // produced are not flaws of the plan.
                collected_flaws.clear();
                prop_state = next_prop_state;
                numeric_state = next_numeric_state;
                break;
            }
            collected_flaws.extend(deviation_flaws);
        }

        if !collected_flaws.is_empty() {
            break;
        }

        step += 1;
    }

    if !collected_flaws.is_empty() {
        return Ok(collected_flaws);
    }

    Ok(get_goal_flaws(
        partitioned,
        ConcreteStateView {
            packer: &state_packer,
            prop: &prop_state,
            numeric: &numeric_state,
        },
        step,
        direction,
    ))
}

pub fn get_execute_entire_plan_flaws(
    task: &dyn AbstractNumericTask,
    partitions: &NumericPartitions,
    wildcard_plan: &WildcardPlanResult,
    direction: SplitDirection,
) -> Result<Vec<Flaw>> {
    let state_packer = std::sync::Arc::new(make_prop_state_packer(task));
    let axiom_evaluator = AxiomEvaluator::new(std::sync::Arc::new(task), state_packer.clone());
    let deltas = numeric_effect_deltas(task);
    let partitioned = PartitionedTask {
        task,
        partitions,
        deltas: &deltas,
    };

    let (mut prop_state, mut numeric_state) =
        get_initial_state(task, &state_packer, &axiom_evaluator)?;

    let mut collected_flaws: Vec<Flaw> = Vec::new();
    let mut step: usize = 1;

    for equivalent_ops in wildcard_plan.wildcard_plan.iter() {
        let expected_abs_numeric_state = &wildcard_plan.abstract_numeric_states[step];
        ensure!(
            step < wildcard_plan.abstract_numeric_states.len(),
            "WildcardPlanResult abstract_numeric_states too short for step {step}"
        );

        let state = ConcreteStateView {
            packer: &state_packer,
            prop: &prop_state,
            numeric: &numeric_state,
        };
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

            let operator_flaws =
                get_progression_precondition_flaws(partitioned, op, state, step, direction);
            if operator_flaws.is_empty() {
                chosen_op = Some(op);
                step_flaws.clear();
                break;
            }
            step_flaws.extend(operator_flaws);
        }

        collected_flaws.extend(step_flaws);

        if let Some(op) = chosen_op.or(fallback_op) {
            let (next_prop_state, next_numeric_state, deviation_flaws) =
                progress_and_get_deviation_flaws(
                    partitioned,
                    state,
                    expected_abs_numeric_state,
                    &axiom_evaluator,
                    op,
                    step,
                    direction,
                )?;
            collected_flaws.extend(deviation_flaws);
            prop_state = next_prop_state;
            numeric_state = next_numeric_state;
        }

        step += 1;
    }

    collected_flaws.extend(get_goal_flaws(
        partitioned,
        ConcreteStateView {
            packer: &state_packer,
            prop: &prop_state,
            numeric: &numeric_state,
        },
        step,
        direction,
    ));

    Ok(collected_flaws)
}

/// The successor of one plan step -- packed propositional half, numeric values
/// -- and the deviation flaws that step exposed.
type ProgressedStateAndFlaws = (Vec<u64>, Vec<f64>, Vec<Flaw>);
/// Apply `op` to `state` and report the successor together with the numeric
/// deviation flaws it exposes. An empty flaw list means the concrete successor
/// lands in the partitions the abstract plan expected.
pub(crate) fn progress_and_get_deviation_flaws(
    partitioned: PartitionedTask<'_>,
    state: ConcreteStateView<'_>,
    expected_abs_numeric_state: &[usize],
    axiom_evaluator: &AxiomEvaluator<'_>,
    op: &Operator,
    step: usize,
    direction: SplitDirection,
) -> Result<ProgressedStateAndFlaws> {
    let mut next_prop_state = state.prop.to_vec();
    let mut next_numeric_state = state.numeric.to_vec();
    crate::evaluation::cegar::progress_concrete_state(
        op,
        axiom_evaluator,
        state.packer,
        &mut next_prop_state,
        &mut next_numeric_state,
    )?;

    let deviation_flaws = get_progression_numeric_deviation_flaws(
        partitioned.task,
        op,
        NumericTransitionStates {
            current: state.numeric,
            successor: &next_numeric_state,
            abstract_successor: expected_abs_numeric_state,
        },
        partitioned.partitions,
        step,
        direction,
    );

    Ok((next_prop_state, next_numeric_state, deviation_flaws))
}

/// The concrete and abstract numeric states either side of one plan step.
pub struct NumericTransitionStates<'a> {
    /// Concrete numeric values before the operator.
    pub current: &'a [f64],
    /// Concrete numeric values after the operator.
    pub successor: &'a [f64],
    /// Partition ids the abstract plan expects after the operator.
    pub abstract_successor: &'a [usize],
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
    task: &dyn AbstractNumericTask,
    op: &Operator,
    states: NumericTransitionStates<'_>,
    partitions: &NumericPartitions,
    step: usize,
    direction: SplitDirection,
) -> Vec<Flaw> {
    let NumericTransitionStates {
        current,
        successor,
        abstract_successor,
    } = states;
    let mut flaws: Vec<Flaw> = Vec::new();

    let num_vars = successor.len().min(abstract_successor.len());
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
                .any(|eff| eff.affected_var_id() == var_id)
                || numeric_dimension_delta_for_operator(task, var_id, op)
                    .is_some_and(|delta| delta.abs() >= 1e-12);
            if !operator_modified_var {
                continue;
            }
        }

        let abstract_value = abstract_successor[var_id];
        let Some(parts) = partitions.partitions(var_id) else {
            continue;
        };
        let Some(correct_abstract_value) = partition_for_value(parts, successor[var_id]) else {
            continue;
        };
        if abstract_value == correct_abstract_value {
            continue;
        }

        let concrete_next_value = successor[var_id];
        let concrete_current_value = current.get(var_id).copied().unwrap_or(concrete_next_value);
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

pub fn get_progression_precondition_flaws(
    partitioned: PartitionedTask<'_>,
    op: &Operator,
    state: ConcreteStateView<'_>,
    step: usize,
    direction: SplitDirection,
) -> Vec<Flaw> {
    let mut out: Vec<Flaw> = Vec::new();
    for pre in op.preconditions().iter() {
        if !fact_is_hold(pre, state.packer, state.prop) {
            out.push(build_prop_flaw_for_fact(
                partitioned,
                pre,
                state.numeric,
                step,
                direction,
            ));
        }
    }
    out
}

pub fn get_goal_flaws(
    partitioned: PartitionedTask<'_>,
    state: ConcreteStateView<'_>,
    step: usize,
    direction: SplitDirection,
) -> Vec<Flaw> {
    let mut out: Vec<Flaw> = Vec::new();
    for requirement in goal_requirements(partitioned.task) {
        if !fact_is_hold(&requirement, state.packer, state.prop) {
            out.push(build_prop_flaw_for_fact(
                partitioned,
                &requirement,
                state.numeric,
                step,
                direction,
            ));
        }
    }
    out
}

/// Build a propositional flaw for `fact`, attaching dependent numeric flaws
/// when `fact` references a comparison-axiom propositional variable. The
/// dependent flaws are computed forward (concrete-value split per variable)
/// or backward (boundary-aligned shell splits) according to `direction`.
fn build_prop_flaw_for_fact(
    partitioned: PartitionedTask<'_>,
    fact: &ExplicitFact,
    numeric_state: &[f64],
    step: usize,
    direction: SplitDirection,
) -> Flaw {
    let PartitionedTask {
        task,
        partitions,
        deltas,
    } = partitioned;
    let dependent_numeric_flaws = if task.numeric_conditions().is_condition_var(fact.var()) {
        match direction {
            SplitDirection::Forward | SplitDirection::ForwardPartitionDeviation => {
                dependent_numeric_flaws_for_comparison_prop_var(
                    task,
                    partitions,
                    fact.var(),
                    numeric_state,
                    step,
                )
            }
            SplitDirection::Backward => dependent_numeric_flaws_backward(
                task,
                deltas,
                partitions,
                fact,
                numeric_state,
                step,
            ),
        }
    } else {
        Vec::new()
    };
    Flaw::Propositional(PropFlaw {
        fact: *fact,
        dependent_numeric_flaws,
        step,
    })
}
