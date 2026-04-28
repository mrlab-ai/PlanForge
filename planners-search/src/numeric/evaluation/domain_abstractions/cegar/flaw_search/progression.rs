#[cfg(test)]
mod tests;

use std::collections::BTreeSet;

use anyhow::{Result, ensure};
use planners_sas::numeric::{
    axioms::AxiomEvaluator,
    numeric_task::{AbstractNumericTask, ExplicitFact, Operator},
    utils::int_packer::IntDoublePacker,
};

use super::{
    Flaw, NumericFlaw, PropFlaw, can_split_numeric_var,
    dependent_numeric_flaws_for_comparison_prop_var, state::progress,
};
use crate::numeric::evaluation::domain_abstractions::{
    domain_abstraction::{ComparisonAxiomIndex, NumericPartitions},
    domain_abstraction_factory::WildcardPlanResult,
    utils::{fact_is_hold, get_initial_state, make_prop_state_packer, partition_for_value},
};

#[allow(unused_assignments)]
pub fn get_progression_flaws(
    task: &dyn AbstractNumericTask,
    partitions: &NumericPartitions,
    wildcard_plan: &WildcardPlanResult,
    execute_entire_plan: bool,
) -> Result<Vec<Flaw>> {
    let comparison_index = ComparisonAxiomIndex::from_task(task)
        .map_err(|e| anyhow::anyhow!("failed to build ComparisonAxiomIndex: {e}"))?;

    let state_packer = make_prop_state_packer(task);
    let axiom_evaluator = AxiomEvaluator::new(task, &state_packer);

    let (mut prop_state, mut numeric_state) =
        get_initial_state(task, &state_packer, &axiom_evaluator)?;
    let mut next_prop_state = None;
    let mut next_numeric_state = None;

    let mut collected_flaws: Vec<Flaw> = Vec::new();
    let mut flawed = false;
    let mut step_num: usize = 1;

    for equivalent_ops in wildcard_plan.wildcard_plan.iter() {
        let expected_abs_numeric_state = &wildcard_plan.abstract_numeric_states[step_num];
        ensure!(
            step_num < wildcard_plan.abstract_numeric_states.len(),
            "WildcardPlanResult abstract_numeric_states too short for step {step_num}"
        );

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
                partitions,
                &comparison_index,
                op,
                &state_packer,
                &prop_state,
                &numeric_state,
            );
            if operator_flaws.is_empty() {
                chosen_op = Some(op);
                (next_prop_state, next_numeric_state, flawed) = progress_and_get_deviation_flaws(
                    &prop_state,
                    &numeric_state,
                    expected_abs_numeric_state,
                    &state_packer,
                    &axiom_evaluator,
                    op,
                    partitions,
                    &mut collected_flaws,
                )?;
                if !flawed {
                    collected_flaws.clear();
                    if !execute_entire_plan {
                        prop_state = next_prop_state.take().unwrap();
                        numeric_state = next_numeric_state.take().unwrap();
                    }
                    break;
                }
            } else {
                collected_flaws.extend(operator_flaws);
            }
        }

        if execute_entire_plan {
            // Progress and find flaws in the fallback operator only if it has
            // not been done in any other operator.
            if let Some(op) = fallback_op
                && chosen_op.is_none()
            {
                (next_prop_state, next_numeric_state, flawed) = progress_and_get_deviation_flaws(
                    &prop_state,
                    &numeric_state,
                    expected_abs_numeric_state,
                    &state_packer,
                    &axiom_evaluator,
                    op,
                    partitions,
                    &mut collected_flaws,
                )?;
            }

            let Some(next_prop) = next_prop_state.take() else {
                break;
            };
            let Some(next_numeric) = next_numeric_state.take() else {
                break;
            };
            prop_state = next_prop;
            numeric_state = next_numeric;
        } else if !collected_flaws.is_empty() {
            break;
        }

        step_num += 1;
    }

    if !execute_entire_plan && !collected_flaws.is_empty() {
        return Ok(collected_flaws);
    }

    let goal_flaws = get_goal_flaws(
        task,
        partitions,
        &comparison_index,
        &state_packer,
        &prop_state,
        &numeric_state,
    );
    if execute_entire_plan {
        collected_flaws.extend(goal_flaws);
        Ok(collected_flaws)
    } else {
        Ok(goal_flaws)
    }
}

type OptionalPropAndNumStateAndFlawed = (Option<Vec<u64>>, Option<Vec<f64>>, bool);
#[allow(clippy::too_many_arguments)]
fn progress_and_get_deviation_flaws(
    prop_state: &[u64],
    numeric_state: &[f64],
    expected_abs_numeric_state: &[usize],
    state_packer: &IntDoublePacker,
    axiom_evaluator: &AxiomEvaluator<'_>,
    op: &Operator,
    partitions: &NumericPartitions,
    collected_flaws: &mut Vec<Flaw>,
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
    );
    if !deviation_flaws.is_empty() {
        collected_flaws.extend(deviation_flaws);
        flawed = true;
    }

    Ok((Some(next_prop_state), Some(next_numeric_state), flawed))
}

pub fn get_progression_numeric_deviation_flaws(
    op: &Operator,
    numeric_current_state: &[f64],
    numeric_successor_state: &[f64],
    abstract_numeric_successor_state: &[usize],
    partitions: &NumericPartitions,
) -> Vec<Flaw> {
    let mut flaws: Vec<Flaw> = Vec::new();

    let num_vars = numeric_successor_state
        .len()
        .min(abstract_numeric_successor_state.len());
    for var_id in 0..num_vars {
        let operator_modified_var = op
            .assignment_effects()
            .iter()
            .any(|eff| eff.affected_var_id() == var_id);
        if !operator_modified_var {
            continue;
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

        let operator_increased_value = concrete_next_value > concrete_current_value;
        let mut include_in_lower = !operator_increased_value;

        if can_split_numeric_var(partitions, var_id, concrete_current_value, include_in_lower) {
            flaws.push(Flaw::Numeric(NumericFlaw {
                numeric_var_id: var_id,
                value: concrete_current_value,
                include_in_lower,
            }));
        } else {
            include_in_lower = !include_in_lower;
            if can_split_numeric_var(partitions, var_id, concrete_current_value, include_in_lower) {
                flaws.push(Flaw::Numeric(NumericFlaw {
                    numeric_var_id: var_id,
                    value: concrete_current_value,
                    include_in_lower,
                }));
            }
        }
    }

    flaws
}

pub fn get_progression_precondition_flaws(
    task: &dyn AbstractNumericTask,
    partitions: &NumericPartitions,
    comparison_index: &ComparisonAxiomIndex,
    op: &Operator,
    packer: &IntDoublePacker,
    buffer: &[u64],
    numeric_state: &[f64],
) -> Vec<Flaw> {
    let mut out: Vec<Flaw> = Vec::new();
    for pre in op.preconditions().iter() {
        if !fact_is_hold(pre, packer, buffer) {
            let prop_var_id = pre.var;
            let dependent_numeric_flaws =
                if comparison_index.is_comparison_axiom_variable(prop_var_id) {
                    dependent_numeric_flaws_for_comparison_prop_var(
                        task,
                        partitions,
                        comparison_index,
                        prop_var_id,
                        numeric_state,
                    )
                } else {
                    vec![]
                };
            out.push(Flaw::Propositional(PropFlaw {
                fact: pre.clone(),
                dependent_numeric_flaws,
            }));
        }
    }
    out
}

fn get_goal_flaws(
    task: &dyn AbstractNumericTask,
    partitions: &NumericPartitions,
    comparison_index: &ComparisonAxiomIndex,
    packer: &IntDoublePacker,
    buffer: &[u64],
    numeric_state: &[f64],
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
            let prop_var_id = goal_fact.var;
            let dependent_numeric_flaws =
                if comparison_index.is_comparison_axiom_variable(prop_var_id) {
                    dependent_numeric_flaws_for_comparison_prop_var(
                        task,
                        partitions,
                        comparison_index,
                        prop_var_id,
                        numeric_state,
                    )
                } else {
                    vec![]
                };
            out.push(Flaw::Propositional(PropFlaw {
                fact: goal_fact.clone(),
                dependent_numeric_flaws,
            }));
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
                let prop_var_id = pre.var;
                let dependent_numeric_flaws =
                    if comparison_index.is_comparison_axiom_variable(prop_var_id) {
                        dependent_numeric_flaws_for_comparison_prop_var(
                            task,
                            partitions,
                            comparison_index,
                            prop_var_id,
                            numeric_state,
                        )
                    } else {
                        vec![]
                    };
                out.push(Flaw::Propositional(PropFlaw {
                    fact: pre.clone(),
                    dependent_numeric_flaws,
                }));
            }
        }
    }
    out
}
