#[cfg(test)]
mod tests;

use std::collections::BTreeSet;

use anyhow::{Result, ensure};
use planners_sas::numeric::{
    axioms::AxiomEvaluator,
    numeric_task::{AbstractNumericTask, ExplicitFact, Operator},
};

use super::{Flaw, NumericFlaw, PropFlaw, can_split_numeric_var};
use crate::numeric::evaluation::domain_abstractions::{
    abstract_operator_generator::DomainMapping,
    cegar::flaw_search::{
        dependent_numeric_flaws_in_interval_for_comparison_prop_var,
        regression::{get_init_state_flaws, get_regression_precondition_flaws},
        state::{FlawSearchState, get_initial_flaw_search_state},
    },
    domain_abstraction::{ComparisonAxiomIndex, NumericPartitions},
    domain_abstraction_factory::WildcardPlanResult,
    utils::{make_prop_state_packer, partitions_for_interval},
};

pub enum SequenceDirection {
    Progression,
    Regression,
    Bidirectional,
}

#[allow(unused_assignments)]
pub fn get_sequence_flaws(
    task: &dyn AbstractNumericTask,
    partitions: &NumericPartitions,
    domain_mapping: &DomainMapping,
    wildcard_plan: &WildcardPlanResult,
    direction: SequenceDirection,
) -> Result<Vec<Flaw>> {
    let mut flaws = Vec::new();
    match direction {
        SequenceDirection::Progression => {
            get_sequence_progression_flaws(
                task,
                partitions,
                domain_mapping,
                wildcard_plan,
                &mut flaws,
            )?;
        }
        SequenceDirection::Regression => {
            get_sequence_regression_flaws(task, domain_mapping, wildcard_plan, &mut flaws)?;
            if flaws.is_empty() {
                // Progression sequence flaws as fallback.
                get_sequence_progression_flaws(
                    task,
                    partitions,
                    domain_mapping,
                    wildcard_plan,
                    &mut flaws,
                )?;
            }
        }
        SequenceDirection::Bidirectional => {
            get_sequence_progression_flaws(
                task,
                partitions,
                domain_mapping,
                wildcard_plan,
                &mut flaws,
            )?;
            get_sequence_regression_flaws(task, domain_mapping, wildcard_plan, &mut flaws)?;
        }
    }

    Ok(flaws)
}

#[allow(unused_assignments)]
fn get_sequence_progression_flaws(
    task: &dyn AbstractNumericTask,
    partitions: &NumericPartitions,
    domain_mapping: &DomainMapping,
    wildcard_plan: &WildcardPlanResult,
    collected_flaws: &mut Vec<Flaw>,
) -> Result<()> {
    let comparison_index = ComparisonAxiomIndex::from_task(task)
        .map_err(|e| anyhow::anyhow!("failed to build ComparisonAxiomIndex: {e}"))?;

    let state_packer = make_prop_state_packer(task);
    let axiom_evaluator = AxiomEvaluator::new(task, &state_packer);

    let mut state =
        get_initial_flaw_search_state(task, &state_packer, &axiom_evaluator, domain_mapping)?;
    let mut next_state = None;

    let mut flawed = false;
    let mut step: usize = 1;

    for equivalent_ops in wildcard_plan.wildcard_plan.iter() {
        let mut step_flaws = Vec::new();
        let expected_abs_numeric_state = &wildcard_plan.abstract_numeric_states[step];
        ensure!(
            step < wildcard_plan.abstract_numeric_states.len(),
            "WildcardPlanResult abstract_numeric_states too short for step {step}"
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
            let operator_flaws = get_progression_sequence_precondition_flaws(
                task,
                partitions,
                &comparison_index,
                op,
                &state,
                step,
            );
            if operator_flaws.is_empty() {
                chosen_op = Some(op);
                (state, next_state, flawed) = progress_and_get_sequence_deviation_flaws(
                    state,
                    expected_abs_numeric_state,
                    &axiom_evaluator,
                    op,
                    partitions,
                    &mut step_flaws,
                    step,
                )?;
                if !flawed {
                    step_flaws.clear();
                    state = next_state.take().unwrap();
                    break;
                }
            } else {
                step_flaws.extend(operator_flaws);
            }
        }

        // Progress and find flaws in the fallback operator only if it has
        // not been done in any other operator.
        if let Some(op) = fallback_op
            && chosen_op.is_none()
        {
            (state, next_state, flawed) = progress_and_get_sequence_deviation_flaws(
                state,
                expected_abs_numeric_state,
                &axiom_evaluator,
                op,
                partitions,
                &mut step_flaws,
                step,
            )?;

            state = next_state.take().unwrap();
            if flawed {
                // Undeviate the flaws.
                for (var, value) in state.numeric.iter_mut().enumerate() {
                    let Some(parts) = partitions.partitions(var) else {
                        continue;
                    };
                    let correct_values = partitions_for_interval(parts, value);
                    if !correct_values.is_empty()
                        && !correct_values.contains(&expected_abs_numeric_state[var])
                    {
                        *value = parts[expected_abs_numeric_state[var]];
                    }
                }
            }
        }

        collected_flaws.extend(step_flaws);

        step += 1;
    }

    collected_flaws.extend(get_goal_sequence_flaws(
        task,
        partitions,
        &comparison_index,
        &state,
        step,
    ));

    Ok(())
}

fn get_sequence_regression_flaws(
    task: &dyn AbstractNumericTask,
    domain_mapping: &DomainMapping,
    wildcard_plan: &WildcardPlanResult,
    collected_flaws: &mut Vec<Flaw>,
) -> Result<()> {
    let state_packer = make_prop_state_packer(task);
    let axiom_evaluator = AxiomEvaluator::new(task, &state_packer);

    let mut state = FlawSearchState::goals_partial_state(task, domain_mapping);

    let mut step: usize = wildcard_plan.wildcard_plan.len();

    // Deviation flaws are not possible because numeric variables are always
    // unbounded.
    for equivalent_ops in wildcard_plan.wildcard_plan.iter().rev() {
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
            let operator_flaws = get_regression_precondition_flaws(op, &state, step);
            if operator_flaws.is_empty() {
                chosen_op = Some(op);
                step_flaws.clear();
                state.regress(op, &axiom_evaluator)?;
                break;
            } else {
                step_flaws.extend(operator_flaws);
            }
        }

        // Regress in the fallback operator only if it has
        // not been done in any other operator.
        if let Some(op) = fallback_op
            && chosen_op.is_none()
        {
            state.regress(op, &axiom_evaluator)?;
        }

        collected_flaws.extend(step_flaws);

        step -= 1;
    }

    state.revert_axioms(&axiom_evaluator)?;
    let init_flaws = get_init_state_flaws(task, &state);
    collected_flaws.extend(init_flaws);

    Ok(())
}

type CurrentNextAndFlawed<'a> = (FlawSearchState<'a>, Option<FlawSearchState<'a>>, bool);
#[allow(clippy::too_many_arguments)]
pub(crate) fn progress_and_get_sequence_deviation_flaws<'a>(
    state: FlawSearchState<'a>,
    expected_abs_numeric_state: &[usize],
    axiom_evaluator: &AxiomEvaluator<'_>,
    op: &Operator,
    partitions: &NumericPartitions,
    collected_flaws: &mut Vec<Flaw>,
    step: usize,
) -> Result<CurrentNextAndFlawed<'a>> {
    let mut next_state = state.clone();
    let mut flawed = false;
    next_state.progress(op, axiom_evaluator)?;

    let deviation_flaws = get_progression_numeric_sequence_deviation_flaws(
        op,
        &state,
        &next_state,
        expected_abs_numeric_state,
        partitions,
        step,
    );
    if !deviation_flaws.is_empty() {
        collected_flaws.extend(deviation_flaws);
        flawed = true;
    }

    Ok((state, Some(next_state), flawed))
}

#[allow(clippy::needless_range_loop)]
pub fn get_progression_numeric_sequence_deviation_flaws(
    op: &Operator,
    current_state: &FlawSearchState,
    successor_state: &FlawSearchState,
    abstract_numeric_successor_state: &[usize],
    partitions: &NumericPartitions,
    step: usize,
) -> Vec<Flaw> {
    let mut flaws: Vec<Flaw> = Vec::new();

    let num_vars = successor_state
        .numeric
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
        let correct_abstract_values =
            partitions_for_interval(parts, &successor_state.numeric[var_id]);
        if correct_abstract_values.is_empty() {
            continue;
        };
        if correct_abstract_values.contains(&abstract_value) {
            continue;
        }

        let interval_next_value = successor_state.numeric[var_id];
        let interval_current_value = current_state
            .numeric
            .get(var_id)
            .copied()
            .unwrap_or(interval_next_value);
        if interval_next_value == interval_current_value {
            continue;
        }

        let operator_increased_value = interval_current_value.lower_is_lower(&interval_next_value);
        let mut include_in_lower = !operator_increased_value;

        if can_split_numeric_var(
            partitions,
            var_id,
            interval_current_value.lower,
            include_in_lower,
        ) {
            flaws.push(Flaw::Numeric(NumericFlaw {
                numeric_var_id: var_id,
                value: interval_current_value.upper,
                include_in_lower,
                step,
            }));
        } else {
            include_in_lower = !include_in_lower;
            if can_split_numeric_var(
                partitions,
                var_id,
                interval_current_value.upper,
                include_in_lower,
            ) {
                flaws.push(Flaw::Numeric(NumericFlaw {
                    numeric_var_id: var_id,
                    value: interval_current_value.lower,
                    include_in_lower,
                    step,
                }));
            }
        }
    }

    flaws
}

pub fn get_progression_sequence_precondition_flaws(
    task: &dyn AbstractNumericTask,
    partitions: &NumericPartitions,
    comparison_index: &ComparisonAxiomIndex,
    op: &Operator,
    state: &FlawSearchState,
    step: usize,
) -> Vec<Flaw> {
    let mut out: Vec<Flaw> = Vec::new();
    for pre in op.preconditions().iter() {
        if !state.fact_is_hold(pre) {
            let prop_var_id = pre.var;
            let dependent_numeric_flaws =
                if comparison_index.is_comparison_axiom_variable(prop_var_id) {
                    dependent_numeric_flaws_in_interval_for_comparison_prop_var(
                        task,
                        partitions,
                        comparison_index,
                        prop_var_id,
                        state,
                        step,
                    )
                } else {
                    vec![]
                };
            out.push(Flaw::Propositional(PropFlaw {
                fact: pre.clone(),
                dependent_numeric_flaws,
                step,
            }));
        }
    }
    out
}

pub fn get_goal_sequence_flaws(
    task: &dyn AbstractNumericTask,
    partitions: &NumericPartitions,
    comparison_index: &ComparisonAxiomIndex,
    state: &FlawSearchState,
    step: usize,
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
        if !state.fact_is_hold(goal_fact) && seen.insert(goal_fact.clone()) {
            let prop_var_id = goal_fact.var;
            let dependent_numeric_flaws =
                if comparison_index.is_comparison_axiom_variable(prop_var_id) {
                    dependent_numeric_flaws_in_interval_for_comparison_prop_var(
                        task,
                        partitions,
                        comparison_index,
                        prop_var_id,
                        state,
                        step,
                    )
                } else {
                    vec![]
                };
            out.push(Flaw::Propositional(PropFlaw {
                fact: goal_fact.clone(),
                dependent_numeric_flaws,
                step,
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
            if !state.fact_is_hold(pre) && seen.insert(pre.clone()) {
                let prop_var_id = pre.var;
                let dependent_numeric_flaws =
                    if comparison_index.is_comparison_axiom_variable(prop_var_id) {
                        dependent_numeric_flaws_in_interval_for_comparison_prop_var(
                            task,
                            partitions,
                            comparison_index,
                            prop_var_id,
                            state,
                            step,
                        )
                    } else {
                        vec![]
                    };
                out.push(Flaw::Propositional(PropFlaw {
                    fact: pre.clone(),
                    dependent_numeric_flaws,
                    step,
                }));
            }
        }
    }
    out
}
