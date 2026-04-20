pub mod flaw_selection;
pub mod progression;
pub mod state;
#[cfg(test)]
mod tests;

use anyhow::{Result, ensure};
use planners_sas::numeric::axioms::AxiomEvaluator;
use planners_sas::numeric::utils::int_packer::IntDoublePacker;
use std::collections::BTreeSet;
use std::fmt;

use planners_sas::numeric::numeric_task::{AbstractNumericTask, ExplicitFact, Operator};

use serde::{Deserialize, Serialize};

use super::{determine_include_in_lower, make_prop_state_packer};
use crate::numeric::evaluation::domain_abstractions::comparison_expression::Interval;
use crate::numeric::evaluation::domain_abstractions::domain_abstraction::{
    ComparisonAxiomIndex, NumericPartitions,
};
use crate::numeric::evaluation::domain_abstractions::domain_abstraction_factory::WildcardPlanResult;
use crate::numeric::evaluation::domain_abstractions::utils::{fact_is_hold, get_initial_state};
use state::progress;

/// Mirrors numeric-FD's `NumericFlaw = tuple<int, ap_float, bool>`.
#[derive(Debug, Clone, PartialEq)]
pub struct NumericFlaw {
    pub numeric_var_id: usize,
    pub value: f64,
    pub include_in_lower: bool,
}

/// Mirrors numeric-FD's `PropFlaw = pair<Fact, vector<NumericFlaw>>`.
#[derive(Debug, Clone, PartialEq)]
pub struct PropFlaw {
    pub fact: ExplicitFact,
    pub dependent_numeric_flaws: Vec<NumericFlaw>,
}

/// Mirrors numeric-FD's `Flaw = variant<PropFlaw, NumericFlaw>`.
#[derive(Debug, Clone, PartialEq)]
pub enum Flaw {
    Propositional(PropFlaw),
    Numeric(NumericFlaw),
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExecEntirePlanMode {
    StopAtFirstFlaw,
    ExecuteEntirePlan,
}

impl fmt::Display for ExecEntirePlanMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StopAtFirstFlaw => write!(f, "stop_at_first_flaw"),
            Self::ExecuteEntirePlan => write!(f, "execute_entire_plan"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DependentNumericRefinement {
    None,
    One,
    All,
}

pub fn get_flaws(
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

    let mut step_flaws: Vec<Flaw> = Vec::new();
    let mut collected_flaws: Vec<Flaw> = Vec::new();
    let mut step_num: usize = 1;

    for equivalent_ops in wildcard_plan.wildcard_plan.iter() {
        ensure!(
            step_num < wildcard_plan.abstract_numeric_states.len(),
            "WildcardPlanResult abstract_numeric_states too short for step {step_num}"
        );
        let expected_abs_numeric_state = &wildcard_plan.abstract_numeric_states[step_num];

        step_flaws.clear();

        if !execute_entire_plan {
            let mut applied = false;
            for &op_id in equivalent_ops.iter() {
                let Some(op) = task.get_operators().get(op_id) else {
                    continue;
                };
                let operator_flaws = get_precondition_flaws(
                    task,
                    partitions,
                    &comparison_index,
                    op,
                    &state_packer,
                    &prop_state,
                    &numeric_state,
                );
                if operator_flaws.is_empty() {
                    let mut candidate_prop_state = prop_state.clone();
                    let numeric_state_before_op = numeric_state.clone();
                    let mut candidate_numeric_state = numeric_state.clone();
                    progress(
                        op,
                        &axiom_evaluator,
                        &state_packer,
                        &mut candidate_prop_state,
                        &mut candidate_numeric_state,
                    )?;

                    let deviation_flaws = get_numeric_deviation_flaws(
                        op,
                        &numeric_state_before_op,
                        &candidate_numeric_state,
                        expected_abs_numeric_state,
                        partitions,
                    );
                    if deviation_flaws.is_empty() {
                        prop_state = candidate_prop_state;
                        numeric_state = candidate_numeric_state;
                        applied = true;
                        step_flaws.clear();
                        break;
                    } else {
                        step_flaws.extend(deviation_flaws);
                    }
                } else {
                    step_flaws.extend(operator_flaws);
                }
            }

            if !applied {
                return Ok(step_flaws.clone());
            }
            step_num += 1;
            continue;
        }

        // Execute_entire_plan mode: keep executing even if flaws are found.
        let mut chosen_op_id: Option<usize> = None;
        let mut fallback_op_id: Option<usize> = None;
        for &op_id in equivalent_ops.iter() {
            if task.get_operators().get(op_id).is_none() {
                continue;
            }
            if fallback_op_id.is_none() {
                fallback_op_id = Some(op_id);
            }
            let op = &task.get_operators()[op_id];
            let operator_flaws = get_precondition_flaws(
                task,
                partitions,
                &comparison_index,
                op,
                &state_packer,
                &prop_state,
                &numeric_state,
            );
            if operator_flaws.is_empty() {
                chosen_op_id = Some(op_id);
                break;
            } else {
                step_flaws.extend(operator_flaws);
            }
        }

        if !step_flaws.is_empty() {
            collected_flaws.append(&mut step_flaws);
        }

        let chosen = chosen_op_id.or(fallback_op_id);
        if let Some(op_id) = chosen {
            let op = &task.get_operators()[op_id];
            let numeric_state_before_op = numeric_state.clone();
            progress(
                op,
                &axiom_evaluator,
                &state_packer,
                &mut prop_state,
                &mut numeric_state,
            )?;

            let deviation_flaws = get_numeric_deviation_flaws(
                op,
                &numeric_state_before_op,
                &numeric_state,
                expected_abs_numeric_state,
                partitions,
            );
            if !deviation_flaws.is_empty() {
                collected_flaws.extend(deviation_flaws);
            }
        }

        step_num += 1;
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

#[allow(unused)]
fn score_flaw(
    flaw: &Flaw,
    domain_sizes: &[usize],
    numeric_domain_sizes: &[usize],
    _abstraction_size: usize,
) -> usize {
    match flaw {
        Flaw::Numeric(nf) => numeric_domain_sizes
            .get(nf.numeric_var_id)
            .copied()
            .unwrap_or(0),
        Flaw::Propositional(pf) => {
            let var_id = pf.fact.var;
            let base = domain_sizes.get(var_id).copied().unwrap_or(0);
            let max_dep = pf
                .dependent_numeric_flaws
                .iter()
                .filter_map(|nf| numeric_domain_sizes.get(nf.numeric_var_id).copied())
                .max()
                .unwrap_or(0);
            base + max_dep
        }
    }
}

fn dependent_numeric_flaws_for_comparison_prop_var(
    task: &dyn AbstractNumericTask,
    partitions: &NumericPartitions,
    comparison_index: &ComparisonAxiomIndex,
    prop_var_id: usize,
    numeric_state: &[f64],
) -> Vec<NumericFlaw> {
    let Some(tree) = comparison_index.comparison_tree(prop_var_id) else {
        return vec![];
    };

    let mut out: Vec<NumericFlaw> = Vec::new();
    for dep_var_id in tree.regular_numeric_var_dependencies(task) {
        let Some(&concrete_value) = numeric_state.get(dep_var_id) else {
            continue;
        };
        let include_in_lower =
            determine_include_in_lower(tree, dep_var_id, concrete_value, numeric_state);

        if can_split_numeric_var(partitions, dep_var_id, concrete_value, include_in_lower) {
            out.push(NumericFlaw {
                numeric_var_id: dep_var_id,
                value: concrete_value,
                include_in_lower,
            });
        } else if can_split_numeric_var(partitions, dep_var_id, concrete_value, !include_in_lower) {
            out.push(NumericFlaw {
                numeric_var_id: dep_var_id,
                value: concrete_value,
                include_in_lower: !include_in_lower,
            });
        }
    }
    out
}

pub fn get_precondition_flaws(
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

fn can_split_numeric_var(
    partitions: &NumericPartitions,
    numeric_var_id: usize,
    value: f64,
    include_in_lower: bool,
) -> bool {
    let Some(parts) = partitions.partitions(numeric_var_id) else {
        return false;
    };
    let Some(part_id) = parts.iter().position(|iv| iv.contains(value)) else {
        return false;
    };
    parts[part_id].can_split_at(value, include_in_lower)
}

pub fn get_numeric_deviation_flaws(
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

fn partition_for_value(partitions: &[Interval], value: f64) -> Option<usize> {
    partitions.iter().position(|iv| iv.contains(value))
}
