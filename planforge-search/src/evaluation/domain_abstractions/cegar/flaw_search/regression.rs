#[cfg(test)]
mod tests;

use anyhow::Result;
use planforge_sas::{
    axioms::AxiomEvaluator,
    numeric_task::{AbstractNumericTask, ExplicitFact, Operator},
};

use super::{Flaw, NumericFlaw, PropFlaw, can_split_numeric_var};
use crate::evaluation::domain_abstractions::{
    abstract_operator_generator::DomainMapping,
    cegar::flaw_search::{numeric_requirement_for_comparison_fact, state::FlawSearchState},
    comparison_expression::{EMPTY_INTERVAL, Interval, UNBOUNDED_INTERVAL},
    domain_abstraction::{ComparisonAxiomIndex, NumericPartitions},
    domain_abstraction_factory::WildcardPlanResult,
    utils::make_prop_state_packer,
};

#[allow(unused_assignments)]
pub fn get_regression_flaws(
    task: &dyn AbstractNumericTask,
    partitions: &NumericPartitions,
    domain_mapping: &DomainMapping,
    wildcard_plan: &WildcardPlanResult,
) -> Result<Vec<Flaw>> {
    let state_packer = std::sync::Arc::new(make_prop_state_packer(task));
    let axiom_evaluator = AxiomEvaluator::new(std::sync::Arc::new(task), state_packer.clone());
    let comparison_index = ComparisonAxiomIndex::from_task(task)
        .map_err(|e| anyhow::anyhow!("failed to build comparison axiom index: {e}"))?;

    let mut state = FlawSearchState::goals_partial_state(task, domain_mapping);
    materialize_comparison_requirements(task, &comparison_index, &mut state);

    let mut collected_flaws: Vec<Flaw> = Vec::new();
    let mut step: usize = wildcard_plan.wildcard_plan.len();

    // Deviation flaws are not possible because numeric variables are always
    // unbounded.
    for equivalent_ops in wildcard_plan.wildcard_plan.iter().rev() {
        for &op_id in equivalent_ops.iter() {
            let Some(op) = task.get_operators().get(op_id) else {
                continue;
            };
            let operator_flaws = get_regression_precondition_flaws(op, &state, step);
            if operator_flaws.is_empty() {
                state.regress(op, &axiom_evaluator)?;
                materialize_comparison_requirements(task, &comparison_index, &mut state);
                collected_flaws.clear();
                break;
            } else {
                collected_flaws.extend(operator_flaws);
            }
        }

        if !collected_flaws.is_empty() {
            break;
        }

        step -= 1;
    }

    if !collected_flaws.is_empty() {
        return Ok(collected_flaws);
    }

    state.revert_axioms(&axiom_evaluator)?;
    let init_flaws = get_init_state_flaws(task, partitions, &state);

    Ok(init_flaws)
}

pub fn get_regression_precondition_flaws(
    op: &Operator,
    state: &FlawSearchState,
    step: usize,
) -> Vec<Flaw> {
    let mut out: Vec<Flaw> = Vec::new();
    for eff in op.effects().iter() {
        if !state.value_is_hold_for_var(eff.var_id(), eff.value()) {
            let eff_var_id = eff.var_id();
            out.push(Flaw::Propositional(PropFlaw {
                fact: ExplicitFact::new(eff_var_id, eff.value()),
                dependent_numeric_flaws: vec![],
                step,
            }));
        }
    }
    out
}

pub fn get_init_state_flaws(
    task: &dyn AbstractNumericTask,
    partitions: &NumericPartitions,
    state: &FlawSearchState,
) -> Vec<Flaw> {
    let initial_prop_state = task.get_initial_propositional_state_values();
    let initial_numeric_state = task.get_initial_numeric_state_values();
    let mut flaws: Vec<Flaw> = Vec::new();
    for (var, value) in initial_prop_state.iter().enumerate() {
        if !state.value_is_hold_for_var(var, *value) {
            flaws.push(Flaw::Propositional(PropFlaw {
                fact: ExplicitFact::new(var, *value),
                dependent_numeric_flaws: vec![],
                step: 0,
            }));
        }
    }

    for (numeric_var_id, requirement) in state.numeric.iter().enumerate() {
        if *requirement == UNBOUNDED_INTERVAL || requirement.is_empty() {
            continue;
        }
        let Some(&initial_value) = initial_numeric_state.get(numeric_var_id) else {
            continue;
        };
        if requirement.contains(initial_value) {
            continue;
        }
        let Some((value, include_in_lower)) =
            split_for_missing_numeric_requirement(*requirement, initial_value)
        else {
            continue;
        };
        if can_split_numeric_var(partitions, numeric_var_id, value, include_in_lower) {
            flaws.push(Flaw::Numeric(NumericFlaw {
                numeric_var_id,
                value,
                include_in_lower,
                step: 0,
            }));
        }
    }

    flaws
}

pub(crate) fn materialize_comparison_requirements(
    task: &dyn AbstractNumericTask,
    comparison_index: &ComparisonAxiomIndex,
    state: &mut FlawSearchState,
) {
    for var in 0..state.concrete_prop.len() {
        let Some(value) = state.concrete_prop[var] else {
            continue;
        };
        let fact = ExplicitFact::new(var, value);
        let Some((numeric_var_id, required_interval)) =
            numeric_requirement_for_comparison_fact(task, comparison_index, &fact)
        else {
            continue;
        };
        state.numeric[numeric_var_id] =
            intersect_intervals(state.numeric[numeric_var_id], required_interval);
    }
}

fn split_for_missing_numeric_requirement(
    requirement: Interval,
    initial_value: f64,
) -> Option<(f64, bool)> {
    if initial_value < requirement.lower
        || (initial_value == requirement.lower && !requirement.lower_closed)
    {
        if requirement.lower.is_finite() {
            return Some((requirement.lower, !requirement.lower_closed));
        }
    }
    if initial_value > requirement.upper
        || (initial_value == requirement.upper && !requirement.upper_closed)
    {
        if requirement.upper.is_finite() {
            return Some((requirement.upper, requirement.upper_closed));
        }
    }
    None
}

fn intersect_intervals(left: Interval, right: Interval) -> Interval {
    if left.is_empty() || right.is_empty() {
        return EMPTY_INTERVAL;
    }

    let (lower, lower_closed) = if left.lower > right.lower {
        (left.lower, left.lower_closed)
    } else if right.lower > left.lower {
        (right.lower, right.lower_closed)
    } else {
        (left.lower, left.lower_closed && right.lower_closed)
    };

    let (upper, upper_closed) = if left.upper < right.upper {
        (left.upper, left.upper_closed)
    } else if right.upper < left.upper {
        (right.upper, right.upper_closed)
    } else {
        (left.upper, left.upper_closed && right.upper_closed)
    };

    Interval::new(lower, upper, lower_closed, upper_closed)
}
