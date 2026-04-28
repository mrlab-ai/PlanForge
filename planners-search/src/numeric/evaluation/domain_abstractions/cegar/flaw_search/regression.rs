#[cfg(test)]
mod tests;

use anyhow::Result;
use planners_sas::numeric::{
    axioms::AxiomEvaluator,
    numeric_task::{AbstractNumericTask, ExplicitFact, Operator},
};

use super::{Flaw, PropFlaw};
use crate::numeric::evaluation::domain_abstractions::{
    abstract_operator_generator::DomainMapping, cegar::flaw_search::state::FlawSearchState,
    domain_abstraction_factory::WildcardPlanResult, utils::make_prop_state_packer,
};

#[allow(unused_assignments)]
pub fn get_regression_flaws(
    task: &dyn AbstractNumericTask,
    domain_mapping: &DomainMapping,
    wildcard_plan: &WildcardPlanResult,
    execute_entire_plan: bool,
) -> Result<Vec<Flaw>> {
    let state_packer = make_prop_state_packer(task);
    let axiom_evaluator = AxiomEvaluator::new(task, &state_packer);

    let mut state = FlawSearchState::goals_partial_state(task, domain_mapping);

    let mut collected_flaws: Vec<Flaw> = Vec::new();

    // Deviation flaws are not possible because numeric variables are always
    // unbounded.
    for equivalent_ops in wildcard_plan.wildcard_plan.iter().rev() {
        let mut chosen_op: Option<&Operator> = None;
        let mut fallback_op: Option<&Operator> = None;
        for &op_id in equivalent_ops.iter() {
            let Some(op) = task.get_operators().get(op_id) else {
                continue;
            };
            if fallback_op.is_none() {
                fallback_op = Some(op);
            }
            let operator_flaws = get_regression_precondition_flaws(op, &state);
            if operator_flaws.is_empty() {
                chosen_op = Some(op);
                state.regress(op, &axiom_evaluator)?;
                collected_flaws.clear();
                break;
            } else {
                collected_flaws.extend(operator_flaws);
            }
        }

        if execute_entire_plan {
            // Regress in the fallback operator only if it has
            // not been done in any other operator.
            if let Some(op) = fallback_op
                && chosen_op.is_none()
            {
                state.regress(op, &axiom_evaluator)?;
            }
        } else if !collected_flaws.is_empty() {
            break;
        }
    }

    if !execute_entire_plan && !collected_flaws.is_empty() {
        return Ok(collected_flaws);
    }

    state.revert_axioms(&axiom_evaluator)?;
    let init_flaws = get_init_state_flaws(task, &state);
    if execute_entire_plan {
        collected_flaws.extend(init_flaws);
        Ok(collected_flaws)
    } else {
        Ok(init_flaws)
    }
}

pub fn get_regression_precondition_flaws(op: &Operator, state: &FlawSearchState) -> Vec<Flaw> {
    let mut out: Vec<Flaw> = Vec::new();
    for eff in op.effects().iter() {
        if !state.value_is_hold_for_var(eff.var_id(), eff.value()) {
            let eff_var_id = eff.var_id();
            out.push(Flaw::Propositional(PropFlaw {
                fact: ExplicitFact::new(eff_var_id, eff.value()),
                dependent_numeric_flaws: vec![],
            }));
        }
    }
    out
}

fn get_init_state_flaws(task: &dyn AbstractNumericTask, state: &FlawSearchState) -> Vec<Flaw> {
    let initial_prop_state = task.get_initial_propositional_state_values();
    let mut flaws: Vec<Flaw> = Vec::new();
    for (var, value) in initial_prop_state.iter().enumerate() {
        if !state.value_is_hold_for_var(var, *value) {
            flaws.push(Flaw::Propositional(PropFlaw {
                fact: ExplicitFact::new(var, *value),
                dependent_numeric_flaws: vec![],
            }));
        }
    }

    flaws
}
