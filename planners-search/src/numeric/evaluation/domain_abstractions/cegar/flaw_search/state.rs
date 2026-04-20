use anyhow::Result;

use planners_sas::numeric::{
    axioms::AxiomEvaluator, numeric_task::Operator, utils::int_packer::IntDoublePacker,
};

use crate::numeric::evaluation::domain_abstractions::utils::fact_is_hold;

pub(crate) fn progress(
    op: &Operator,
    axiom_evaluator: &AxiomEvaluator,
    packer: &IntDoublePacker,
    prop_state: &mut [u64],
    numeric_state: &mut [f64],
) -> Result<()> {
    // Propositional effects (respect conditions).
    for eff in op.effects().iter() {
        let mut ok = true;
        for cond in eff.conditions().iter() {
            if !fact_is_hold(cond, packer, prop_state) {
                ok = false;
                break;
            }
        }
        if ok {
            packer.set(prop_state, eff.var_id(), eff.value() as u64);
        }
    }

    // Numeric assignment effects.
    for eff in op.assignment_effects().iter() {
        if eff.is_conditional() {
            let mut ok = true;
            for cond in eff.conditions().iter() {
                if !fact_is_hold(cond, packer, prop_state) {
                    ok = false;
                    break;
                }
            }
            if !ok {
                continue;
            }
        }

        let assignment_var_id = eff.var_id();
        let affected_var_id = eff.affected_var_id();
        if assignment_var_id >= numeric_state.len() || affected_var_id >= numeric_state.len() {
            continue;
        }
        let operand = numeric_state[assignment_var_id];
        numeric_state[affected_var_id] =
            planners_sas::numeric::numeric_task::AssignmentOperation::apply(
                numeric_state[affected_var_id],
                eff.operation(),
                operand,
            );
    }

    axiom_evaluator
        .evaluate_arithmetic_axioms(numeric_state)
        .map_err(|e| {
            anyhow::anyhow!("failed to evaluate arithmetic axioms after operator: {e:?}")
        })?;
    axiom_evaluator
        .evaluate(prop_state, numeric_state)
        .map_err(|e| anyhow::anyhow!("failed to evaluate axioms after operator: {e:?}"))?;

    Ok(())
}
