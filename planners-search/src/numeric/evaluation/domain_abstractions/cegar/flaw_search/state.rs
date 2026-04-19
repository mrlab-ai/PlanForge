use planners_sas::numeric::{numeric_task::Operator, utils::int_packer::IntDoublePacker};

use crate::numeric::evaluation::domain_abstractions::utils::fact_is_hold;

pub(crate) fn apply_operator_to_state(
    op: &Operator,
    packer: &IntDoublePacker,
    buffer: &mut [u64],
    numeric_state: &mut [f64],
) {
    // Propositional effects (respect conditions).
    for eff in op.effects().iter() {
        let mut ok = true;
        for cond in eff.conditions().iter() {
            if !fact_is_hold(cond, packer, buffer) {
                ok = false;
                break;
            }
        }
        if ok {
            packer.set(buffer, eff.var_id(), eff.value() as u64);
        }
    }

    // Numeric assignment effects.
    for eff in op.assignment_effects().iter() {
        if eff.is_conditional() {
            let mut ok = true;
            for cond in eff.conditions().iter() {
                if !fact_is_hold(cond, packer, buffer) {
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
}
