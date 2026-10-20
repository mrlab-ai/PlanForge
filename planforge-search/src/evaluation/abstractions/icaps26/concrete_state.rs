//! Concrete states as the ICAPS 2026 abstractions replay them: propositional
//! values packed, effects applied in their stored order. Kept apart from
//! [`planforge_sas::explicit_state`] so the frozen experiments stay
//! reproducible.

use super::*;
use planforge_sas::numeric_task::Operator;

/// A concrete state, closed under the task's axioms, with propositional values
/// packed by [`StatePacker::propositional`].
#[derive(Debug, Clone, PartialEq)]
pub(super) struct ConcreteState {
    pub(super) propositions: Vec<u64>,
    pub(super) numeric: Vec<NumericValue>,
}

pub(super) fn make_prop_state_packer(task: &dyn AbstractNumericTask) -> StatePacker {
    StatePacker::propositional(task)
}

pub(super) fn get_initial_state(
    task: &dyn AbstractNumericTask,
    state_packer: &StatePacker,
    axiom_evaluator: &AxiomEvaluator,
) -> Result<(Vec<u64>, Vec<NumericValue>)> {
    let mut buffer = vec![0u64; state_packer.num_bins()];
    for (var_id, &val) in task
        .get_initial_propositional_state_values()
        .iter()
        .enumerate()
    {
        state_packer.set(&mut buffer, var_id, val.index() as u64);
    }
    let mut numeric_state: Vec<NumericValue> = task.get_initial_numeric_state_values().to_vec();
    for value in &mut numeric_state {
        *value = float_tolerance::canonicalize(*value);
    }

    axiom_evaluator
        .evaluate(&mut buffer, &mut numeric_state)
        .map_err(|e| anyhow::anyhow!("failed to evaluate axioms for initial state: {e:?}"))?;

    Ok((buffer, numeric_state))
}

/// Applies `op` to the packed state effect by effect, each effect reading the
/// state the effects before it left.
pub(super) fn progress_concrete_state(
    op: &Operator,
    axiom_evaluator: &AxiomEvaluator,
    packer: &StatePacker,
    prop_state: &mut [u64],
    numeric_state: &mut [NumericValue],
) -> Result<()> {
    for effect in op.effects() {
        if effect
            .conditions()
            .iter()
            .all(|condition| packer.get(prop_state, condition.var()) == condition.value() as u64)
        {
            packer.set(
                prop_state,
                effect.var_id().index(),
                effect.value().index() as u64,
            );
        }
    }

    for effect in op.assignment_effects() {
        if effect.is_conditional()
            && !effect.conditions().iter().all(|condition| {
                packer.get(prop_state, condition.var()) == condition.value() as u64
            })
        {
            continue;
        }
        let source_var = effect.var_id();
        let affected_var = effect.affected_var_id();
        ensure!(
            source_var.index() < numeric_state.len() && affected_var.index() < numeric_state.len(),
            "operator {} numeric effect references vars ({source_var}, {affected_var}) outside {} numeric variables",
            op.name(),
            numeric_state.len()
        );
        numeric_state[affected_var.index()] =
            float_tolerance::canonicalize(AssignmentOperation::apply(
                numeric_state[affected_var.index()],
                effect.operation(),
                numeric_state[source_var.index()],
            ));
    }

    axiom_evaluator
        .evaluate(prop_state, numeric_state)
        .map_err(|error| anyhow::anyhow!("failed to evaluate axioms: {error:?}"))?;
    Ok(())
}
