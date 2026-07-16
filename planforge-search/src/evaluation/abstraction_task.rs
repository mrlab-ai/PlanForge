use anyhow::{Context, Result, ensure};
use planforge_sas::numeric_task::{
    AbstractNumericTask, AssignmentOperation, NumericType, Operator,
};

/// Validates the concrete-operator fragment shared by domain and Cartesian
/// abstractions. Unsupported task input is rejected before either backend
/// constructs transitions.
pub(crate) fn validate_abstraction_operator(
    task: &dyn AbstractNumericTask,
    operator: &Operator,
    operator_id: usize,
) -> Result<()> {
    let mut propositional_effect_by_var = vec![None; task.get_num_variables()];
    for (effect_id, effect) in operator.effects().iter().enumerate() {
        ensure!(
            effect.var_id() < task.get_num_variables(),
            "operator {operator_id} ({}) propositional effect {effect_id} targets missing variable {}",
            operator.name(),
            effect.var_id()
        );
        ensure!(
            effect.conditions().is_empty(),
            "numeric-fd parity: conditional propositional or numeric effects are unsupported in abstraction generation"
        );
        ensure!(
            propositional_effect_by_var[effect.var_id()]
                .replace(effect_id)
                .is_none(),
            "operator {operator_id} ({}) has multiple propositional effects on variable {}",
            operator.name(),
            effect.var_id()
        );
    }

    let numeric_variables = task.numeric_variables();
    let initial_numeric = task.get_initial_numeric_state_values();
    let mut numeric_effect_by_var = vec![None; numeric_variables.len()];
    for (effect_id, effect) in operator.assignment_effects().iter().enumerate() {
        ensure!(
            !effect.is_conditional() && effect.conditions().is_empty(),
            "numeric-fd parity: conditional propositional or numeric effects are unsupported in abstraction generation"
        );
        ensure!(
            effect.affected_var_id() < numeric_variables.len(),
            "operator {operator_id} ({}) numeric effect {effect_id} targets missing variable {}",
            operator.name(),
            effect.affected_var_id()
        );
        ensure!(
            numeric_effect_by_var[effect.affected_var_id()]
                .replace(effect_id)
                .is_none(),
            "operator {operator_id} ({}) has multiple numeric effects on variable {}",
            operator.name(),
            effect.affected_var_id()
        );
        let affected_type = numeric_variables[effect.affected_var_id()].get_type();
        ensure!(
            matches!(affected_type, NumericType::Regular | NumericType::Cost),
            "operator {operator_id} ({}) numeric effect {effect_id} targets {:?} variable {}",
            operator.name(),
            affected_type,
            effect.affected_var_id()
        );
        let rhs_var_id = effect.var_id();
        let rhs_variable = numeric_variables.get(rhs_var_id).with_context(|| {
            format!(
                "operator {operator_id} ({}) numeric effect {effect_id} reads missing RHS variable {rhs_var_id}",
                operator.name()
            )
        })?;
        ensure!(
            rhs_variable.get_type() == &NumericType::Constant,
            "numeric-fd parity: assignment effects require constant RHS, got {:?} for numeric var {}",
            rhs_variable.get_type(),
            rhs_var_id
        );
        let rhs = *initial_numeric.get(rhs_var_id).with_context(|| {
            format!("missing initial value for constant numeric variable {rhs_var_id}")
        })?;
        ensure!(
            rhs.is_finite(),
            "operator {operator_id} ({}) numeric effect {effect_id} has non-finite constant RHS {rhs}",
            operator.name()
        );
        ensure!(
            !matches!(effect.operation(), AssignmentOperation::Divide) || rhs != 0.0,
            "operator {operator_id} ({}) numeric effect {effect_id} divides by zero",
            operator.name()
        );
    }
    Ok(())
}
