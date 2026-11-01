//! The concrete operators domain and Cartesian abstractions support.

use anyhow::{Context, Result, ensure};

use crate::ensure_supported;

use planforge_sas::numeric_task::{
    AbstractNumericTask, AssignmentOperation, NumericType, Operator, OperatorIndex, ZERO_VALUE,
};

/// Validates the concrete-operator fragment shared by domain and Cartesian
/// abstractions. Unsupported task input is rejected before either backend
/// constructs transitions.
pub(crate) fn validate_abstraction_operator(
    task: &dyn AbstractNumericTask,
    operator: &Operator,
    operator_id: OperatorIndex,
) -> Result<()> {
    let mut propositional_effect_by_var = vec![None; task.variables().len()];
    for (effect_id, effect) in operator.effects().iter().enumerate() {
        ensure!(
            effect.var_id().index() < task.variables().len(),
            "operator {operator_id} ({}) propositional effect {effect_id} targets missing variable {}",
            operator.name(),
            effect.var_id()
        );
        ensure_supported!(
            effect.conditions().is_empty(),
            "operator {operator_id} ({}) has a conditional propositional effect",
            operator.name()
        );
        ensure_supported!(
            propositional_effect_by_var[effect.var_id().index()]
                .replace(effect_id)
                .is_none(),
            "operator {operator_id} ({}) has multiple propositional effects on variable {}",
            operator.name(),
            effect.var_id()
        );
    }

    let numeric_variables = task.numeric_variables();
    let initial_numeric = task.initial_numeric_values();
    let mut numeric_effect_by_var = vec![None; numeric_variables.len()];
    for (effect_id, effect) in operator.assignment_effects().iter().enumerate() {
        ensure_supported!(
            !effect.is_conditional() && effect.conditions().is_empty(),
            "operator {operator_id} ({}) has a conditional numeric effect",
            operator.name()
        );
        ensure!(
            effect.affected_var_id().index() < numeric_variables.len(),
            "operator {operator_id} ({}) numeric effect {effect_id} targets missing variable {}",
            operator.name(),
            effect.affected_var_id()
        );
        ensure_supported!(
            numeric_effect_by_var[effect.affected_var_id().index()]
                .replace(effect_id)
                .is_none(),
            "operator {operator_id} ({}) has multiple numeric effects on variable {}",
            operator.name(),
            effect.affected_var_id()
        );
        let affected_type = numeric_variables[effect.affected_var_id().index()].numeric_type();
        ensure_supported!(
            matches!(affected_type, NumericType::Regular | NumericType::Cost),
            "operator {operator_id} ({}) numeric effect {effect_id} targets {:?} variable {}",
            operator.name(),
            affected_type,
            effect.affected_var_id()
        );
        let rhs_var_id = effect.var_id();
        let rhs_variable = numeric_variables.get(rhs_var_id.index()).with_context(|| {
            format!(
                "operator {operator_id} ({}) numeric effect {effect_id} reads missing RHS variable {rhs_var_id}",
                operator.name()
            )
        })?;
        ensure_supported!(
            rhs_variable.numeric_type() == NumericType::Constant,
            "operator {operator_id} ({}) numeric effect {effect_id} has a {:?} argument, not a constant",
            operator.name(),
            rhs_variable.numeric_type()
        );
        let rhs = *initial_numeric.get(rhs_var_id.index()).with_context(|| {
            format!("missing initial value for constant numeric variable {rhs_var_id}")
        })?;
        ensure!(
            rhs.is_finite(),
            "operator {operator_id} ({}) numeric effect {effect_id} has non-finite constant RHS {rhs}",
            operator.name()
        );
        ensure!(
            !matches!(effect.operation(), AssignmentOperation::Divide) || rhs != ZERO_VALUE,
            "operator {operator_id} ({}) numeric effect {effect_id} divides by zero",
            operator.name()
        );
    }
    Ok(())
}
