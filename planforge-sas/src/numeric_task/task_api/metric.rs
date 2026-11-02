//! The metric: its value in a state and what each operator costs under it.

use super::*;

pub fn evaluate_metric_from_values(task: &Task, numeric_values: &[NumericValue]) -> NumericValue {
    let metric_var_id = task.metric().var_id();
    match metric_var_id {
        Some(var_id) => *numeric_values.get(var_id.index()).unwrap_or_else(|| {
            panic!(
                "metric variable {} is out of bounds for {} numeric values",
                var_id.index(),
                numeric_values.len()
            )
        }),
        None => NumericValue::new(0.0),
    }
}

pub fn propagate_assignment_axiom_values(
    task: &Task,
    numeric_values: &mut [NumericValue],
) -> Result<(), AssignmentAxiomError> {
    // Assignment axioms are stored in dependency-layer order, so each RHS is
    // complete when it is visited and one forward pass closes the values.
    for axiom in task.assignment_axioms() {
        let affected_var_id = axiom.get_affected_var_id().index();
        assert!(
            affected_var_id < numeric_values.len(),
            "assignment axiom target {affected_var_id} is out of bounds for {} numeric values",
            numeric_values.len()
        );
        axiom.update_values(numeric_values)?;
    }
    Ok(())
}

/// The metric cost of every operator of `task` applied in its initial state,
/// as search charges it there: the effects apply simultaneously, conditional
/// ones where their conditions hold, and the successor is closed under the
/// axioms. Without a metric, every operator's own cost.
pub fn metric_operator_costs(task: &Task) -> Vec<f64> {
    if !task.metric().use_metric() {
        return task
            .operators()
            .iter()
            .map(|operator| operator.cost().value() as f64)
            .collect();
    }
    let states = ExplicitStateSpace::new(task);
    let initial_state = states
        .initial_state()
        .unwrap_or_else(|error| panic!("the initial state violates the axioms: {error}"));
    let initial_metric = evaluate_metric_from_values(task, &initial_state.numeric);
    let mut successor = ExplicitState::default();
    task.operators()
        .iter()
        .map(|operator| {
            states
                .apply_effects_into(&initial_state, operator, &mut successor)
                .unwrap_or_else(|error| {
                    panic!(
                        "operator {} leads the initial state into an axiom violation: {error}",
                        operator.name()
                    )
                });
            let metric = evaluate_metric_from_values(task, &successor.numeric);
            let cost = task
                .metric()
                .step_cost(initial_metric.value(), metric.value());
            assert!(
                cost >= 0.0,
                "operator {} has negative metric cost {cost}, which search does not support",
                operator.name()
            );
            cost
        })
        .collect()
}

pub(super) fn linear_metric_operator_cost_expression(
    task: &TaskData,
    operator_id: OperatorIndex,
) -> Option<crate::utils::linear_effects::LinearExpression> {
    if !task.metric().use_metric() {
        return None;
    }

    let metric_var_id = task.metric().var_id().unwrap();
    let metric_variable = task.numeric_variables().get(metric_var_id.index())?;
    if metric_variable.numeric_type() != NumericType::Cost {
        return None;
    }

    let operator = task
        .operators()
        .get(operator_id.index())
        .unwrap_or_else(|| {
            panic!("operator id {operator_id} is out of bounds for linear metric-cost extraction")
        });
    let metric_direction = if task.metric().is_min() { 1.0 } else { -1.0 };
    let mut linear_cost_expression = None;

    for assignment_effect in operator.assignment_effects() {
        if assignment_effect.affected_var_id() != metric_var_id {
            continue;
        }
        if assignment_effect.is_conditional() || !assignment_effect.conditions().is_empty() {
            continue;
        }

        let source_expression = task
            .linearize_numeric_var(assignment_effect.var_id())
            .unwrap_or_else(|error| {
                panic!(
                    "failed to linearize metric-cost source variable {} for operator {operator_id}: {error}",
                    assignment_effect.var_id().index()
                )
            });
        let candidate = match assignment_effect.operation() {
            AssignmentOperation::Plus => source_expression.scale(metric_direction),
            AssignmentOperation::Minus => source_expression.scale(-metric_direction),
            AssignmentOperation::Assign
            | AssignmentOperation::Times
            | AssignmentOperation::Divide => continue,
        };

        if candidate
            .coefficients
            .iter()
            .all(|&coefficient| coefficient == 0.0)
        {
            continue;
        }

        if linear_cost_expression.is_some() {
            todo!(
                "multiple unconditional linear metric-cost effects for operator {operator_id} are not implemented yet"
            );
        }
        linear_cost_expression = Some(candidate);
    }

    linear_cost_expression
}
