use std::collections::HashMap;

use anyhow::{Result, bail, ensure};
use planforge_sas::numeric_task::{
    AbstractNumericTask, AssignmentEffect, AssignmentOperation, NumericType, Operator,
};
use planforge_sas::utils::linear_effects::{LinearExpression, linearize_numeric_var};

use super::comparison_expression::ComparisonTree;

const EPSILON: f64 = 1e-12;

/// A derived affine expression whose value changes by a deterministic constant
/// under every operator. Such an expression is an exact numeric coordinate of
/// the original task and can therefore be partitioned without compiling a new
/// task or losing correlations between its regular numeric dependencies.
#[derive(Clone, Debug)]
pub(crate) struct AdditiveNumericView {
    expression: LinearExpression,
    operator_deltas: Vec<f64>,
}

impl AdditiveNumericView {
    pub(crate) fn operator_delta(&self, operator_id: usize) -> Result<f64> {
        self.operator_deltas
            .get(operator_id)
            .copied()
            .ok_or_else(|| {
                anyhow::anyhow!("missing additive-view delta for operator {operator_id}")
            })
    }

    pub(crate) fn evaluate(&self, numeric_values: &[f64]) -> f64 {
        self.expression.evaluate(numeric_values)
    }
}

pub(crate) fn initial_numeric_values_with_additive_views(
    task: &dyn AbstractNumericTask,
) -> Vec<f64> {
    let mut values = task.get_initial_numeric_state_values().to_vec();
    for numeric_var_id in 0..task.numeric_variables().len() {
        let Some(view) = analyze_additive_numeric_view(task, numeric_var_id) else {
            continue;
        };
        values[numeric_var_id] = view.evaluate(&values);
    }
    values
}

/// Active derived views, indexed by the task's numeric variable IDs.
#[derive(Clone, Debug)]
pub(crate) struct AdditiveNumericViews {
    by_numeric_var: Vec<Option<AdditiveNumericView>>,
}

impl AdditiveNumericViews {
    pub(crate) fn for_active_dimensions(
        task: &dyn AbstractNumericTask,
        numeric_domain_sizes: &[usize],
    ) -> Result<Self> {
        ensure!(
            numeric_domain_sizes.len() == task.numeric_variables().len(),
            "numeric-domain-size count {} does not match task numeric-variable count {}",
            numeric_domain_sizes.len(),
            task.numeric_variables().len()
        );
        let mut by_numeric_var = vec![None; numeric_domain_sizes.len()];
        for (numeric_var_id, &domain_size) in numeric_domain_sizes.iter().enumerate() {
            if task.numeric_variables()[numeric_var_id].get_type() != &NumericType::Derived {
                continue;
            }
            let view = analyze_additive_numeric_view(task, numeric_var_id);
            if domain_size > 1 && view.is_none() {
                bail!(
                    "derived numeric variable {numeric_var_id} ({}) was refined, but is not an affine coordinate with deterministic additive operator effects",
                    task.numeric_variables()[numeric_var_id].name()
                );
            }
            by_numeric_var[numeric_var_id] = view;
        }
        Ok(Self { by_numeric_var })
    }

    pub(crate) fn get(&self, numeric_var_id: usize) -> Option<&AdditiveNumericView> {
        self.by_numeric_var
            .get(numeric_var_id)
            .and_then(Option::as_ref)
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (usize, &AdditiveNumericView)> {
        self.by_numeric_var
            .iter()
            .enumerate()
            .filter_map(|(numeric_var_id, view)| view.as_ref().map(|view| (numeric_var_id, view)))
    }
}

pub(crate) fn analyze_additive_numeric_view(
    task: &dyn AbstractNumericTask,
    numeric_var_id: usize,
) -> Option<AdditiveNumericView> {
    if task.numeric_variables().get(numeric_var_id)?.get_type() != &NumericType::Derived {
        return None;
    }
    if numeric_expression_depends_on_cost(task, numeric_var_id) {
        return None;
    }
    let expression = linearize_numeric_var(task, numeric_var_id).ok()?;
    if expression.is_constant()
        || !expression.constant.is_finite()
        || expression
            .coefficients
            .iter()
            .any(|coefficient| !coefficient.is_finite())
    {
        return None;
    }
    for dependency in expression
        .coefficients
        .iter()
        .enumerate()
        .filter_map(|(var_id, coefficient)| (coefficient.abs() >= EPSILON).then_some(var_id))
    {
        if task.numeric_variables()[dependency].get_type() != &NumericType::Regular {
            return None;
        }
    }

    let operator_deltas = task
        .get_operators()
        .iter()
        .map(|operator| additive_view_delta_for_operator(task, &expression, operator))
        .collect::<Option<Vec<_>>>()?;
    Some(AdditiveNumericView {
        expression,
        operator_deltas,
    })
}

fn numeric_expression_depends_on_cost(
    task: &dyn AbstractNumericTask,
    numeric_var_id: usize,
) -> bool {
    fn visit(task: &dyn AbstractNumericTask, numeric_var_id: usize, visiting: &mut [bool]) -> bool {
        let Some(variable) = task.numeric_variables().get(numeric_var_id) else {
            return true;
        };
        match variable.get_type() {
            NumericType::Cost => true,
            NumericType::Regular | NumericType::Constant => false,
            NumericType::Derived => {
                if visiting[numeric_var_id] {
                    return true;
                }
                let mut axioms = task
                    .assignment_axioms()
                    .iter()
                    .filter(|axiom| axiom.get_affected_var_id() == numeric_var_id);
                let Some(axiom) = axioms.next() else {
                    return true;
                };
                if axioms.next().is_some() {
                    return true;
                }
                visiting[numeric_var_id] = true;
                let depends_on_cost = visit(task, axiom.get_left_var_id(), visiting)
                    || visit(task, axiom.get_right_var_id(), visiting);
                visiting[numeric_var_id] = false;
                depends_on_cost
            }
        }
    }

    visit(
        task,
        numeric_var_id,
        &mut vec![false; task.numeric_variables().len()],
    )
}

pub(crate) fn is_refinable_numeric_dimension(
    task: &dyn AbstractNumericTask,
    numeric_var_id: usize,
) -> bool {
    task.numeric_variables()
        .get(numeric_var_id)
        .is_some_and(|variable| match variable.get_type() {
            NumericType::Regular => true,
            NumericType::Derived => analyze_additive_numeric_view(task, numeric_var_id).is_some(),
            NumericType::Constant | NumericType::Cost => false,
        })
}

/// Prefer exact task-level coordinates at a comparison's two roots. If a
/// nonlinear or non-additive derived root cannot serve as a coordinate, fall
/// back to its regular leaves, preserving the existing general behavior.
pub(crate) fn comparison_refinement_dimensions(
    task: &dyn AbstractNumericTask,
    tree: &ComparisonTree,
) -> Vec<usize> {
    let mut direct = [tree.left_numeric_var_id, tree.right_numeric_var_id]
        .into_iter()
        .filter(|&numeric_var_id| is_refinable_numeric_dimension(task, numeric_var_id))
        .collect::<Vec<_>>();
    direct.sort_unstable();
    direct.dedup();
    if !direct.is_empty() {
        return direct;
    }
    tree.regular_numeric_var_dependencies(task)
}

/// Every refined coordinate that can constrain evaluation of `tree`.
pub(crate) fn active_comparison_dimensions(
    task: &dyn AbstractNumericTask,
    tree: &ComparisonTree,
    numeric_domain_sizes: &[usize],
    additive_views: &AdditiveNumericViews,
) -> Vec<usize> {
    let mut dimensions = tree
        .regular_numeric_var_dependencies(task)
        .into_iter()
        .filter(|&numeric_var_id| {
            numeric_domain_sizes
                .get(numeric_var_id)
                .is_some_and(|&size| size > 1)
        })
        .collect::<Vec<_>>();
    for node in &tree.nodes {
        let super::comparison_expression::ComparisonTreeNode::Arith {
            result_numeric_var_id,
            ..
        } = node
        else {
            continue;
        };
        if numeric_domain_sizes
            .get(*result_numeric_var_id)
            .is_some_and(|&size| size > 1)
            && additive_views.get(*result_numeric_var_id).is_some()
        {
            dimensions.push(*result_numeric_var_id);
        }
    }
    dimensions.sort_unstable();
    dimensions.dedup();
    dimensions
}

pub(crate) fn numeric_effect_deltas(task: &dyn AbstractNumericTask) -> HashMap<usize, Vec<f64>> {
    let mut deltas: HashMap<usize, Vec<f64>> = HashMap::new();
    for (numeric_var_id, numeric_var) in task.numeric_variables().iter().enumerate() {
        match numeric_var.get_type() {
            NumericType::Regular => {
                for operator in task.get_operators() {
                    if let Some(delta) =
                        regular_additive_delta_for_operator(task, numeric_var_id, operator)
                        && delta.abs() >= EPSILON
                    {
                        deltas.entry(numeric_var_id).or_default().push(delta);
                    }
                }
            }
            NumericType::Derived => {
                let Some(view) = analyze_additive_numeric_view(task, numeric_var_id) else {
                    continue;
                };
                for delta in view.operator_deltas {
                    if delta.abs() >= EPSILON {
                        deltas.entry(numeric_var_id).or_default().push(delta);
                    }
                }
            }
            NumericType::Constant | NumericType::Cost => {}
        }
    }
    for values in deltas.values_mut() {
        values.sort_by(|left, right| left.total_cmp(right));
        values.dedup_by(|left, right| (*left - *right).abs() < EPSILON);
    }
    deltas
}

pub(crate) fn numeric_dimension_delta_for_operator(
    task: &dyn AbstractNumericTask,
    numeric_var_id: usize,
    operator: &Operator,
) -> Option<f64> {
    match task.numeric_variables().get(numeric_var_id)?.get_type() {
        NumericType::Regular => regular_additive_delta_for_operator(task, numeric_var_id, operator),
        NumericType::Derived => {
            let view = analyze_additive_numeric_view(task, numeric_var_id)?;
            additive_view_delta_for_operator(task, &view.expression, operator)
        }
        NumericType::Constant | NumericType::Cost => None,
    }
}

pub(crate) fn is_operator_invariant_regular_dimension(
    task: &dyn AbstractNumericTask,
    numeric_var_id: usize,
) -> bool {
    task.numeric_variables()
        .get(numeric_var_id)
        .is_some_and(|variable| variable.get_type() == &NumericType::Regular)
        && task.get_operators().iter().all(|operator| {
            regular_additive_delta_for_operator(task, numeric_var_id, operator)
                .is_some_and(|delta| delta.abs() < EPSILON)
        })
}

fn additive_view_delta_for_operator(
    task: &dyn AbstractNumericTask,
    expression: &LinearExpression,
    operator: &Operator,
) -> Option<f64> {
    let mut delta = 0.0;
    for (numeric_var_id, &coefficient) in expression.coefficients.iter().enumerate() {
        if coefficient.abs() < EPSILON {
            continue;
        }
        delta += coefficient * regular_additive_delta_for_operator(task, numeric_var_id, operator)?;
    }
    delta.is_finite().then_some(delta)
}

fn regular_additive_delta_for_operator(
    task: &dyn AbstractNumericTask,
    numeric_var_id: usize,
    operator: &Operator,
) -> Option<f64> {
    let mut matching = operator
        .assignment_effects()
        .iter()
        .filter(|effect| effect.affected_var_id() == numeric_var_id);
    let Some(effect) = matching.next() else {
        return Some(0.0);
    };
    if matching.next().is_some() || effect.is_conditional() || !effect.conditions().is_empty() {
        return None;
    }
    constant_effect_delta(task, effect)
}

fn constant_effect_delta(task: &dyn AbstractNumericTask, effect: &AssignmentEffect) -> Option<f64> {
    let rhs_variable = task.numeric_variables().get(effect.var_id())?;
    if rhs_variable.get_type() != &NumericType::Constant {
        return None;
    }
    let rhs = *task
        .get_initial_numeric_state_values()
        .get(effect.var_id())?;
    if !rhs.is_finite() {
        return None;
    }
    match effect.operation() {
        AssignmentOperation::Plus => Some(rhs),
        AssignmentOperation::Minus => Some(-rhs),
        AssignmentOperation::Assign | AssignmentOperation::Times | AssignmentOperation::Divide => {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use planforge_sas::axioms::{AssignmentAxiom, CalOperator};
    use planforge_sas::numeric_task::{
        AssignmentEffect, AssignmentOperation, ExplicitFact, Metric, NumericRootTask, NumericType,
        NumericVariable, Operator,
    };

    use super::*;

    fn affine_sum_task(operation: AssignmentOperation) -> NumericRootTask {
        let numeric_variables = vec![
            NumericVariable::new("x".into(), NumericType::Regular, None),
            NumericVariable::new("y".into(), NumericType::Regular, None),
            NumericVariable::new("x_plus_y".into(), NumericType::Derived, None),
            NumericVariable::new("one".into(), NumericType::Constant, None),
        ];
        let operator = Operator::new(
            "change-x".into(),
            vec![],
            vec![],
            vec![AssignmentEffect::new(0, operation, 3, false, vec![])],
            1,
        );
        NumericRootTask::new(
            4,
            Metric::new(true, None),
            vec![],
            numeric_variables,
            vec![],
            vec![],
            vec![],
            vec![0.0, 0.0, 0.0, 1.0],
            vec![operator],
            vec![],
            vec![],
            vec![AssignmentAxiom::new(2, CalOperator::Sum, 0, 1)],
            ExplicitFact::new(0, 0),
        )
    }

    #[test]
    fn affine_sum_is_an_exact_additive_coordinate() {
        let task = affine_sum_task(AssignmentOperation::Plus);
        let view = analyze_additive_numeric_view(&task, 2).expect("x+y should be additive");

        assert_eq!(view.expression.coefficients, vec![1.0, 1.0, 0.0, 0.0]);
        assert_eq!(view.operator_delta(0).unwrap(), 1.0);
    }

    #[test]
    fn non_additive_effect_rejects_affine_coordinate() {
        let task = affine_sum_task(AssignmentOperation::Assign);

        assert!(analyze_additive_numeric_view(&task, 2).is_none());
        assert!(AdditiveNumericViews::for_active_dimensions(&task, &[1, 1, 2, 1]).is_err());
    }

    #[test]
    fn cost_dependent_expression_is_not_an_additive_view() {
        let numeric_variables = vec![
            NumericVariable::new("x".into(), NumericType::Regular, None),
            NumericVariable::new("accumulated-cost".into(), NumericType::Cost, None),
            NumericVariable::new("x-plus-cost".into(), NumericType::Derived, None),
        ];
        let task = NumericRootTask::new(
            4,
            Metric::new(true, Some(1)),
            vec![],
            numeric_variables,
            vec![],
            vec![],
            vec![],
            vec![0.0, 0.0, 0.0],
            vec![],
            vec![],
            vec![],
            vec![AssignmentAxiom::new(2, CalOperator::Sum, 0, 1)],
            ExplicitFact::new(0, 0),
        );

        assert!(analyze_additive_numeric_view(&task, 2).is_none());
    }
}
