#[cfg(test)]
mod tests;

use std::collections::HashMap;

use anyhow::{Result, bail, ensure};
use planforge_sas::numeric_task::{
    AbstractNumericTask, AssignmentEffect, AssignmentOperation, NumericType, NumericValue,
    Operator, OperatorIndex, VariableIndex, ZERO_VALUE,
};
use planforge_sas::utils::linear_effects::{LinearExpression, linearize_numeric_var};

use planforge_sas::numeric_conditions::{ConditionNode, NumericCondition};

const EPSILON: f64 = 1e-12;

/// A derived affine expression whose value changes by a deterministic constant
/// under every operator. Such an expression is an exact numeric coordinate of
/// the original task and can therefore be partitioned without compiling a new
/// task or losing correlations between its regular numeric dependencies.
#[derive(Clone, Debug)]
pub(crate) struct AdditiveNumericView {
    expression: LinearExpression,
    operator_deltas: Vec<NumericValue>,
}

impl AdditiveNumericView {
    pub(crate) fn operator_delta(&self, operator_id: OperatorIndex) -> Result<NumericValue> {
        self.operator_deltas
            .get(operator_id.index())
            .copied()
            .ok_or_else(|| {
                anyhow::anyhow!("missing additive-view delta for operator {operator_id}")
            })
    }

    pub(crate) fn operator_deltas(&self) -> &[NumericValue] {
        &self.operator_deltas
    }

    pub(crate) fn evaluate(&self, numeric_values: &[NumericValue]) -> NumericValue {
        self.expression.evaluate(numeric_values)
    }
}

pub(crate) fn initial_numeric_values_with_additive_views(
    task: &dyn AbstractNumericTask,
) -> Vec<NumericValue> {
    let mut values = task.get_initial_numeric_state_values().to_vec();
    for numeric_var_id in 0..task.numeric_variables().len() {
        let Some(view) =
            analyze_additive_numeric_view(task, VariableIndex::from_usize(numeric_var_id))
        else {
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
            let view =
                analyze_additive_numeric_view(task, VariableIndex::from_usize(numeric_var_id));
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
    numeric_var_id: VariableIndex,
) -> Option<AdditiveNumericView> {
    if task
        .numeric_variables()
        .get(numeric_var_id.index())?
        .get_type()
        != &NumericType::Derived
    {
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
    numeric_var_id: VariableIndex,
) -> bool {
    fn visit(
        task: &dyn AbstractNumericTask,
        numeric_var_id: VariableIndex,
        visiting: &mut [bool],
    ) -> bool {
        let Some(variable) = task.numeric_variables().get(numeric_var_id.index()) else {
            return true;
        };
        match variable.get_type() {
            NumericType::Cost => true,
            NumericType::Regular | NumericType::Constant => false,
            NumericType::Derived => {
                if visiting[numeric_var_id.index()] {
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
                visiting[numeric_var_id.index()] = true;
                let depends_on_cost = visit(task, axiom.get_left_var_id(), visiting)
                    || visit(task, axiom.get_right_var_id(), visiting);
                visiting[numeric_var_id.index()] = false;
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
    numeric_var_id: VariableIndex,
) -> bool {
    task.numeric_variables()
        .get(numeric_var_id.index())
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
    tree: &NumericCondition,
) -> Vec<VariableIndex> {
    let mut direct = [tree.left_numeric_var_id(), tree.right_numeric_var_id()]
        .into_iter()
        .filter(|&numeric_var_id| is_refinable_numeric_dimension(task, numeric_var_id))
        .collect::<Vec<_>>();
    direct.sort_unstable();
    direct.dedup();
    if !direct.is_empty() {
        return direct;
    }
    tree.regular_numeric_var_dependencies().to_vec()
}

/// Every refined coordinate that can constrain evaluation of `tree`.
pub(crate) fn active_comparison_dimensions(
    tree: &NumericCondition,
    numeric_domain_sizes: &[usize],
    additive_views: &AdditiveNumericViews,
) -> Vec<VariableIndex> {
    let mut dimensions = tree
        .regular_numeric_var_dependencies()
        .iter()
        .copied()
        .filter(|&numeric_var_id| {
            numeric_domain_sizes
                .get(numeric_var_id.index())
                .is_some_and(|&size| size > 1)
        })
        .collect::<Vec<_>>();
    for node in tree.nodes() {
        let ConditionNode::Arith {
            result_numeric_var_id,
            ..
        } = node
        else {
            continue;
        };
        if numeric_domain_sizes
            .get(result_numeric_var_id.index())
            .is_some_and(|&size| size > 1)
            && additive_views.get(result_numeric_var_id.index()).is_some()
        {
            dimensions.push(*result_numeric_var_id);
        }
    }
    dimensions.sort_unstable();
    dimensions.dedup();
    dimensions
}

pub(crate) fn numeric_effect_deltas(
    task: &dyn AbstractNumericTask,
) -> HashMap<VariableIndex, Vec<NumericValue>> {
    let mut deltas: HashMap<VariableIndex, Vec<NumericValue>> = HashMap::new();
    for (numeric_var_id, numeric_var) in task.numeric_variables().iter().enumerate() {
        let numeric_var_id = VariableIndex::from_usize(numeric_var_id);
        match numeric_var.get_type() {
            NumericType::Regular => {
                for operator in task.get_operators() {
                    if let Some(delta) =
                        regular_additive_delta_for_operator(task, numeric_var_id, operator)
                        && delta.value().abs() >= EPSILON
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
                    if delta.value().abs() >= EPSILON {
                        deltas.entry(numeric_var_id).or_default().push(delta);
                    }
                }
            }
            NumericType::Constant | NumericType::Cost => {}
        }
    }
    for values in deltas.values_mut() {
        values.sort_by(|left, right| left.value().total_cmp(&right.value()));
        values.dedup_by(|left, right| (left.value() - right.value()).abs() < EPSILON);
    }
    deltas
}

pub(crate) fn numeric_dimension_delta_for_operator(
    task: &dyn AbstractNumericTask,
    numeric_var_id: VariableIndex,
    operator: &Operator,
) -> Option<NumericValue> {
    match task
        .numeric_variables()
        .get(numeric_var_id.index())?
        .get_type()
    {
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
    numeric_var_id: VariableIndex,
) -> bool {
    task.numeric_variables()
        .get(numeric_var_id.index())
        .is_some_and(|variable| variable.get_type() == &NumericType::Regular)
        && task.get_operators().iter().all(|operator| {
            regular_additive_delta_for_operator(task, numeric_var_id, operator)
                .is_some_and(|delta| delta.value().abs() < EPSILON)
        })
}

fn additive_view_delta_for_operator(
    task: &dyn AbstractNumericTask,
    expression: &LinearExpression,
    operator: &Operator,
) -> Option<NumericValue> {
    let mut delta: f64 = 0.0;
    for (numeric_var_id, &coefficient) in expression.coefficients.iter().enumerate() {
        if coefficient.abs() < EPSILON {
            continue;
        }
        delta += coefficient
            * regular_additive_delta_for_operator(
                task,
                VariableIndex::from_usize(numeric_var_id),
                operator,
            )?
            .value();
    }
    delta.is_finite().then_some(NumericValue::new(delta))
}

fn regular_additive_delta_for_operator(
    task: &dyn AbstractNumericTask,
    numeric_var_id: VariableIndex,
    operator: &Operator,
) -> Option<NumericValue> {
    let mut matching = operator
        .assignment_effects()
        .iter()
        .filter(|effect| effect.affected_var_id() == numeric_var_id);
    let Some(effect) = matching.next() else {
        return Some(ZERO_VALUE);
    };
    if matching.next().is_some() || effect.is_conditional() || !effect.conditions().is_empty() {
        return None;
    }
    constant_effect_delta(task, effect)
}

fn constant_effect_delta(
    task: &dyn AbstractNumericTask,
    effect: &AssignmentEffect,
) -> Option<NumericValue> {
    let rhs_variable = task.numeric_variables().get(effect.var_id().index())?;
    if rhs_variable.get_type() != &NumericType::Constant {
        return None;
    }
    let rhs = *task
        .get_initial_numeric_state_values()
        .get(effect.var_id().index())?;
    if !rhs.value().is_finite() {
        return None;
    }
    match effect.operation() {
        AssignmentOperation::Plus => Some(rhs),
        AssignmentOperation::Minus => Some(NumericValue::new(-rhs.value())),
        AssignmentOperation::Assign | AssignmentOperation::Times | AssignmentOperation::Divide => {
            None
        }
    }
}
