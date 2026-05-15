#[cfg(test)]
mod tests;

use std::collections::{HashMap, HashSet};

use planforge_sas::numeric::axioms::ComparisonAxiom;
use planforge_sas::numeric::numeric_task::{
    AbstractNumericTask, AssignmentOperation, NumericType, Operator,
};

use crate::numeric::evaluation::domain_abstractions::comparison_expression::{
    ArithOp, ComparisonTree, ComparisonTreeNode,
};

use super::projected_task::{
    AuxiliaryNumericVar, build_assignment_axiom_lookup, build_auxiliary_numeric_vars,
};

#[derive(Debug, Clone, Copy)]
struct NumericCondition {
    var_id: usize,
    constant: f64,
}

#[derive(Debug, Clone, Copy)]
struct LinearizedExpr {
    var_id: Option<usize>,
    multiplier: f64,
    summand: f64,
}

impl LinearizedExpr {
    fn constant(value: f64) -> Self {
        Self {
            var_id: None,
            multiplier: 0.0,
            summand: value,
        }
    }

    fn variable(var_id: usize) -> Self {
        Self {
            var_id: Some(var_id),
            multiplier: 1.0,
            summand: 0.0,
        }
    }

    fn is_constant(self) -> bool {
        self.var_id.is_none()
    }
}

pub struct NumericSizeEstimator {
    approximate_domain_sizes: Vec<usize>,
}

impl NumericSizeEstimator {
    pub fn new(task: &dyn AbstractNumericTask) -> Self {
        let assignment_lookup = build_assignment_axiom_lookup(task);
        let base_initial_numeric_values = task.get_initial_numeric_state_values().to_vec();
        let auxiliary_numeric_vars =
            build_auxiliary_numeric_vars(task, &assignment_lookup, &base_initial_numeric_values)
                .unwrap_or_default();

        let conditions =
            collect_numeric_conditions(task, &base_initial_numeric_values, &auxiliary_numeric_vars);
        let helper_space_len = task.numeric_variables().len() + auxiliary_numeric_vars.len();

        Self {
            approximate_domain_sizes: (0..helper_space_len)
                .map(|numeric_var_id| {
                    estimate_numeric_domain_size(
                        task,
                        numeric_var_id,
                        &base_initial_numeric_values,
                        &auxiliary_numeric_vars,
                        &conditions,
                    )
                })
                .collect(),
        }
    }

    pub fn estimate_domain_size(&self, numeric_var_id: usize) -> usize {
        self.approximate_domain_sizes
            .get(numeric_var_id)
            .copied()
            .unwrap_or(1)
            .max(1)
    }

    pub fn helper_space_len(&self) -> usize {
        self.approximate_domain_sizes.len()
    }
}

fn collect_numeric_conditions(
    task: &dyn AbstractNumericTask,
    base_initial_numeric_values: &[f64],
    auxiliary_numeric_vars: &[AuxiliaryNumericVar],
) -> Vec<NumericCondition> {
    let mut comparison_axioms_by_var = HashMap::new();
    let mut numeric_condition_vars = HashSet::new();

    for (comparison_axiom_id, comparison_axiom) in task.comparison_axioms().iter().enumerate() {
        let affected_var_id = comparison_axiom.get_affected_var_id();
        comparison_axioms_by_var.insert(affected_var_id, comparison_axiom_id);
        numeric_condition_vars.insert(affected_var_id);
    }

    let mut conditions = Vec::new();
    for operator in task.get_operators() {
        for fact in operator.preconditions() {
            let fact_var_id = fact.var();
            if !numeric_condition_vars.contains(&fact_var_id) {
                continue;
            }
            if let Some(&comparison_axiom_id) = comparison_axioms_by_var.get(&fact_var_id)
                && let Some(condition) = build_numeric_condition(
                    task,
                    comparison_axiom_id,
                    base_initial_numeric_values,
                    auxiliary_numeric_vars,
                )
            {
                conditions.push(condition);
            }
        }
    }

    for goal_id in 0..task.get_num_goals() {
        let goal = task.get_goal_fact(goal_id);
        let goal_var_id = goal.var();
        if !numeric_condition_vars.contains(&goal_var_id) {
            continue;
        }
        if let Some(&comparison_axiom_id) = comparison_axioms_by_var.get(&goal_var_id)
            && let Some(condition) = build_numeric_condition(
                task,
                comparison_axiom_id,
                base_initial_numeric_values,
                auxiliary_numeric_vars,
            )
        {
            conditions.push(condition);
        }
    }

    conditions
}

fn build_numeric_condition(
    task: &dyn AbstractNumericTask,
    comparison_axiom_id: usize,
    base_initial_numeric_values: &[f64],
    auxiliary_numeric_vars: &[AuxiliaryNumericVar],
) -> Option<NumericCondition> {
    let comparison_axiom = task.comparison_axioms().get(comparison_axiom_id)?;
    if let Some(condition) = build_helper_numeric_condition_from_axiom(
        task,
        comparison_axiom,
        base_initial_numeric_values,
        auxiliary_numeric_vars,
    ) {
        return Some(condition);
    }

    let tree = ComparisonTree::from_task(task, comparison_axiom_id).ok()?;
    let lhs = linearize_tree_node(
        task,
        &tree,
        tree.left_root,
        base_initial_numeric_values,
        auxiliary_numeric_vars,
    )?;
    let rhs = linearize_tree_node(
        task,
        &tree,
        tree.right_root,
        base_initial_numeric_values,
        auxiliary_numeric_vars,
    )?;

    match (lhs.var_id, rhs.var_id) {
        (Some(_), None) => build_variable_to_constant_condition(lhs, rhs.summand),
        (None, Some(_)) => build_variable_to_constant_condition(rhs, lhs.summand),
        _ => None,
    }
}

fn build_helper_numeric_condition_from_axiom(
    task: &dyn AbstractNumericTask,
    comparison_axiom: &ComparisonAxiom,
    base_initial_numeric_values: &[f64],
    auxiliary_numeric_vars: &[AuxiliaryNumericVar],
) -> Option<NumericCondition> {
    let left_var_id = comparison_axiom.get_left_var_id();
    let right_var_id = comparison_axiom.get_right_var_id();

    helper_condition_from_side_pair(
        task,
        left_var_id,
        right_var_id,
        base_initial_numeric_values,
        auxiliary_numeric_vars,
    )
    .or_else(|| {
        helper_condition_from_side_pair(
            task,
            right_var_id,
            left_var_id,
            base_initial_numeric_values,
            auxiliary_numeric_vars,
        )
    })
}

fn helper_condition_from_side_pair(
    task: &dyn AbstractNumericTask,
    candidate_derived_var_id: usize,
    candidate_constant_var_id: usize,
    base_initial_numeric_values: &[f64],
    auxiliary_numeric_vars: &[AuxiliaryNumericVar],
) -> Option<NumericCondition> {
    if task
        .numeric_variables()
        .get(candidate_derived_var_id)?
        .get_type()
        != &NumericType::Derived
    {
        return None;
    }

    if !matches!(
        task.numeric_variables()
            .get(candidate_constant_var_id)?
            .get_type(),
        NumericType::Constant | NumericType::Cost
    ) {
        return None;
    }

    let helper_var_id = auxiliary_numeric_vars
        .iter()
        .find(|auxiliary_numeric_var| {
            auxiliary_numeric_var.source_numeric_var_id == candidate_derived_var_id
        })?
        .helper_id;

    let constant = *base_initial_numeric_values.get(candidate_constant_var_id)?;
    Some(NumericCondition {
        var_id: helper_var_id,
        constant,
    })
}

fn build_variable_to_constant_condition(
    variable_expr: LinearizedExpr,
    constant_expr_value: f64,
) -> Option<NumericCondition> {
    let var_id = variable_expr.var_id?;
    if variable_expr.multiplier.abs() <= 1e-12 {
        return None;
    }
    let constant = (constant_expr_value - variable_expr.summand) / variable_expr.multiplier;
    Some(NumericCondition { var_id, constant })
}

fn linearize_tree_node(
    task: &dyn AbstractNumericTask,
    tree: &ComparisonTree,
    node_id: usize,
    base_initial_numeric_values: &[f64],
    auxiliary_numeric_vars: &[AuxiliaryNumericVar],
) -> Option<LinearizedExpr> {
    match &tree.nodes[node_id] {
        ComparisonTreeNode::Leaf { numeric_var_id } => {
            let var_id = *numeric_var_id;
            match task.numeric_variables().get(var_id)?.get_type() {
                NumericType::Regular => Some(LinearizedExpr::variable(var_id)),
                NumericType::Constant | NumericType::Cost => Some(LinearizedExpr::constant(
                    *base_initial_numeric_values.get(var_id)?,
                )),
                NumericType::Derived => auxiliary_numeric_vars
                    .iter()
                    .find(|auxiliary_numeric_var| {
                        auxiliary_numeric_var.source_numeric_var_id == var_id
                    })
                    .map(|auxiliary_numeric_var| {
                        LinearizedExpr::variable(auxiliary_numeric_var.helper_id)
                    }),
            }
        }
        ComparisonTreeNode::Arith {
            op, left, right, ..
        } => {
            let lhs = linearize_tree_node(
                task,
                tree,
                *left,
                base_initial_numeric_values,
                auxiliary_numeric_vars,
            )?;
            let rhs = linearize_tree_node(
                task,
                tree,
                *right,
                base_initial_numeric_values,
                auxiliary_numeric_vars,
            )?;
            combine_linearized_exprs(lhs, *op, rhs)
        }
    }
}

fn combine_linearized_exprs(
    lhs: LinearizedExpr,
    op: ArithOp,
    rhs: LinearizedExpr,
) -> Option<LinearizedExpr> {
    match op {
        ArithOp::Add => combine_add_sub(lhs, rhs, 1.0),
        ArithOp::Sub => combine_add_sub(lhs, rhs, -1.0),
        ArithOp::Mul => {
            if lhs.is_constant() {
                Some(scale_linearized_expr(rhs, lhs.summand))
            } else if rhs.is_constant() {
                Some(scale_linearized_expr(lhs, rhs.summand))
            } else {
                None
            }
        }
        ArithOp::Div => {
            if rhs.is_constant() && rhs.summand.abs() > 1e-12 {
                Some(scale_linearized_expr(lhs, 1.0 / rhs.summand))
            } else {
                None
            }
        }
    }
}

fn combine_add_sub(
    lhs: LinearizedExpr,
    rhs: LinearizedExpr,
    rhs_sign: f64,
) -> Option<LinearizedExpr> {
    match (lhs.var_id, rhs.var_id) {
        (None, None) => Some(LinearizedExpr::constant(
            lhs.summand + rhs_sign * rhs.summand,
        )),
        (Some(var_id), None) => Some(LinearizedExpr {
            var_id: Some(var_id),
            multiplier: lhs.multiplier,
            summand: lhs.summand + rhs_sign * rhs.summand,
        }),
        (None, Some(var_id)) => Some(LinearizedExpr {
            var_id: Some(var_id),
            multiplier: rhs_sign * rhs.multiplier,
            summand: lhs.summand + rhs_sign * rhs.summand,
        }),
        (Some(lhs_var_id), Some(rhs_var_id)) if lhs_var_id == rhs_var_id => Some(LinearizedExpr {
            var_id: Some(lhs_var_id),
            multiplier: lhs.multiplier + rhs_sign * rhs.multiplier,
            summand: lhs.summand + rhs_sign * rhs.summand,
        }),
        _ => None,
    }
}

fn scale_linearized_expr(expr: LinearizedExpr, factor: f64) -> LinearizedExpr {
    LinearizedExpr {
        var_id: expr.var_id,
        multiplier: expr.multiplier * factor,
        summand: expr.summand * factor,
    }
}

fn estimate_numeric_domain_size(
    task: &dyn AbstractNumericTask,
    numeric_var_id: usize,
    base_initial_numeric_values: &[f64],
    auxiliary_numeric_vars: &[AuxiliaryNumericVar],
    conditions: &[NumericCondition],
) -> usize {
    let helper_space_len = task.numeric_variables().len() + auxiliary_numeric_vars.len();
    if numeric_var_id >= helper_space_len {
        return 1;
    }

    let mut increments = Vec::new();
    let mut decrements = Vec::new();
    let mut min_const = 0.0_f64;
    let mut max_const = 0.0_f64;
    let mut min_change = f64::INFINITY;
    let mut max_pos_change = 0.0_f64;
    let mut max_neg_change = 0.0_f64;

    for operator in task.get_operators() {
        if let Some(additive_effects) = approximate_additive_effects(
            task,
            operator,
            base_initial_numeric_values,
            auxiliary_numeric_vars,
        ) {
            let effect = additive_effects[numeric_var_id];
            if effect > 0.0 {
                increments.push(effect);
                min_change = min_change.min(effect);
                max_pos_change = max_pos_change.max(effect);
            } else if effect < 0.0 {
                decrements.push(effect);
                min_change = min_change.min(effect.abs());
                max_neg_change = max_neg_change.min(effect);
            }
        }

        if numeric_var_id < task.numeric_variables().len() {
            for assignment_effect in operator.assignment_effects() {
                if assignment_effect.affected_var_id() != numeric_var_id {
                    continue;
                }
                if assignment_effect.operation() != &AssignmentOperation::Assign {
                    continue;
                }
                let source_var_id = assignment_effect.var_id();
                if source_var_id >= base_initial_numeric_values.len() {
                    continue;
                }
                if !matches!(
                    task.numeric_variables()[source_var_id].get_type(),
                    NumericType::Constant | NumericType::Cost
                ) {
                    continue;
                }
                let assigned_value = base_initial_numeric_values[source_var_id];
                min_const = min_const.min(assigned_value);
                max_const = max_const.max(assigned_value);
            }
        }
    }

    let initial_value = if numeric_var_id < task.numeric_variables().len() {
        base_initial_numeric_values[numeric_var_id]
    } else {
        auxiliary_numeric_vars[numeric_var_id - task.numeric_variables().len()].initial_value
    };
    min_const = min_const.min(initial_value);
    max_const = max_const.max(initial_value);

    for condition in conditions {
        if condition.var_id == numeric_var_id {
            min_const = min_const.min(condition.constant);
            max_const = max_const.max(condition.constant);
        }
    }

    min_const += max_neg_change;
    max_const += max_pos_change;

    let min_increment = if !increments.is_empty() && !decrements.is_empty() {
        let mut best_increment = f64::INFINITY;
        for increment in &increments {
            for decrement in &decrements {
                best_increment = best_increment.min((increment + decrement).abs());
            }
        }
        if best_increment <= 1e-12 {
            min_change
        } else {
            best_increment
        }
    } else {
        min_change
    };

    if !min_increment.is_finite() || min_increment <= 1e-12 {
        return 1;
    }

    ((((max_const - min_const).abs() / min_increment) + 1.0) as usize).max(1)
}

fn approximate_additive_effects(
    task: &dyn AbstractNumericTask,
    operator: &Operator,
    base_initial_numeric_values: &[f64],
    auxiliary_numeric_vars: &[AuxiliaryNumericVar],
) -> Option<Vec<f64>> {
    let helper_space_len = task.numeric_variables().len() + auxiliary_numeric_vars.len();
    let mut additive_effects = vec![0.0; helper_space_len];
    let mut has_assign_effect = false;

    for effect in operator.assignment_effects() {
        if effect.is_conditional() {
            return None;
        }

        let source_var_id = effect.var_id();
        if source_var_id >= task.numeric_variables().len() {
            return None;
        }
        let source_type = task.numeric_variables()[source_var_id].get_type();
        let source_value = base_initial_numeric_values[source_var_id];
        let affected_var_id = effect.affected_var_id();
        if affected_var_id >= task.numeric_variables().len() {
            return None;
        }

        match effect.operation() {
            AssignmentOperation::Assign => {
                has_assign_effect = true;
            }
            AssignmentOperation::Plus => {
                if source_type != &NumericType::Constant {
                    return None;
                }
                additive_effects[affected_var_id] += source_value;
            }
            AssignmentOperation::Minus => {
                if source_type != &NumericType::Constant {
                    return None;
                }
                additive_effects[affected_var_id] -= source_value;
            }
            AssignmentOperation::Times | AssignmentOperation::Divide => {
                return None;
            }
        }
    }

    if has_assign_effect {
        return None;
    }

    for auxiliary_numeric_var in auxiliary_numeric_vars {
        additive_effects[auxiliary_numeric_var.helper_id] = auxiliary_numeric_var
            .expr
            .evaluate_ignore_additive_consts(&additive_effects);
    }

    Some(additive_effects)
}
