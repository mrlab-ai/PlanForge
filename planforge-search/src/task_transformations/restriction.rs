//! The restricted task: every numeric comparison and the metric read state
//! variables, with derived arithmetic folded into them.

#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet, HashMap};

use anyhow::{Context, Result, anyhow, ensure};
use planforge_sas::axioms::{AssignmentAxiom, ComparisonAxiom};
use planforge_sas::numeric_task::{
    AssignmentEffect, AssignmentOperation, ExplicitVariable, Metric, NumericType, NumericValue,
    NumericVariable, ONE_VALUE, Operator, OperatorIndex, Task, TaskParts, VariableIndex,
    ZERO_VALUE,
};
use tracing::info;

pub fn validate_restricted_task(task: &Task) -> std::result::Result<(), String> {
    if let Some(variable) = task.metric().var_id()
        && task.numeric_type(variable) == NumericType::Derived
    {
        return Err(format!(
            "task is not restricted: metric variable {variable} is derived"
        ));
    }
    for (comparison_axiom_id, comparison_axiom) in task.comparison_axioms().iter().enumerate() {
        for (side, numeric_var_id) in [
            ("left", comparison_axiom.get_left_var_id()),
            ("right", comparison_axiom.get_right_var_id()),
        ] {
            let numeric_var = task.numeric_variables().get(numeric_var_id.index()).ok_or_else(|| {
                format!(
                    "comparison axiom {comparison_axiom_id} has invalid {side} numeric variable {numeric_var_id}"
                )
            })?;
            if numeric_var.numeric_type() == NumericType::Derived {
                return Err(format!(
                    "task is not restricted: comparison axiom {comparison_axiom_id} has derived {side} operand {numeric_var_id}; use `--restrict-task` or provide an already restricted task"
                ));
            }
        }
    }
    for (operator_id, operator) in task.operators().iter().enumerate() {
        for (effect_id, effect) in operator.assignment_effects().iter().enumerate() {
            let affected = task
                .numeric_variables()
                .get(effect.affected_var_id().index())
                .ok_or_else(|| {
                    format!(
                        "operator {operator_id} ({}) numeric effect {effect_id} has invalid target {}",
                        operator.name(),
                        effect.affected_var_id()
                    )
                })?;
            if !matches!(
                affected.numeric_type(),
                NumericType::Regular | NumericType::Cost
            ) {
                return Err(format!(
                    "task is not restricted: operator {operator_id} ({}) numeric effect {effect_id} targets unsupported variable {} ({:?}); use `--restrict-task` or provide an already restricted task",
                    operator.name(),
                    effect.affected_var_id(),
                    affected.numeric_type()
                ));
            }

            let source = task
                .numeric_variables()
                .get(effect.var_id().index())
                .ok_or_else(|| {
                    format!(
                        "operator {operator_id} ({}) numeric effect {effect_id} has invalid source {}",
                        operator.name(),
                        effect.var_id()
                    )
                })?;
            if source.numeric_type() == NumericType::Derived {
                return Err(format!(
                    "task is not restricted: operator {operator_id} ({}) numeric effect {effect_id} reads derived variable {}; use `--restrict-task` or provide an already restricted task",
                    operator.name(),
                    effect.var_id()
                ));
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
struct AffineExpression {
    coefficients: Vec<NumericValue>,
    constant: NumericValue,
}

impl AffineExpression {
    fn zero(num_vars: usize) -> Self {
        Self {
            coefficients: vec![ZERO_VALUE; num_vars],
            constant: ZERO_VALUE,
        }
    }

    fn var(var_id: VariableIndex, num_vars: usize) -> Self {
        let mut expr = Self::zero(num_vars);
        expr.coefficients[var_id.index()] = ONE_VALUE;
        expr
    }

    fn constant(value: NumericValue, num_vars: usize) -> Self {
        let mut expr = Self::zero(num_vars);
        expr.constant = value;
        expr
    }

    fn sub(mut self, rhs: Self) -> Self {
        for (lhs, rhs) in self.coefficients.iter_mut().zip(rhs.coefficients) {
            *lhs -= rhs;
        }
        self.constant -= rhs.constant;
        self
    }

    fn non_zero_vars(&self) -> Vec<VariableIndex> {
        self.coefficients
            .iter()
            .enumerate()
            .filter_map(|(var_id, &coefficient)| {
                (coefficient != ZERO_VALUE).then_some(VariableIndex::from_usize(var_id))
            })
            .collect()
    }

    fn evaluate(&self, numeric_values: &[NumericValue]) -> Result<NumericValue> {
        ensure!(
            numeric_values.len() >= self.coefficients.len(),
            "numeric state too short for transformed task projection: {} < {}",
            numeric_values.len(),
            self.coefficients.len()
        );
        Ok(NumericValue::affine(
            self.coefficients
                .iter()
                .zip(numeric_values)
                .filter(|(coefficient, _)| **coefficient != ZERO_VALUE)
                .map(|(coefficient, value)| (coefficient.value(), *value)),
            self.constant.value(),
        ))
    }

    fn apply_effects(
        &self,
        additive_deltas: &[NumericValue],
        assigned_constants: &[Option<NumericValue>],
    ) -> Self {
        let mut out = self.clone();
        out.constant = NumericValue::affine(
            self.coefficients
                .iter()
                .enumerate()
                .filter_map(|(var_id, &coefficient)| {
                    if coefficient == ZERO_VALUE {
                        return None;
                    }
                    let value = match assigned_constants[var_id] {
                        Some(value) => {
                            out.coefficients[var_id] = ZERO_VALUE;
                            value
                        }
                        None => additive_deltas[var_id],
                    };
                    Some((coefficient.value(), value))
                }),
            self.constant.value(),
        );
        out
    }
}

pub fn build_restricted_task(task: &Task) -> Result<Option<Task>> {
    let restriction_reason = match validate_restricted_task(task) {
        Ok(()) => {
            info!("restricted task: task already satisfies restricted-task invariants");
            return Ok(None);
        }
        Err(reason) => reason,
    };
    let num_numeric_vars = task.numeric_variables().len();
    if num_numeric_vars == 0 {
        crate::unsupported!(
            "task violates restricted-task invariants ({restriction_reason}), but has no numeric comparison roots that can be promoted"
        );
    }

    let initial_numeric = task.initial_numeric_values().to_vec();
    // Every linear comparison is already a direct threshold. Promote each
    // shared multi-variable linear part once.
    let mut root_var_ids = BTreeSet::new();
    let mut root_exprs = BTreeMap::new();
    for comparison_axiom in task.comparison_axioms() {
        let operands = [
            comparison_axiom.get_left_var_id(),
            comparison_axiom.get_right_var_id(),
        ];
        let is_derived = |var_id: VariableIndex| task.numeric_type(var_id) == NumericType::Derived;
        for var_id in operands {
            root_var_ids.insert(var_id);
            if is_derived(var_id) {
                root_exprs.insert(var_id, affine_expression(task, var_id)?);
            }
        }
    }

    let mut numeric_var_ids: Vec<VariableIndex> = root_var_ids
        .iter()
        .copied()
        .filter(|&numeric_var_id| {
            !matches!(task.numeric_type(numeric_var_id), NumericType::Derived)
        })
        .collect();
    if let Some(metric_var_id) = task.metric().var_id() {
        if let Some(expr) = derived_metric_expression(task, metric_var_id)? {
            crate::ensure_supported!(
                !root_exprs.contains_key(&metric_var_id),
                "metric variable {metric_var_id} is also a numeric comparison operand"
            );
            root_exprs.insert(metric_var_id, expr);
        }
        numeric_var_ids.push(metric_var_id);
    }
    numeric_var_ids.extend(root_exprs.keys().copied());
    numeric_var_ids.sort_unstable();
    numeric_var_ids.dedup();

    build_task(task, &initial_numeric, &root_exprs, &numeric_var_ids)
}

fn build_task(
    task: &Task,
    initial_numeric: &[NumericValue],
    root_exprs: &BTreeMap<VariableIndex, AffineExpression>,
    numeric_var_ids: &[VariableIndex],
) -> Result<Option<Task>> {
    let num_original_numeric = task.numeric_variables().len();
    let mut original_to_transformed = vec![None; num_original_numeric];
    let mut transformed_to_expr = Vec::new();
    let mut numeric_variables = Vec::new();
    let mut numeric_initial = Vec::new();

    for &original_id in numeric_var_ids {
        let expr = root_exprs.get(&original_id).cloned().unwrap_or_else(|| {
            match task.numeric_type(original_id) {
                NumericType::Constant => AffineExpression::constant(
                    initial_numeric[original_id.index()],
                    num_original_numeric,
                ),
                NumericType::Cost => AffineExpression::var(original_id, num_original_numeric),
                _ => AffineExpression::var(original_id, num_original_numeric),
            }
        });
        let transformed_id = numeric_variables.len();
        original_to_transformed[original_id.index()] =
            Some(VariableIndex::from_usize(transformed_id));
        let original_name = task.numeric_variables()[original_id.index()].name();
        let (name, numeric_type) = if task.metric().var_id() == Some(original_id) {
            let numeric_type = if root_exprs.contains_key(&original_id) {
                NumericType::Cost
            } else {
                task.numeric_type(original_id)
            };
            (original_name.to_string(), numeric_type)
        } else if root_exprs.contains_key(&original_id) {
            (
                format!("{}|{}", restricted_shape_prefix(&expr), original_name),
                NumericType::Regular,
            )
        } else {
            (original_name.to_string(), task.numeric_type(original_id))
        };
        numeric_variables.push(NumericVariable::new(name, numeric_type, None));
        numeric_initial.push(expr.evaluate(initial_numeric)?);
        transformed_to_expr.push(expr);
    }

    let mut constants = ConstantPool::default();
    let mut operators = Vec::with_capacity(task.operators().len());
    for operator in task.operators() {
        operators.push(transform_operator(
            task,
            operator,
            initial_numeric,
            root_exprs,
            &original_to_transformed,
            &mut TransformedNumerics {
                constants: &mut constants,
                variables: &mut numeric_variables,
                initial: &mut numeric_initial,
                to_expr: &mut transformed_to_expr,
            },
        )?);
    }

    let transformed = |var_id: VariableIndex| {
        original_to_transformed[var_id.index()]
            .expect("every comparison operand is a restricted numeric variable")
    };
    let comparison_axioms = task
        .comparison_axioms()
        .iter()
        .map(|comparison| {
            ComparisonAxiom::new(
                comparison.get_affected_var_id(),
                transformed(comparison.get_left_var_id()),
                transformed(comparison.get_right_var_id()),
                comparison.get_operator().clone(),
            )
        })
        .collect::<Vec<_>>();
    let variables = renumber_propositional_axiom_layers(task, &comparison_axioms);

    let metric_var_id = task.metric().var_id().and_then(|var_id| {
        original_to_transformed
            .get(var_id.index())
            .and_then(|mapped| *mapped)
    });

    let transformed_task = Task::new(TaskParts {
        metric: Metric::new(task.metric().is_min(), metric_var_id),
        variables,
        numeric_variables,
        goals: task.goals().to_vec(),
        mutexes: task.mutexes().to_vec(),
        state: task.initial_propositional_values().to_vec(),
        numeric_state: numeric_initial,
        operators,
        axioms: task.axioms().to_vec(),
        comparison_axioms,
        assignment_axioms: Vec::<AssignmentAxiom>::new(),
        global_constraint: None,
    });
    validate_restricted_task(&transformed_task)
        .map_err(|reason| anyhow::anyhow!("restricted task construction failed: {reason}"))?;

    ensure_operator_costs_unchanged(task, &transformed_task)?;

    Ok(Some(transformed_task))
}

/// The affine expression of a metric that an assignment axiom derives, as in
/// `(:metric minimize (+ (labour) (pollution)))`.
///
/// Restricted tasks carry no assignment axioms, so such a metric would never
/// change and every operator would cost 0. The caller keeps it as a cost
/// variable backed by this expression, whose effects are the metric's
/// increments.
fn derived_metric_expression(
    task: &Task,
    metric_var_id: VariableIndex,
) -> Result<Option<AffineExpression>> {
    if !matches!(task.numeric_type(metric_var_id), NumericType::Derived) {
        return Ok(None);
    }
    affine_expression(task, metric_var_id)
        .with_context(|| format!("metric variable {metric_var_id} is not affine"))
        .map(Some)
}

fn ensure_operator_costs_unchanged(source_task: &Task, transformed_task: &Task) -> Result<()> {
    ensure!(
        source_task.operators().len() == transformed_task.operators().len(),
        "restricted task changed the operator count"
    );
    for (operator_id, source) in source_task.operators().iter().enumerate() {
        let operator_index = OperatorIndex::from_usize(operator_id);
        let source_cost = source_task.operator_cost(operator_index);
        let transformed_cost = transformed_task.operator_cost(operator_index);
        ensure!(
            (source_cost - transformed_cost).abs()
                <= planforge_sas::utils::float_tolerance::AGREEMENT_EPSILON
                    * source_cost.abs().max(transformed_cost.abs()).max(1.0),
            "restricted task changed metric cost of operator {operator_id} ({}): {source_cost} -> {transformed_cost}",
            source.name()
        );
    }
    Ok(())
}

fn renumber_propositional_axiom_layers(
    task: &Task,
    comparison_axioms: &[ComparisonAxiom],
) -> Vec<ExplicitVariable> {
    if comparison_axioms.is_empty() {
        return task.variables().to_vec();
    }

    let comparison_vars = comparison_axioms
        .iter()
        .map(|axiom| axiom.get_affected_var_id())
        .collect::<BTreeSet<_>>();
    let remaining_layers = task
        .variables()
        .iter()
        .enumerate()
        .filter_map(|(var_id, variable)| {
            (!comparison_vars.contains(&VariableIndex::from_usize(var_id)))
                .then_some(variable.axiom_layer())
                .flatten()
        })
        .collect::<BTreeSet<_>>();
    let layer_map = remaining_layers
        .into_iter()
        .enumerate()
        .map(|(index, layer)| (layer, index + 1))
        .collect::<BTreeMap<_, _>>();

    task.variables()
        .iter()
        .enumerate()
        .map(|(var_id, variable)| {
            let new_layer = if comparison_vars.contains(&VariableIndex::from_usize(var_id)) {
                Some(0)
            } else {
                variable
                    .axiom_layer()
                    .map(|layer| *layer_map.get(&layer).expect("layer map is complete"))
            };
            variable.with_axiom_layer(new_layer)
        })
        .collect()
}

fn transform_operator(
    task: &Task,
    operator: &Operator,
    initial_numeric: &[NumericValue],
    root_exprs: &BTreeMap<VariableIndex, AffineExpression>,
    original_to_transformed: &[Option<VariableIndex>],
    numerics: &mut TransformedNumerics<'_>,
) -> Result<Operator> {
    let has_assignment = operator
        .assignment_effects()
        .iter()
        .any(|effect| matches!(effect.operation(), AssignmentOperation::Assign));
    if has_assignment {
        return transform_operator_with_assignment(
            task,
            operator,
            initial_numeric,
            original_to_transformed,
            numerics,
        );
    }

    let mut deltas = vec![ZERO_VALUE; task.numeric_variables().len()];
    for effect in operator.assignment_effects() {
        crate::ensure_supported!(
            !effect.is_conditional() && effect.conditions().is_empty(),
            "restricted task does not support conditional numeric effects in operator {}",
            operator.name()
        );
        let source_value = match task.numeric_type(effect.var_id()) {
            NumericType::Constant => initial_numeric[effect.var_id().index()],
            _ => crate::unsupported!(
                "restricted task requires constant RHS numeric effects in operator {}",
                operator.name()
            ),
        };
        match effect.operation() {
            AssignmentOperation::Plus => deltas[effect.affected_var_id().index()] += source_value,
            AssignmentOperation::Minus => deltas[effect.affected_var_id().index()] -= source_value,
            AssignmentOperation::Assign => crate::unsupported!(
                "restricted task does not support assignment numeric effects in operator {}",
                operator.name()
            ),
            AssignmentOperation::Times | AssignmentOperation::Divide => crate::unsupported!(
                "restricted task does not support non-additive numeric effects in operator {}",
                operator.name()
            ),
        }
    }

    let transformed_deltas = original_to_transformed
        .iter()
        .enumerate()
        .filter_map(|(original_id, &mapped)| {
            mapped.map(|transformed_id| {
                let delta = root_exprs
                    .get(&VariableIndex::from_usize(original_id))
                    .map(|expr| {
                        NumericValue::affine(
                            expr.coefficients
                                .iter()
                                .zip(&deltas)
                                .map(|(coefficient, delta)| (coefficient.value(), *delta)),
                            0.0,
                        )
                    })
                    .unwrap_or(deltas[original_id]);
                (transformed_id, delta)
            })
        })
        .collect::<Vec<_>>();
    let mut assignment_effects = Vec::new();
    for (transformed_id, delta) in transformed_deltas {
        if delta == ZERO_VALUE {
            continue;
        }
        let const_id = numerics.intern_constant(delta, task.numeric_variables().len());
        assignment_effects.push(AssignmentEffect::new(
            transformed_id,
            AssignmentOperation::Plus,
            const_id,
            false,
            vec![],
        ));
    }

    Ok(Operator::new(
        operator.name().to_string(),
        operator.preconditions().clone(),
        operator.effects().clone(),
        assignment_effects,
        operator.cost(),
    ))
}

fn transform_operator_with_assignment(
    task: &Task,
    operator: &Operator,
    initial_numeric: &[NumericValue],
    original_to_transformed: &[Option<VariableIndex>],
    numerics: &mut TransformedNumerics<'_>,
) -> Result<Operator> {
    let mut additive_deltas = vec![ZERO_VALUE; task.numeric_variables().len()];
    let mut assigned_constants = vec![None; task.numeric_variables().len()];
    for effect in operator.assignment_effects() {
        crate::ensure_supported!(
            !effect.is_conditional() && effect.conditions().is_empty(),
            "restricted task does not support conditional numeric effects in operator {}",
            operator.name()
        );
        let source_value = match task.numeric_type(effect.var_id()) {
            NumericType::Constant => initial_numeric[effect.var_id().index()],
            _ => crate::unsupported!(
                "restricted task requires constant RHS numeric effects in operator {}",
                operator.name()
            ),
        };
        match effect.operation() {
            AssignmentOperation::Plus => {
                crate::ensure_supported!(
                    assigned_constants[effect.affected_var_id().index()].is_none(),
                    "restricted task does not support mixed assignment/additive numeric effects on one variable in operator {}",
                    operator.name()
                );
                additive_deltas[effect.affected_var_id().index()] += source_value;
            }
            AssignmentOperation::Minus => {
                crate::ensure_supported!(
                    assigned_constants[effect.affected_var_id().index()].is_none(),
                    "restricted task does not support mixed assignment/additive numeric effects on one variable in operator {}",
                    operator.name()
                );
                additive_deltas[effect.affected_var_id().index()] -= source_value;
            }
            AssignmentOperation::Assign => {
                crate::ensure_supported!(
                    (additive_deltas[effect.affected_var_id().index()] == ZERO_VALUE)
                        && assigned_constants[effect.affected_var_id().index()].is_none(),
                    "restricted task does not support multiple numeric effects on an assigned variable in operator {}",
                    operator.name()
                );
                assigned_constants[effect.affected_var_id().index()] = Some(source_value);
            }
            AssignmentOperation::Times | AssignmentOperation::Divide => crate::unsupported!(
                "restricted task does not support non-additive numeric effects in operator {}",
                operator.name()
            ),
        }
    }

    let coordinate_ids = original_to_transformed
        .iter()
        .filter_map(|&mapped| mapped)
        .collect::<Vec<_>>();
    let transformed_effects = coordinate_ids
        .into_iter()
        .map(|transformed_id| {
            let expr = &numerics.to_expr[transformed_id.index()];
            let successor_expr = expr.apply_effects(&additive_deltas, &assigned_constants);
            let delta_expr = successor_expr.clone().sub(expr.clone());
            (transformed_id, successor_expr, delta_expr)
        })
        .collect::<Vec<(VariableIndex, AffineExpression, AffineExpression)>>();
    let mut assignment_effects = Vec::new();
    for (transformed_id, successor_expr, delta_expr) in transformed_effects {
        if delta_expr.non_zero_vars().is_empty() {
            let delta = delta_expr.constant;
            if delta == ZERO_VALUE {
                continue;
            }
            let const_id = numerics.intern_constant(delta, task.numeric_variables().len());
            assignment_effects.push(AssignmentEffect::new(
                transformed_id,
                AssignmentOperation::Plus,
                const_id,
                false,
                vec![],
            ));
            continue;
        }
        crate::ensure_supported!(
            successor_expr.non_zero_vars().is_empty(),
            "restricted task cannot express assignment effect on transformed numeric variable {transformed_id} in operator {}",
            operator.name()
        );
        let const_id =
            numerics.intern_constant(successor_expr.constant, task.numeric_variables().len());
        assignment_effects.push(AssignmentEffect::new(
            transformed_id,
            AssignmentOperation::Assign,
            const_id,
            false,
            vec![],
        ));
    }

    Ok(Operator::new(
        operator.name().to_string(),
        operator.preconditions().clone(),
        operator.effects().clone(),
        assignment_effects,
        operator.cost(),
    ))
}

#[derive(Default)]
struct ConstantPool {
    by_value: HashMap<NumericValue, VariableIndex>,
}

/// The numeric side of the task under construction.
///
/// `variables`, `initial` and `to_expr` are parallel — one entry per
/// transformed numeric variable — and interning a constant appends to all
/// three, so an operator transform always needs the whole set rather than any
/// one of them.
struct TransformedNumerics<'a> {
    constants: &'a mut ConstantPool,
    variables: &'a mut Vec<NumericVariable>,
    initial: &'a mut Vec<NumericValue>,
    to_expr: &'a mut Vec<AffineExpression>,
}

impl TransformedNumerics<'_> {
    /// Id of the constant numeric variable holding `value`, creating it on
    /// first use.
    fn intern_constant(
        &mut self,
        value: NumericValue,
        num_original_numeric: usize,
    ) -> VariableIndex {
        self.constants.get_or_insert(
            value,
            self.variables,
            self.initial,
            self.to_expr,
            num_original_numeric,
        )
    }
}

impl ConstantPool {
    fn get_or_insert(
        &mut self,
        value: NumericValue,
        numeric_variables: &mut Vec<NumericVariable>,
        numeric_initial: &mut Vec<NumericValue>,
        transformed_to_expr: &mut Vec<AffineExpression>,
        num_original_numeric: usize,
    ) -> VariableIndex {
        if let Some(&id) = self.by_value.get(&value) {
            return id;
        }
        let id = VariableIndex::from_usize(numeric_variables.len());
        numeric_variables.push(NumericVariable::new(
            format!("restricted-const-{id}"),
            NumericType::Constant,
            None,
        ));
        numeric_initial.push(value);
        transformed_to_expr.push(AffineExpression::constant(value, num_original_numeric));
        self.by_value.insert(value, id);
        id
    }
}

fn affine_expression(task: &Task, variable: VariableIndex) -> Result<AffineExpression> {
    let expression = task
        .linearize_numeric_var(variable)
        .map_err(|error| anyhow!("{error}"))?;
    Ok(AffineExpression {
        coefficients: expression
            .coefficients
            .into_iter()
            .map(NumericValue::new)
            .collect(),
        constant: NumericValue::new(expression.constant),
    })
}

fn restricted_shape_prefix(expr: &AffineExpression) -> String {
    let mut coefficients = expr
        .coefficients
        .iter()
        .copied()
        .filter(|&coefficient| coefficient != ZERO_VALUE)
        .collect::<Vec<_>>();
    coefficients.sort();
    let shape = coefficients
        .iter()
        .map(|coefficient| coefficient.to_string())
        .collect::<Vec<_>>()
        .join(",");
    format!("rt-shape:{shape}")
}
