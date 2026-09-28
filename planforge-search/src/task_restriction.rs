#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet, HashMap};

use anyhow::{Context, Result, anyhow, bail, ensure};
use planforge_sas::axioms::{AssignmentAxiom, CalOperator, ComparisonAxiom, ComparisonOperator};
use planforge_sas::numeric_task::{
    AbstractNumericTask, AssignmentEffect, AssignmentOperation, AxiomIndex, ExplicitFact,
    ExplicitVariable, Metric, NumericRootTask, NumericRootTaskParts, NumericTaskExt, NumericType,
    NumericValue, NumericVariable, Operator, VariableIndex,
    metric_operator_cost_from_initial_values,
};
use planforge_sas::utils::float_tolerance;
use tracing::info;

#[derive(Debug)]
pub struct RestrictedTask {
    task: NumericRootTask,
}

impl RestrictedTask {
    pub fn task(&self) -> &NumericRootTask {
        &self.task
    }

    pub fn into_task(self) -> NumericRootTask {
        self.task
    }
}

pub fn validate_restricted_task(task: &dyn AbstractNumericTask) -> std::result::Result<(), String> {
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
            if numeric_var.get_type() == &NumericType::Derived {
                return Err(format!(
                    "task is not restricted: comparison axiom {comparison_axiom_id} has derived {side} operand {numeric_var_id}; use `--restrict-task` or provide an already restricted task"
                ));
            }
        }
    }
    for (operator_id, operator) in task.get_operators().iter().enumerate() {
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
                affected.get_type(),
                NumericType::Regular | NumericType::Cost
            ) {
                return Err(format!(
                    "task is not restricted: operator {operator_id} ({}) numeric effect {effect_id} targets unsupported variable {} ({:?}); use `--restrict-task` or provide an already restricted task",
                    operator.name(),
                    effect.affected_var_id(),
                    affected.get_type()
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
            if source.get_type() == &NumericType::Derived {
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
            coefficients: vec![NumericValue::new(0.0); num_vars],
            constant: NumericValue::new(0.0),
        }
    }

    fn var(var_id: VariableIndex, num_vars: usize) -> Self {
        let mut expr = Self::zero(num_vars);
        expr.coefficients[var_id.index()] = NumericValue::new(1.0);
        expr
    }

    fn constant(value: NumericValue, num_vars: usize) -> Self {
        let mut expr = Self::zero(num_vars);
        expr.constant = value;
        expr
    }

    fn add(mut self, rhs: Self) -> Self {
        for (lhs, rhs) in self.coefficients.iter_mut().zip(rhs.coefficients) {
            *lhs = NumericValue::new(lhs.value() + rhs.value());
        }
        self.constant = NumericValue::new(self.constant.value() + rhs.constant.value());
        self
    }

    fn sub(mut self, rhs: Self) -> Self {
        for (lhs, rhs) in self.coefficients.iter_mut().zip(rhs.coefficients) {
            *lhs = NumericValue::new(lhs.value() - rhs.value());
        }
        self.constant = NumericValue::new(self.constant.value() - rhs.constant.value());
        self
    }

    fn scale(mut self, factor: f64) -> Self {
        for coefficient in &mut self.coefficients {
            *coefficient = NumericValue::new(coefficient.value() * factor);
        }
        self.constant = NumericValue::new(self.constant.value() * factor);
        self
    }

    fn non_zero_vars(&self) -> Vec<VariableIndex> {
        self.coefficients
            .iter()
            .enumerate()
            .filter_map(|(var_id, &coefficient)| {
                (!approx_eq(coefficient.value(), 0.0)).then_some(VariableIndex::from_usize(var_id))
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
        let mut value = self.constant;
        for (var_id, &coefficient) in self.coefficients.iter().enumerate() {
            if !approx_eq(coefficient.value(), 0.0) {
                value = NumericValue::new(
                    value.value() + coefficient.value() * numeric_values[var_id].value(),
                );
            }
        }
        Ok(value)
    }

    fn apply_effects(
        &self,
        additive_deltas: &[f64],
        assigned_constants: &[Option<NumericValue>],
    ) -> Self {
        let mut out = Self::constant(self.constant, self.coefficients.len());
        for (var_id, &coefficient) in self.coefficients.iter().enumerate() {
            if approx_eq(coefficient.value(), 0.0) {
                continue;
            }
            if let Some(value) = assigned_constants[var_id] {
                out.constant =
                    NumericValue::new(out.constant.value() + coefficient.value() * value.value());
            } else {
                out.coefficients[var_id] =
                    NumericValue::new(out.coefficients[var_id].value() + coefficient.value());
                out.constant = NumericValue::new(
                    out.constant.value() + coefficient.value() * additive_deltas[var_id],
                );
            }
        }
        out
    }
}

pub fn build_restricted_task(task: &dyn AbstractNumericTask) -> Result<Option<RestrictedTask>> {
    let restriction_reason = match validate_restricted_task(task) {
        Ok(()) => {
            info!("restricted task: task already satisfies restricted-task invariants");
            return Ok(None);
        }
        Err(reason) => reason,
    };
    let num_numeric_vars = task.numeric_variables().len();
    if num_numeric_vars == 0 || task.comparison_axioms().is_empty() {
        bail!(
            "task violates restricted-task invariants ({restriction_reason}), but has no numeric comparison roots that can be promoted"
        );
    }

    let initial_numeric = task.get_initial_numeric_state_values().to_vec();
    let assignment_lookup = task
        .assignment_axiom_lookup()
        .map_err(|error| anyhow!("invalid assignment axiom layout: {error}"))?;
    let mut linearizer = Linearizer {
        task,
        assignment_lookup,
        initial_numeric: &initial_numeric,
        memo: vec![None; num_numeric_vars],
        visiting: vec![false; num_numeric_vars],
    };

    let mut root_var_ids = BTreeSet::new();
    for condition in task.numeric_conditions().iter() {
        root_var_ids.insert(condition.left_numeric_var_id());
        root_var_ids.insert(condition.right_numeric_var_id());
    }

    let mut root_exprs = BTreeMap::new();
    for &root_var_id in &root_var_ids {
        let expr = linearizer.linearize(root_var_id)?;
        match task.numeric_variables()[root_var_id.index()].get_type() {
            NumericType::Regular | NumericType::Constant | NumericType::Cost => {}
            NumericType::Derived => {
                root_exprs.insert(root_var_id, expr);
            }
        }
    }

    if root_exprs.is_empty() {
        bail!(
            "task violates restricted-task invariants ({restriction_reason}), but no derived numeric comparison roots can be promoted"
        );
    }

    let mut numeric_var_ids: Vec<VariableIndex> = root_var_ids
        .iter()
        .copied()
        .filter(|&numeric_var_id| {
            !matches!(
                task.numeric_variables()[numeric_var_id.index()].get_type(),
                NumericType::Derived
            )
        })
        .collect();
    numeric_var_ids.extend(root_exprs.keys().copied());
    if let Some(metric_var_id) = task.metric().var_id() {
        numeric_var_ids.push(metric_var_id);
    }
    numeric_var_ids.sort_unstable();
    numeric_var_ids.dedup();

    build_task(task, &initial_numeric, &root_exprs, &numeric_var_ids)
}

/// Builds the restricted representation consumed by the ICAPS 2026 Cartesian
/// artifact: regular variables remain Cartesian dimensions and unary affine
/// conditions become thresholds on those variables.
pub fn build_icaps26_restricted_task(
    task: &dyn AbstractNumericTask,
) -> Result<Option<RestrictedTask>> {
    if task.comparison_axioms().is_empty() {
        validate_restricted_task(task).map_err(anyhow::Error::msg)?;
        info!("ICAPS 2026 restricted task: no numeric conditions to normalize");
        return Ok(None);
    }

    let initial_numeric = task.get_initial_numeric_state_values().to_vec();
    let num_original_numeric = task.numeric_variables().len();
    let mut linearizer = Linearizer {
        task,
        assignment_lookup: task
            .assignment_axiom_lookup()
            .map_err(|error| anyhow!("invalid assignment axiom layout: {error}"))?,
        initial_numeric: &initial_numeric,
        memo: vec![None; num_original_numeric],
        visiting: vec![false; num_original_numeric],
    };
    let mut used_comparison_vars = BTreeSet::new();
    for operator in task.get_operators() {
        used_comparison_vars.extend(operator.preconditions().iter().map(ExplicitFact::var_index));
    }
    used_comparison_vars
        .extend((0..task.get_num_goals()).map(|goal_id| task.get_goal_fact(goal_id).var_index()));
    for axiom in task.axioms() {
        used_comparison_vars.extend(axiom.conditions().iter().map(ExplicitFact::var_index));
    }

    let mut comparisons = Vec::new();
    for (comparison_axiom_id, comparison_axiom) in task.comparison_axioms().iter().enumerate() {
        if !used_comparison_vars.contains(&comparison_axiom.get_affected_var_id()) {
            continue;
        }
        let difference = linearizer
            .linearize(comparison_axiom.get_left_var_id())?
            .sub(linearizer.linearize(comparison_axiom.get_right_var_id())?);
        comparisons.push((comparison_axiom_id, difference));
    }

    let retained_original_ids = task
        .numeric_variables()
        .iter()
        .enumerate()
        .filter_map(|(var_id, variable)| {
            (!matches!(variable.get_type(), NumericType::Derived))
                .then_some(VariableIndex::from_usize(var_id))
        })
        .collect::<Vec<_>>();
    let mut original_to_transformed = vec![None; num_original_numeric];
    let mut numeric_variables = Vec::with_capacity(retained_original_ids.len());
    let mut numeric_initial = Vec::with_capacity(retained_original_ids.len());
    let mut transformed_to_expr = Vec::with_capacity(retained_original_ids.len());
    let mut constants = ConstantPool::default();
    for original_id in retained_original_ids {
        let transformed_id = numeric_variables.len();
        original_to_transformed[original_id.index()] =
            Some(VariableIndex::from_usize(transformed_id));
        let variable = task.numeric_variables()[original_id.index()].clone();
        let value = initial_numeric[original_id.index()];
        if matches!(variable.get_type(), NumericType::Constant) {
            constants.by_bits.insert(
                float_tolerance::canonicalize_nv(value).value().to_bits(),
                VariableIndex::from_usize(transformed_id),
            );
            transformed_to_expr.push(AffineExpression::constant(value, num_original_numeric));
        } else {
            transformed_to_expr.push(AffineExpression::var(original_id, num_original_numeric));
        }
        numeric_variables.push(variable);
        numeric_initial.push(value);
    }

    let mut normalized_comparisons = Vec::with_capacity(comparisons.len());
    for (comparison_axiom_id, difference) in comparisons {
        let comparison_axiom = &task.comparison_axioms()[comparison_axiom_id];
        let dependencies = difference.non_zero_vars();
        let (left, threshold, operator) = if dependencies.len() == 1 {
            let original_var_id = dependencies[0];
            ensure!(
                matches!(
                    task.numeric_variables()[original_var_id.index()].get_type(),
                    NumericType::Regular
                ),
                "ICAPS 2026 comparison axiom {comparison_axiom_id} depends on non-regular numeric variable {original_var_id}"
            );
            let coefficient = difference.coefficients[original_var_id.index()];
            ensure!(
                coefficient.value().is_finite() && !approx_eq(coefficient.value(), 0.0),
                "ICAPS 2026 comparison axiom {comparison_axiom_id} has invalid coefficient {coefficient}"
            );
            (
                original_to_transformed[original_var_id.index()]
                    .expect("retained ICAPS regular variable has no transformed id"),
                float_tolerance::canonicalize(-difference.constant.value() / coefficient.value()),
                if coefficient.value() > 0.0 {
                    comparison_axiom.get_operator().clone()
                } else {
                    reverse_comparison(comparison_axiom.get_operator())
                },
            )
        } else {
            ensure!(
                dependencies.iter().all(|&var_id| matches!(
                    task.numeric_variables()[var_id.index()].get_type(),
                    NumericType::Regular
                )),
                "ICAPS 2026 comparison axiom {comparison_axiom_id} has non-regular dependencies {dependencies:?}"
            );
            let auxiliary_id = numeric_variables.len();
            numeric_variables.push(NumericVariable::new(
                format!("icaps26-condition-{comparison_axiom_id}"),
                NumericType::Regular,
                None,
            ));
            numeric_initial.push(difference.evaluate(&initial_numeric)?);
            transformed_to_expr.push(difference);
            (
                VariableIndex::from_usize(auxiliary_id),
                0.0,
                comparison_axiom.get_operator().clone(),
            )
        };
        ensure!(
            threshold.is_finite(),
            "ICAPS 2026 comparison axiom {comparison_axiom_id} has non-finite threshold {threshold}"
        );
        normalized_comparisons.push((
            AxiomIndex::from_usize(comparison_axiom_id),
            left,
            threshold,
            operator,
        ));
    }

    let root_exprs = BTreeMap::new();
    let mut operators = Vec::with_capacity(task.get_operators().len());
    for operator in task.get_operators() {
        operators.push(transform_operator(
            task,
            operator,
            &initial_numeric,
            &root_exprs,
            &original_to_transformed,
            EffectCoordinateMode::AllTransformed,
            &mut TransformedNumerics {
                constants: &mut constants,
                variables: &mut numeric_variables,
                initial: &mut numeric_initial,
                to_expr: &mut transformed_to_expr,
            },
        )?);
    }

    let mut comparison_axioms = Vec::with_capacity(normalized_comparisons.len());
    for (comparison_axiom_id, left, threshold, operator) in normalized_comparisons {
        let right = constants.get_or_insert(
            NumericValue::new(threshold),
            &mut numeric_variables,
            &mut numeric_initial,
            &mut transformed_to_expr,
            num_original_numeric,
        );
        comparison_axioms.push(ComparisonAxiom::new(
            task.comparison_axioms()[comparison_axiom_id.index()].get_affected_var_id(),
            left,
            right,
            operator,
        ));
    }

    let variables = renumber_propositional_axiom_layers(task, &comparison_axioms);
    let metric_var_id = task.metric().var_id().and_then(|var_id| {
        original_to_transformed
            .get(var_id.index())
            .and_then(|mapped| *mapped)
    });
    let transformed_task = NumericRootTask::new(NumericRootTaskParts {
        version: 1,
        metric: Metric::new(task.metric().is_min(), metric_var_id),
        variables,
        numeric_variables,
        goals: (0..task.get_num_goals())
            .map(|goal_id| *task.get_goal_fact(goal_id))
            .collect(),
        mutexes: vec![],
        state: task.get_initial_propositional_state_values().to_vec(),
        numeric_state: numeric_initial,
        operators,
        axioms: task.axioms().clone(),
        comparison_axioms,
        assignment_axioms: Vec::new(),
        global_constraint: ExplicitFact::propositional(0, 0),
    });
    validate_restricted_task(&transformed_task).map_err(|reason| {
        anyhow::anyhow!("ICAPS 2026 restricted task construction failed: {reason}")
    })?;
    ensure_operator_costs_unchanged(task, &transformed_task)?;
    Ok(Some(RestrictedTask {
        task: transformed_task,
    }))
}

fn reverse_comparison(operator: &ComparisonOperator) -> ComparisonOperator {
    match operator {
        ComparisonOperator::LessThan => ComparisonOperator::GreaterThan,
        ComparisonOperator::LessThanOrEqual => ComparisonOperator::GreaterThanOrEqual,
        ComparisonOperator::Equal => ComparisonOperator::Equal,
        ComparisonOperator::GreaterThanOrEqual => ComparisonOperator::LessThanOrEqual,
        ComparisonOperator::GreaterThan => ComparisonOperator::LessThan,
        ComparisonOperator::UnEqual => ComparisonOperator::UnEqual,
    }
}

fn build_task(
    task: &dyn AbstractNumericTask,
    initial_numeric: &[NumericValue],
    root_exprs: &BTreeMap<VariableIndex, AffineExpression>,
    numeric_var_ids: &[VariableIndex],
) -> Result<Option<RestrictedTask>> {
    let num_original_numeric = task.numeric_variables().len();
    let mut original_to_transformed = vec![None; num_original_numeric];
    let mut transformed_to_expr = Vec::new();
    let mut numeric_variables = Vec::new();
    let mut numeric_initial = Vec::new();

    for &original_id in numeric_var_ids {
        let expr = root_exprs.get(&original_id).cloned().unwrap_or_else(|| {
            match task.numeric_variables()[original_id.index()].get_type() {
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
        let name = if root_exprs.contains_key(&original_id) {
            format!("{}|{}", restricted_shape_prefix(&expr), original_name)
        } else {
            original_name.to_string()
        };
        let numeric_type = if root_exprs.contains_key(&original_id) {
            NumericType::Regular
        } else {
            *task.numeric_variables()[original_id.index()].get_type()
        };
        numeric_variables.push(NumericVariable::new(name, numeric_type, None));
        numeric_initial.push(expr.evaluate(initial_numeric)?);
        transformed_to_expr.push(expr);
    }

    let mut constants = ConstantPool::default();
    let mut operators = Vec::with_capacity(task.get_operators().len());
    for operator in task.get_operators() {
        operators.push(transform_operator(
            task,
            operator,
            initial_numeric,
            root_exprs,
            &original_to_transformed,
            EffectCoordinateMode::OriginalMapping,
            &mut TransformedNumerics {
                constants: &mut constants,
                variables: &mut numeric_variables,
                initial: &mut numeric_initial,
                to_expr: &mut transformed_to_expr,
            },
        )?);
    }

    let mut comparison_axioms = Vec::new();
    for comparison_axiom in task.comparison_axioms() {
        let Some(left) = original_to_transformed[comparison_axiom.get_left_var_id().index()] else {
            continue;
        };
        let Some(right) = original_to_transformed[comparison_axiom.get_right_var_id().index()]
        else {
            continue;
        };
        comparison_axioms.push(ComparisonAxiom::new(
            comparison_axiom.get_affected_var_id(),
            left,
            right,
            comparison_axiom.get_operator().clone(),
        ));
    }
    let variables = renumber_propositional_axiom_layers(task, &comparison_axioms);

    let metric_var_id = task.metric().var_id().and_then(|var_id| {
        original_to_transformed
            .get(var_id.index())
            .and_then(|mapped| *mapped)
    });

    let transformed_task = NumericRootTask::new(NumericRootTaskParts {
        version: 1,
        metric: Metric::new(task.metric().is_min(), metric_var_id),
        variables,
        numeric_variables,
        goals: (0..task.get_num_goals())
            .map(|goal_id| *task.get_goal_fact(goal_id))
            .collect(),
        mutexes: vec![],
        state: task.get_initial_propositional_state_values().to_vec(),
        numeric_state: numeric_initial,
        operators,
        axioms: task.axioms().clone(),
        comparison_axioms,
        assignment_axioms: Vec::<AssignmentAxiom>::new(),
        global_constraint: ExplicitFact::propositional(0, 0),
    });
    validate_restricted_task(&transformed_task)
        .map_err(|reason| anyhow::anyhow!("restricted task construction failed: {reason}"))?;

    ensure_operator_costs_unchanged(task, &transformed_task)?;

    Ok(Some(RestrictedTask {
        task: transformed_task,
    }))
}

fn ensure_operator_costs_unchanged(
    source_task: &dyn AbstractNumericTask,
    transformed_task: &dyn AbstractNumericTask,
) -> Result<()> {
    ensure!(
        source_task.get_operators().len() == transformed_task.get_operators().len(),
        "restricted task changed the operator count"
    );
    for (operator_id, (source, transformed)) in source_task
        .get_operators()
        .iter()
        .zip(transformed_task.get_operators())
        .enumerate()
    {
        let source_cost = metric_operator_cost_from_initial_values(source_task, source);
        let transformed_cost =
            metric_operator_cost_from_initial_values(transformed_task, transformed);
        ensure!(
            approx_eq(source_cost.value(), transformed_cost.value()),
            "restricted task changed metric cost of operator {operator_id} ({}): {source_cost} -> {transformed_cost}",
            source.name()
        );
    }
    Ok(())
}

fn renumber_propositional_axiom_layers(
    task: &dyn AbstractNumericTask,
    comparison_axioms: &[ComparisonAxiom],
) -> Vec<ExplicitVariable> {
    if comparison_axioms.is_empty() {
        return task.variables().clone();
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

#[derive(Clone, Copy)]
enum EffectCoordinateMode {
    OriginalMapping,
    AllTransformed,
}

fn transform_operator(
    task: &dyn AbstractNumericTask,
    operator: &Operator,
    initial_numeric: &[NumericValue],
    root_exprs: &BTreeMap<VariableIndex, AffineExpression>,
    original_to_transformed: &[Option<VariableIndex>],
    coordinate_mode: EffectCoordinateMode,
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
            coordinate_mode,
            numerics,
        );
    }

    let mut deltas = vec![NumericValue::new(0.0); task.numeric_variables().len()];
    for effect in operator.assignment_effects() {
        ensure!(
            !effect.is_conditional() && effect.conditions().is_empty(),
            "restricted task does not support conditional numeric effects in operator {}",
            operator.name()
        );
        let source_value = match task.numeric_variables()[effect.var_id().index()].get_type() {
            NumericType::Constant => initial_numeric[effect.var_id().index()],
            _ => bail!(
                "restricted task requires constant RHS numeric effects in operator {}",
                operator.name()
            ),
        };
        match effect.operation() {
            AssignmentOperation::Plus => {
                deltas[effect.affected_var_id().index()] = NumericValue::new(
                    deltas[effect.affected_var_id().index()].value() + source_value.value(),
                )
            }
            AssignmentOperation::Minus => {
                deltas[effect.affected_var_id().index()] = NumericValue::new(
                    deltas[effect.affected_var_id().index()].value() - source_value.value(),
                )
            }
            AssignmentOperation::Assign => bail!(
                "restricted task does not support assignment numeric effects in operator {}",
                operator.name()
            ),
            AssignmentOperation::Times | AssignmentOperation::Divide => bail!(
                "restricted task does not support non-additive numeric effects in operator {}",
                operator.name()
            ),
        }
    }

    let transformed_deltas = match coordinate_mode {
        EffectCoordinateMode::OriginalMapping => original_to_transformed
            .iter()
            .enumerate()
            .filter_map(|(original_id, &mapped)| {
                mapped.map(|transformed_id| {
                    let delta = root_exprs
                        .get(&VariableIndex::from_usize(original_id))
                        .map(|expr| {
                            NumericValue::new(
                                expr.coefficients
                                    .iter()
                                    .zip(deltas.iter())
                                    .map(|(&coefficient, &delta)| {
                                        coefficient.value() * delta.value()
                                    })
                                    .sum::<f64>(),
                            )
                        })
                        .unwrap_or(deltas[original_id]);
                    (transformed_id, delta)
                })
            })
            .collect::<Vec<_>>(),
        EffectCoordinateMode::AllTransformed => numerics
            .to_expr
            .iter()
            .enumerate()
            .map(|(transformed_id, expr)| {
                let delta = NumericValue::new(
                    expr.coefficients
                        .iter()
                        .zip(deltas.iter())
                        .map(|(&coefficient, &delta)| coefficient.value() * delta.value())
                        .sum::<f64>(),
                );
                (VariableIndex::from_usize(transformed_id), delta)
            })
            .collect(),
    };
    let mut assignment_effects = Vec::new();
    for (transformed_id, delta) in transformed_deltas {
        if approx_eq(delta.value(), 0.0) {
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
    task: &dyn AbstractNumericTask,
    operator: &Operator,
    initial_numeric: &[NumericValue],
    original_to_transformed: &[Option<VariableIndex>],
    coordinate_mode: EffectCoordinateMode,
    numerics: &mut TransformedNumerics<'_>,
) -> Result<Operator> {
    let mut additive_deltas = vec![0.0; task.numeric_variables().len()];
    let mut assigned_constants = vec![None; task.numeric_variables().len()];
    for effect in operator.assignment_effects() {
        ensure!(
            !effect.is_conditional() && effect.conditions().is_empty(),
            "restricted task does not support conditional numeric effects in operator {}",
            operator.name()
        );
        let source_value = match task.numeric_variables()[effect.var_id().index()].get_type() {
            NumericType::Constant => initial_numeric[effect.var_id().index()],
            _ => bail!(
                "restricted task requires constant RHS numeric effects in operator {}",
                operator.name()
            ),
        };
        match effect.operation() {
            AssignmentOperation::Plus => {
                ensure!(
                    assigned_constants[effect.affected_var_id().index()].is_none(),
                    "restricted task does not support mixed assignment/additive numeric effects on one variable in operator {}",
                    operator.name()
                );
                additive_deltas[effect.affected_var_id().index()] += source_value.value();
            }
            AssignmentOperation::Minus => {
                ensure!(
                    assigned_constants[effect.affected_var_id().index()].is_none(),
                    "restricted task does not support mixed assignment/additive numeric effects on one variable in operator {}",
                    operator.name()
                );
                additive_deltas[effect.affected_var_id().index()] -= source_value.value();
            }
            AssignmentOperation::Assign => {
                ensure!(
                    approx_eq(additive_deltas[effect.affected_var_id().index()], 0.0)
                        && assigned_constants[effect.affected_var_id().index()].is_none(),
                    "restricted task does not support multiple numeric effects on an assigned variable in operator {}",
                    operator.name()
                );
                assigned_constants[effect.affected_var_id().index()] = Some(source_value);
            }
            AssignmentOperation::Times | AssignmentOperation::Divide => bail!(
                "restricted task does not support non-additive numeric effects in operator {}",
                operator.name()
            ),
        }
    }

    let coordinate_ids = match coordinate_mode {
        EffectCoordinateMode::OriginalMapping => original_to_transformed
            .iter()
            .filter_map(|&mapped| mapped)
            .collect::<Vec<_>>(),
        EffectCoordinateMode::AllTransformed => (0..numerics.to_expr.len())
            .map(VariableIndex::from_usize)
            .collect(),
    };
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
            if approx_eq(delta.value(), 0.0) {
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
        ensure!(
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
    by_bits: HashMap<u64, VariableIndex>,
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
        let value = float_tolerance::canonicalize_nv(value);
        let bits = value.value().to_bits();
        if let Some(&id) = self.by_bits.get(&bits) {
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
        self.by_bits.insert(bits, id);
        id
    }
}

struct Linearizer<'a> {
    task: &'a dyn AbstractNumericTask,
    assignment_lookup: Vec<Option<AxiomIndex>>,
    initial_numeric: &'a [NumericValue],
    memo: Vec<Option<AffineExpression>>,
    visiting: Vec<bool>,
}

impl Linearizer<'_> {
    fn linearize(&mut self, numeric_var_id: VariableIndex) -> Result<AffineExpression> {
        ensure!(
            numeric_var_id.index() < self.task.numeric_variables().len(),
            "numeric variable {numeric_var_id} out of bounds"
        );
        if let Some(expr) = &self.memo[numeric_var_id.index()] {
            return Ok(expr.clone());
        }
        ensure!(
            !self.visiting[numeric_var_id.index()],
            "cycle in numeric assignment axioms at numeric variable {numeric_var_id}"
        );
        self.visiting[numeric_var_id.index()] = true;

        let num_vars = self.task.numeric_variables().len();
        let expr = match self.task.numeric_variables()[numeric_var_id.index()].get_type() {
            NumericType::Regular => AffineExpression::var(numeric_var_id, num_vars),
            NumericType::Constant => {
                AffineExpression::constant(self.initial_numeric[numeric_var_id.index()], num_vars)
            }
            NumericType::Cost => AffineExpression::var(numeric_var_id, num_vars),
            NumericType::Derived => {
                let axiom_id =
                    self.assignment_lookup[numeric_var_id.index()].with_context(|| {
                        format!("missing assignment axiom for numeric variable {numeric_var_id}")
                    })?;
                let axiom = &self.task.assignment_axioms()[axiom_id.index()];
                let lhs = self.linearize(axiom.get_left_var_id())?;
                let rhs = self.linearize(axiom.get_right_var_id())?;
                match axiom.get_operator() {
                    CalOperator::Sum => lhs.add(rhs),
                    CalOperator::Difference => lhs.sub(rhs),
                    CalOperator::Product => {
                        if rhs.non_zero_vars().is_empty() {
                            lhs.scale(rhs.constant.value())
                        } else if lhs.non_zero_vars().is_empty() {
                            rhs.scale(lhs.constant.value())
                        } else {
                            bail!("numeric assignment axiom {axiom_id} is nonlinear")
                        }
                    }
                    CalOperator::Division => {
                        ensure!(
                            rhs.non_zero_vars().is_empty() && !approx_eq(rhs.constant.value(), 0.0),
                            "numeric assignment axiom {axiom_id} has non-constant or zero divisor"
                        );
                        lhs.scale(1.0 / rhs.constant.value())
                    }
                }
            }
        };

        self.visiting[numeric_var_id.index()] = false;
        self.memo[numeric_var_id.index()] = Some(expr.clone());
        Ok(expr)
    }
}

fn approx_eq(lhs: f64, rhs: f64) -> bool {
    (lhs - rhs).abs() <= 1e-12
}

fn restricted_shape_prefix(expr: &AffineExpression) -> String {
    let mut coefficients = expr
        .coefficients
        .iter()
        .copied()
        .filter(|coefficient| !approx_eq(coefficient.value(), 0.0))
        .collect::<Vec<_>>();
    coefficients.sort_by(|left, right| left.value().total_cmp(&right.value()));
    let shape = coefficients
        .iter()
        .map(|coefficient| coefficient.to_string())
        .collect::<Vec<_>>()
        .join(",");
    format!("rt-shape:{shape}")
}
