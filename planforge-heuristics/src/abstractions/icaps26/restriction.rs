//! The restricted task the ICAPS 2026 artifact consumes, frozen with the rest
//! of this module: regular variables stay Cartesian dimensions, a unary affine
//! condition becomes a threshold on its variable, and every other condition a
//! threshold on an auxiliary variable of its own. `--transform icaps26_restrict` builds it
//! instead of the standard restriction whenever the search uses
//! `icaps26_cartesian`.

#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet, HashMap};

use anyhow::{Context, Result, anyhow, ensure};
use planforge_sas::axioms::{ComparisonAxiom, ComparisonOperator};
use planforge_sas::numeric_task::{
    AssignmentEffect, AssignmentOperation, AxiomIndex, ExplicitFact, ExplicitVariable, Metric,
    NumericType, NumericValue, NumericVariable, ONE_VALUE, Operator, OperatorIndex, Task,
    TaskParts, VariableIndex, ZERO_VALUE,
};
use planforge_search::config::EmptyConfig;
use planforge_search::task_transformations::restriction::validate_restricted_task;
use planforge_search::task_transformations::{TransformError, TransformationFeature};
use tracing::info;

/// `icaps26_restrict()`: the restriction `icaps26_cartesian` requires.
pub struct Icaps26RestrictFeature;

impl TransformationFeature for Icaps26RestrictFeature {
    const NAME: &'static str = "icaps26_restrict";
    type Config = EmptyConfig;

    fn apply(_: &EmptyConfig, task: Task) -> std::result::Result<Task, TransformError> {
        Ok(build_icaps26_restricted_task(&task)
            .map_err(crate::abstractions::transform_error)?
            .unwrap_or(task))
    }
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

/// Builds the restricted representation consumed by the ICAPS 2026 Cartesian
/// artifact: regular variables remain Cartesian dimensions and unary affine
/// conditions become thresholds on those variables.
pub(crate) fn build_icaps26_restricted_task(task: &Task) -> Result<Option<Task>> {
    if task.comparison_axioms().is_empty() {
        validate_restricted_task(task).map_err(anyhow::Error::msg)?;
        info!("ICAPS 2026 restricted task: no numeric conditions to normalize");
        return Ok(None);
    }

    let initial_numeric = task.initial_numeric_values().to_vec();
    let num_original_numeric = task.numeric_variables().len();
    let mut used_comparison_vars = BTreeSet::new();
    for operator in task.operators() {
        used_comparison_vars.extend(operator.preconditions().iter().map(ExplicitFact::var_index));
    }
    used_comparison_vars.extend(task.goals().iter().map(ExplicitFact::var_index));
    for axiom in task.axioms() {
        used_comparison_vars.extend(axiom.conditions().iter().map(ExplicitFact::var_index));
    }

    let mut comparisons = Vec::new();
    for (comparison_axiom_id, comparison_axiom) in task.comparison_axioms().iter().enumerate() {
        if !used_comparison_vars.contains(&comparison_axiom.affected_var_id()) {
            continue;
        }
        let difference = affine_expression(task, comparison_axiom.left_var_id())?
            .sub(affine_expression(task, comparison_axiom.right_var_id())?);
        comparisons.push((comparison_axiom_id, difference));
    }

    let retained_original_ids = task
        .numeric_variables()
        .iter()
        .enumerate()
        .filter_map(|(var_id, variable)| {
            (!matches!(variable.numeric_type(), NumericType::Derived))
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
        if matches!(variable.numeric_type(), NumericType::Constant) {
            constants
                .by_value
                .insert(value, VariableIndex::from_usize(transformed_id));
            transformed_to_expr.push(AffineExpression::constant(value, num_original_numeric));
        } else {
            transformed_to_expr.push(AffineExpression::var(original_id, num_original_numeric));
        }
        numeric_variables.push(variable);
        numeric_initial.push(value);
    }
    if let Some(metric_var_id) = task.metric().var_id()
        && let Some(expr) = derived_metric_expression(task, metric_var_id)?
    {
        original_to_transformed[metric_var_id.index()] =
            Some(VariableIndex::from_usize(numeric_variables.len()));
        numeric_variables.push(NumericVariable::new(
            task.numeric_variables()[metric_var_id.index()]
                .name()
                .to_string(),
            NumericType::Cost,
            None,
        ));
        numeric_initial.push(expr.evaluate(&initial_numeric)?);
        transformed_to_expr.push(expr);
    }

    let mut normalized_comparisons = Vec::with_capacity(comparisons.len());
    for (comparison_axiom_id, difference) in comparisons {
        let comparison_axiom = &task.comparison_axioms()[comparison_axiom_id];
        let dependencies = difference.non_zero_vars();
        let (left, threshold, operator) = if dependencies.len() == 1 {
            let original_var_id = dependencies[0];
            planforge_search::ensure_supported!(
                matches!(task.numeric_type(original_var_id), NumericType::Regular),
                "ICAPS 2026 comparison axiom {comparison_axiom_id} depends on non-regular numeric variable {original_var_id}"
            );
            let coefficient = difference.coefficients[original_var_id.index()];
            ensure!(
                coefficient.is_finite() && (coefficient != ZERO_VALUE),
                "ICAPS 2026 comparison axiom {comparison_axiom_id} has invalid coefficient {coefficient}"
            );
            (
                original_to_transformed[original_var_id.index()]
                    .expect("retained ICAPS regular variable has no transformed id"),
                -difference.constant / coefficient,
                if coefficient > ZERO_VALUE {
                    comparison_axiom.operator().clone()
                } else {
                    reverse_comparison(comparison_axiom.operator())
                },
            )
        } else {
            planforge_search::ensure_supported!(
                dependencies
                    .iter()
                    .all(|&var_id| matches!(task.numeric_type(var_id), NumericType::Regular)),
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
                ZERO_VALUE,
                comparison_axiom.operator().clone(),
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

    let mut operators = Vec::with_capacity(task.operators().len());
    for operator in task.operators() {
        operators.push(transform_operator(
            task,
            operator,
            &initial_numeric,
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
            threshold,
            &mut numeric_variables,
            &mut numeric_initial,
            &mut transformed_to_expr,
            num_original_numeric,
        );
        comparison_axioms.push(ComparisonAxiom::new(
            task.comparison_axioms()[comparison_axiom_id.index()].affected_var_id(),
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
        assignment_axioms: Vec::new(),
        global_constraint: None,
    });
    validate_restricted_task(&transformed_task).map_err(|reason| {
        anyhow::anyhow!("ICAPS 2026 restricted task construction failed: {reason}")
    })?;
    ensure_operator_costs_unchanged(task, &transformed_task)?;
    Ok(Some(transformed_task))
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
        .map(|axiom| axiom.affected_var_id())
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
    numerics: &mut TransformedNumerics<'_>,
) -> Result<Operator> {
    let has_assignment = operator
        .assignment_effects()
        .iter()
        .any(|effect| matches!(effect.operation(), AssignmentOperation::Assign));
    if has_assignment {
        return transform_operator_with_assignment(task, operator, initial_numeric, numerics);
    }

    let mut deltas = vec![ZERO_VALUE; task.numeric_variables().len()];
    for effect in operator.assignment_effects() {
        planforge_search::ensure_supported!(
            !effect.is_conditional() && effect.conditions().is_empty(),
            "restricted task does not support conditional numeric effects in operator {}",
            operator.name()
        );
        let source_value = match task.numeric_type(effect.var_id()) {
            NumericType::Constant => initial_numeric[effect.var_id().index()],
            _ => planforge_search::unsupported!(
                "restricted task requires constant RHS numeric effects in operator {}",
                operator.name()
            ),
        };
        match effect.operation() {
            AssignmentOperation::Plus => deltas[effect.affected_var_id().index()] += source_value,
            AssignmentOperation::Minus => deltas[effect.affected_var_id().index()] -= source_value,
            AssignmentOperation::Assign => planforge_search::unsupported!(
                "restricted task does not support assignment numeric effects in operator {}",
                operator.name()
            ),
            AssignmentOperation::Times | AssignmentOperation::Divide => {
                planforge_search::unsupported!(
                    "restricted task does not support non-additive numeric effects in operator {}",
                    operator.name()
                )
            }
        }
    }

    let transformed_deltas = numerics
        .to_expr
        .iter()
        .enumerate()
        .map(|(transformed_id, expr)| {
            let delta = NumericValue::affine(
                expr.coefficients
                    .iter()
                    .zip(&deltas)
                    .map(|(coefficient, delta)| (coefficient.value(), *delta)),
                0.0,
            );
            (VariableIndex::from_usize(transformed_id), delta)
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
    numerics: &mut TransformedNumerics<'_>,
) -> Result<Operator> {
    let mut additive_deltas = vec![ZERO_VALUE; task.numeric_variables().len()];
    let mut assigned_constants = vec![None; task.numeric_variables().len()];
    for effect in operator.assignment_effects() {
        planforge_search::ensure_supported!(
            !effect.is_conditional() && effect.conditions().is_empty(),
            "restricted task does not support conditional numeric effects in operator {}",
            operator.name()
        );
        let source_value = match task.numeric_type(effect.var_id()) {
            NumericType::Constant => initial_numeric[effect.var_id().index()],
            _ => planforge_search::unsupported!(
                "restricted task requires constant RHS numeric effects in operator {}",
                operator.name()
            ),
        };
        match effect.operation() {
            AssignmentOperation::Plus => {
                planforge_search::ensure_supported!(
                    assigned_constants[effect.affected_var_id().index()].is_none(),
                    "restricted task does not support mixed assignment/additive numeric effects on one variable in operator {}",
                    operator.name()
                );
                additive_deltas[effect.affected_var_id().index()] += source_value;
            }
            AssignmentOperation::Minus => {
                planforge_search::ensure_supported!(
                    assigned_constants[effect.affected_var_id().index()].is_none(),
                    "restricted task does not support mixed assignment/additive numeric effects on one variable in operator {}",
                    operator.name()
                );
                additive_deltas[effect.affected_var_id().index()] -= source_value;
            }
            AssignmentOperation::Assign => {
                planforge_search::ensure_supported!(
                    (additive_deltas[effect.affected_var_id().index()] == ZERO_VALUE)
                        && assigned_constants[effect.affected_var_id().index()].is_none(),
                    "restricted task does not support multiple numeric effects on an assigned variable in operator {}",
                    operator.name()
                );
                assigned_constants[effect.affected_var_id().index()] = Some(source_value);
            }
            AssignmentOperation::Times | AssignmentOperation::Divide => {
                planforge_search::unsupported!(
                    "restricted task does not support non-additive numeric effects in operator {}",
                    operator.name()
                )
            }
        }
    }

    let coordinate_ids = (0..numerics.to_expr.len()).map(VariableIndex::from_usize);
    let transformed_effects = coordinate_ids
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
        planforge_search::ensure_supported!(
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
