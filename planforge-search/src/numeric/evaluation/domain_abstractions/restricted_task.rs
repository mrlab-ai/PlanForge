use std::collections::{BTreeMap, BTreeSet, HashMap};

use anyhow::{Context, Result, bail, ensure};
use planforge_sas::numeric::axioms::{AssignmentAxiom, CalOperator, ComparisonAxiom};
use planforge_sas::numeric::numeric_task::{
    AbstractNumericTask, AssignmentEffect, AssignmentOperation, ExplicitFact, ExplicitVariable,
    Metric, NumericRootTask, NumericType, NumericVariable, Operator,
};
use planforge_sas::numeric::utils::float_tolerance;
use tracing::info;

use super::comparison_expression::ComparisonTree;

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

#[derive(Debug, Clone, PartialEq)]
struct AffineExpression {
    coefficients: Vec<f64>,
    constant: f64,
}

impl AffineExpression {
    fn zero(num_vars: usize) -> Self {
        Self {
            coefficients: vec![0.0; num_vars],
            constant: 0.0,
        }
    }

    fn var(var_id: usize, num_vars: usize) -> Self {
        let mut expr = Self::zero(num_vars);
        expr.coefficients[var_id] = 1.0;
        expr
    }

    fn constant(value: f64, num_vars: usize) -> Self {
        let mut expr = Self::zero(num_vars);
        expr.constant = value;
        expr
    }

    fn add(mut self, rhs: Self) -> Self {
        for (lhs, rhs) in self.coefficients.iter_mut().zip(rhs.coefficients) {
            *lhs += rhs;
        }
        self.constant += rhs.constant;
        self
    }

    fn sub(mut self, rhs: Self) -> Self {
        for (lhs, rhs) in self.coefficients.iter_mut().zip(rhs.coefficients) {
            *lhs -= rhs;
        }
        self.constant -= rhs.constant;
        self
    }

    fn scale(mut self, factor: f64) -> Self {
        for coefficient in &mut self.coefficients {
            *coefficient *= factor;
        }
        self.constant *= factor;
        self
    }

    fn non_zero_vars(&self) -> Vec<usize> {
        self.coefficients
            .iter()
            .enumerate()
            .filter_map(|(var_id, &coefficient)| (!approx_eq(coefficient, 0.0)).then_some(var_id))
            .collect()
    }

    fn evaluate(&self, numeric_values: &[f64]) -> Result<f64> {
        ensure!(
            numeric_values.len() >= self.coefficients.len(),
            "numeric state too short for transformed task projection: {} < {}",
            numeric_values.len(),
            self.coefficients.len()
        );
        let mut value = self.constant;
        for (var_id, &coefficient) in self.coefficients.iter().enumerate() {
            if !approx_eq(coefficient, 0.0) {
                value += coefficient * numeric_values[var_id];
            }
        }
        Ok(value)
    }

    fn apply_effects(&self, additive_deltas: &[f64], assigned_constants: &[Option<f64>]) -> Self {
        let mut out = Self::constant(self.constant, self.coefficients.len());
        for (var_id, &coefficient) in self.coefficients.iter().enumerate() {
            if approx_eq(coefficient, 0.0) {
                continue;
            }
            if let Some(value) = assigned_constants[var_id] {
                out.constant += coefficient * value;
            } else {
                out.coefficients[var_id] += coefficient;
                out.constant += coefficient * additive_deltas[var_id];
            }
        }
        out
    }
}

pub fn build_restricted_task(task: &dyn AbstractNumericTask) -> Result<Option<RestrictedTask>> {
    let num_numeric_vars = task.numeric_variables().len();
    if num_numeric_vars == 0 || task.comparison_axioms().is_empty() {
        info!("restricted task: domain has no promotable roots, using original task");
        return Ok(None);
    }

    let initial_numeric = task.get_initial_numeric_state_values().to_vec();
    let assignment_lookup = build_assignment_lookup(task);
    let mut linearizer = Linearizer {
        task,
        assignment_lookup,
        initial_numeric: &initial_numeric,
        memo: vec![None; num_numeric_vars],
        visiting: vec![false; num_numeric_vars],
    };

    let mut root_var_ids = BTreeSet::new();
    for comparison_axiom_id in 0..task.comparison_axioms().len() {
        let tree = ComparisonTree::from_task(task, comparison_axiom_id).map_err(|e| {
            anyhow::anyhow!("failed to inspect comparison axiom {comparison_axiom_id}: {e:?}")
        })?;
        root_var_ids.insert(tree.left_numeric_var_id);
        root_var_ids.insert(tree.right_numeric_var_id);
    }

    let mut root_exprs = BTreeMap::new();
    for &root_var_id in &root_var_ids {
        let expr = linearizer.linearize(root_var_id)?;
        match task.numeric_variables()[root_var_id].get_type() {
            NumericType::Regular | NumericType::Constant | NumericType::Cost => {}
            NumericType::Derived => {
                let deps = expr.non_zero_vars();
                if deps.is_empty() {
                    continue;
                }
                root_exprs.insert(root_var_id, expr);
            }
        }
    }

    if root_exprs.is_empty() {
        info!("restricted task: domain has no promotable roots, using original task");
        return Ok(None);
    }

    let mut numeric_var_ids: Vec<usize> = root_var_ids
        .iter()
        .copied()
        .filter(|&numeric_var_id| {
            !matches!(
                task.numeric_variables()[numeric_var_id].get_type(),
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

fn build_task(
    task: &dyn AbstractNumericTask,
    initial_numeric: &[f64],
    root_exprs: &BTreeMap<usize, AffineExpression>,
    numeric_var_ids: &[usize],
) -> Result<Option<RestrictedTask>> {
    let num_original_numeric = task.numeric_variables().len();
    let mut original_to_transformed = vec![None; num_original_numeric];
    let mut transformed_to_expr = Vec::new();
    let mut numeric_variables = Vec::new();
    let mut numeric_initial = Vec::new();

    for &original_id in numeric_var_ids {
        let expr = root_exprs.get(&original_id).cloned().unwrap_or_else(|| {
            match task.numeric_variables()[original_id].get_type() {
                NumericType::Constant | NumericType::Cost => {
                    AffineExpression::constant(initial_numeric[original_id], num_original_numeric)
                }
                _ => AffineExpression::var(original_id, num_original_numeric),
            }
        });
        let transformed_id = numeric_variables.len();
        original_to_transformed[original_id] = Some(transformed_id);
        let original_name = task.numeric_variables()[original_id].name();
        let name = if root_exprs.contains_key(&original_id) {
            format!("{}|{}", restricted_shape_prefix(&expr), original_name)
        } else {
            original_name.to_string()
        };
        let numeric_type = if root_exprs.contains_key(&original_id) {
            NumericType::Regular
        } else {
            task.numeric_variables()[original_id].get_type().clone()
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
            &mut constants,
            &mut numeric_variables,
            &mut numeric_initial,
            &mut transformed_to_expr,
        )?);
    }

    let mut comparison_axioms = Vec::new();
    for comparison_axiom in task.comparison_axioms() {
        let Some(left) = original_to_transformed[comparison_axiom.get_left_var_id()] else {
            continue;
        };
        let Some(right) = original_to_transformed[comparison_axiom.get_right_var_id()] else {
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
            .get(var_id)
            .and_then(|mapped| *mapped)
    });

    let task = NumericRootTask::new(
        1,
        Metric::new(task.metric().is_min(), metric_var_id),
        variables,
        numeric_variables,
        (0..task.get_num_goals())
            .map(|goal_id| task.get_goal_fact(goal_id).clone())
            .collect(),
        vec![],
        task.get_initial_propositional_state_values().to_vec(),
        numeric_initial,
        operators,
        task.axioms().clone(),
        comparison_axioms,
        Vec::<AssignmentAxiom>::new(),
        ExplicitFact::new(0, 0),
    );

    Ok(Some(RestrictedTask { task }))
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
            (!comparison_vars.contains(&var_id))
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
            let new_layer = if comparison_vars.contains(&var_id) {
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
    task: &dyn AbstractNumericTask,
    operator: &Operator,
    initial_numeric: &[f64],
    root_exprs: &BTreeMap<usize, AffineExpression>,
    original_to_transformed: &[Option<usize>],
    constants: &mut ConstantPool,
    numeric_variables: &mut Vec<NumericVariable>,
    numeric_initial: &mut Vec<f64>,
    transformed_to_expr: &mut Vec<AffineExpression>,
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
            constants,
            numeric_variables,
            numeric_initial,
            transformed_to_expr,
        );
    }

    let mut deltas = vec![0.0; task.numeric_variables().len()];
    for effect in operator.assignment_effects() {
        ensure!(
            !effect.is_conditional() && effect.conditions().is_empty(),
            "restricted task does not support conditional numeric effects in operator {}",
            operator.name()
        );
        let source_value = match task.numeric_variables()[effect.var_id()].get_type() {
            NumericType::Constant | NumericType::Cost => initial_numeric[effect.var_id()],
            _ => bail!(
                "restricted task requires constant RHS numeric effects in operator {}",
                operator.name()
            ),
        };
        match effect.operation() {
            AssignmentOperation::Plus => deltas[effect.affected_var_id()] += source_value,
            AssignmentOperation::Minus => deltas[effect.affected_var_id()] -= source_value,
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

    let mut assignment_effects = Vec::new();
    for (original_id, mapped) in original_to_transformed.iter().enumerate() {
        let Some(transformed_id) = *mapped else {
            continue;
        };
        let delta = root_exprs
            .get(&original_id)
            .map(|expr| {
                expr.coefficients
                    .iter()
                    .zip(deltas.iter())
                    .map(|(&coefficient, &delta)| coefficient * delta)
                    .sum::<f64>()
            })
            .unwrap_or(deltas[original_id]);
        if approx_eq(delta, 0.0) {
            continue;
        }
        let const_id = constants.get_or_insert(
            delta,
            numeric_variables,
            numeric_initial,
            transformed_to_expr,
            task.numeric_variables().len(),
        );
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
    initial_numeric: &[f64],
    original_to_transformed: &[Option<usize>],
    constants: &mut ConstantPool,
    numeric_variables: &mut Vec<NumericVariable>,
    numeric_initial: &mut Vec<f64>,
    transformed_to_expr: &mut Vec<AffineExpression>,
) -> Result<Operator> {
    let mut additive_deltas = vec![0.0; task.numeric_variables().len()];
    let mut assigned_constants = vec![None; task.numeric_variables().len()];
    for effect in operator.assignment_effects() {
        ensure!(
            !effect.is_conditional() && effect.conditions().is_empty(),
            "restricted task does not support conditional numeric effects in operator {}",
            operator.name()
        );
        let source_value = match task.numeric_variables()[effect.var_id()].get_type() {
            NumericType::Constant | NumericType::Cost => initial_numeric[effect.var_id()],
            _ => bail!(
                "restricted task requires constant RHS numeric effects in operator {}",
                operator.name()
            ),
        };
        match effect.operation() {
            AssignmentOperation::Plus => {
                ensure!(
                    assigned_constants[effect.affected_var_id()].is_none(),
                    "restricted task does not support mixed assignment/additive numeric effects on one variable in operator {}",
                    operator.name()
                );
                additive_deltas[effect.affected_var_id()] += source_value;
            }
            AssignmentOperation::Minus => {
                ensure!(
                    assigned_constants[effect.affected_var_id()].is_none(),
                    "restricted task does not support mixed assignment/additive numeric effects on one variable in operator {}",
                    operator.name()
                );
                additive_deltas[effect.affected_var_id()] -= source_value;
            }
            AssignmentOperation::Assign => {
                ensure!(
                    approx_eq(additive_deltas[effect.affected_var_id()], 0.0)
                        && assigned_constants[effect.affected_var_id()].is_none(),
                    "restricted task does not support multiple numeric effects on an assigned variable in operator {}",
                    operator.name()
                );
                assigned_constants[effect.affected_var_id()] = Some(source_value);
            }
            AssignmentOperation::Times | AssignmentOperation::Divide => bail!(
                "restricted task does not support non-additive numeric effects in operator {}",
                operator.name()
            ),
        }
    }

    let mut assignment_effects = Vec::new();
    for (original_id, mapped) in original_to_transformed.iter().enumerate() {
        let Some(transformed_id) = *mapped else {
            continue;
        };
        let expr = &transformed_to_expr[transformed_id];
        let successor_expr = expr.apply_effects(&additive_deltas, &assigned_constants);
        let delta_expr = successor_expr.clone().sub(expr.clone());
        if delta_expr.non_zero_vars().is_empty() {
            let delta = delta_expr.constant;
            if approx_eq(delta, 0.0) {
                continue;
            }
            let const_id = constants.get_or_insert(
                delta,
                numeric_variables,
                numeric_initial,
                transformed_to_expr,
                task.numeric_variables().len(),
            );
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
            "restricted task cannot express assignment effect on transformed numeric variable {original_id} in operator {}",
            operator.name()
        );
        let const_id = constants.get_or_insert(
            successor_expr.constant,
            numeric_variables,
            numeric_initial,
            transformed_to_expr,
            task.numeric_variables().len(),
        );
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
    by_bits: HashMap<u64, usize>,
}

impl ConstantPool {
    fn get_or_insert(
        &mut self,
        value: f64,
        numeric_variables: &mut Vec<NumericVariable>,
        numeric_initial: &mut Vec<f64>,
        transformed_to_expr: &mut Vec<AffineExpression>,
        num_original_numeric: usize,
    ) -> usize {
        let bits = float_tolerance::canonical_bits(value);
        if let Some(&id) = self.by_bits.get(&bits) {
            return id;
        }
        let id = numeric_variables.len();
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
    assignment_lookup: Vec<Option<usize>>,
    initial_numeric: &'a [f64],
    memo: Vec<Option<AffineExpression>>,
    visiting: Vec<bool>,
}

impl Linearizer<'_> {
    fn linearize(&mut self, numeric_var_id: usize) -> Result<AffineExpression> {
        ensure!(
            numeric_var_id < self.task.numeric_variables().len(),
            "numeric variable {numeric_var_id} out of bounds"
        );
        if let Some(expr) = &self.memo[numeric_var_id] {
            return Ok(expr.clone());
        }
        ensure!(
            !self.visiting[numeric_var_id],
            "cycle in numeric assignment axioms at numeric variable {numeric_var_id}"
        );
        self.visiting[numeric_var_id] = true;

        let num_vars = self.task.numeric_variables().len();
        let expr = match self.task.numeric_variables()[numeric_var_id].get_type() {
            NumericType::Regular => AffineExpression::var(numeric_var_id, num_vars),
            NumericType::Constant | NumericType::Cost => {
                AffineExpression::constant(self.initial_numeric[numeric_var_id], num_vars)
            }
            NumericType::Derived => {
                let axiom_id = self.assignment_lookup[numeric_var_id].with_context(|| {
                    format!("missing assignment axiom for numeric variable {numeric_var_id}")
                })?;
                let axiom = &self.task.assignment_axioms()[axiom_id];
                let lhs = self.linearize(axiom.get_left_var_id())?;
                let rhs = self.linearize(axiom.get_right_var_id())?;
                match axiom.get_operator() {
                    CalOperator::Sum => lhs.add(rhs),
                    CalOperator::Difference => lhs.sub(rhs),
                    CalOperator::Product => {
                        if rhs.non_zero_vars().is_empty() {
                            lhs.scale(rhs.constant)
                        } else if lhs.non_zero_vars().is_empty() {
                            rhs.scale(lhs.constant)
                        } else {
                            bail!("numeric assignment axiom {axiom_id} is nonlinear")
                        }
                    }
                    CalOperator::Division => {
                        ensure!(
                            rhs.non_zero_vars().is_empty() && !approx_eq(rhs.constant, 0.0),
                            "numeric assignment axiom {axiom_id} has non-constant or zero divisor"
                        );
                        lhs.scale(1.0 / rhs.constant)
                    }
                }
            }
        };

        self.visiting[numeric_var_id] = false;
        self.memo[numeric_var_id] = Some(expr.clone());
        Ok(expr)
    }
}

fn build_assignment_lookup(task: &dyn AbstractNumericTask) -> Vec<Option<usize>> {
    let mut lookup = vec![None; task.numeric_variables().len()];
    for (axiom_id, axiom) in task.assignment_axioms().iter().enumerate() {
        if axiom.get_affected_var_id() < lookup.len() {
            lookup[axiom.get_affected_var_id()] = Some(axiom_id);
        }
    }
    lookup
}

fn approx_eq(lhs: f64, rhs: f64) -> bool {
    (lhs - rhs).abs() <= 1e-12
}

fn restricted_shape_prefix(expr: &AffineExpression) -> String {
    let mut coefficients = expr
        .coefficients
        .iter()
        .copied()
        .filter(|coefficient| !approx_eq(*coefficient, 0.0))
        .collect::<Vec<_>>();
    coefficients.sort_by(|left, right| left.total_cmp(right));
    let shape = coefficients
        .iter()
        .map(|coefficient| coefficient.to_string())
        .collect::<Vec<_>>()
        .join(",");
    format!("rt-shape:{shape}")
}

#[cfg(test)]
mod tests {
    use planforge_sas::numeric::axioms::ComparisonOperator;
    use planforge_sas::numeric::numeric_task::ExplicitVariable;

    use super::*;

    #[test]
    fn restricted_task_lifts_derived_condition_root_and_maps_effects() {
        let variables = vec![ExplicitVariable::new(
            2,
            "cmp".into(),
            vec!["true".into(), "false".into()],
            Some(0),
            1,
        )];
        let numeric_variables = vec![
            NumericVariable::new("x".into(), NumericType::Regular, None),
            NumericVariable::new("y".into(), NumericType::Regular, None),
            NumericVariable::new("u".into(), NumericType::Derived, Some(0)),
            NumericVariable::new("limit".into(), NumericType::Constant, None),
            NumericVariable::new("one".into(), NumericType::Constant, None),
        ];
        let operator = Operator::new(
            "inc-x".into(),
            vec![],
            vec![],
            vec![AssignmentEffect::new(
                0,
                AssignmentOperation::Plus,
                4,
                false,
                vec![],
            )],
            1,
        );
        let task = NumericRootTask::new(
            1,
            Metric::new(true, None),
            variables,
            numeric_variables,
            vec![ExplicitFact::new(0, 0)],
            vec![],
            vec![1],
            vec![2.0, 3.0, 5.0, 10.0, 1.0],
            vec![operator],
            vec![],
            vec![ComparisonAxiom::new(
                0,
                2,
                3,
                ComparisonOperator::LessThanOrEqual,
            )],
            vec![AssignmentAxiom::new(2, CalOperator::Sum, 0, 1)],
            ExplicitFact::new(0, 0),
        );

        let restricted = build_restricted_task(&task)
            .unwrap()
            .expect("task should be restricted");
        let transformed = restricted.task();

        assert_eq!(transformed.numeric_variables().len(), 3);
        assert!(transformed.numeric_variables()[0].name().ends_with("|u"));
        assert_eq!(transformed.numeric_variables()[1].name(), "limit");
        assert_eq!(transformed.get_variable_axiom_layer(0), Ok(Some(0)));
        assert_eq!(
            transformed.get_initial_numeric_state_values().as_slice(),
            &[5.0, 10.0, 1.0]
        );
        assert_eq!(transformed.comparison_axioms()[0].get_left_var_id(), 0);
        assert_eq!(transformed.comparison_axioms()[0].get_right_var_id(), 1);

        let assignment_effects = transformed.get_operators()[0].assignment_effects();
        assert_eq!(assignment_effects.len(), 1);
        assert_eq!(assignment_effects[0].affected_var_id(), 0);
        assert_eq!(assignment_effects[0].var_id(), 2);
        assert_eq!(
            transformed.get_initial_numeric_state_values()[assignment_effects[0].var_id()],
            1.0
        );
    }

    #[test]
    fn restricted_task_supports_assignment_to_constant_when_views_stay_simple() {
        let variables = vec![ExplicitVariable::new(
            2,
            "cmp".into(),
            vec!["true".into(), "false".into()],
            Some(0),
            1,
        )];
        let numeric_variables = vec![
            NumericVariable::new("fuel".into(), NumericType::Regular, None),
            NumericVariable::new("capacity".into(), NumericType::Constant, None),
            NumericVariable::new("capacity-minus-fuel".into(), NumericType::Derived, Some(0)),
        ];
        let operator = Operator::new(
            "refuel".into(),
            vec![],
            vec![],
            vec![AssignmentEffect::new(
                0,
                AssignmentOperation::Assign,
                1,
                false,
                vec![],
            )],
            1,
        );
        let task = NumericRootTask::new(
            1,
            Metric::new(true, None),
            variables,
            numeric_variables,
            vec![ExplicitFact::new(0, 0)],
            vec![],
            vec![1],
            vec![4000.0, 6000.0, 2000.0],
            vec![operator],
            vec![],
            vec![ComparisonAxiom::new(
                0,
                2,
                1,
                ComparisonOperator::GreaterThan,
            )],
            vec![AssignmentAxiom::new(2, CalOperator::Difference, 1, 0)],
            ExplicitFact::new(0, 0),
        );

        let restricted = build_restricted_task(&task)
            .unwrap()
            .expect("task should be restricted");
        let transformed = restricted.task();
        let assignment_effects = transformed.get_operators()[0].assignment_effects();

        assert_eq!(assignment_effects.len(), 1);
        assert_eq!(assignment_effects[0].affected_var_id(), 1);
        assert_eq!(
            assignment_effects[0].operation(),
            &AssignmentOperation::Assign
        );
        assert_eq!(
            transformed.get_initial_numeric_state_values()[assignment_effects[0].var_id()],
            0.0
        );
    }

    #[test]
    fn restricted_task_returns_none_when_domain_has_no_derived_roots() {
        let variables = vec![ExplicitVariable::new(
            2,
            "cmp".into(),
            vec!["true".into(), "false".into()],
            Some(0),
            1,
        )];
        let numeric_variables = vec![
            NumericVariable::new("x".into(), NumericType::Regular, None),
            NumericVariable::new("limit".into(), NumericType::Constant, None),
        ];
        let task = NumericRootTask::new(
            1,
            Metric::new(true, None),
            variables,
            numeric_variables,
            vec![ExplicitFact::new(0, 0)],
            vec![],
            vec![1],
            vec![2.0, 10.0],
            vec![],
            vec![],
            vec![ComparisonAxiom::new(
                0,
                0,
                1,
                ComparisonOperator::LessThanOrEqual,
            )],
            vec![],
            ExplicitFact::new(0, 0),
        );

        assert!(build_restricted_task(&task).unwrap().is_none());
    }

    #[test]
    fn restricted_task_reports_unsupported_effect_as_error() {
        let variables = vec![ExplicitVariable::new(
            2,
            "cmp".into(),
            vec!["true".into(), "false".into()],
            Some(0),
            1,
        )];
        let numeric_variables = vec![
            NumericVariable::new("x".into(), NumericType::Regular, None),
            NumericVariable::new("y".into(), NumericType::Regular, None),
            NumericVariable::new("u".into(), NumericType::Derived, Some(0)),
            NumericVariable::new("limit".into(), NumericType::Constant, None),
        ];
        let operator = Operator::new(
            "scale-x".into(),
            vec![],
            vec![],
            vec![AssignmentEffect::new(
                0,
                AssignmentOperation::Times,
                3,
                false,
                vec![],
            )],
            1,
        );
        let task = NumericRootTask::new(
            1,
            Metric::new(true, None),
            variables,
            numeric_variables,
            vec![ExplicitFact::new(0, 0)],
            vec![],
            vec![1],
            vec![2.0, 3.0, 5.0, 10.0],
            vec![operator],
            vec![],
            vec![ComparisonAxiom::new(
                0,
                2,
                3,
                ComparisonOperator::LessThanOrEqual,
            )],
            vec![AssignmentAxiom::new(2, CalOperator::Sum, 0, 1)],
            ExplicitFact::new(0, 0),
        );

        let error = build_restricted_task(&task).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("does not support non-additive numeric effects"),
            "{error:#}"
        );
    }
}
