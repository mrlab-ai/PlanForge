use std::fmt;

use crate::numeric::axioms::{AssignmentAxiom, CalOperator};
use crate::numeric::numeric_task::{
    AbstractNumericTask, AssignmentEffect, AssignmentOperation, Fact, NumericType,
};

#[derive(Debug, Clone, PartialEq)]
pub struct LinearExpression {
    pub coefficients: Vec<f64>,
    pub constant: f64,
}

impl LinearExpression {
    pub fn zero(num_numeric_vars: usize) -> Self {
        Self {
            coefficients: vec![0.0; num_numeric_vars],
            constant: 0.0,
        }
    }

    pub fn variable(num_numeric_vars: usize, numeric_var_id: usize) -> Self {
        let mut coefficients = vec![0.0; num_numeric_vars];
        coefficients[numeric_var_id] = 1.0;
        Self {
            coefficients,
            constant: 0.0,
        }
    }

    pub fn constant(num_numeric_vars: usize, value: f64) -> Self {
        Self {
            coefficients: vec![0.0; num_numeric_vars],
            constant: value,
        }
    }

    pub fn is_constant(&self) -> bool {
        self.coefficients
            .iter()
            .all(|coefficient| coefficient.abs() < 1e-12)
    }

    pub fn add(&self, rhs: &Self) -> Self {
        let mut result = self.clone();
        for (left, right) in result.coefficients.iter_mut().zip(rhs.coefficients.iter()) {
            *left += right;
        }
        result.constant += rhs.constant;
        result
    }

    pub fn subtract(&self, rhs: &Self) -> Self {
        let mut result = self.clone();
        for (left, right) in result.coefficients.iter_mut().zip(rhs.coefficients.iter()) {
            *left -= right;
        }
        result.constant -= rhs.constant;
        result
    }

    pub fn scale(&self, factor: f64) -> Self {
        let mut result = self.clone();
        for coefficient in &mut result.coefficients {
            *coefficient *= factor;
        }
        result.constant *= factor;
        result
    }

    pub fn evaluate(&self, numeric_values: &[f64]) -> f64 {
        self.constant
            + self
                .coefficients
                .iter()
                .zip(numeric_values.iter())
                .map(|(coefficient, value)| coefficient * value)
                .sum::<f64>()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LinearNumericEffect {
    pub affected_var_id: usize,
    pub source_var_id: usize,
    pub operation: AssignmentOperation,
    pub conditions: Vec<Fact>,
    pub is_conditional: bool,
    pub delta: LinearExpression,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinearizationError {
    InvalidNumericVarId {
        numeric_var_id: usize,
        len: usize,
    },
    InvalidOperatorId {
        operator_id: usize,
        len: usize,
    },
    InvalidAssignmentEffectId {
        assignment_effect_id: usize,
        len: usize,
    },
    MissingAssignmentAxiom {
        numeric_var_id: usize,
    },
    CycleDetected {
        numeric_var_id: usize,
    },
    NonLinearAssignmentAxiom {
        numeric_var_id: usize,
        operator: &'static str,
    },
    NonLinearAssignmentEffect {
        affected_var_id: usize,
        operator: &'static str,
    },
    DivisionByZeroConstant {
        numeric_var_id: usize,
    },
}

impl fmt::Display for LinearizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidNumericVarId {
                numeric_var_id,
                len,
            } => {
                write!(f, "invalid numeric var id {numeric_var_id}; len={len}")
            }
            Self::InvalidOperatorId { operator_id, len } => {
                write!(f, "invalid operator id {operator_id}; len={len}")
            }
            Self::InvalidAssignmentEffectId {
                assignment_effect_id,
                len,
            } => write!(
                f,
                "invalid assignment effect id {assignment_effect_id}; len={len}"
            ),
            Self::MissingAssignmentAxiom { numeric_var_id } => {
                write!(
                    f,
                    "missing assignment axiom for derived numeric var {numeric_var_id}"
                )
            }
            Self::CycleDetected { numeric_var_id } => {
                write!(
                    f,
                    "cycle detected while expanding numeric var {numeric_var_id}"
                )
            }
            Self::NonLinearAssignmentAxiom {
                numeric_var_id,
                operator,
            } => write!(
                f,
                "numeric var {numeric_var_id} depends on non-linear assignment axiom operator {operator}"
            ),
            Self::NonLinearAssignmentEffect {
                affected_var_id,
                operator,
            } => write!(
                f,
                "assignment effect on numeric var {affected_var_id} is non-linear under operator {operator}"
            ),
            Self::DivisionByZeroConstant { numeric_var_id } => write!(
                f,
                "numeric var {numeric_var_id} depends on division by a zero constant"
            ),
        }
    }
}

pub fn build_assignment_axiom_lookup<T: AbstractNumericTask + ?Sized>(
    task: &T,
) -> Vec<Option<usize>> {
    let mut lookup = vec![None; task.numeric_variables().len()];
    for (axiom_id, axiom) in task.assignment_axioms().iter().enumerate() {
        let affected_var_id = axiom.get_affected_var_id() as usize;
        if affected_var_id < lookup.len() {
            lookup[affected_var_id] = Some(axiom_id);
        }
    }
    lookup
}

pub fn linearize_numeric_var<T: AbstractNumericTask + ?Sized>(
    task: &T,
    numeric_var_id: usize,
) -> Result<LinearExpression, LinearizationError> {
    let assignment_lookup = build_assignment_axiom_lookup(task);
    let initial_numeric_values = task.get_initial_numeric_state_values().to_vec();
    let mut visiting = vec![false; task.numeric_variables().len()];
    linearize_numeric_var_with_lookup(
        task,
        numeric_var_id,
        &assignment_lookup,
        &initial_numeric_values,
        &mut visiting,
    )
}

pub fn linearize_operator_assignment_effects<T: AbstractNumericTask + ?Sized>(
    task: &T,
    operator_id: usize,
) -> Result<Vec<LinearNumericEffect>, LinearizationError> {
    let operators = task.get_operators();
    let operator = operators
        .get(operator_id)
        .ok_or(LinearizationError::InvalidOperatorId {
            operator_id,
            len: operators.len(),
        })?;
    let assignment_lookup = build_assignment_axiom_lookup(task);
    let initial_numeric_values = task.get_initial_numeric_state_values().to_vec();
    let num_numeric_vars = task.numeric_variables().len();
    let mut effects = Vec::with_capacity(operator.assignment_effects().len());

    for (assignment_effect_id, assignment_effect) in
        operator.assignment_effects().iter().enumerate()
    {
        let mut visiting = vec![false; num_numeric_vars];
        let source_var_id = assignment_effect.var_id() as usize;
        let source_expression = linearize_numeric_var_with_lookup(
            task,
            source_var_id,
            &assignment_lookup,
            &initial_numeric_values,
            &mut visiting,
        )?;
        let affected_var_id = assignment_effect.affected_var_id() as usize;
        if affected_var_id >= num_numeric_vars {
            return Err(LinearizationError::InvalidAssignmentEffectId {
                assignment_effect_id,
                len: operator.assignment_effects().len(),
            });
        }

        let delta = linearize_assignment_delta(
            num_numeric_vars,
            affected_var_id,
            assignment_effect,
            &source_expression,
        )?;
        effects.push(LinearNumericEffect {
            affected_var_id,
            source_var_id,
            operation: assignment_effect.operation().clone(),
            conditions: assignment_effect.conditions().clone(),
            is_conditional: assignment_effect.is_conditional(),
            delta,
        });
    }

    Ok(effects)
}

fn linearize_numeric_var_with_lookup<T: AbstractNumericTask + ?Sized>(
    task: &T,
    numeric_var_id: usize,
    assignment_lookup: &[Option<usize>],
    initial_numeric_values: &[f64],
    visiting: &mut [bool],
) -> Result<LinearExpression, LinearizationError> {
    let num_numeric_vars = task.numeric_variables().len();
    if numeric_var_id >= num_numeric_vars {
        return Err(LinearizationError::InvalidNumericVarId {
            numeric_var_id,
            len: num_numeric_vars,
        });
    }
    if visiting[numeric_var_id] {
        return Err(LinearizationError::CycleDetected { numeric_var_id });
    }

    let numeric_var = &task.numeric_variables()[numeric_var_id];
    match numeric_var.get_type() {
        NumericType::Regular => Ok(LinearExpression::variable(num_numeric_vars, numeric_var_id)),
        NumericType::Constant | NumericType::Cost => Ok(LinearExpression::constant(
            num_numeric_vars,
            *initial_numeric_values.get(numeric_var_id).unwrap_or(&0.0),
        )),
        NumericType::Derived => {
            let axiom_id = assignment_lookup[numeric_var_id]
                .ok_or(LinearizationError::MissingAssignmentAxiom { numeric_var_id })?;
            let axiom = task
                .assignment_axioms()
                .get(axiom_id)
                .ok_or(LinearizationError::MissingAssignmentAxiom { numeric_var_id })?;
            visiting[numeric_var_id] = true;
            let lhs = linearize_numeric_var_with_lookup(
                task,
                axiom.get_left_var_id() as usize,
                assignment_lookup,
                initial_numeric_values,
                visiting,
            )?;
            let rhs = linearize_numeric_var_with_lookup(
                task,
                axiom.get_right_var_id() as usize,
                assignment_lookup,
                initial_numeric_values,
                visiting,
            )?;
            visiting[numeric_var_id] = false;
            linearize_assignment_axiom_expr(num_numeric_vars, numeric_var_id, axiom, &lhs, &rhs)
        }
    }
}

fn linearize_assignment_axiom_expr(
    num_numeric_vars: usize,
    numeric_var_id: usize,
    axiom: &AssignmentAxiom,
    lhs: &LinearExpression,
    rhs: &LinearExpression,
) -> Result<LinearExpression, LinearizationError> {
    match axiom.get_operator() {
        CalOperator::Sum => Ok(lhs.add(rhs)),
        CalOperator::Difference => Ok(lhs.subtract(rhs)),
        CalOperator::Product => {
            if lhs.is_constant() {
                Ok(rhs.scale(lhs.constant))
            } else if rhs.is_constant() {
                Ok(lhs.scale(rhs.constant))
            } else {
                Err(LinearizationError::NonLinearAssignmentAxiom {
                    numeric_var_id,
                    operator: "*",
                })
            }
        }
        CalOperator::Division => {
            if !rhs.is_constant() {
                return Err(LinearizationError::NonLinearAssignmentAxiom {
                    numeric_var_id,
                    operator: "/",
                });
            }
            if rhs.constant.abs() < 1e-12 {
                return Err(LinearizationError::DivisionByZeroConstant { numeric_var_id });
            }
            Ok(lhs.scale(1.0 / rhs.constant))
        }
    }
}

fn linearize_assignment_delta(
    num_numeric_vars: usize,
    affected_var_id: usize,
    assignment_effect: &AssignmentEffect,
    source_expression: &LinearExpression,
) -> Result<LinearExpression, LinearizationError> {
    let target_expression = LinearExpression::variable(num_numeric_vars, affected_var_id);
    match assignment_effect.operation() {
        AssignmentOperation::Assign => Ok(source_expression.subtract(&target_expression)),
        AssignmentOperation::Plus => Ok(source_expression.clone()),
        AssignmentOperation::Minus => Ok(source_expression.scale(-1.0)),
        AssignmentOperation::Times => {
            if !source_expression.is_constant() {
                return Err(LinearizationError::NonLinearAssignmentEffect {
                    affected_var_id,
                    operator: "*",
                });
            }
            Ok(target_expression.scale(source_expression.constant - 1.0))
        }
        AssignmentOperation::Divide => {
            if !source_expression.is_constant() {
                return Err(LinearizationError::NonLinearAssignmentEffect {
                    affected_var_id,
                    operator: "/",
                });
            }
            if source_expression.constant.abs() < 1e-12 {
                return Err(LinearizationError::DivisionByZeroConstant {
                    numeric_var_id: affected_var_id,
                });
            }
            Ok(target_expression.scale(1.0 / source_expression.constant - 1.0))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::axioms::{AssignmentAxiom, ComparisonAxiom, PropositionalAxiom};
    use crate::numeric::numeric_task::{
        AssignmentEffect, ExplicitVariable, Metric, NumericRootTask, NumericVariable, Operator,
    };

    fn simple_var(name: &str, values: &[&str], axiom_layer: i32) -> ExplicitVariable {
        ExplicitVariable::new(
            values.len() as u32,
            name.to_string(),
            values.iter().map(|value| value.to_string()).collect(),
            axiom_layer,
            0,
        )
    }

    fn base_task(
        numeric_variables: Vec<NumericVariable>,
        operators: Vec<Operator>,
        assignment_axioms: Vec<AssignmentAxiom>,
        numeric_state: Vec<f64>,
    ) -> NumericRootTask {
        NumericRootTask::new(
            3,
            Metric::new(true, -1),
            vec![simple_var("p", &["f", "t"], -1)],
            numeric_variables,
            vec![Fact::new(0, 1)],
            vec![],
            vec![0],
            numeric_state,
            operators,
            vec![],
            Vec::<ComparisonAxiom>::new(),
            assignment_axioms,
            (0, 0),
        )
    }

    #[test]
    fn linearizes_derived_sum_expression() {
        let task = base_task(
            vec![
                NumericVariable::new("x".to_string(), NumericType::Regular, -1),
                NumericVariable::new("y".to_string(), NumericType::Regular, -1),
                NumericVariable::new("sum".to_string(), NumericType::Derived, -1),
            ],
            vec![],
            vec![AssignmentAxiom::new(2, CalOperator::Sum, 0, 1)],
            vec![1.0, 2.0, 3.0],
        );

        let expression = linearize_numeric_var(&task, 2).expect("derived sum should be linear");

        assert_eq!(expression.coefficients, vec![1.0, 1.0, 0.0]);
        assert_eq!(expression.constant, 0.0);
    }

    #[test]
    fn linearizes_operator_assignment_effect_through_derived_var() {
        let operators = vec![Operator::new(
            "increase-z".to_string(),
            vec![],
            vec![],
            vec![AssignmentEffect::new(
                3,
                AssignmentOperation::Plus,
                2,
                false,
                vec![],
            )],
            1,
        )];
        let task = base_task(
            vec![
                NumericVariable::new("x".to_string(), NumericType::Regular, -1),
                NumericVariable::new("y".to_string(), NumericType::Regular, -1),
                NumericVariable::new("sum".to_string(), NumericType::Derived, -1),
                NumericVariable::new("z".to_string(), NumericType::Regular, -1),
            ],
            operators,
            vec![AssignmentAxiom::new(2, CalOperator::Sum, 0, 1)],
            vec![1.0, 2.0, 3.0, 0.0],
        );

        let effects = linearize_operator_assignment_effects(&task, 0)
            .expect("operator effect through derived sum should be linear");

        assert_eq!(effects.len(), 1);
        assert_eq!(effects[0].affected_var_id, 3);
        assert_eq!(effects[0].delta.coefficients, vec![1.0, 1.0, 0.0, 0.0]);
        assert_eq!(effects[0].delta.constant, 0.0);
    }

    #[test]
    fn rejects_non_linear_product_of_two_variables() {
        let task = base_task(
            vec![
                NumericVariable::new("x".to_string(), NumericType::Regular, -1),
                NumericVariable::new("y".to_string(), NumericType::Regular, -1),
                NumericVariable::new("prod".to_string(), NumericType::Derived, -1),
            ],
            vec![],
            vec![AssignmentAxiom::new(2, CalOperator::Product, 0, 1)],
            vec![2.0, 3.0, 6.0],
        );

        let error = linearize_numeric_var(&task, 2)
            .expect_err("product of two variables should not linearize");

        assert!(matches!(
            error,
            LinearizationError::NonLinearAssignmentAxiom {
                numeric_var_id: 2,
                operator: "*"
            }
        ));
    }
}
