use super::*;
use crate::axioms::{AssignmentAxiom, ComparisonAxiom};
use crate::numeric_task::{
    AssignmentEffect, ExplicitVariable, Metric, NumericRootTask, NumericVariable, Operator,
};

fn simple_var(name: &str, values: &[&str], axiom_layer: Option<usize>) -> ExplicitVariable {
    ExplicitVariable::new(
        values.len(),
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
        Metric::new(true, None),
        vec![simple_var("p", &["f", "t"], None)],
        numeric_variables,
        vec![ExplicitFact::new(0, 1)],
        vec![],
        vec![0],
        numeric_state,
        operators,
        vec![],
        Vec::<ComparisonAxiom>::new(),
        assignment_axioms,
        ExplicitFact::new(0, 0),
    )
}

#[test]
fn linearizes_derived_sum_expression() {
    let task = base_task(
        vec![
            NumericVariable::new("x".to_string(), NumericType::Regular, None),
            NumericVariable::new("y".to_string(), NumericType::Regular, None),
            NumericVariable::new("sum".to_string(), NumericType::Derived, None),
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
            NumericVariable::new("x".to_string(), NumericType::Regular, None),
            NumericVariable::new("y".to_string(), NumericType::Regular, None),
            NumericVariable::new("sum".to_string(), NumericType::Derived, None),
            NumericVariable::new("z".to_string(), NumericType::Regular, None),
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
            NumericVariable::new("x".to_string(), NumericType::Regular, None),
            NumericVariable::new("y".to_string(), NumericType::Regular, None),
            NumericVariable::new("prod".to_string(), NumericType::Derived, None),
        ],
        vec![],
        vec![AssignmentAxiom::new(2, CalOperator::Product, 0, 1)],
        vec![2.0, 3.0, 6.0],
    );

    let error =
        linearize_numeric_var(&task, 2).expect_err("product of two variables should not linearize");

    assert!(matches!(
        error,
        LinearizationError::NonLinearAssignmentAxiom {
            numeric_var_id: 2,
            operator: "*"
        }
    ));
}
