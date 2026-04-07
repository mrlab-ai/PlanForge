use planners_sas::numeric::axioms::{AssignmentAxiom, CalOperator};
use planners_sas::numeric::numeric_task::{ExplicitVariable, Metric, NumericVariable};
use planners_sas::numeric::{
    axioms::ComparisonOperator,
    numeric_task::{ExplicitFact, NumericRootTask},
};

use super::*;

#[test]
fn estimates_regular_numeric_domain_size_from_bounds_and_effects() {
    let task = NumericRootTask::new(
        1,
        Metric::new(true, None),
        vec![ExplicitVariable::new(
            3,
            "cmp".to_string(),
            vec!["t".to_string(), "f".to_string(), "u".to_string()],
            Some(0),
            2,
        )],
        vec![
            NumericVariable::new("c1".to_string(), NumericType::Constant, None),
            NumericVariable::new("x".to_string(), NumericType::Regular, None),
        ],
        vec![ExplicitFact::new(0, 0)],
        vec![],
        vec![2],
        vec![1.0, 0.0],
        vec![Operator::new(
            "inc".to_string(),
            vec![],
            vec![],
            vec![planners_sas::numeric::numeric_task::AssignmentEffect::new(
                1,
                AssignmentOperation::Plus,
                0,
                false,
                vec![],
            )],
            1,
        )],
        vec![],
        vec![ComparisonAxiom::new(
            0,
            1,
            0,
            ComparisonOperator::GreaterThanOrEqual,
        )],
        vec![],
        ExplicitFact::new(0, 0),
    );

    let estimator = NumericSizeEstimator::new(&task);

    assert_eq!(estimator.estimate_domain_size(1), 3);
}

#[test]
fn estimates_helper_numeric_domain_size_from_derived_expression_effects() {
    let task = NumericRootTask::new(
        1,
        Metric::new(true, None),
        vec![ExplicitVariable::new(
            3,
            "cmp".to_string(),
            vec!["t".to_string(), "f".to_string(), "u".to_string()],
            Some(0),
            2,
        )],
        vec![
            NumericVariable::new("c1".to_string(), NumericType::Constant, None),
            NumericVariable::new("c5".to_string(), NumericType::Constant, None),
            NumericVariable::new("x".to_string(), NumericType::Regular, None),
            NumericVariable::new("y".to_string(), NumericType::Regular, None),
            NumericVariable::new("sum".to_string(), NumericType::Derived, Some(0)),
        ],
        vec![ExplicitFact::new(0, 0)],
        vec![],
        vec![2],
        vec![1.0, 5.0, 0.0, 1.0, 0.0],
        vec![Operator::new(
            "inc-x".to_string(),
            vec![],
            vec![],
            vec![planners_sas::numeric::numeric_task::AssignmentEffect::new(
                2,
                AssignmentOperation::Plus,
                0,
                false,
                vec![],
            )],
            1,
        )],
        vec![],
        vec![ComparisonAxiom::new(
            0,
            4,
            1,
            ComparisonOperator::GreaterThanOrEqual,
        )],
        vec![AssignmentAxiom::new(4, CalOperator::Sum, 2, 3)],
        ExplicitFact::new(0, 0),
    );

    let estimator = NumericSizeEstimator::new(&task);

    assert_eq!(estimator.helper_space_len(), 6);
    assert_eq!(estimator.estimate_domain_size(5), 7);
}
