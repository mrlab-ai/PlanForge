use planforge_sas::axioms::{AssignmentAxiom, CalOperator, ComparisonAxiom, ComparisonOperator};
use planforge_sas::numeric_task::{
    ExplicitFact, ExplicitValueIndex, ExplicitVariable, Metric, NumericRootTask,
    NumericRootTaskParts, NumericValue, NumericVariable, Operator, OperatorCost, VariableIndex,
};

use super::*;

#[test]
fn estimates_regular_numeric_domain_size_from_bounds_and_effects() {
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 1,
        metric: Metric::new(true, None),
        variables: vec![ExplicitVariable::new(
            3,
            "cmp".to_string(),
            vec!["t".to_string(), "f".to_string(), "u".to_string()],
            Some(0),
            ExplicitValueIndex::new(2),
        )],
        numeric_variables: vec![
            NumericVariable::new("c1".to_string(), NumericType::Constant, None),
            NumericVariable::new("x".to_string(), NumericType::Regular, None),
        ],
        goals: vec![ExplicitFact::propositional(0, 0)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(2)],
        numeric_state: vec![NumericValue::new(1.0), NumericValue::new(0.0)],
        operators: vec![Operator::new(
            "inc".to_string(),
            vec![],
            vec![],
            vec![planforge_sas::numeric_task::AssignmentEffect::new(
                VariableIndex::from_usize(1),
                AssignmentOperation::Plus,
                VariableIndex::from_usize(0),
                false,
                vec![],
            )],
            OperatorCost::new(1),
        )],
        axioms: vec![],
        comparison_axioms: vec![ComparisonAxiom::new(
            VariableIndex::new(0),
            VariableIndex::new(1),
            VariableIndex::new(0),
            ComparisonOperator::GreaterThanOrEqual,
        )],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    let estimator = NumericSizeEstimator::new(&task);

    assert_eq!(
        estimator.estimate_domain_size(VariableIndex::from_usize(1)),
        3
    );
}

#[test]
#[should_panic(expected = "numeric PDB size estimation requires a restricted task")]
fn rejects_unrestricted_numeric_conditions() {
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 1,
        metric: Metric::new(true, None),
        variables: vec![ExplicitVariable::new(
            3,
            "cmp".to_string(),
            vec!["t".to_string(), "f".to_string(), "u".to_string()],
            Some(1),
            ExplicitValueIndex::new(2),
        )],
        numeric_variables: vec![
            NumericVariable::new("c1".to_string(), NumericType::Constant, None),
            NumericVariable::new("c5".to_string(), NumericType::Constant, None),
            NumericVariable::new("x".to_string(), NumericType::Regular, None),
            NumericVariable::new("y".to_string(), NumericType::Regular, None),
            NumericVariable::new("sum".to_string(), NumericType::Derived, Some(0)),
        ],
        goals: vec![ExplicitFact::propositional(0, 0)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(2)],
        numeric_state: vec![
            NumericValue::new(1.0),
            NumericValue::new(5.0),
            NumericValue::new(0.0),
            NumericValue::new(1.0),
            NumericValue::new(0.0),
        ],
        operators: vec![Operator::new(
            "inc-x".to_string(),
            vec![],
            vec![],
            vec![planforge_sas::numeric_task::AssignmentEffect::new(
                VariableIndex::from_usize(2),
                AssignmentOperation::Plus,
                VariableIndex::from_usize(0),
                false,
                vec![],
            )],
            OperatorCost::new(1),
        )],
        axioms: vec![],
        comparison_axioms: vec![ComparisonAxiom::new(
            VariableIndex::new(0),
            VariableIndex::new(4),
            VariableIndex::new(1),
            ComparisonOperator::GreaterThanOrEqual,
        )],
        assignment_axioms: vec![AssignmentAxiom::new(
            VariableIndex::from_usize(4),
            CalOperator::Sum,
            VariableIndex::from_usize(2),
            VariableIndex::from_usize(3),
        )],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    let _ = NumericSizeEstimator::new(&task);
}
