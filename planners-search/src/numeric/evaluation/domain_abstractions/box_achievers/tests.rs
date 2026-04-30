use super::*;

use planners_sas::numeric::axioms::{ComparisonAxiom, ComparisonOperator};
use planners_sas::numeric::numeric_task::{
    Effect, ExplicitVariable, Metric, NumericRootTask, NumericVariable, Operator,
};

#[test]
fn detects_goal_relevant_numeric_box_achiever() {
    let variables = vec![
        ExplicitVariable::new(
            3,
            "x_ge_1".into(),
            vec!["true".into(), "false".into(), "unknown".into()],
            Some(0),
            2,
        ),
        ExplicitVariable::new(
            3,
            "x_le_5".into(),
            vec!["true".into(), "false".into(), "unknown".into()],
            Some(0),
            2,
        ),
        ExplicitVariable::new(2, "saved".into(), vec!["no".into(), "yes".into()], None, 0),
    ];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("c1".into(), NumericType::Constant, None),
        NumericVariable::new("c5".into(), NumericType::Constant, None),
    ];
    let comparison_axioms = vec![
        ComparisonAxiom::new(0, 0, 1, ComparisonOperator::GreaterThanOrEqual),
        ComparisonAxiom::new(1, 0, 2, ComparisonOperator::LessThanOrEqual),
    ];
    let operator = Operator::new(
        "save".into(),
        vec![ExplicitFact::new(0, COMPARISON_TRUE_VAL), ExplicitFact::new(1, COMPARISON_TRUE_VAL)],
        vec![Effect::new(vec![], 2, Some(0), 1)],
        vec![],
        1,
    );
    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        numeric_variables,
        vec![ExplicitFact::new(2, 1)],
        vec![],
        vec![2, 2, 0],
        vec![0.0, 1.0, 5.0],
        vec![operator],
        vec![],
        comparison_axioms,
        vec![],
        ExplicitFact::new(0, 0),
    );

    let achievers = detect_numeric_box_achievers(&task);
    assert_eq!(achievers.len(), 1);
    assert_eq!(achievers[0].operator_id, 0);
    assert_eq!(achievers[0].achieved_facts, vec![ExplicitFact::new(2, 1)]);
    assert_eq!(achievers[0].bounds.len(), 1);
    assert_eq!(achievers[0].bounds[0].0, 0);
    assert_eq!(achievers[0].bounds[0].1, Interval::new(1.0, 5.0, true, true));
}

#[test]
fn ignores_non_goal_achievers_and_non_constant_bounds() {
    let variables = vec![
        ExplicitVariable::new(
            3,
            "x_lt_y".into(),
            vec!["true".into(), "false".into(), "unknown".into()],
            Some(0),
            2,
        ),
        ExplicitVariable::new(2, "flag".into(), vec!["off".into(), "on".into()], None, 0),
    ];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("y".into(), NumericType::Regular, None),
    ];
    let comparison_axioms = vec![ComparisonAxiom::new(0, 0, 1, ComparisonOperator::LessThan)];
    let operator = Operator::new(
        "toggle".into(),
        vec![ExplicitFact::new(0, COMPARISON_TRUE_VAL)],
        vec![Effect::new(vec![], 1, Some(0), 1)],
        vec![],
        1,
    );
    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        numeric_variables,
        vec![],
        vec![],
        vec![2, 0],
        vec![0.0, 0.0],
        vec![operator],
        vec![],
        comparison_axioms,
        vec![],
        ExplicitFact::new(0, 0),
    );

    assert!(detect_numeric_box_achievers(&task).is_empty());
}