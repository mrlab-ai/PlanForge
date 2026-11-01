use planforge_sas::axioms::{AssignmentAxiom, CalOperator, ComparisonOperator};
use planforge_sas::numeric_task::{ExplicitValueIndex, OperatorCost};

use super::*;

#[test]
fn a_multivariate_condition_becomes_a_threshold_on_an_auxiliary_variable() {
    let variables = vec![ExplicitVariable::new(
        2,
        "cmp".into(),
        vec!["true".into(), "false".into()],
        Some(1),
        ExplicitValueIndex::new(1),
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
            VariableIndex::from_usize(0),
            AssignmentOperation::Plus,
            VariableIndex::from_usize(4),
            false,
            vec![],
        )],
        OperatorCost::new(1),
    );
    let task = Task::new(TaskParts {
        metric: Metric::new(true, None),
        variables,
        numeric_variables,
        goals: vec![ExplicitFact::propositional(0, 0)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(1)],
        numeric_state: vec![
            NumericValue::new(2.0),
            NumericValue::new(3.0),
            NumericValue::new(5.0),
            NumericValue::new(10.0),
            NumericValue::new(1.0),
        ],
        operators: vec![operator],
        axioms: vec![],
        comparison_axioms: vec![ComparisonAxiom::new(
            VariableIndex::from_usize(0),
            VariableIndex::from_usize(2),
            VariableIndex::from_usize(3),
            ComparisonOperator::LessThanOrEqual,
        )],
        assignment_axioms: vec![AssignmentAxiom::new(
            VariableIndex::from_usize(2),
            CalOperator::Sum,
            VariableIndex::from_usize(0),
            VariableIndex::from_usize(1),
        )],
        global_constraint: Some(ExplicitFact::propositional(0, 0)),
    });
    let icaps = build_icaps26_restricted_task(&task)
        .unwrap()
        .expect("ICAPS translation should materialize the multivariate condition");
    let icaps = &icaps;
    let auxiliary_id = icaps
        .numeric_variables()
        .iter()
        .position(|variable| variable.name() == "icaps26-condition-0")
        .expect("ICAPS condition auxiliary is missing");
    assert_eq!(
        icaps.initial_numeric_values()[auxiliary_id],
        NumericValue::new(-5.0)
    );
    assert_eq!(
        icaps.comparison_axioms()[0].get_left_var_id(),
        VariableIndex::from_usize(auxiliary_id)
    );
    let auxiliary_effect = icaps.operators()[0]
        .assignment_effects()
        .iter()
        .find(|effect| effect.affected_var_id() == VariableIndex::from_usize(auxiliary_id))
        .expect("operator must update the ICAPS condition auxiliary");
    assert_eq!(auxiliary_effect.operation(), &AssignmentOperation::Plus);
    assert_eq!(
        icaps.initial_numeric_values()[auxiliary_effect.var_id().index()],
        NumericValue::new(1.0)
    );
}

#[test]
fn the_restriction_preserves_a_derived_metric_cost() {
    let restricted = build_icaps26_restricted_task(&task_with_derived_metric())
        .unwrap()
        .expect("task should be restricted");
    assert_derived_metric_cost_preserved(&restricted);
}

/// Settlers: the metric is a sum of fluents, derived by an assignment
/// axiom, rather than a `total-cost` variable the operators increase. A
/// translator mutex group spans the comparison's head and a location.
fn task_with_derived_metric() -> Task {
    let variables = vec![
        ExplicitVariable::new(
            2,
            "cmp".into(),
            vec!["true".into(), "false".into()],
            Some(1),
            ExplicitValueIndex::new(1),
        ),
        ExplicitVariable::new(
            2,
            "location".into(),
            vec!["mine".into(), "town".into()],
            None,
            ExplicitValueIndex::new(0),
        ),
    ];
    let numeric_variables = vec![
        NumericVariable::new("fuel".into(), NumericType::Regular, None),
        NumericVariable::new("capacity".into(), NumericType::Constant, None),
        NumericVariable::new("capacity-minus-fuel".into(), NumericType::Derived, Some(0)),
        NumericVariable::new("one".into(), NumericType::Constant, None),
        NumericVariable::new("labour".into(), NumericType::Regular, None),
        NumericVariable::new("pollution".into(), NumericType::Regular, None),
        NumericVariable::new("metric".into(), NumericType::Derived, Some(0)),
    ];
    let operator = Operator::new(
        "breakstone".into(),
        vec![],
        vec![],
        vec![
            AssignmentEffect::new(
                VariableIndex::from_usize(4),
                AssignmentOperation::Plus,
                VariableIndex::from_usize(3),
                false,
                vec![],
            ),
            AssignmentEffect::new(
                VariableIndex::from_usize(5),
                AssignmentOperation::Plus,
                VariableIndex::from_usize(3),
                false,
                vec![],
            ),
        ],
        OperatorCost::new(0),
    );
    Task::new(TaskParts {
        metric: Metric::new(true, Some(VariableIndex::from_usize(6))),
        variables,
        numeric_variables,
        goals: vec![ExplicitFact::propositional(0, 0)],
        mutexes: vec![vec![
            ExplicitFact::propositional(1, 0),
            ExplicitFact::propositional(1, 1),
        ]],
        state: vec![ExplicitValueIndex::new(1), ExplicitValueIndex::new(0)],
        numeric_state: vec![
            NumericValue::new(4.0),
            NumericValue::new(6.0),
            NumericValue::new(2.0),
            NumericValue::new(1.0),
            NumericValue::new(0.0),
            NumericValue::new(0.0),
            NumericValue::new(0.0),
        ],
        operators: vec![operator],
        axioms: vec![],
        comparison_axioms: vec![ComparisonAxiom::new(
            VariableIndex::from_usize(0),
            VariableIndex::from_usize(2),
            VariableIndex::from_usize(3),
            ComparisonOperator::GreaterThan,
        )],
        assignment_axioms: vec![
            AssignmentAxiom::new(
                VariableIndex::from_usize(2),
                CalOperator::Difference,
                VariableIndex::from_usize(1),
                VariableIndex::from_usize(0),
            ),
            AssignmentAxiom::new(
                VariableIndex::from_usize(6),
                CalOperator::Sum,
                VariableIndex::from_usize(4),
                VariableIndex::from_usize(5),
            ),
        ],
        global_constraint: Some(ExplicitFact::propositional(0, 0)),
    })
}

/// The restriction keeps the metric's cost and, since it keeps every
/// propositional variable id, the translator's mutex groups unchanged.
fn assert_derived_metric_cost_preserved(transformed: &Task) {
    assert_eq!(transformed.mutexes(), task_with_derived_metric().mutexes());
    let metric_var_id = transformed.metric().var_id().unwrap();
    assert_eq!(transformed.numeric_type(metric_var_id), NumericType::Cost);
    assert_eq!(transformed.operator_cost(OperatorIndex::new(0)), 2.0);
}
