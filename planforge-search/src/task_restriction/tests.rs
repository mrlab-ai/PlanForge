use planforge_sas::axioms::ComparisonOperator;
use planforge_sas::numeric_task::{
    ExplicitValueIndex, ExplicitVariable, NumericValue, OperatorCost, VariableIndex,
};

use super::*;

#[test]
fn restricted_task_lifts_derived_condition_root_and_maps_effects() {
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
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 1,
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
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    let restricted = build_restricted_task(&task)
        .unwrap()
        .expect("task should be restricted");
    let transformed = restricted.task();

    assert_eq!(transformed.numeric_variables().len(), 3);
    assert!(transformed.numeric_variables()[0].name().ends_with("|u"));
    assert_eq!(transformed.numeric_variables()[1].name(), "limit");
    assert_eq!(
        transformed.get_variable_axiom_layer(VariableIndex::from_usize(0)),
        Ok(Some(0))
    );
    assert_eq!(
        transformed.get_initial_numeric_state_values(),
        &[
            NumericValue::new(5.0),
            NumericValue::new(10.0),
            NumericValue::new(1.0)
        ]
    );
    assert_eq!(
        transformed.comparison_axioms()[0].get_left_var_id(),
        VariableIndex::from_usize(0)
    );
    assert_eq!(
        transformed.comparison_axioms()[0].get_right_var_id(),
        VariableIndex::from_usize(1)
    );

    let assignment_effects = transformed.get_operators()[0].assignment_effects();
    assert_eq!(assignment_effects.len(), 1);
    assert_eq!(
        assignment_effects[0].affected_var_id(),
        VariableIndex::from_usize(0)
    );
    assert_eq!(assignment_effects[0].var_id(), VariableIndex::from_usize(2));
    assert_eq!(
        transformed.get_initial_numeric_state_values()[assignment_effects[0].var_id().index()],
        NumericValue::new(1.0)
    );
    assert!(
        build_restricted_task(transformed).unwrap().is_none(),
        "applying task restriction twice must be a no-op"
    );

    let icaps = build_icaps26_restricted_task(&task)
        .unwrap()
        .expect("ICAPS translation should materialize the multivariate condition");
    let icaps = icaps.task();
    let auxiliary_id = icaps
        .numeric_variables()
        .iter()
        .position(|variable| variable.name() == "icaps26-condition-0")
        .expect("ICAPS condition auxiliary is missing");
    assert_eq!(
        icaps.get_initial_numeric_state_values()[auxiliary_id],
        NumericValue::new(-5.0)
    );
    assert_eq!(
        icaps.comparison_axioms()[0].get_left_var_id(),
        VariableIndex::from_usize(auxiliary_id)
    );
    let auxiliary_effect = icaps.get_operators()[0]
        .assignment_effects()
        .iter()
        .find(|effect| effect.affected_var_id() == VariableIndex::from_usize(auxiliary_id))
        .expect("operator must update the ICAPS condition auxiliary");
    assert_eq!(auxiliary_effect.operation(), &AssignmentOperation::Plus);
    assert_eq!(
        icaps.get_initial_numeric_state_values()[auxiliary_effect.var_id().index()],
        NumericValue::new(1.0)
    );
}

#[test]
fn restricted_task_supports_assignment_to_constant_when_views_stay_simple() {
    let variables = vec![ExplicitVariable::new(
        2,
        "cmp".into(),
        vec!["true".into(), "false".into()],
        Some(1),
        ExplicitValueIndex::new(1),
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
            VariableIndex::from_usize(0),
            AssignmentOperation::Assign,
            VariableIndex::from_usize(1),
            false,
            vec![],
        )],
        OperatorCost::new(1),
    );
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 1,
        metric: Metric::new(true, None),
        variables,
        numeric_variables,
        goals: vec![ExplicitFact::propositional(0, 0)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(1)],
        numeric_state: vec![
            NumericValue::new(4000.0),
            NumericValue::new(6000.0),
            NumericValue::new(2000.0),
        ],
        operators: vec![operator],
        axioms: vec![],
        comparison_axioms: vec![ComparisonAxiom::new(
            VariableIndex::from_usize(0),
            VariableIndex::from_usize(2),
            VariableIndex::from_usize(1),
            ComparisonOperator::GreaterThan,
        )],
        assignment_axioms: vec![AssignmentAxiom::new(
            VariableIndex::from_usize(2),
            CalOperator::Difference,
            VariableIndex::from_usize(1),
            VariableIndex::from_usize(0),
        )],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    let restricted = build_restricted_task(&task)
        .unwrap()
        .expect("task should be restricted");
    let transformed = restricted.task();
    let assignment_effects = transformed.get_operators()[0].assignment_effects();

    assert_eq!(assignment_effects.len(), 1);
    assert_eq!(
        assignment_effects[0].affected_var_id(),
        VariableIndex::from_usize(1)
    );
    assert_eq!(
        assignment_effects[0].operation(),
        &AssignmentOperation::Assign
    );
    assert_eq!(
        transformed.get_initial_numeric_state_values()[assignment_effects[0].var_id().index()],
        NumericValue::new(0.0)
    );
}

#[test]
fn restricted_task_preserves_metric_increment_on_assignment_operator() {
    let variables = vec![ExplicitVariable::new(
        2,
        "cmp".into(),
        vec!["true".into(), "false".into()],
        Some(1),
        ExplicitValueIndex::new(1),
    )];
    let numeric_variables = vec![
        NumericVariable::new("fuel".into(), NumericType::Regular, None),
        NumericVariable::new("capacity".into(), NumericType::Constant, None),
        NumericVariable::new("capacity-minus-fuel".into(), NumericType::Derived, Some(0)),
        NumericVariable::new("one".into(), NumericType::Constant, None),
        NumericVariable::new("total-cost".into(), NumericType::Cost, None),
    ];
    let operator = Operator::new(
        "refuel".into(),
        vec![],
        vec![],
        vec![
            AssignmentEffect::new(
                VariableIndex::from_usize(0),
                AssignmentOperation::Assign,
                VariableIndex::from_usize(1),
                false,
                vec![],
            ),
            AssignmentEffect::new(
                VariableIndex::from_usize(4),
                AssignmentOperation::Plus,
                VariableIndex::from_usize(3),
                false,
                vec![],
            ),
        ],
        OperatorCost::new(0),
    );
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 1,
        metric: Metric::new(true, Some(VariableIndex::from_usize(4))),
        variables,
        numeric_variables,
        goals: vec![ExplicitFact::propositional(0, 0)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(1)],
        numeric_state: vec![
            NumericValue::new(4.0),
            NumericValue::new(6.0),
            NumericValue::new(2.0),
            NumericValue::new(1.0),
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
        assignment_axioms: vec![AssignmentAxiom::new(
            VariableIndex::from_usize(2),
            CalOperator::Difference,
            VariableIndex::from_usize(1),
            VariableIndex::from_usize(0),
        )],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    let restricted = build_restricted_task(&task)
        .unwrap()
        .expect("task should be restricted");
    let transformed = restricted.task();
    assert_eq!(
        metric_operator_cost_from_initial_values(&task, &task.get_operators()[0]),
        NumericValue::new(1.0)
    );
    assert_eq!(
        metric_operator_cost_from_initial_values(transformed, &transformed.get_operators()[0]),
        NumericValue::new(1.0)
    );
    let metric_var_id = transformed.metric().var_id().unwrap();
    assert!(
        transformed.get_operators()[0]
            .assignment_effects()
            .iter()
            .any(|effect| effect.affected_var_id() == metric_var_id)
    );
}

#[test]
fn restricted_task_returns_none_when_domain_has_no_derived_roots() {
    let variables = vec![ExplicitVariable::new(
        2,
        "cmp".into(),
        vec!["true".into(), "false".into()],
        Some(0),
        ExplicitValueIndex::new(1),
    )];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("limit".into(), NumericType::Constant, None),
    ];
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 1,
        metric: Metric::new(true, None),
        variables,
        numeric_variables,
        goals: vec![ExplicitFact::propositional(0, 0)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(1)],
        numeric_state: vec![NumericValue::new(2.0), NumericValue::new(10.0)],
        operators: vec![],
        axioms: vec![],
        comparison_axioms: vec![ComparisonAxiom::new(
            VariableIndex::from_usize(0),
            VariableIndex::from_usize(0),
            VariableIndex::from_usize(1),
            ComparisonOperator::LessThanOrEqual,
        )],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    assert!(build_restricted_task(&task).unwrap().is_none());
}

#[test]
fn restricted_task_reports_unsupported_effect_as_error() {
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
    ];
    let operator = Operator::new(
        "scale-x".into(),
        vec![],
        vec![],
        vec![AssignmentEffect::new(
            VariableIndex::from_usize(0),
            AssignmentOperation::Times,
            VariableIndex::from_usize(3),
            false,
            vec![],
        )],
        OperatorCost::new(1),
    );
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 1,
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
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    let error = build_restricted_task(&task).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("does not support non-additive numeric effects"),
        "{error:#}"
    );
}
