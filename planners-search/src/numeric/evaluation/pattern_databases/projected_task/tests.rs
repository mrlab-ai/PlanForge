use planners_sas::numeric::axioms::{ComparisonOperator, PropositionalAxiom};
use planners_sas::numeric::numeric_task::{AssignmentOperation, NumericRootTask};

use super::*;

fn simple_var(name: &str, axiom_layer: Option<usize>) -> ExplicitVariable {
    ExplicitVariable::new(
        2,
        name.to_string(),
        vec![format!("{name}=0"), format!("{name}=1")],
        axiom_layer,
        1,
    )
}

fn sample_task() -> NumericRootTask {
    let variables = vec![
        simple_var("p", None),
        ExplicitVariable::new(
            3,
            "cmp".to_string(),
            vec![
                "cmp-true".to_string(),
                "cmp-false".to_string(),
                "cmp-unk".to_string(),
            ],
            Some(0),
            2,
        ),
        simple_var("goal_marker", Some(1)),
    ];
    let numeric_variables = vec![
        NumericVariable::new("const10".to_string(), NumericType::Constant, None),
        NumericVariable::new("x".to_string(), NumericType::Regular, None),
        NumericVariable::new("sum".to_string(), NumericType::Derived, Some(0)),
    ];
    let operators = vec![Operator::new(
        "inc-x".to_string(),
        vec![ExplicitFact::new(0, 0)],
        vec![],
        vec![AssignmentEffect::new(
            1,
            AssignmentOperation::Plus,
            0,
            false,
            vec![],
        )],
        1,
    )];
    let axioms = vec![PropositionalAxiom::new(
        vec![ExplicitFact::new(1, 0)],
        2,
        1,
        0,
    )];
    let comparison_axioms = vec![ComparisonAxiom::new(
        1,
        2,
        0,
        ComparisonOperator::GreaterThanOrEqual,
    )];
    let assignment_axioms = vec![AssignmentAxiom::new(2, CalOperator::Sum, 1, 0)];

    NumericRootTask::new(
        1,
        Metric::new(true, None),
        variables,
        numeric_variables,
        vec![ExplicitFact::new(2, 0)],
        vec![],
        vec![0, 2, 1],
        vec![10.0, 0.0, 10.0],
        operators,
        axioms,
        comparison_axioms,
        assignment_axioms,
        ExplicitFact::new(0, 0),
    )
}

#[test]
fn projected_task_closes_over_relevant_numeric_and_goal_axiom_vars() {
    let task = sample_task();
    let pattern = Pattern {
        regular: vec![0],
        numeric: vec![1],
    };

    let projected = ProjectedTask::new(&task, &pattern).unwrap();

    assert_eq!(projected.get_num_variables(), 3);
    assert_eq!(projected.numeric_variables().len(), 3);
    assert_eq!(projected.get_num_operators(), 1);
    assert_eq!(projected.get_num_cmp_axioms(), 1);
    assert_eq!(projected.get_num_axioms(), 1);
    assert_eq!(projected.get_num_goals(), 1);

    let init_num = projected.get_initial_numeric_state_values();
    assert_eq!(init_num.as_slice(), &[0.0, 10.0, 10.0]);
}

#[test]
fn projected_task_accepts_subtraction_based_numeric_conditions() {
    let variables = vec![simple_var("p", None), simple_var("cmp", Some(0))];
    let numeric_variables = vec![
        NumericVariable::new("const1".to_string(), NumericType::Constant, None),
        NumericVariable::new("x".to_string(), NumericType::Regular, None),
        NumericVariable::new("diff".to_string(), NumericType::Derived, Some(0)),
    ];
    let task = NumericRootTask::new(
        1,
        Metric::new(true, None),
        variables,
        numeric_variables,
        vec![],
        vec![],
        vec![0, 1],
        vec![1.0, 2.0, 1.0],
        vec![],
        vec![],
        vec![ComparisonAxiom::new(
            1,
            2,
            0,
            ComparisonOperator::GreaterThanOrEqual,
        )],
        vec![AssignmentAxiom::new(2, CalOperator::Difference, 1, 0)],
        ExplicitFact::new(0, 0),
    );

    let projected = ProjectedTask::new(
        &task,
        &Pattern {
            regular: vec![0],
            numeric: vec![1],
        },
    )
    .unwrap();

    assert_eq!(projected.get_num_cmp_axioms(), 1);
    let init_num = projected.get_initial_numeric_state_values();
    assert_eq!(init_num.as_slice(), &[2.0, 1.0, 1.0]);
}

#[test]
fn projected_task_rejects_raw_derived_numeric_pattern_vars() {
    let task = sample_task();

    let result = ProjectedTask::new(
        &task,
        &Pattern {
            regular: vec![0],
            numeric: vec![2],
        },
    );

    assert!(matches!(
        result,
        Err(ProjectedTaskBuildError::UnsupportedPatternNumericVarType {
            numeric_var_id: 2,
            numeric_type: NumericType::Derived,
        })
    ));
}

#[test]
fn projected_task_helper_pattern_var_gets_lifted_additive_effect() {
    let variables = vec![simple_var("goal", None)];
    let numeric_variables = vec![
        NumericVariable::new("const5".to_string(), NumericType::Constant, None),
        NumericVariable::new("x".to_string(), NumericType::Regular, None),
        NumericVariable::new("y".to_string(), NumericType::Regular, None),
        NumericVariable::new("sum".to_string(), NumericType::Derived, Some(0)),
    ];
    let operators = vec![Operator::new(
        "inc-x".to_string(),
        vec![],
        vec![],
        vec![AssignmentEffect::new(
            1,
            AssignmentOperation::Plus,
            0,
            false,
            vec![],
        )],
        1,
    )];
    let task = NumericRootTask::new(
        1,
        Metric::new(true, None),
        variables,
        numeric_variables,
        vec![],
        vec![],
        vec![0],
        vec![5.0, 2.0, 3.0, 5.0],
        operators,
        vec![],
        vec![],
        vec![AssignmentAxiom::new(3, CalOperator::Sum, 1, 2)],
        ExplicitFact::new(0, 0),
    );

    let helper_pattern_numeric_id = task.numeric_variables().len();
    let projected = ProjectedTask::new(
        &task,
        &Pattern {
            regular: vec![],
            numeric: vec![helper_pattern_numeric_id],
        },
    )
    .unwrap();

    assert_eq!(projected.numeric_variables().len(), 2);
    assert_eq!(
        projected.numeric_variables()[0].get_type(),
        &NumericType::Regular
    );

    let initial_numeric_values = projected.get_initial_numeric_state_values();
    assert_eq!(initial_numeric_values.as_slice(), &[5.0, 5.0]);
    drop(initial_numeric_values);

    assert_eq!(projected.get_num_operators(), 1);
    let op = &projected.get_operators()[0];
    assert_eq!(op.assignment_effects().len(), 1);
    assert_eq!(op.assignment_effects()[0].affected_var_id(), 0);
    assert_eq!(
        op.assignment_effects()[0].operation(),
        &AssignmentOperation::Plus
    );
    assert_eq!(op.assignment_effects()[0].var_id(), 1);
}

#[test]
fn projected_task_relayers_helper_backed_comparison_chain() {
    let variables = vec![
        simple_var("goal", Some(6)),
        ExplicitVariable::new(
            3,
            "cmp".to_string(),
            vec![
                "cmp-true".to_string(),
                "cmp-false".to_string(),
                "cmp-unk".to_string(),
            ],
            Some(5),
            2,
        ),
    ];
    let numeric_variables = vec![
        NumericVariable::new("zero".to_string(), NumericType::Constant, None),
        NumericVariable::new("x".to_string(), NumericType::Regular, None),
        NumericVariable::new("y".to_string(), NumericType::Regular, None),
        NumericVariable::new("sum".to_string(), NumericType::Derived, Some(4)),
    ];
    let task = NumericRootTask::new(
        1,
        Metric::new(true, None),
        variables,
        numeric_variables,
        vec![ExplicitFact::new(0, 1)],
        vec![],
        vec![0, 2],
        vec![0.0, 1.0, 2.0, 3.0],
        vec![],
        vec![PropositionalAxiom::new(
            vec![ExplicitFact::new(1, 0)],
            0,
            0,
            1,
        )],
        vec![ComparisonAxiom::new(
            1,
            3,
            0,
            ComparisonOperator::GreaterThanOrEqual,
        )],
        vec![AssignmentAxiom::new(3, CalOperator::Sum, 1, 2)],
        ExplicitFact::new(0, 0),
    );

    let helper_var_id = task.numeric_variables().len();
    let projected = ProjectedTask::new(
        &task,
        &Pattern {
            regular: vec![0],
            numeric: vec![helper_var_id],
        },
    )
    .unwrap();

    assert_eq!(projected.get_variable_axiom_layer(1).unwrap(), Some(0));
    assert_eq!(projected.get_variable_axiom_layer(0).unwrap(), Some(1));
    projected.evaluated_initial_state_values().unwrap();
}
