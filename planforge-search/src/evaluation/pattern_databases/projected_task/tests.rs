use planforge_sas::axioms::{
    AssignmentAxiom, CalOperator, ComparisonAxiom, ComparisonOperator, PropositionalAxiom,
};
use planforge_sas::numeric_task::{
    AssignmentEffect, AssignmentOperation, ExplicitFact, ExplicitVariable, Metric, NumericRootTask,
    NumericType, NumericVariable, Operator,
};

use super::*;

fn variable(name: &str, axiom_layer: Option<usize>) -> ExplicitVariable {
    ExplicitVariable::new(
        2,
        name.to_string(),
        vec![format!("{name}=0"), format!("{name}=1")],
        axiom_layer,
        1,
    )
}

fn restricted_sample_task() -> NumericRootTask {
    NumericRootTask::new(
        1,
        Metric::new(true, None),
        vec![
            variable("p", None),
            ExplicitVariable::new(
                3,
                "cmp".to_string(),
                vec![
                    "cmp-true".to_string(),
                    "cmp-false".to_string(),
                    "cmp-unknown".to_string(),
                ],
                Some(0),
                2,
            ),
            variable("goal-marker", Some(1)),
        ],
        vec![
            NumericVariable::new("limit".to_string(), NumericType::Constant, None),
            NumericVariable::new("x".to_string(), NumericType::Regular, None),
        ],
        vec![ExplicitFact::new(2, 0)],
        vec![],
        vec![0, 2, 1],
        vec![10.0, 0.0],
        vec![Operator::new(
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
        )],
        vec![PropositionalAxiom::new(
            vec![ExplicitFact::new(1, 0)],
            2,
            1,
            0,
        )],
        vec![ComparisonAxiom::new(
            1,
            1,
            0,
            ComparisonOperator::GreaterThanOrEqual,
        )],
        vec![],
        ExplicitFact::new(0, 0),
    )
}

#[test]
fn projection_builds_a_compact_restricted_transition_system() {
    let task = restricted_sample_task();
    let projected = ProjectedTask::new(
        &task,
        &Pattern {
            regular: vec![0],
            numeric: vec![1],
        },
    )
    .unwrap();

    assert_eq!(projected.get_num_variables(), 2);
    assert_eq!(projected.numeric_variables().len(), 2);
    assert_eq!(projected.get_num_operators(), 1);
    assert_eq!(projected.get_num_cmp_axioms(), 1);
    assert_eq!(projected.get_num_axioms(), 0);
    assert_eq!(projected.get_num_goals(), 1);
    assert_eq!(
        projected.get_initial_numeric_state_values().as_slice(),
        &[0.0, 10.0]
    );
}

#[test]
fn projection_rejects_an_unrestricted_task() {
    let task = restricted_sample_task();
    let variables = task.variables().clone();
    let mut numeric_variables = task.numeric_variables().clone();
    numeric_variables.push(NumericVariable::new(
        "derived-x".to_string(),
        NumericType::Derived,
        Some(0),
    ));
    let unrestricted = NumericRootTask::new(
        1,
        task.metric().clone(),
        variables,
        numeric_variables,
        vec![ExplicitFact::new(2, 0)],
        vec![],
        task.get_initial_propositional_state_values().to_vec(),
        vec![10.0, 0.0, 0.0],
        task.get_operators().clone(),
        task.axioms().clone(),
        vec![ComparisonAxiom::new(
            1,
            2,
            0,
            ComparisonOperator::GreaterThanOrEqual,
        )],
        vec![AssignmentAxiom::new(2, CalOperator::Sum, 1, 0)],
        ExplicitFact::new(0, 0),
    );

    let result = ProjectedTask::new(
        &unrestricted,
        &Pattern {
            regular: vec![],
            numeric: vec![1],
        },
    );
    assert!(matches!(
        result,
        Err(ProjectedTaskBuildError::UnrestrictedTask { .. })
    ));
}

#[test]
fn projection_rejects_derived_pattern_variables() {
    let task = NumericRootTask::new(
        1,
        Metric::new(true, None),
        vec![variable("p", None)],
        vec![
            NumericVariable::new("x".to_string(), NumericType::Regular, None),
            NumericVariable::new("zero".to_string(), NumericType::Constant, None),
            NumericVariable::new("derived-x".to_string(), NumericType::Derived, Some(0)),
        ],
        vec![],
        vec![],
        vec![0],
        vec![1.0, 0.0, 1.0],
        vec![],
        vec![],
        vec![],
        vec![AssignmentAxiom::new(2, CalOperator::Sum, 0, 1)],
        ExplicitFact::new(0, 0),
    );

    let result = ProjectedTask::new(
        &task,
        &Pattern {
            regular: vec![],
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
fn projection_closes_over_numeric_effect_sources() {
    let task = NumericRootTask::new(
        1,
        Metric::new(true, None),
        vec![variable("p", None)],
        vec![
            NumericVariable::new("x".to_string(), NumericType::Regular, None),
            NumericVariable::new("step".to_string(), NumericType::Regular, None),
        ],
        vec![],
        vec![],
        vec![0],
        vec![1.0, 2.0],
        vec![Operator::new(
            "increase".to_string(),
            vec![],
            vec![],
            vec![AssignmentEffect::new(
                0,
                AssignmentOperation::Plus,
                1,
                false,
                vec![],
            )],
            1,
        )],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    );

    let projected = ProjectedTask::new(
        &task,
        &Pattern {
            regular: vec![],
            numeric: vec![0],
        },
    )
    .unwrap();

    assert_eq!(projected.numeric_variables().len(), 2);
    assert_eq!(projected.pattern_numeric_projected_ids(), &[0]);
    let effect = &projected.get_operators()[0].assignment_effects()[0];
    assert_eq!(effect.affected_var_id(), 0);
    assert_eq!(effect.var_id(), 1);
}

#[test]
fn projection_computes_transitive_numeric_effect_source_closure() {
    let task = NumericRootTask::new(
        1,
        Metric::new(true, None),
        vec![variable("p", None)],
        vec![
            NumericVariable::new("x".to_string(), NumericType::Regular, None),
            NumericVariable::new("y".to_string(), NumericType::Regular, None),
            NumericVariable::new("z".to_string(), NumericType::Regular, None),
        ],
        vec![],
        vec![],
        vec![0],
        vec![0.0, 1.0, 2.0],
        vec![
            Operator::new(
                "update-y-first".to_string(),
                vec![],
                vec![],
                vec![AssignmentEffect::new(
                    1,
                    AssignmentOperation::Plus,
                    2,
                    false,
                    vec![],
                )],
                1,
            ),
            Operator::new(
                "update-x-second".to_string(),
                vec![],
                vec![],
                vec![AssignmentEffect::new(
                    0,
                    AssignmentOperation::Plus,
                    1,
                    false,
                    vec![],
                )],
                1,
            ),
        ],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    );

    let projected = ProjectedTask::new(
        &task,
        &Pattern {
            regular: vec![],
            numeric: vec![0],
        },
    )
    .unwrap();

    assert_eq!(projected.numeric_variables().len(), 3);
    assert_eq!(projected.get_num_operators(), 2);
    assert_eq!(
        projected.get_operators()[0].assignment_effects()[0].var_id(),
        2
    );
}

#[test]
fn projection_closes_over_selected_comparison_operands() {
    let task = NumericRootTask::new(
        1,
        Metric::new(true, None),
        vec![ExplicitVariable::new(
            3,
            "cmp".to_string(),
            vec![
                "true".to_string(),
                "false".to_string(),
                "unknown".to_string(),
            ],
            Some(0),
            2,
        )],
        vec![
            NumericVariable::new("x".to_string(), NumericType::Regular, None),
            NumericVariable::new("limit".to_string(), NumericType::Constant, None),
        ],
        vec![ExplicitFact::new(0, 0)],
        vec![],
        vec![2],
        vec![1.0, 5.0],
        vec![],
        vec![],
        vec![ComparisonAxiom::new(
            0,
            0,
            1,
            ComparisonOperator::GreaterThanOrEqual,
        )],
        vec![],
        ExplicitFact::new(0, 0),
    );

    let projected = ProjectedTask::new(
        &task,
        &Pattern {
            regular: vec![0],
            numeric: vec![],
        },
    )
    .unwrap();

    assert_eq!(projected.numeric_variables().len(), 2);
    assert_eq!(projected.get_num_cmp_axioms(), 1);
    assert_eq!(projected.get_num_goals(), 1);
}

#[test]
fn projected_axioms_drop_omitted_conditions_admissibly() {
    let task = NumericRootTask::new(
        1,
        Metric::new(true, None),
        vec![
            variable("condition", None),
            variable("derived-goal", Some(0)),
        ],
        vec![],
        vec![ExplicitFact::new(1, 0)],
        vec![],
        vec![0, 1],
        vec![],
        vec![],
        vec![PropositionalAxiom::new(
            vec![ExplicitFact::new(0, 0)],
            1,
            1,
            0,
        )],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    );

    let projected = ProjectedTask::new(
        &task,
        &Pattern {
            regular: vec![1],
            numeric: vec![],
        },
    )
    .unwrap();

    assert_eq!(projected.get_num_variables(), 1);
    assert_eq!(projected.get_num_axioms(), 1);
    assert!(projected.axioms()[0].conditions().is_empty());
}

#[test]
fn source_state_projection_is_a_direct_index_mapping() {
    let task = restricted_sample_task();
    let projected = ProjectedTask::new(
        &task,
        &Pattern {
            regular: vec![0],
            numeric: vec![1],
        },
    )
    .unwrap();
    let propositional = vec![1, 2, 1];
    let numeric = vec![10.0, 7.0];

    let expected = projected
        .project_state_values(&propositional, &numeric)
        .unwrap();
    let mut projected_prop = Vec::new();
    let mut projected_numeric = Vec::new();
    projected
        .project_state_values_from_source_numeric_into(
            &propositional,
            &numeric,
            &mut projected_prop,
            &mut projected_numeric,
        )
        .unwrap();

    assert_eq!((projected_prop, projected_numeric), expected);
}
