use planforge_sas::axioms::{
    AssignmentAxiom, CalOperator, ComparisonAxiom, ComparisonOperator, PropositionalAxiom,
};
use planforge_sas::numeric_conditions::ConditionValue;
use planforge_sas::numeric_task::{
    AssignmentEffect, AssignmentOperation, ExplicitFact, ExplicitVariable, Metric, NumericRootTask,
    NumericRootTaskParts, NumericType, NumericVariable, Operator,
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
    NumericRootTask::new(NumericRootTaskParts {
        version: 1,
        metric: Metric::new(true, None),
        variables: vec![
            variable("p", None),
            ExplicitVariable::new(
                ConditionValue::DOMAIN_SIZE,
                "cmp".to_string(),
                vec!["cmp-true".to_string(), "cmp-false".to_string()],
                Some(0),
                ConditionValue::False.as_usize(),
            ),
            variable("goal-marker", Some(1)),
        ],
        numeric_variables: vec![
            NumericVariable::new("limit".to_string(), NumericType::Constant, None),
            NumericVariable::new("x".to_string(), NumericType::Regular, None),
        ],
        goals: vec![ExplicitFact::propositional(2, 0)],
        mutexes: vec![],
        state: vec![0, 2, 1],
        numeric_state: vec![10.0, 0.0],
        operators: vec![Operator::new(
            "inc-x".to_string(),
            vec![ExplicitFact::propositional(0, 0)],
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
        axioms: vec![PropositionalAxiom::new(
            vec![ExplicitFact::propositional(1, 0)],
            2,
            1,
            0,
        )],
        comparison_axioms: vec![ComparisonAxiom::new(
            1,
            1,
            0,
            ComparisonOperator::GreaterThanOrEqual,
        )],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    })
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
    assert_eq!(projected.get_initial_numeric_state_values(), &[0.0, 10.0]);
}

#[test]
fn projection_rejects_an_unrestricted_task() {
    let task = restricted_sample_task();
    let mut numeric_variables = task.numeric_variables().clone();
    numeric_variables.push(NumericVariable::new(
        "derived-x".to_string(),
        NumericType::Derived,
        Some(0),
    ));
    // The derived numeric variable takes axiom layer 0, so the comparison and
    // the propositional axiom above it each move up one layer.
    let variables: Vec<ExplicitVariable> = task
        .variables()
        .iter()
        .map(|variable| variable.with_axiom_layer(variable.axiom_layer().map(|layer| layer + 1)))
        .collect();
    let unrestricted = NumericRootTask::new(NumericRootTaskParts {
        version: 1,
        metric: task.metric().clone(),
        variables,
        numeric_variables,
        goals: vec![ExplicitFact::propositional(2, 0)],
        mutexes: vec![],
        state: task.get_initial_propositional_state_values().to_vec(),
        numeric_state: vec![10.0, 0.0, 0.0],
        operators: task.get_operators().clone(),
        axioms: task.axioms().clone(),
        comparison_axioms: vec![ComparisonAxiom::new(
            1,
            2,
            0,
            ComparisonOperator::GreaterThanOrEqual,
        )],
        assignment_axioms: vec![AssignmentAxiom::new(2, CalOperator::Sum, 1, 0)],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

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
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 1,
        metric: Metric::new(true, None),
        variables: vec![variable("p", None)],
        numeric_variables: vec![
            NumericVariable::new("x".to_string(), NumericType::Regular, None),
            NumericVariable::new("zero".to_string(), NumericType::Constant, None),
            NumericVariable::new("derived-x".to_string(), NumericType::Derived, Some(0)),
        ],
        goals: vec![],
        mutexes: vec![],
        state: vec![0],
        numeric_state: vec![1.0, 0.0, 1.0],
        operators: vec![],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![AssignmentAxiom::new(2, CalOperator::Sum, 0, 1)],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

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
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 1,
        metric: Metric::new(true, None),
        variables: vec![variable("p", None)],
        numeric_variables: vec![
            NumericVariable::new("x".to_string(), NumericType::Regular, None),
            NumericVariable::new("step".to_string(), NumericType::Regular, None),
        ],
        goals: vec![],
        mutexes: vec![],
        state: vec![0],
        numeric_state: vec![1.0, 2.0],
        operators: vec![Operator::new(
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
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

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
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 1,
        metric: Metric::new(true, None),
        variables: vec![variable("p", None)],
        numeric_variables: vec![
            NumericVariable::new("x".to_string(), NumericType::Regular, None),
            NumericVariable::new("y".to_string(), NumericType::Regular, None),
            NumericVariable::new("z".to_string(), NumericType::Regular, None),
        ],
        goals: vec![],
        mutexes: vec![],
        state: vec![0],
        numeric_state: vec![0.0, 1.0, 2.0],
        operators: vec![
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
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

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
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 1,
        metric: Metric::new(true, None),
        variables: vec![ExplicitVariable::new(
            ConditionValue::DOMAIN_SIZE,
            "cmp".to_string(),
            vec!["true".to_string(), "false".to_string()],
            Some(0),
            ConditionValue::False.as_usize(),
        )],
        numeric_variables: vec![
            NumericVariable::new("x".to_string(), NumericType::Regular, None),
            NumericVariable::new("limit".to_string(), NumericType::Constant, None),
        ],
        goals: vec![ExplicitFact::propositional(0, 0)],
        mutexes: vec![],
        state: vec![2],
        numeric_state: vec![1.0, 5.0],
        operators: vec![],
        axioms: vec![],
        comparison_axioms: vec![ComparisonAxiom::new(
            0,
            0,
            1,
            ComparisonOperator::GreaterThanOrEqual,
        )],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

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

/// A task whose comparison verdict is *false* in the initial state, read at that
/// value by the propositional axioms one layer above it.
///
/// This is the shape the translator emits for a derived predicate with a numeric
/// body: `cmp` is the two-valued condition variable, `refuted` carries the rule
/// that refutes the derived atom when the comparison fails, and `unproven` needs
/// a second literal — `blocker=0`, which the initial state does not hold — on top
/// of the same verdict.
fn task_reading_a_failed_comparison() -> NumericRootTask {
    NumericRootTask::new(NumericRootTaskParts {
        version: 1,
        metric: Metric::new(true, None),
        variables: vec![
            variable("global-constraint", None),
            ExplicitVariable::new(
                ConditionValue::DOMAIN_SIZE,
                "cmp".to_string(),
                vec!["cmp-true".to_string(), "cmp-false".to_string()],
                Some(0),
                ConditionValue::False.as_usize(),
            ),
            variable("blocker", None),
            variable("refuted", Some(1)),
            variable("unproven", Some(1)),
        ],
        numeric_variables: vec![
            NumericVariable::new("x".to_string(), NumericType::Regular, None),
            NumericVariable::new("limit".to_string(), NumericType::Constant, None),
        ],
        goals: vec![ExplicitFact::propositional(3, 0)],
        mutexes: vec![],
        // `blocker` is 1, so the `blocker=0` condition of `unproven` never holds.
        state: vec![0, ConditionValue::False.as_usize(), 1, 1, 1],
        numeric_state: vec![1.0, 5.0],
        operators: vec![],
        axioms: vec![
            PropositionalAxiom::new(vec![ExplicitFact::propositional(1, 1)], 3, 1, 0),
            PropositionalAxiom::new(
                vec![
                    ExplicitFact::propositional(1, 1),
                    ExplicitFact::propositional(2, 0),
                ],
                4,
                1,
                0,
            ),
        ],
        comparison_axioms: vec![ComparisonAxiom::new(
            1,
            0,
            1,
            ComparisonOperator::GreaterThanOrEqual,
        )],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    })
}

/// A failed comparison is announced to the Horn rules exactly once.
///
/// The condition variable is *computed*, not proven: the comparison pass writes
/// its verdict and the closure seeds the queue with it. Admitting it to negation
/// by failure as well — as this crate's own copy of the evaluator used to do,
/// excluding only the deepest layer and not the comparison layer — announces the
/// same literal twice. Each announcement decrements one condition counter, so a
/// two-condition rule fires one condition short, and a rule that already fired
/// on the first announcement decrements a counter that is already zero.
#[test]
fn a_failed_comparison_verdict_reaches_the_horn_rules_once() {
    let task = task_reading_a_failed_comparison();
    let projected = ProjectedTask::new(
        &task,
        &Pattern {
            regular: vec![0, 1, 2, 3, 4],
            numeric: vec![0],
        },
    )
    .unwrap();
    assert_eq!(
        projected
            .variables()
            .iter()
            .map(ExplicitVariable::axiom_layer)
            .collect::<Vec<Option<usize>>>(),
        vec![None, Some(0), None, Some(1), Some(1)],
        "the comparison layer has to sit strictly below the propositional one, or \
         excluding the deepest layer would already exclude the comparison"
    );

    let (values, _numeric) = projected.evaluated_initial_state_values().unwrap();
    assert_eq!(
        values[1],
        ConditionValue::False.as_usize(),
        "1 >= 5 is false"
    );
    assert_eq!(
        values[3], 0,
        "the refuting rule reads the verdict and fires"
    );
    assert_eq!(
        values[4], 1,
        "`blocker=0` is not satisfied, so the two-condition rule must not fire"
    );
}

#[test]
fn projected_axioms_drop_omitted_conditions_admissibly() {
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 1,
        metric: Metric::new(true, None),
        variables: vec![
            variable("condition", None),
            variable("derived-goal", Some(0)),
        ],
        numeric_variables: vec![],
        goals: vec![ExplicitFact::propositional(1, 0)],
        mutexes: vec![],
        state: vec![0, 1],
        numeric_state: vec![],
        operators: vec![],
        axioms: vec![PropositionalAxiom::new(
            vec![ExplicitFact::propositional(0, 0)],
            1,
            1,
            0,
        )],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

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
