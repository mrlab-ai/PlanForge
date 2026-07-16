use planforge_sas::axioms::{
    AssignmentAxiom, CalOperator, ComparisonAxiom, ComparisonOperator, PropositionalAxiom,
};
use planforge_sas::numeric_task::{
    AssignmentEffect, AssignmentOperation, Effect, ExplicitFact, ExplicitVariable, Metric,
    NumericRootTask, NumericType, NumericVariable, Operator,
};

use super::{CartesianAbstractionConfig, CartesianAbstractionGenerator, CartesianStopReason};

#[test]
fn refines_through_propositional_axiom_goal_support() {
    let variables = vec![
        ExplicitVariable::new(
            2,
            "x-at-limit".into(),
            vec!["true".into(), "false".into()],
            Some(0),
            1,
        ),
        ExplicitVariable::new(
            2,
            "goal".into(),
            vec!["true".into(), "false".into()],
            Some(1),
            1,
        ),
    ];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("limit".into(), NumericType::Constant, None),
        NumericVariable::new("one".into(), NumericType::Constant, None),
    ];
    let increment = Operator::new(
        "increment".into(),
        vec![],
        vec![],
        vec![AssignmentEffect::new(
            0,
            AssignmentOperation::Plus,
            2,
            false,
            vec![],
        )],
        1,
    );
    let task = NumericRootTask::new(
        1,
        Metric::new(true, None),
        variables,
        numeric_variables,
        vec![ExplicitFact::new(1, 0)],
        vec![],
        vec![1, 1],
        vec![0.0, 2.0, 1.0],
        vec![increment],
        vec![PropositionalAxiom::new(
            vec![ExplicitFact::new(0, 0)],
            1,
            1,
            0,
        )],
        vec![ComparisonAxiom::new(
            0,
            0,
            1,
            ComparisonOperator::GreaterThanOrEqual,
        )],
        vec![],
        ExplicitFact::new(0, 0),
    );

    let abstraction = CartesianAbstractionGenerator::new(CartesianAbstractionConfig {
        max_states: 16,
        ..Default::default()
    })
    .unwrap()
    .generate(&task)
    .unwrap();

    assert_eq!(
        abstraction.distance_table.distances[abstraction.distance_table.initial_state_hash],
        2.0
    );
    assert!(abstraction.metadata.solved_by_self);
    assert_eq!(
        abstraction.metadata.stop_reason,
        CartesianStopReason::ConcretePlan
    );
}

#[test]
fn supports_snp_assignment_axiom_comparisons() {
    let variables = vec![comparison_variable("sum-at-limit")];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("one".into(), NumericType::Constant, None),
        NumericVariable::new("sum".into(), NumericType::Derived, None),
        NumericVariable::new("limit".into(), NumericType::Constant, None),
    ];
    let increment = Operator::new(
        "increment".into(),
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
    );
    let task = numeric_goal_task(
        variables,
        numeric_variables,
        vec![0.0, 1.0, 0.0, 3.0],
        vec![increment],
        vec![ComparisonAxiom::new(
            0,
            2,
            3,
            ComparisonOperator::GreaterThanOrEqual,
        )],
        vec![AssignmentAxiom::new(2, CalOperator::Sum, 0, 1)],
    );

    assert_solved_with_h(&task, 2.0);
}

#[test]
fn supports_nonlinear_snp_assignment_axiom_comparisons() {
    let variables = vec![comparison_variable("product-at-nine")];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("y".into(), NumericType::Regular, None),
        NumericVariable::new("one".into(), NumericType::Constant, None),
        NumericVariable::new("product".into(), NumericType::Derived, None),
        NumericVariable::new("nine".into(), NumericType::Constant, None),
    ];
    let increment_both = Operator::new(
        "increment-both".into(),
        vec![],
        vec![],
        vec![
            AssignmentEffect::new(0, AssignmentOperation::Plus, 2, false, vec![]),
            AssignmentEffect::new(1, AssignmentOperation::Plus, 2, false, vec![]),
        ],
        1,
    );
    let task = numeric_goal_task(
        variables,
        numeric_variables,
        vec![1.0, 1.0, 1.0, 0.0, 9.0],
        vec![increment_both],
        vec![ComparisonAxiom::new(
            0,
            3,
            4,
            ComparisonOperator::GreaterThanOrEqual,
        )],
        vec![AssignmentAxiom::new(3, CalOperator::Product, 0, 1)],
    );

    assert_solved_with_h(&task, 2.0);
}

#[test]
fn supports_comparisons_between_regular_variables() {
    let variables = vec![comparison_variable("x-at-y")];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("y".into(), NumericType::Regular, None),
        NumericVariable::new("one".into(), NumericType::Constant, None),
    ];
    let increment = Operator::new(
        "increment-x".into(),
        vec![],
        vec![],
        vec![AssignmentEffect::new(
            0,
            AssignmentOperation::Plus,
            2,
            false,
            vec![],
        )],
        1,
    );
    let task = numeric_goal_task(
        variables,
        numeric_variables,
        vec![0.0, 2.0, 1.0],
        vec![increment],
        vec![ComparisonAxiom::new(
            0,
            0,
            1,
            ComparisonOperator::GreaterThanOrEqual,
        )],
        vec![],
    );

    assert_solved_with_h(&task, 2.0);
}

#[test]
fn supports_multiplicative_numeric_effects() {
    let variables = vec![comparison_variable("x-at-four")];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("two".into(), NumericType::Constant, None),
        NumericVariable::new("four".into(), NumericType::Constant, None),
    ];
    let double = Operator::new(
        "double".into(),
        vec![],
        vec![],
        vec![AssignmentEffect::new(
            0,
            AssignmentOperation::Times,
            1,
            false,
            vec![],
        )],
        1,
    );
    let task = numeric_goal_task(
        variables,
        numeric_variables,
        vec![1.0, 2.0, 4.0],
        vec![double],
        vec![ComparisonAxiom::new(
            0,
            0,
            2,
            ComparisonOperator::GreaterThanOrEqual,
        )],
        vec![],
    );

    assert_solved_with_h(&task, 2.0);
}

#[test]
fn refines_failed_default_derived_goal() {
    let variables = vec![
        ExplicitVariable::new(2, "base".into(), vec!["on".into(), "off".into()], None, 0),
        ExplicitVariable::new(
            2,
            "derived".into(),
            vec!["active".into(), "default".into()],
            Some(0),
            1,
        ),
    ];
    let turn_off = Operator::new(
        "turn-off".into(),
        vec![],
        vec![Effect::new(vec![], 0, None, 1)],
        vec![],
        1,
    );
    let task = NumericRootTask::new(
        1,
        Metric::new(true, None),
        variables,
        vec![],
        vec![ExplicitFact::new(1, 1)],
        vec![],
        vec![0, 1],
        vec![],
        vec![turn_off],
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

    assert_solved_with_h(&task, 1.0);
}

#[test]
fn state_limit_returns_an_admissible_partial_snp_abstraction() {
    let variables = vec![comparison_variable("sum-at-limit")];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("one".into(), NumericType::Constant, None),
        NumericVariable::new("sum".into(), NumericType::Derived, None),
        NumericVariable::new("limit".into(), NumericType::Constant, None),
    ];
    let increment = Operator::new(
        "increment".into(),
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
    );
    let task = numeric_goal_task(
        variables,
        numeric_variables,
        vec![0.0, 1.0, 0.0, 3.0],
        vec![increment],
        vec![ComparisonAxiom::new(
            0,
            2,
            3,
            ComparisonOperator::GreaterThanOrEqual,
        )],
        vec![AssignmentAxiom::new(2, CalOperator::Sum, 0, 1)],
    );

    let abstraction = CartesianAbstractionGenerator::new(CartesianAbstractionConfig {
        max_states: 1,
        ..Default::default()
    })
    .unwrap()
    .generate(&task)
    .unwrap();
    let h = abstraction.distance_table.distances[abstraction.distance_table.initial_state_hash];
    assert!(h <= 2.0);
    assert!(!abstraction.metadata.solved_by_self);
    assert_eq!(
        abstraction.metadata.stop_reason,
        CartesianStopReason::StateLimit
    );
    assert!(abstraction.metadata.pending_flaw.is_some());
}

fn comparison_variable(name: &str) -> ExplicitVariable {
    ExplicitVariable::new(
        3,
        name.into(),
        vec!["true".into(), "false".into(), "unknown".into()],
        Some(0),
        2,
    )
}

fn numeric_goal_task(
    variables: Vec<ExplicitVariable>,
    numeric_variables: Vec<NumericVariable>,
    initial_numeric: Vec<f64>,
    operators: Vec<Operator>,
    comparison_axioms: Vec<ComparisonAxiom>,
    assignment_axioms: Vec<AssignmentAxiom>,
) -> NumericRootTask {
    NumericRootTask::new(
        1,
        Metric::new(true, None),
        variables,
        numeric_variables,
        vec![ExplicitFact::new(0, 0)],
        vec![],
        vec![2],
        initial_numeric,
        operators,
        vec![],
        comparison_axioms,
        assignment_axioms,
        ExplicitFact::new(0, 0),
    )
}

fn assert_solved_with_h(task: &NumericRootTask, expected_h: f64) {
    let abstraction = CartesianAbstractionGenerator::new(CartesianAbstractionConfig {
        max_states: 64,
        ..Default::default()
    })
    .unwrap()
    .generate(task)
    .unwrap();
    assert_eq!(
        abstraction.distance_table.distances[abstraction.distance_table.initial_state_hash],
        expected_h
    );
    assert!(abstraction.metadata.solved_by_self);
    assert_eq!(
        abstraction.metadata.stop_reason,
        CartesianStopReason::ConcretePlan
    );
}
