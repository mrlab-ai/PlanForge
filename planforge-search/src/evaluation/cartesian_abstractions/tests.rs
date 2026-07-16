use planforge_sas::axioms::{ComparisonAxiom, ComparisonOperator, PropositionalAxiom};
use planforge_sas::numeric_task::{
    AssignmentEffect, AssignmentOperation, ExplicitFact, ExplicitVariable, Metric, NumericRootTask,
    NumericType, NumericVariable, Operator,
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
