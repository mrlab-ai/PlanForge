use super::*;

#[test]
fn regression_splits_at_comparison_target_while_progression_splits_at_witness() {
    let task = numeric_goal_task(
        vec![comparison_variable("x-at-least-three")],
        vec![
            NumericVariable::new("x".into(), NumericType::Regular, None),
            NumericVariable::new("one".into(), NumericType::Constant, None),
            NumericVariable::new("three".into(), NumericType::Constant, None),
        ],
        vec![
            NumericValue::new(0.0),
            NumericValue::new(1.0),
            NumericValue::new(3.0),
        ],
        vec![Operator::new(
            "increment".into(),
            vec![],
            vec![],
            vec![AssignmentEffect::new(
                VariableIndex::from_usize(0),
                AssignmentOperation::Plus,
                VariableIndex::from_usize(1),
                false,
                vec![],
            )],
            OperatorCost::new(1),
        )],
        vec![ComparisonAxiom::new(
            VariableIndex::from_usize(0),
            VariableIndex::from_usize(0),
            VariableIndex::from_usize(2),
            ComparisonOperator::GreaterThanOrEqual,
        )],
        vec![],
    );

    let first_numeric_boundary = |direction| {
        let abstraction = CartesianAbstractionGenerator::new(CartesianAbstractionConfig {
            max_states: 2,
            split_kind: direction,
            ..Default::default()
        })
        .unwrap()
        .generate(&task)
        .unwrap();
        abstraction
            .hierarchy
            .nodes
            .iter()
            .find_map(|node| match node {
                RefinementNode::Numeric { boundary, .. } => Some(*boundary),
                _ => None,
            })
            .expect("one Cartesian numeric split")
    };

    assert_eq!(
        first_numeric_boundary(SplitKind::Concrete),
        NumericValue::new(0.0)
    );
    assert_eq!(
        first_numeric_boundary(SplitKind::Wanted),
        NumericValue::new(3.0)
    );
}

/// CEGAR with regression flaws alone, first or whole-plan, reaches a concrete
/// optimal plan on the tasks where regression and progression differ, with
/// both split kinds.
#[test]
fn regression_cegar_solves_the_regression_tasks_optimally() {
    use crate::evaluation::abstractions::common::cegar::regression::test_tasks::{
        diagonal_task, jump_task, linear_goal_task,
    };
    for (name, task, optimal_cost) in [
        ("jump", jump_task(), 4.0),
        ("linear goal", linear_goal_task(), 2.0),
        ("diagonal", diagonal_task(), 2.0),
    ] {
        for flaw_kind in [FlawKind::Regression, FlawKind::RegressEntirePlan] {
            for split_kind in [SplitKind::Concrete, SplitKind::Wanted] {
                let abstraction = CartesianAbstractionGenerator::new(CartesianAbstractionConfig {
                    max_states: 1_000,
                    flaw_kind,
                    split_kind,
                    ..Default::default()
                })
                .unwrap()
                .generate(&task)
                .unwrap();
                assert_eq!(
                    abstraction.metadata.stop_reason,
                    CegarStopReason::ConcretePlan,
                    "{name}, {flaw_kind}, {split_kind:?}"
                );
                assert_eq!(
                    abstraction.distance_table.distances
                        [abstraction.distance_table.initial_state_hash],
                    optimal_cost,
                    "{name}, {flaw_kind}, {split_kind:?}"
                );
            }
        }
    }
}
