use super::*;
use crate::evaluation::abstractions::cegar::FlawKind;
use crate::evaluation::abstractions::cegar::regression::test_tasks::{
    diagonal_task, jump_task, linear_goal_task,
};
use crate::evaluation::abstractions::domain::cegar::CegarConfig;
use crate::evaluation::abstractions::domain::generator::DomainAbstractionGenerator;

/// Domain-abstraction CEGAR with regression flaws alone reaches a concrete
/// optimal plan on the tasks where regression and progression differ, in
/// both split directions.
#[test]
fn regression_cegar_solves_the_regression_tasks_optimally() {
    for (name, task, optimal_cost) in [
        ("jump", jump_task(), 4.0),
        ("linear goal", linear_goal_task(), 2.0),
        ("diagonal", diagonal_task(), 2.0),
    ] {
        for direction in [SplitDirection::Forward, SplitDirection::Backward] {
            let abstraction = DomainAbstractionGenerator::new(CegarConfig {
                max_abstraction_size: 10_000,
                flaw_kind: FlawKind::Regression,
                split_direction: Some(direction),
                ..Default::default()
            })
            .unwrap()
            .generate(&task)
            .unwrap();
            assert!(
                abstraction.metadata.solved_by_self,
                "{name}, {direction}: no concrete plan"
            );
            assert_eq!(
                abstraction.distance_table.distances[abstraction.distance_table.initial_state_hash],
                optimal_cost,
                "{name}, {direction}"
            );
        }
    }
}
