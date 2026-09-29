use super::*;
use planforge_sas::numeric_task::NumericRootTask;
use std::sync::Arc;

fn pddl_task(directory: &str, problem: &str) -> TaskRef<'static> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../tests/assets")
        .join(directory);
    let task: NumericRootTask = planforge_translate::translate_to_task(
        root.join("domain.pddl").to_str().unwrap(),
        root.join(problem).to_str().unwrap(),
    )
    .unwrap();
    Arc::new(task)
}

fn generous_limits() -> EnumerationLimits {
    EnumerationLimits {
        max_states: 100_000,
        max_transitions: 1_000_000,
        max_time: Duration::from_secs(30),
    }
}

#[test]
fn blocks_two_has_exact_goal_distances() {
    let graph = enumerate_state_space(
        pddl_task(
            "strips-pddl-files/blocks-minimal",
            "probBLOCKS-2-reverse.pddl",
        ),
        generous_limits(),
    )
    .unwrap();
    assert_eq!(graph.num_states(), 5);
    assert_eq!(graph.h_star[0], 4.0);
    for (&is_goal, &distance) in graph.goal_states.iter().zip(&graph.h_star) {
        if is_goal {
            assert_eq!(distance, 0.0);
        }
    }
}

#[test]
fn blocks_eight_has_the_expected_complete_state_space() {
    let graph = enumerate_state_space(
        pddl_task("strips-pddl-files/blocks-8-0", "probBLOCKS-8-0.pddl"),
        EnumerationLimits {
            max_states: 700_000,
            max_transitions: 2_100_000,
            max_time: Duration::from_secs(30),
        },
    )
    .unwrap();
    assert_eq!(graph.num_states(), 695_417);
    assert_eq!(graph.h_star[0], 18.0);
}

#[test]
fn unreachable_goal_states_are_exact_dead_ends() {
    let graph = enumerate_state_space(
        pddl_task("adl/unreachable-goal", "problem.pddl"),
        generous_limits(),
    )
    .unwrap();
    assert!(graph.h_star.iter().all(|distance| distance.is_infinite()));
    assert_eq!(graph.summary().dead_end_count, graph.num_states());
}

#[test]
fn a_state_bound_returns_no_success_shaped_partial_graph() {
    let error = enumerate_state_space(
        pddl_task(
            "strips-pddl-files/blocks-minimal",
            "probBLOCKS-2-reverse.pddl",
        ),
        EnumerationLimits {
            max_states: 1,
            ..generous_limits()
        },
    )
    .unwrap_err();
    assert!(matches!(
        error,
        StateSpaceEnumerationError::StateLimit { limit: 1, .. }
    ));
}

#[test]
fn a_transition_bound_returns_no_success_shaped_partial_graph() {
    let error = enumerate_state_space(
        pddl_task(
            "strips-pddl-files/blocks-minimal",
            "probBLOCKS-2-reverse.pddl",
        ),
        EnumerationLimits {
            max_transitions: 1,
            ..generous_limits()
        },
    )
    .unwrap_err();
    assert!(matches!(
        error,
        StateSpaceEnumerationError::TransitionLimit { limit: 1, .. }
    ));
}

#[test]
fn a_time_bound_returns_no_success_shaped_partial_graph() {
    let error = enumerate_state_space(
        pddl_task(
            "strips-pddl-files/blocks-minimal",
            "probBLOCKS-2-reverse.pddl",
        ),
        EnumerationLimits {
            max_time: Duration::ZERO,
            ..generous_limits()
        },
    )
    .unwrap_err();
    assert!(matches!(
        error,
        StateSpaceEnumerationError::TimeLimit { limit, .. } if limit == Duration::ZERO
    ));
}
