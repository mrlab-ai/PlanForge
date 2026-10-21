use super::*;

impl ShortestPaths {
    /// Shortest paths that make every one of `states` states a goal: for
    /// tests that read only the regions and growth of a view.
    pub(in crate::evaluation::abstractions::cartesian) fn all_goals(states: usize) -> Self {
        Self {
            distances: vec![0.0; states],
            generating_transition: vec![None; states],
            dependents: vec![Vec::new(); states],
            dependent_positions: vec![None; states],
            is_goal: vec![true; states],
            invalid: vec![false; states],
        }
    }
}

#[test]
fn shortest_path_dependency_positions_survive_swap_removal() {
    let first = TransitionKey {
        source: 0,
        concrete_op_id: OperatorIndex::new(0),
        target: 2,
    };
    let second = TransitionKey {
        source: 1,
        concrete_op_id: OperatorIndex::new(1),
        target: 2,
    };
    let mut shortest_paths = ShortestPaths {
        distances: vec![1.0, 1.0, 0.0],
        generating_transition: vec![Some(first), Some(second), None],
        dependents: vec![vec![], vec![], vec![0, 1]],
        dependent_positions: vec![Some(0), Some(1), None],
        is_goal: vec![false, false, true],
        invalid: vec![false; 3],
    };

    shortest_paths.remove_generating_transition(0);
    assert_eq!(shortest_paths.dependents[2], [1]);
    assert_eq!(shortest_paths.dependent_positions[1], Some(0));
    shortest_paths.remove_generating_transition(1);
    assert!(shortest_paths.dependents[2].is_empty());
}
