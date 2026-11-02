use super::*;
use crate::successor_generator::StandardGroundedTree;
use crate::test_tasks::one_step_task;

#[test]
fn reach_records_the_cheapest_path_and_reopens_closed_nodes() {
    let task = one_step_task();
    let mut space = SearchSpace::new(task.clone(), Box::new(StandardGroundedTree::new(&task)));
    let initial = space.initial_state();
    assert_eq!(space.reach(&initial, None, 0.0), Reached::New);
    let mut successors = Vec::new();
    space
        .generate_successors(&initial, &mut successors)
        .unwrap();
    let [successor] = &successors[..] else {
        panic!("the one-step task has one successor, got {successors:?}");
    };
    assert_eq!(successor.cost, 1.0);
    let parent = Some((initial.get_id(), successor.operator));

    assert_eq!(space.reach(&successor.state, parent, 3.0), Reached::New);
    let goal = successor.state.get_id();
    assert!(space.is_goal(goal));
    assert_eq!(
        space.reach(&successor.state, parent, 3.0),
        Reached::Duplicate
    );
    space.close(goal);
    assert_eq!(
        space.reach(&successor.state, parent, 1.0),
        Reached::Cheaper { reopened: true }
    );
    let node = space.node(goal).unwrap();
    assert_eq!(node.parent, parent);
    assert_eq!(node.g, 1.0);
    assert!(!node.is_closed && node.is_goal && !node.is_dead_end);

    space.mark_dead_end(goal);
    assert_eq!(
        space.reach(&successor.state, parent, 0.5),
        Reached::Duplicate
    );
    assert!(space.node(goal).unwrap().is_dead_end);
    assert_eq!(space.trace_path(goal).len(), 1);
    assert!(space.node(goal + 1).is_none());
}
