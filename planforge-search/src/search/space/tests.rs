use planforge_sas::numeric_task::OperatorIndex;

use super::{SearchNodeInfo, SearchSpace};

#[test]
fn compact_node_storage_round_trips_and_updates_status() {
    let mut space = SearchSpace::new();
    space.set_node(
        7,
        SearchNodeInfo {
            parent_state: Some(3),
            parent_operator_id: Some(OperatorIndex::new(11)),
            g_value: 4.5,
            is_dead_end: false,
            is_closed: false,
            is_goal: true,
        },
    );

    let node = space.node(7).expect("stored node must exist");
    assert_eq!(node.parent_state, Some(3));
    assert_eq!(node.parent_operator_id, Some(OperatorIndex::new(11)));
    assert_eq!(node.g_value, 4.5);
    assert!(!node.is_dead_end);
    assert!(!node.is_closed);
    assert!(node.is_goal);
    assert!(space.is_goal(7));

    space.mark_dead_end(7);
    space.mark_closed(7);
    let node = space.node(7).expect("updated node must exist");
    assert!(node.is_dead_end);
    assert!(node.is_closed);
    assert!(!space.contains_node(6));
}

#[test]
fn empty_preferred_snapshots_do_not_allocate_per_state_storage() {
    let mut space = SearchSpace::new();
    space.store_preferred(1_000_000, &[]);
    assert!(space.preferred_ranges.is_empty());

    space.store_preferred(3, &[OperatorIndex::new(2), OperatorIndex::new(5)]);
    assert_eq!(space.preferred_ranges.len(), 4);
    let range = space.take_preferred(3).expect("preferred range must exist");
    assert!(space.preferred_contains(range, OperatorIndex::new(2)));
    assert!(space.preferred_contains(range, OperatorIndex::new(5)));
    assert!(!space.preferred_contains(range, OperatorIndex::new(7)));
}
