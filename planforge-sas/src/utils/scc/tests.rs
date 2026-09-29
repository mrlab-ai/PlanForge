use crate::numeric_task::VariableIndex;

use super::Scc;

/// `0 -> 1 -> 2 -> 0` is one component; `3` can reach it and must come
/// first, because the components come in reverse topological order.
#[test]
fn a_cycle_and_its_predecessor_are_ordered_by_dependency() {
    let sccs = Scc::new(vec![
        vec![VariableIndex::new(1)],
        vec![VariableIndex::new(2)],
        vec![VariableIndex::new(0)],
        vec![VariableIndex::new(0)],
    ])
    .get_result();

    assert_eq!(sccs, vec![vec![3], vec![0, 1, 2]]);
}

/// Nodes with no edges are their own components. Nothing orders them, so
/// they come out in the reverse of the order they were searched in, which is
/// what reversing the finished components leaves.
#[test]
fn isolated_nodes_are_singleton_components() {
    let sccs = Scc::new(vec![Vec::new(), Vec::new(), Vec::new()]).get_result();

    assert_eq!(sccs, vec![vec![2], vec![1], vec![0]]);
}

/// Two components, the second reachable from the first.
#[test]
fn two_cycles_in_a_chain_stay_separate() {
    // 0 <-> 1  ->  2 <-> 3
    let sccs = Scc::new(vec![
        vec![VariableIndex::new(1)],
        vec![VariableIndex::new(0), VariableIndex::new(2)],
        vec![VariableIndex::new(3)],
        vec![VariableIndex::new(2)],
    ])
    .get_result();

    assert_eq!(sccs, vec![vec![0, 1], vec![2, 3]]);
}

/// A node reached again from a component that is already closed must not
/// pull that component's number into the new one.
#[test]
fn an_edge_into_a_closed_component_does_not_merge_it() {
    // 0 -> 1, 2 -> 1: 1 closes first, then 0 and 2 each stay singletons.
    let sccs = Scc::new(vec![
        vec![VariableIndex::new(1)],
        Vec::new(),
        vec![VariableIndex::new(1)],
    ])
    .get_result();

    assert_eq!(sccs, vec![vec![2], vec![0], vec![1]]);
}

/// The recursive formulation overflowed the stack on a path this long,
/// which is why the search keeps its own path.
#[test]
fn a_long_path_does_not_exhaust_the_stack() {
    let node_count: usize = 400_000;
    let mut graph: Vec<Vec<VariableIndex>> = (1..node_count)
        .map(|next| vec![VariableIndex::from_usize(next)])
        .collect();
    graph.push(Vec::new());

    let sccs = Scc::new(graph).get_result();

    assert_eq!(sccs.len(), node_count);
    assert_eq!(sccs[0], vec![0]);
    assert_eq!(sccs[node_count - 1], vec![node_count - 1]);
}
