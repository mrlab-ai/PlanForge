//! Tarjan's strongly connected components, iteratively.
//!
//! The recursion depth of the textbook formulation is the length of the longest
//! simple path in the graph, which on large tasks exceeds the stack. Mainline
//! Fast Downward hit the same wall in Python and rewrote `sccs.py` iteratively;
//! this is the same rewrite, and it visits the graph in exactly the same order,
//! so the components and their order are unchanged.

#[cfg(test)]
mod tests;

use crate::numeric_task::VariableIndex;

/// Depth-first search over an adjacency list.
#[derive(Debug, Clone)]
pub struct Scc {
    graph: Vec<Vec<VariableIndex>>,
}

/// One entry of the explicit search path: a node, and how many of its
/// successors have been dealt with.
struct Visit {
    node: usize,
    next_successor: usize,
}

impl Scc {
    pub fn new(graph: Vec<Vec<VariableIndex>>) -> Self {
        Self { graph }
    }

    /// The strongly connected components, each in the order its nodes were
    /// first reached, and the components themselves in reverse topological
    /// order: a component comes before the components it can reach.
    pub fn get_result(self) -> Vec<Vec<usize>> {
        let node_count = self.graph.len();
        // The order the nodes were first reached in. `None` is "not reached".
        let mut dfs_numbers: Vec<Option<u32>> = vec![None; node_count];
        // The lowest number reachable from the node, which identifies the root
        // of its component. Only meaningful once the node has been reached.
        let mut dfs_minima: Vec<u32> = vec![0; node_count];
        // Where the node sits on `open_nodes`, or `None` once its component has
        // been closed.
        let mut stack_indices: Vec<Option<usize>> = vec![None; node_count];
        // The nodes whose component is not decided yet.
        let mut open_nodes: Vec<usize> = Vec::with_capacity(node_count);
        let mut path: Vec<Visit> = Vec::new();
        let mut sccs: Vec<Vec<usize>> = Vec::new();
        let mut next_dfs_number = 0u32;

        for initial in 0..node_count {
            if dfs_numbers[initial].is_some() {
                continue;
            }
            path.push(Visit {
                node: initial,
                next_successor: 0,
            });

            while let Some(&Visit {
                node,
                next_successor,
            }) = path.last()
            {
                // A node is reached the first time the loop sees it, whether it
                // was put on the path as a search root or as a successor.
                let node_number = *dfs_numbers[node].get_or_insert_with(|| {
                    let number = next_dfs_number;
                    next_dfs_number += 1;
                    dfs_minima[node] = number;
                    stack_indices[node] = Some(open_nodes.len());
                    open_nodes.push(node);
                    number
                });

                if let Some(&successor) = self.graph[node].get(next_successor) {
                    path.last_mut()
                        .expect("the node whose successor this is, is on the path")
                        .next_successor += 1;
                    match dfs_numbers[successor.index()] {
                        None => path.push(Visit {
                            node: successor.index(),
                            next_successor: 0,
                        }),
                        // A successor already assigned to a closed component
                        // says nothing about this node's component.
                        Some(successor_number) => {
                            if successor_number < node_number
                                && stack_indices[successor.index()].is_some()
                            {
                                dfs_minima[node] = dfs_minima[node].min(successor_number);
                            }
                        }
                    }
                    continue;
                }

                // Every successor of `node` has been dealt with.
                path.pop();
                if let Some(parent) = path.last() {
                    dfs_minima[parent.node] = dfs_minima[parent.node].min(dfs_minima[node]);
                }
                if dfs_minima[node] == node_number {
                    let first =
                        stack_indices[node].expect("the root of a component is still an open node");
                    for &member in &open_nodes[first..] {
                        stack_indices[member] = None;
                    }
                    sccs.push(open_nodes.split_off(first));
                }
            }
        }

        assert!(
            open_nodes.is_empty(),
            "{} nodes were left without a component",
            open_nodes.len()
        );
        sccs.reverse();
        sccs
    }
}
