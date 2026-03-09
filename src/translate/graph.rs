/// Port of graph.py
use std::collections::{HashMap, HashSet};

/// Python: class Graph(object)
pub struct Graph {
    pub nodes: Vec<usize>,
    pub edges: HashMap<usize, HashSet<usize>>,
}

impl Graph {
    pub fn new(nodes: Vec<usize>) -> Self {
        let edges = nodes.iter().map(|&n| (n, HashSet::new())).collect();
        Graph { nodes, edges }
    }

    /// Python: def connect(self, u, v)
    /// Adds an undirected edge between u and v.
    pub fn connect(&mut self, u: usize, v: usize) {
        self.edges.entry(u).or_insert_with(HashSet::new).insert(v);
        self.edges.entry(v).or_insert_with(HashSet::new).insert(u);
    }

    /// Python: def connected_components(self)
    /// Returns the connected components as a list of sets (uses DFS).
    pub fn connected_components(&self) -> Vec<HashSet<usize>> {
        let mut remaining: HashSet<usize> = self.nodes.iter().cloned().collect();
        let mut result = vec![];

        while !remaining.is_empty() {
            let start = *remaining.iter().next().unwrap();
            let mut component = HashSet::new();
            let mut stack = vec![start];

            while let Some(node) = stack.pop() {
                if component.insert(node) {
                    remaining.remove(&node);
                    if let Some(neighbors) = self.edges.get(&node) {
                        for &neighbor in neighbors {
                            if !component.contains(&neighbor) {
                                stack.push(neighbor);
                            }
                        }
                    }
                }
            }
            result.push(component);
        }
        result
    }
}

/// Python: def transitive_closure(pairs)
/// Warshall's algorithm for transitive closure.
pub fn transitive_closure(pairs: &[(usize, usize)]) -> HashSet<(usize, usize)> {
    // Collect all nodes
    let mut nodes = HashSet::new();
    for &(a, b) in pairs {
        nodes.insert(a);
        nodes.insert(b);
    }

    let mut closure: HashSet<(usize, usize)> = pairs.iter().cloned().collect();

    let nodes_vec: Vec<usize> = nodes.into_iter().collect();

    // Warshall's algorithm
    for &k in &nodes_vec {
        for &i in &nodes_vec {
            for &j in &nodes_vec {
                if closure.contains(&(i, k)) && closure.contains(&(k, j)) {
                    closure.insert((i, j));
                }
            }
        }
    }

    closure
}
