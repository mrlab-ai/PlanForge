// Minimal Graph utilities stub (port target for python/translate/graph.py)
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct Graph<T>
where
    T: Clone + Eq + std::hash::Hash,
{
    pub nodes: HashSet<T>,
    pub edges: HashMap<T, HashSet<T>>,
}

impl<T> Graph<T>
where
    T: Clone + Eq + std::hash::Hash,
{
    pub fn new() -> Self {
        Self {
            nodes: HashSet::new(),
            edges: HashMap::new(),
        }
    }

    pub fn add_edge(&mut self, from: T, to: T) {
        self.nodes.insert(from.clone());
        self.nodes.insert(to.clone());
        self.edges
            .entry(from)
            .or_insert_with(HashSet::new)
            .insert(to);
    }

    pub fn successors(&self, node: &T) -> Option<&HashSet<T>> {
        self.edges.get(node)
    }

    // simple connected components using DFS (non-exhaustive)
    pub fn connected_components(&self) -> Vec<Vec<T>> {
        let mut visited = HashSet::new();
        let mut comps = Vec::new();
        for node in &self.nodes {
            if !visited.contains(node) {
                let mut stack = vec![node.clone()];
                let mut comp = Vec::new();
                while let Some(n) = stack.pop() {
                    if visited.insert(n.clone()) {
                        comp.push(n.clone());
                        if let Some(succ) = self.edges.get(&n) {
                            for v in succ {
                                stack.push(v.clone());
                            }
                        }
                    }
                }
                // leave as-is; deterministic ordering depends on T implementing Ord
                comps.push(comp);
            }
        }
        comps
    }
}
