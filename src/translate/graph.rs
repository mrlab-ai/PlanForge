//! Graph utilities for planning
//! Port of python/translate/graph.py

use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, Clone)]
pub struct Graph<T> {
    pub nodes: HashSet<T>,
    pub edges: HashMap<T, HashSet<T>>,
}

impl<T: Clone + Eq + std::hash::Hash> Graph<T> {
    pub fn new() -> Self {
        Self {
            nodes: HashSet::new(),
            edges: HashMap::new(),
        }
    }

    pub fn add_node(&mut self, node: T) {
        self.nodes.insert(node.clone());
        self.edges.entry(node).or_insert_with(HashSet::new);
    }

    pub fn add_edge(&mut self, from: T, to: T) {
        self.add_node(from.clone());
        self.add_node(to.clone());
        self.edges.get_mut(&from).unwrap().insert(to);
    }

    pub fn get_successors(&self, node: &T) -> Option<&HashSet<T>> {
        self.edges.get(node)
    }

    pub fn topological_sort(&self) -> Result<Vec<T>, String> {
        let mut in_degree: HashMap<T, usize> = HashMap::new();
        let mut queue = VecDeque::new();
        let mut result = Vec::new();

        // Calculate in-degrees
        for node in &self.nodes {
            in_degree.insert(node.clone(), 0);
        }
        
        for successors in self.edges.values() {
            for successor in successors {
                *in_degree.get_mut(successor).unwrap() += 1;
            }
        }

        // Find nodes with no incoming edges
        for (node, degree) in &in_degree {
            if *degree == 0 {
                queue.push_back(node.clone());
            }
        }

        // Process nodes
        while let Some(node) = queue.pop_front() {
            result.push(node.clone());

            if let Some(successors) = self.edges.get(&node) {
                for successor in successors {
                    let degree = in_degree.get_mut(successor).unwrap();
                    *degree -= 1;
                    if *degree == 0 {
                        queue.push_back(successor.clone());
                    }
                }
            }
        }

        if result.len() != self.nodes.len() {
            Err("Graph contains cycles".to_string())
        } else {
            Ok(result)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topological_sort() {
        let mut graph = Graph::new();
        graph.add_edge(1, 2);
        graph.add_edge(1, 3);
        graph.add_edge(2, 4);
        graph.add_edge(3, 4);
        
        let sorted = graph.topological_sort().unwrap();
        assert_eq!(sorted[0], 1);
        assert_eq!(sorted[sorted.len() - 1], 4);
    }
}
