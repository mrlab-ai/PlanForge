use std::collections::BTreeMap;

use tracing::{Level, debug};

#[derive(Debug, Clone)]
pub struct MaxDag {
    weighted_graph: Vec<Vec<(usize, u64)>>,
}

impl MaxDag {
    pub fn new(graph: Vec<Vec<(usize, u64)>>) -> Self {
        Self {
            weighted_graph: graph,
        }
    }

    #[allow(clippy::needless_range_loop)]
    pub fn get_result(&self) -> Vec<usize> {
        let num_nodes = self.weighted_graph.len();
        if tracing::enabled!(Level::DEBUG) {
            for i in 0..num_nodes {
                debug!("From {}:", i);
                for trans in &self.weighted_graph[i] {
                    debug!(" {} [weight {}]", trans.0, trans.1);
                }
                debug!("");
            }
        }

        let mut incoming_weights: Vec<u64> = vec![0; num_nodes];
        for weighted_edges in &self.weighted_graph {
            for edge in weighted_edges {
                incoming_weights[edge.0] += edge.1;
            }
        }

        let mut heap: BTreeMap<(u64, usize), usize> = BTreeMap::new();
        let mut heap_positions: Vec<(u64, usize)> = Vec::new();
        for node in 0..num_nodes {
            debug!("node {} has {} edges", node, incoming_weights[node]);
            let key = (incoming_weights[node], node);
            heap.insert(key, node);
            heap_positions.push(key);
        }

        let mut done: Vec<bool> = vec![false; num_nodes];
        let mut result: Vec<usize> = Vec::new();

        while !heap.is_empty() {
            let first_key = *heap.keys().next().unwrap();
            let removed = heap.remove(&first_key).unwrap();
            debug!("minimal element is {}", removed);
            done[removed] = true;
            result.push(removed);
            let succs = &self.weighted_graph[removed];
            for succ in succs {
                let target = succ.0;
                if !done[target] {
                    let mut arc_weight = succ.1;
                    while arc_weight >= 100000 {
                        arc_weight -= 100000;
                    }
                    let old_key = heap_positions[target];
                    let new_weight = old_key.0 - arc_weight;
                    heap.remove(&old_key);
                    let new_key = (new_weight, target);
                    heap.insert(new_key, target);
                    heap_positions[target] = new_key;
                    debug!("node {} has now {} edges", target, new_weight);
                }
            }
        }

        if tracing::enabled!(Level::DEBUG) {
            debug!("result: ");
            for r in &result {
                debug!("{} - ", r);
            }
            debug!("");
        }
        result
    }
}
