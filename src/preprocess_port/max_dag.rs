use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct MaxDag {
    weighted_graph: Vec<Vec<(i32, i32)>>,
    debug: bool,
}

impl MaxDag {
    pub fn new(graph: Vec<Vec<(i32, i32)>>) -> Self {
        Self {
            weighted_graph: graph,
            debug: false,
        }
    }

    pub fn get_result(&self) -> Vec<i32> {
        let num_nodes = self.weighted_graph.len();
        if self.debug {
            for i in 0..num_nodes {
                print!("From {}:", i);
                for trans in &self.weighted_graph[i] {
                    print!(" {} [weight {}]", trans.0, trans.1);
                }
                println!();
            }
        }

        let mut incoming_weights: Vec<i32> = vec![0; num_nodes];
        for weighted_edges in &self.weighted_graph {
            for edge in weighted_edges {
                incoming_weights[edge.0 as usize] += edge.1;
            }
        }

        let mut heap: BTreeMap<(i32, i32), i32> = BTreeMap::new();
        let mut heap_positions: Vec<(i32, i32)> = Vec::new();
        for node in 0..num_nodes {
            if self.debug {
                println!("node {} has {} edges", node, incoming_weights[node]);
            }
            let key = (incoming_weights[node], node as i32);
            heap.insert(key, node as i32);
            heap_positions.push(key);
        }

        let mut done: Vec<bool> = vec![false; num_nodes];
        let mut result: Vec<i32> = Vec::new();

        while !heap.is_empty() {
            let first_key = *heap.keys().next().unwrap();
            let removed = heap.remove(&first_key).unwrap();
            if self.debug {
                println!("minimal element is {}", removed);
            }
            done[removed as usize] = true;
            result.push(removed);
            let succs = &self.weighted_graph[removed as usize];
            for succ in succs {
                let target = succ.0 as usize;
                if !done[target] {
                    let mut arc_weight = succ.1;
                    while arc_weight >= 100000 {
                        arc_weight -= 100000;
                    }
                    let old_key = heap_positions[target];
                    let new_weight = old_key.0 - arc_weight;
                    heap.remove(&old_key);
                    let new_key = (new_weight, target as i32);
                    heap.insert(new_key, target as i32);
                    heap_positions[target] = new_key;
                    if self.debug {
                        println!("node {} has now {} edges", target, new_weight);
                    }
                }
            }
        }

        if self.debug {
            print!("result: ");
            for r in &result {
                print!("{} - ", r);
            }
            println!();
        }
        result
    }
}
