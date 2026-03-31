#[derive(Debug, Clone)]
pub struct Scc {
    graph: Vec<Vec<usize>>,
    dfs_numbers: Vec<i32>,
    dfs_minima: Vec<i32>,
    stack_indices: Vec<i32>,
    stack: Vec<usize>,
    sccs: Vec<Vec<usize>>,
    current_dfs_number: i32,
}

impl Scc {
    pub fn new(graph: Vec<Vec<usize>>) -> Self {
        Self {
            graph,
            dfs_numbers: Vec::new(),
            dfs_minima: Vec::new(),
            stack_indices: Vec::new(),
            stack: Vec::new(),
            sccs: Vec::new(),
            current_dfs_number: 0,
        }
    }

    pub fn get_result(mut self) -> Vec<Vec<usize>> {
        let node_count = self.graph.len();
        self.dfs_numbers = vec![-1; node_count];
        self.dfs_minima = vec![-1; node_count];
        self.stack_indices = vec![-1; node_count];
        self.stack.reserve(node_count);
        self.current_dfs_number = 0;

        for i in 0..node_count {
            if self.dfs_numbers[i] == -1 {
                self.dfs(i);
            }
        }

        self.sccs.reverse();
        self.sccs
    }

    fn dfs(&mut self, vertex: usize) {
        let vertex_dfs_number = self.current_dfs_number;
        self.current_dfs_number += 1;
        let v = vertex;
        self.dfs_numbers[v] = vertex_dfs_number;
        self.dfs_minima[v] = vertex_dfs_number;
        self.stack_indices[v] = self.stack.len() as i32;
        self.stack.push(vertex);

        let successors = self.graph[v].clone();
        for succ in successors {
            let succ_idx = succ;
            let succ_dfs_number = self.dfs_numbers[succ_idx];
            if succ_dfs_number == -1 {
                self.dfs(succ);
                self.dfs_minima[v] = self.dfs_minima[v].min(self.dfs_minima[succ_idx]);
            } else if succ_dfs_number < vertex_dfs_number && self.stack_indices[succ_idx] != -1 {
                self.dfs_minima[v] = self.dfs_minima[v].min(succ_dfs_number);
            }
        }

        if self.dfs_minima[v] == vertex_dfs_number {
            let stack_index = self.stack_indices[v] as usize;
            let num_stack_entries = self.stack.len();
            let mut scc: Vec<usize> = Vec::new();
            for i in stack_index..num_stack_entries {
                let node = self.stack[i];
                scc.push(node);
                self.stack_indices[node] = -1;
            }
            self.stack.truncate(stack_index);
            self.sccs.push(scc);
        }
    }
}
