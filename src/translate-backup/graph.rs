#[derive(Debug, Clone)]
pub struct Graph {
    pub adjacency: Vec<Vec<usize>>,
}

impl Graph {
    pub fn new(size: usize) -> Self {
        Self {
            adjacency: vec![Vec::new(); size],
        }
    }

    pub fn add_edge(&mut self, from: usize, to: usize) {
        if let Some(list) = self.adjacency.get_mut(from) {
            list.push(to);
        }
    }
}
