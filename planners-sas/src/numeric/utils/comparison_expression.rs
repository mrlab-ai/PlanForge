#[cfg(test)]
mod tests;

#[derive(Copy, Clone, Debug)]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    Div, // optional; handle divide-by-zero as needed
}

impl ArithOp {
    #[inline]
    fn apply(self, lhs: f64, rhs: f64) -> f64 {
        match self {
            ArithOp::Add => lhs + rhs,
            ArithOp::Sub => lhs - rhs,
            ArithOp::Mul => lhs * rhs,
            ArithOp::Div => lhs / rhs, // You may want to check for rhs == 0.0
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum CompOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

impl CompOp {
    #[inline]
    fn apply(self, lhs: f64, rhs: f64) -> bool {
        match self {
            CompOp::Lt => lhs < rhs,
            CompOp::Le => lhs <= rhs,
            CompOp::Gt => lhs > rhs,
            CompOp::Ge => lhs >= rhs,
            CompOp::Eq => lhs == rhs,
            CompOp::Ne => lhs != rhs,
        }
    }
}

// NodeId is an index into Expr::nodes
pub type NodeId = usize;

#[derive(Clone, Debug)]
pub enum Node {
    // Leaf reads a value from inputs[input_idx].
    Leaf {
        input_idx: usize,
        val_cache_idx: usize, // index into arith_cache
    },
    // Internal arithmetic node.
    Arith {
        op: ArithOp,
        left: NodeId,
        right: NodeId,
        val_cache_idx: usize, // index into arith_cache
    },
    // Root comparison node (only the root should be this variant).
    CompareRoot {
        op: CompOp,
        left: NodeId,
        right: NodeId,
        cmp_cache_idx: usize, // index into cmp_cache
    },
}

pub struct Expr {
    nodes: Vec<Node>,
    root: NodeId,
    // Caches:
    // - For arithmetic results (f64)
    arith_cache: Vec<(bool, f64)>, // (is_computed, value)
    // - For comparison result (bool)
    cmp_cache: Vec<(bool, bool)>, // (is_computed, value)
}

impl Expr {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            root: 0,
            arith_cache: Vec::new(),
            cmp_cache: Vec::new(),
        }
    }

    // Builder helpers ---------------------------------------------------------

    fn alloc_arith_cache_slot(&mut self) -> usize {
        let idx = self.arith_cache.len();
        self.arith_cache.push((false, 0.0));
        idx
    }

    fn alloc_cmp_cache_slot(&mut self) -> usize {
        let idx = self.cmp_cache.len();
        self.cmp_cache.push((false, false));
        idx
    }

    pub fn add_leaf(&mut self, input_idx: usize) -> NodeId {
        let val_cache_idx = self.alloc_arith_cache_slot();
        let id = self.nodes.len();
        self.nodes.push(Node::Leaf {
            input_idx,
            val_cache_idx,
        });
        id
    }

    pub fn add_arith(&mut self, op: ArithOp, left: NodeId, right: NodeId) -> NodeId {
        let val_cache_idx = self.alloc_arith_cache_slot();
        let id = self.nodes.len();
        self.nodes.push(Node::Arith {
            op,
            left,
            right,
            val_cache_idx,
        });
        id
    }

    // Creates the root as a comparison node and sets `self.root`.
    pub fn set_root_compare(&mut self, op: CompOp, left: NodeId, right: NodeId) -> NodeId {
        let cmp_cache_idx = self.alloc_cmp_cache_slot();
        let id = self.nodes.len();
        self.nodes.push(Node::CompareRoot {
            op,
            left,
            right,
            cmp_cache_idx,
        });
        self.root = id;
        id
    }

    // Evaluation --------------------------------------------------------------

    // Public entry-point: evaluates from the root node.
    // Clears caches first, so results always reflect the provided inputs.
    pub fn evaluate(&mut self, inputs: &[f64]) -> bool {
        self.clear_caches();
        self.eval_root_compare(self.root, inputs)
    }

    fn clear_caches(&mut self) {
        for c in &mut self.arith_cache {
            c.0 = false;
            // c.1 can be left as-is; it’s ignored when c.0 == false
        }
        for c in &mut self.cmp_cache {
            c.0 = false;
        }
    }

    fn eval_root_compare(&mut self, id: NodeId, inputs: &[f64]) -> bool {
        match self.nodes[id] {
            Node::CompareRoot {
                op,
                left,
                right,
                cmp_cache_idx,
            } => {
                // Return cached if available
                if self.cmp_cache[cmp_cache_idx].0 {
                    return self.cmp_cache[cmp_cache_idx].1;
                }
                let lhs = self.eval_arith(left, inputs);
                let rhs = self.eval_arith(right, inputs);
                let res = op.apply(lhs, rhs);
                self.cmp_cache[cmp_cache_idx] = (true, res);
                res
            }
            _ => panic!("Root must be a CompareRoot node"),
        }
    }

    fn eval_arith(&mut self, id: NodeId, inputs: &[f64]) -> f64 {
        match self.nodes[id] {
            Node::Leaf {
                input_idx,
                val_cache_idx,
            } => {
                if self.arith_cache[val_cache_idx].0 {
                    return self.arith_cache[val_cache_idx].1;
                }
                let v = inputs[input_idx];
                self.arith_cache[val_cache_idx] = (true, v);
                v
            }
            Node::Arith {
                op,
                left,
                right,
                val_cache_idx,
            } => {
                if self.arith_cache[val_cache_idx].0 {
                    return self.arith_cache[val_cache_idx].1;
                }
                let lhs = self.eval_arith(left, inputs);
                let rhs = self.eval_arith(right, inputs);
                let v = op.apply(lhs, rhs);
                self.arith_cache[val_cache_idx] = (true, v);
                v
            }
            Node::CompareRoot { .. } => {
                panic!("Arithmetic evaluation called on comparison root")
            }
        }
    }
}



