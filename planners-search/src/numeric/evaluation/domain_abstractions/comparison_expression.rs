#[cfg(test)]
mod tests;

use planners_sas::numeric::axioms::{CalOperator, ComparisonOperator};
use planners_sas::numeric::numeric_task::{AbstractNumericTask, NumericType};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Interval {
    pub lower: f64,
    pub upper: f64,
    pub lower_closed: bool,
    pub upper_closed: bool,
}

impl Interval {
    #[inline]
    pub fn new(lower: f64, upper: f64, lower_closed: bool, upper_closed: bool) -> Self {
        Self {
            lower,
            upper,
            lower_closed,
            upper_closed,
        }
        .normalized()
    }

    #[inline]
    pub fn closed(lower: f64, upper: f64) -> Self {
        Self::new(lower, upper, true, true)
    }

    #[inline]
    pub fn open(lower: f64, upper: f64) -> Self {
        Self::new(lower, upper, false, false)
    }

    #[inline]
    pub fn singleton(value: f64) -> Self {
        Self {
            lower: value,
            upper: value,
            lower_closed: true,
            upper_closed: true,
        }
    }

    #[inline]
    pub fn unbounded() -> Self {
        Self {
            lower: f64::NEG_INFINITY,
            upper: f64::INFINITY,
            lower_closed: false,
            upper_closed: false,
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        if self.lower.is_nan() || self.upper.is_nan() {
            return true;
        }
        if self.lower > self.upper {
            return true;
        }
        if self.lower == self.upper && !(self.lower_closed && self.upper_closed) {
            return true;
        }
        false
    }

    #[inline]
    pub fn contains(&self, value: f64) -> bool {
        if value.is_nan() || self.is_empty() {
            return false;
        }

        let lower_ok = if value > self.lower {
            true
        } else if value == self.lower {
            self.lower_closed
        } else {
            false
        };

        let upper_ok = if value < self.upper {
            true
        } else if value == self.upper {
            self.upper_closed
        } else {
            false
        };

        lower_ok && upper_ok
    }

    #[inline]
    pub fn can_split_at(&self, value: f64, include_in_lower: bool) -> bool {
        if self.is_empty() || value.is_nan() || value.is_infinite() {
            return false;
        }
        if !self.contains(value) {
            return false;
        }
        if self.is_singleton() {
            return false;
        }

        let lower = Interval::new(self.lower, value, self.lower_closed, include_in_lower);
        let upper = Interval::new(value, self.upper, !include_in_lower, self.upper_closed);
        !lower.is_empty() && !upper.is_empty() && lower != *self && upper != *self
    }

    #[inline]
    fn normalized(mut self) -> Self {
        if self.lower.is_infinite() && self.lower.is_sign_negative() {
            self.lower_closed = false;
        }
        if self.upper.is_infinite() && self.upper.is_sign_positive() {
            self.upper_closed = false;
        }

        if self.is_empty() {
            // Canonical empty interval.
            self.lower = 1.0;
            self.upper = 0.0;
            self.lower_closed = false;
            self.upper_closed = false;
        }
        self
    }

    #[inline]
    fn min_bound(&self) -> (f64, bool) {
        (self.lower, self.lower_closed)
    }

    #[inline]
    fn max_bound(&self) -> (f64, bool) {
        (self.upper, self.upper_closed)
    }

    #[inline]
    fn is_singleton(&self) -> bool {
        self.lower == self.upper && self.lower_closed && self.upper_closed
    }
}

impl std::ops::Add for Interval {
    type Output = Interval;

    #[inline]
    fn add(self, rhs: Interval) -> Interval {
        if self.is_empty() || rhs.is_empty() {
            return Interval::open(1.0, 0.0);
        }

        // Performance-oriented over-approximation:
        // If one side is unbounded above and the other side is known non-negative,
        // the result is guaranteed to stay within the unbounded interval.
        if self.upper.is_infinite()
            && self.upper.is_sign_positive()
            && rhs.lower >= 0.0
            && !rhs.lower.is_nan()
        {
            return self;
        }
        if rhs.upper.is_infinite()
            && rhs.upper.is_sign_positive()
            && self.lower >= 0.0
            && !self.lower.is_nan()
        {
            return rhs;
        }

        Interval {
            lower: self.lower + rhs.lower,
            upper: self.upper + rhs.upper,
            lower_closed: self.lower_closed && rhs.lower_closed,
            upper_closed: self.upper_closed && rhs.upper_closed,
        }
        .normalized()
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl ArithOp {
    #[inline]
    fn apply(self, lhs: f64, rhs: f64) -> f64 {
        match self {
            ArithOp::Add => lhs + rhs,
            ArithOp::Sub => lhs - rhs,
            ArithOp::Mul => lhs * rhs,
            ArithOp::Div => lhs / rhs,
        }
    }

    #[inline]
    pub fn apply_interval(self, lhs: Interval, rhs: Interval) -> Interval {
        match self {
            ArithOp::Add => lhs + rhs,
            ArithOp::Sub => {
                if lhs.is_empty() || rhs.is_empty() {
                    return Interval::open(1.0, 0.0);
                }
                Interval {
                    lower: lhs.lower - rhs.upper,
                    upper: lhs.upper - rhs.lower,
                    lower_closed: lhs.lower_closed && rhs.upper_closed,
                    upper_closed: lhs.upper_closed && rhs.lower_closed,
                }
                .normalized()
            }
            ArithOp::Mul => {
                if lhs.is_empty() || rhs.is_empty() {
                    return Interval::open(1.0, 0.0);
                }
                // Conservative multiplication (ignores some endpoint openness details).
                let candidates = [
                    lhs.lower * rhs.lower,
                    lhs.lower * rhs.upper,
                    lhs.upper * rhs.lower,
                    lhs.upper * rhs.upper,
                ];
                let mut lo = f64::INFINITY;
                let mut hi = f64::NEG_INFINITY;
                for &v in &candidates {
                    if v.is_nan() {
                        continue;
                    }
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
                if lo == f64::INFINITY {
                    return Interval::unbounded();
                }
                Interval {
                    lower: lo,
                    upper: hi,
                    lower_closed: false,
                    upper_closed: false,
                }
                .normalized()
            }
            ArithOp::Div => {
                if lhs.is_empty() || rhs.is_empty() {
                    return Interval::open(1.0, 0.0);
                }

                // If divisor contains 0, we conservatively give up.
                let (rlo, rlo_c) = rhs.min_bound();
                let (rhi, rhi_c) = rhs.max_bound();
                let contains_zero =
                    (rlo < 0.0 && rhi > 0.0) || (rlo == 0.0 && rlo_c) || (rhi == 0.0 && rhi_c);
                if contains_zero {
                    return Interval::unbounded();
                }

                // Reciprocal interval.
                let inv_lo = 1.0 / rhs.upper;
                let inv_hi = 1.0 / rhs.lower;
                let inv = if inv_lo <= inv_hi {
                    Interval {
                        lower: inv_lo,
                        upper: inv_hi,
                        lower_closed: rhs.upper_closed,
                        upper_closed: rhs.lower_closed,
                    }
                } else {
                    Interval {
                        lower: inv_hi,
                        upper: inv_lo,
                        lower_closed: rhs.lower_closed,
                        upper_closed: rhs.upper_closed,
                    }
                }
                .normalized();
                ArithOp::Mul.apply_interval(lhs, inv)
            }
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
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

    #[inline]
    fn apply_interval(self, lhs: Interval, rhs: Interval) -> Option<bool> {
        if lhs.is_empty() || rhs.is_empty() {
            return Some(false);
        }

        let (lmin, lmin_c) = lhs.min_bound();
        let (lmax, lmax_c) = lhs.max_bound();
        let (rmin, rmin_c) = rhs.min_bound();
        let (rmax, rmax_c) = rhs.max_bound();

        // Helpers for strict bound comparisons.
        let max_lt_min = |amax: f64, amax_c: bool, bmin: f64, bmin_c: bool| -> bool {
            (amax < bmin) || (amax == bmin && (!amax_c || !bmin_c))
        };
        let min_ge_max = |amin: f64, amin_c: bool, bmax: f64, bmax_c: bool| -> bool {
            (amin > bmax) || (amin == bmax && (amin_c && bmax_c))
        };

        match self {
            CompOp::Lt => {
                if max_lt_min(lmax, lmax_c, rmin, rmin_c) {
                    Some(true)
                } else if min_ge_max(lmin, lmin_c, rmax, rmax_c) {
                    Some(false)
                } else {
                    None
                }
            }
            CompOp::Le => {
                if lmax <= rmin {
                    Some(true)
                } else if lmin > rmax {
                    Some(false)
                } else {
                    None
                }
            }
            CompOp::Gt => CompOp::Lt.apply_interval(rhs, lhs).map(|b| b),
            CompOp::Ge => CompOp::Le.apply_interval(rhs, lhs).map(|b| b),
            CompOp::Eq => {
                if lhs.is_singleton() && rhs.is_singleton() && lmin == rmin {
                    Some(true)
                } else if max_lt_min(lmax, lmax_c, rmin, rmin_c)
                    || max_lt_min(rmax, rmax_c, lmin, lmin_c)
                {
                    Some(false)
                } else {
                    None
                }
            }
            CompOp::Ne => {
                if lhs.is_singleton() && rhs.is_singleton() && lmin == rmin {
                    Some(false)
                } else if max_lt_min(lmax, lmax_c, rmin, rmin_c)
                    || max_lt_min(rmax, rmax_c, lmin, lmin_c)
                {
                    Some(true)
                } else {
                    None
                }
            }
        }
    }
}

// NodeId is an index into Expr::nodes
pub type NodeId = usize;

#[derive(Clone, Debug)]
pub enum Node {
    Leaf {
        input_idx: usize,
        val_cache_idx: usize,
    },
    Arith {
        op: ArithOp,
        left: NodeId,
        right: NodeId,
        val_cache_idx: usize,
    },
    CompareRoot {
        op: CompOp,
        left: NodeId,
        right: NodeId,
        cmp_cache_idx: usize,
    },
}

pub struct Expr {
    nodes: Vec<Node>,
    root: NodeId,
    arith_cache: Vec<(bool, f64)>,
    arith_interval_cache: Vec<(bool, Interval)>,
    cmp_cache: Vec<(bool, bool)>,
    cmp_interval_cache: Vec<(bool, Option<bool>)>,
}

impl Expr {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            root: 0,
            arith_cache: Vec::new(),
            arith_interval_cache: Vec::new(),
            cmp_cache: Vec::new(),
            cmp_interval_cache: Vec::new(),
        }
    }

    fn alloc_arith_cache_slot(&mut self) -> usize {
        let idx = self.arith_cache.len();
        self.arith_cache.push((false, 0.0));
        self.arith_interval_cache
            .push((false, Interval::open(1.0, 0.0)));
        idx
    }

    fn alloc_cmp_cache_slot(&mut self) -> usize {
        let idx = self.cmp_cache.len();
        self.cmp_cache.push((false, false));
        self.cmp_interval_cache.push((false, None));
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

    pub fn evaluate(&mut self, inputs: &[f64]) -> bool {
        self.clear_point_caches();
        self.eval_root_compare(self.root, inputs)
    }

    pub fn evaluate_interval(&mut self, inputs: &[Interval]) -> Option<bool> {
        self.clear_interval_caches();
        self.eval_root_compare_interval(self.root, inputs)
    }

    fn clear_point_caches(&mut self) {
        for c in &mut self.arith_cache {
            c.0 = false;
        }
        for c in &mut self.cmp_cache {
            c.0 = false;
        }
    }

    fn clear_interval_caches(&mut self) {
        for c in &mut self.arith_interval_cache {
            c.0 = false;
        }
        for c in &mut self.cmp_interval_cache {
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

    fn eval_root_compare_interval(&mut self, id: NodeId, inputs: &[Interval]) -> Option<bool> {
        match self.nodes[id] {
            Node::CompareRoot {
                op,
                left,
                right,
                cmp_cache_idx,
            } => {
                if self.cmp_interval_cache[cmp_cache_idx].0 {
                    return self.cmp_interval_cache[cmp_cache_idx].1;
                }
                let lhs = self.eval_arith_interval(left, inputs);
                let rhs = self.eval_arith_interval(right, inputs);
                let res = op.apply_interval(lhs, rhs);
                self.cmp_interval_cache[cmp_cache_idx] = (true, res);
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
            Node::CompareRoot { .. } => panic!("Arithmetic evaluation called on comparison root"),
        }
    }

    fn eval_arith_interval(&mut self, id: NodeId, inputs: &[Interval]) -> Interval {
        match self.nodes[id] {
            Node::Leaf {
                input_idx,
                val_cache_idx,
            } => {
                if self.arith_interval_cache[val_cache_idx].0 {
                    return self.arith_interval_cache[val_cache_idx].1;
                }
                let v = inputs[input_idx];
                self.arith_interval_cache[val_cache_idx] = (true, v);
                v
            }
            Node::Arith {
                op,
                left,
                right,
                val_cache_idx,
            } => {
                if self.arith_interval_cache[val_cache_idx].0 {
                    return self.arith_interval_cache[val_cache_idx].1;
                }
                let lhs = self.eval_arith_interval(left, inputs);
                let rhs = self.eval_arith_interval(right, inputs);
                let v = op.apply_interval(lhs, rhs);
                self.arith_interval_cache[val_cache_idx] = (true, v);
                v
            }
            Node::CompareRoot { .. } => panic!("Arithmetic evaluation called on comparison root"),
        }
    }
}

pub type ComparisonTreeNodeId = usize;

#[derive(Debug, Clone, PartialEq)]
pub enum ComparisonTreeNode {
    Leaf {
        numeric_var_id: i32,
    },
    Arith {
        result_numeric_var_id: i32,
        assignment_axiom_id: usize,
        op: ArithOp,
        left_numeric_var_id: i32,
        right_numeric_var_id: i32,
        left: ComparisonTreeNodeId,
        right: ComparisonTreeNodeId,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ComparisonTree {
    pub comparison_axiom_id: usize,
    pub affected_var_id: i32,
    pub op: CompOp,
    pub left_numeric_var_id: i32,
    pub right_numeric_var_id: i32,
    pub nodes: Vec<ComparisonTreeNode>,
    pub left_root: ComparisonTreeNodeId,
    pub right_root: ComparisonTreeNodeId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComparisonTreeBuildError {
    InvalidComparisonAxiomId { provided: usize, len: usize },
    InvalidNumericVarId { provided: i32, len: usize },
    CycleDetected { numeric_var_id: i32 },
}

impl ComparisonTree {
    pub fn from_task(
        task: &dyn AbstractNumericTask,
        comparison_axiom_id: usize,
    ) -> Result<Self, ComparisonTreeBuildError> {
        let cmp_axioms = task.comparison_axioms();
        if comparison_axiom_id >= cmp_axioms.len() {
            return Err(ComparisonTreeBuildError::InvalidComparisonAxiomId {
                provided: comparison_axiom_id,
                len: cmp_axioms.len(),
            });
        }

        let num_numeric_vars = task.numeric_variables().len();
        let mut affected_to_assignment_axiom: Vec<Option<usize>> = vec![None; num_numeric_vars];
        for (axiom_id, ax) in task.assignment_axioms().iter().enumerate() {
            let affected = ax.get_affected_var_id() as usize;
            if affected < num_numeric_vars {
                affected_to_assignment_axiom[affected] = Some(axiom_id);
            }
        }

        let cmp = &cmp_axioms[comparison_axiom_id];
        let op = comp_op_from_axiom(cmp.get_operator());
        let affected_var_id = cmp.get_affected_var_id();
        let left_numeric_var_id = cmp.get_left_var_id();
        let right_numeric_var_id = cmp.get_right_var_id();

        let mut nodes: Vec<ComparisonTreeNode> = Vec::new();
        let mut memo: Vec<Option<ComparisonTreeNodeId>> = vec![None; num_numeric_vars];
        let mut visiting: Vec<bool> = vec![false; num_numeric_vars];

        let left_root = build_numeric_tree_node(
            task,
            left_numeric_var_id,
            &affected_to_assignment_axiom,
            &mut nodes,
            &mut memo,
            &mut visiting,
        )?;
        let right_root = build_numeric_tree_node(
            task,
            right_numeric_var_id,
            &affected_to_assignment_axiom,
            &mut nodes,
            &mut memo,
            &mut visiting,
        )?;

        Ok(Self {
            comparison_axiom_id,
            affected_var_id,
            op,
            left_numeric_var_id,
            right_numeric_var_id,
            nodes,
            left_root,
            right_root,
        })
    }

    pub fn regular_numeric_var_dependencies(&self, task: &dyn AbstractNumericTask) -> Vec<i32> {
        let num_numeric_vars = task.numeric_variables().len();
        let mut seen: Vec<bool> = vec![false; num_numeric_vars];
        let mut out: Vec<i32> = Vec::new();

        let mut stack: Vec<ComparisonTreeNodeId> = vec![self.left_root, self.right_root];
        while let Some(node_id) = stack.pop() {
            match &self.nodes[node_id] {
                ComparisonTreeNode::Leaf { numeric_var_id } => {
                    if let Ok(idx) = usize::try_from(*numeric_var_id) {
                        if idx < num_numeric_vars
                            && !seen[idx]
                            && task.numeric_variables()[idx].get_type() == &NumericType::Regular
                        {
                            seen[idx] = true;
                            out.push(*numeric_var_id);
                        }
                    }
                }
                ComparisonTreeNode::Arith { left, right, .. } => {
                    stack.push(*left);
                    stack.push(*right);
                }
            }
        }

        out.sort_unstable();
        out.dedup();
        out
    }

    pub fn evaluate_interval(&self, inputs: &[Interval]) -> Option<bool> {
        let lhs = self.eval_node_interval(self.left_root, inputs);
        let rhs = self.eval_node_interval(self.right_root, inputs);
        self.op.apply_interval(lhs, rhs)
    }

    pub fn evaluate_point(&self, inputs: &[f64]) -> bool {
        let lhs = self.eval_node_point(self.left_root, inputs);
        let rhs = self.eval_node_point(self.right_root, inputs);
        self.op.apply(lhs, rhs)
    }

    fn eval_node_interval(&self, node_id: ComparisonTreeNodeId, inputs: &[Interval]) -> Interval {
        match &self.nodes[node_id] {
            ComparisonTreeNode::Leaf { numeric_var_id } => {
                let idx = usize::try_from(*numeric_var_id)
                    .unwrap_or_else(|_| panic!("negative numeric_var_id {numeric_var_id}"));
                inputs[idx]
            }
            ComparisonTreeNode::Arith {
                op, left, right, ..
            } => {
                let lhs = self.eval_node_interval(*left, inputs);
                let rhs = self.eval_node_interval(*right, inputs);
                op.apply_interval(lhs, rhs)
            }
        }
    }

    fn eval_node_point(&self, node_id: ComparisonTreeNodeId, inputs: &[f64]) -> f64 {
        match &self.nodes[node_id] {
            ComparisonTreeNode::Leaf { numeric_var_id } => {
                let idx = usize::try_from(*numeric_var_id)
                    .unwrap_or_else(|_| panic!("negative numeric_var_id {numeric_var_id}"));
                inputs[idx]
            }
            ComparisonTreeNode::Arith {
                op, left, right, ..
            } => {
                let lhs = self.eval_node_point(*left, inputs);
                let rhs = self.eval_node_point(*right, inputs);
                op.apply(lhs, rhs)
            }
        }
    }
}

fn comp_op_from_axiom(op: &ComparisonOperator) -> CompOp {
    match op {
        ComparisonOperator::LessThan => CompOp::Lt,
        ComparisonOperator::LessThanOrEqual => CompOp::Le,
        ComparisonOperator::Equal => CompOp::Eq,
        ComparisonOperator::GreaterThanOrEqual => CompOp::Ge,
        ComparisonOperator::GreaterThan => CompOp::Gt,
        ComparisonOperator::UnEqual => CompOp::Ne,
    }
}

fn arith_op_from_axiom(op: &CalOperator) -> ArithOp {
    match op {
        CalOperator::Sum => ArithOp::Add,
        CalOperator::Difference => ArithOp::Sub,
        CalOperator::Product => ArithOp::Mul,
        CalOperator::Division => ArithOp::Div,
    }
}

fn build_numeric_tree_node(
    task: &dyn AbstractNumericTask,
    numeric_var_id: i32,
    affected_to_assignment_axiom: &[Option<usize>],
    nodes: &mut Vec<ComparisonTreeNode>,
    memo: &mut Vec<Option<ComparisonTreeNodeId>>,
    visiting: &mut Vec<bool>,
) -> Result<ComparisonTreeNodeId, ComparisonTreeBuildError> {
    let idx = usize::try_from(numeric_var_id).map_err(|_| {
        ComparisonTreeBuildError::InvalidNumericVarId {
            provided: numeric_var_id,
            len: affected_to_assignment_axiom.len(),
        }
    })?;
    if idx >= affected_to_assignment_axiom.len() {
        return Err(ComparisonTreeBuildError::InvalidNumericVarId {
            provided: numeric_var_id,
            len: affected_to_assignment_axiom.len(),
        });
    }

    if let Some(node_id) = memo[idx] {
        return Ok(node_id);
    }

    if visiting[idx] {
        return Err(ComparisonTreeBuildError::CycleDetected { numeric_var_id });
    }
    visiting[idx] = true;

    let node_id = if let Some(assignment_axiom_id) = affected_to_assignment_axiom[idx] {
        let ax = &task.assignment_axioms()[assignment_axiom_id];
        let op = arith_op_from_axiom(ax.get_operator());
        let left_numeric_var_id = ax.get_left_var_id() as i32;
        let right_numeric_var_id = ax.get_right_var_id() as i32;

        let left = build_numeric_tree_node(
            task,
            left_numeric_var_id,
            affected_to_assignment_axiom,
            nodes,
            memo,
            visiting,
        )?;
        let right = build_numeric_tree_node(
            task,
            right_numeric_var_id,
            affected_to_assignment_axiom,
            nodes,
            memo,
            visiting,
        )?;

        let node_id = nodes.len();
        nodes.push(ComparisonTreeNode::Arith {
            result_numeric_var_id: numeric_var_id,
            assignment_axiom_id,
            op,
            left_numeric_var_id,
            right_numeric_var_id,
            left,
            right,
        });
        node_id
    } else {
        let node_id = nodes.len();
        nodes.push(ComparisonTreeNode::Leaf { numeric_var_id });
        node_id
    };

    visiting[idx] = false;
    memo[idx] = Some(node_id);
    Ok(node_id)
}
