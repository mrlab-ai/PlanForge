#[cfg(test)]
mod tests;

use planforge_sas::numeric::{
    numeric_task::{AbstractNumericTask, ExplicitFact},
    utils::errors::ConstructError,
};
use std::collections::VecDeque;
use std::fmt::Debug;

type Condition<'a> = Vec<&'a ExplicitFact>;

/// Index into the `SuccessorTree`'s internal node storage. The high bit
/// flags the node kind (`1` = leaf, `0` = branch); the remaining 31 bits
/// are the slot within the corresponding `Vec`. Sentinel `INVALID` marks
/// "no child" without paying for `Option<NodeId>` (which would be 8 B per
/// reference instead of 4).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct NodeId(u32);

impl NodeId {
    const LEAF_BIT: u32 = 1 << 31;
    const INDEX_MASK: u32 = !Self::LEAF_BIT;
    const INVALID: NodeId = NodeId(u32::MAX);

    #[inline]
    fn branch(idx: usize) -> Self {
        debug_assert!(idx < Self::LEAF_BIT as usize);
        Self(idx as u32)
    }
    #[inline]
    fn leaf(idx: usize) -> Self {
        debug_assert!(idx < Self::LEAF_BIT as usize);
        Self((idx as u32) | Self::LEAF_BIT)
    }
    #[inline]
    fn is_leaf(self) -> bool {
        self.0 & Self::LEAF_BIT != 0
    }
    #[inline]
    fn index(self) -> usize {
        (self.0 & Self::INDEX_MASK) as usize
    }
    #[inline]
    fn is_valid(self) -> bool {
        self != Self::INVALID
    }
}

/// Single branch node in the precondition decision tree.
///
/// `value_children` is indexed by the concrete value of `var_id`. Entries
/// outside the variable's actual range are filled with `NodeId::INVALID`
/// so the lookup can be a single bounds check. `default_child` collects
/// operators that don't constrain this variable and is `INVALID` when no
/// such operator subtree exists.
///
/// Children are stored as 4-byte `NodeId`s rather than `Box<dyn Node>`
/// fat pointers (16 B), and immediate-operator lists / value-child lists
/// are `Box<[T]>` (2 words) rather than `Vec<T>` (3 words plus growth
/// capacity).
#[derive(Debug)]
struct BranchEntry {
    var_id: u32,
    immediate_operators: Box<[u32]>,
    value_children: Box<[NodeId]>,
    default_child: NodeId,
}

#[derive(Debug)]
struct LeafEntry {
    applicable_operators: Box<[u32]>,
}

/// Decision tree returning the ids of the operators applicable in a given
/// state. Callers resolve ids to `&Operator` via `task.get_operators()`.
///
/// All nodes live in two `Vec`s on the same heap allocation each, so
/// constructing the tree on tasks with hundreds of thousands of operators
/// performs O(n) allocations total instead of O(n) `Box::new` calls. The
/// tree holds no reference to the task: it borrows the task only during
/// construction and stores plain operator ids.
pub struct SuccessorTree {
    branches: Vec<BranchEntry>,
    leaves: Vec<LeafEntry>,
    /// Index 0 in `leaves` is a shared empty leaf reused for every
    /// (branch, value) slot with no applicable operators. The previous
    /// implementation allocated a separate `Box<LeafNode>` for each — on
    /// minecraft-sword-advanced/prob_30x30_5 that meant ~110 k identical
    /// heap objects.
    empty_leaf: NodeId,
    root: NodeId,
}

impl Debug for SuccessorTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SuccessorTree")
            .field("branches", &self.branches.len())
            .field("leaves", &self.leaves.len())
            .field("root", &self.root)
            .finish()
    }
}

impl SuccessorTree {
    pub fn new(task: &dyn AbstractNumericTask) -> Self {
        let mut builder = TreeBuilder::new(task);
        let mut queue: VecDeque<u32> = (0..task.get_operators().len() as u32).collect();
        let root = builder
            .construct(0, &mut queue)
            .expect("successor-tree construction failed");
        SuccessorTree {
            branches: builder.branches,
            leaves: builder.leaves,
            empty_leaf: builder.empty_leaf,
            root,
        }
    }

    /// Append the ids of all operators applicable in `state` to `out`.
    pub fn get_applicable_operators(&self, state: &[usize], out: &mut Vec<u32>) {
        self.walk(self.root, state, out);
    }

    fn walk(&self, id: NodeId, state: &[usize], out: &mut Vec<u32>) {
        // Shared empty leaf is a fast no-op; many branches point to it.
        if id == self.empty_leaf {
            return;
        }
        if id.is_leaf() {
            let leaf = &self.leaves[id.index()];
            out.extend_from_slice(&leaf.applicable_operators);
            return;
        }
        let branch = &self.branches[id.index()];
        out.extend_from_slice(&branch.immediate_operators);
        let value = state[branch.var_id as usize];
        if let Some(&child) = branch.value_children.get(value)
            && child.is_valid()
        {
            self.walk(child, state, out);
        }
        if branch.default_child.is_valid() {
            self.walk(branch.default_child, state, out);
        }
    }
}

// -----------------------------------------------------------------------
// Construction
// -----------------------------------------------------------------------

struct TreeBuilder<'a> {
    task: &'a dyn AbstractNumericTask,
    branches: Vec<BranchEntry>,
    leaves: Vec<LeafEntry>,
    empty_leaf: NodeId,
    conditions: Vec<Condition<'a>>,
    next_condition_by_operator: Vec<usize>,
}

impl<'a> TreeBuilder<'a> {
    fn new(task: &'a dyn AbstractNumericTask) -> Self {
        let operators = task.get_operators();
        let mut conditions = Vec::with_capacity(operators.len());
        let mut next_condition_by_operator = Vec::with_capacity(operators.len());

        for operator in operators.iter() {
            let mut condition: Vec<&ExplicitFact> = operator.preconditions().iter().collect();
            // PARITY(numeric-fd): sort conditions by variable id (only).
            condition.sort_unstable_by_key(|f| f.var());
            conditions.push(condition);
            next_condition_by_operator.push(0);
        }

        let mut leaves: Vec<LeafEntry> = Vec::new();
        leaves.push(LeafEntry {
            applicable_operators: Box::from([] as [u32; 0]),
        });
        let empty_leaf = NodeId::leaf(0);

        TreeBuilder {
            task,
            branches: Vec::new(),
            leaves,
            empty_leaf,
            conditions,
            next_condition_by_operator,
        }
    }

    fn push_branch(&mut self, branch: BranchEntry) -> NodeId {
        let id = NodeId::branch(self.branches.len());
        self.branches.push(branch);
        id
    }

    fn push_leaf(&mut self, ops: Vec<u32>) -> NodeId {
        if ops.is_empty() {
            return self.empty_leaf;
        }
        let id = NodeId::leaf(self.leaves.len());
        self.leaves.push(LeafEntry {
            applicable_operators: ops.into_boxed_slice(),
        });
        id
    }

    fn construct(
        &mut self,
        mut branch_var_id: usize,
        queue: &mut VecDeque<u32>,
    ) -> Result<NodeId, ConstructError> {
        if queue.is_empty() {
            return Ok(self.empty_leaf);
        }
        loop {
            if branch_var_id >= self.task.variables().len() {
                let ops: Vec<u32> = queue.drain(..).collect();
                return Ok(self.push_leaf(ops));
            }

            let branch_var = &self.task.variables()[branch_var_id];
            let num_children = branch_var.domain_size();

            let mut operators_for_value: Vec<VecDeque<u32>> = vec![VecDeque::new(); num_children];
            let mut default_operators: VecDeque<u32> = VecDeque::new();
            let mut applicable_operators: Vec<u32> = Vec::new();

            let mut all_ops_immediate = true;
            let mut var_interesting = false;

            while let Some(op_id) = queue.pop_front() {
                let op_idx = op_id as usize;
                let condition_index = self.next_condition_by_operator[op_idx];

                if condition_index >= self.conditions[op_idx].len() {
                    var_interesting = true;
                    applicable_operators.push(op_id);
                } else {
                    all_ops_immediate = false;
                    let fact = &self.conditions[op_idx][condition_index];
                    if fact.var() == branch_var_id {
                        var_interesting = true;
                        let mut new_index = condition_index;
                        while new_index < self.conditions[op_idx].len()
                            && self.conditions[op_idx][new_index].var() == branch_var_id
                        {
                            new_index += 1;
                        }
                        self.next_condition_by_operator[op_idx] = new_index;
                        operators_for_value[fact.value()].push_back(op_id);
                    } else {
                        default_operators.push_back(op_id);
                    }
                }
            }

            if all_ops_immediate {
                return Ok(self.push_leaf(applicable_operators));
            } else if var_interesting {
                let mut value_children: Vec<NodeId> = Vec::with_capacity(operators_for_value.len());
                for mut ops in operators_for_value.into_iter() {
                    let child = self.construct(branch_var_id + 1, &mut ops)?;
                    value_children.push(child);
                }
                let default_child = self.construct(branch_var_id + 1, &mut default_operators)?;
                let immediate = applicable_operators.into_boxed_slice();
                let var_id_u32: u32 = branch_var_id
                    .try_into()
                    .expect("variable id overflows u32 — sas task has too many variables");
                return Ok(self.push_branch(BranchEntry {
                    var_id: var_id_u32,
                    immediate_operators: immediate,
                    value_children: value_children.into_boxed_slice(),
                    default_child,
                }));
            } else {
                branch_var_id += 1;
                std::mem::swap(&mut default_operators, queue);
            }
        }
    }
}

// -----------------------------------------------------------------------
// Backward-compatible facade for callers that still want
// `GroundedSuccessorGenerator::new(...)` + `.construct(...)` (tests and
// pattern databases). All real work is delegated to the arena-backed
// builder.
// -----------------------------------------------------------------------

pub struct GroundedSuccessorGenerator<'a> {
    builder: TreeBuilder<'a>,
}

impl<'a> GroundedSuccessorGenerator<'a> {
    pub fn new(task: &'a dyn AbstractNumericTask) -> Self {
        Self {
            builder: TreeBuilder::new(task),
        }
    }

    pub fn construct_node_from_task<T: AbstractNumericTask>(task: &T) -> SuccessorTree {
        SuccessorTree::new(task)
    }

    /// Finishes the in-progress construction and returns the rooted
    /// `SuccessorTree`. The first call seeds the queue with every operator
    /// in the task; subsequent recursive calls (from tests) are wrapped via
    /// `construct_at`.
    pub fn construct(
        &mut self,
        branch_var_id: &mut usize,
        queue: &mut VecDeque<u32>,
    ) -> Result<NodeRef, ConstructError> {
        let id = self.builder.construct(*branch_var_id, queue)?;
        Ok(NodeRef { id })
    }

    /// Returns a `SuccessorTree` consuming the in-progress builder. Roots
    /// at the supplied `NodeRef`.
    pub fn into_tree(self, root: NodeRef) -> SuccessorTree {
        let TreeBuilder {
            branches,
            leaves,
            empty_leaf,
            ..
        } = self.builder;
        SuccessorTree {
            branches,
            leaves,
            empty_leaf,
            root: root.id,
        }
    }
}

/// Opaque handle into the in-progress `GroundedSuccessorGenerator`. Use
/// `into_tree` to convert into a usable `SuccessorTree`. Construction-time
/// only — search-engine and pattern-database hot paths use `SuccessorTree`
/// directly.
#[derive(Debug, Copy, Clone)]
pub struct NodeRef {
    id: NodeId,
}
