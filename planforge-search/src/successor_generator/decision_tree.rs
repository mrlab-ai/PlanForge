//! The decision tree of Helmert (2006) that Fast Downward generates
//! successors with: branches test one variable each, and the operators whose
//! preconditions a path fixes sit at its end.

use planforge_sas::numeric_task::{
    ExplicitFact, ExplicitValueIndex, OperatorIndex, Task, VariableIndex,
};
use planforge_sas::utils::errors::ConstructError;
use std::collections::VecDeque;
use std::fmt::Debug;

use super::SuccessorGenerator;

type Condition<'a> = Vec<&'a ExplicitFact>;

/// Index into the `DecisionTree`'s internal node storage. The high bit
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
    var_id: VariableIndex,
    immediate_operators: Box<[OperatorIndex]>,
    value_children: Box<[NodeId]>,
    default_child: NodeId,
}

#[derive(Debug)]
struct LeafEntry {
    applicable_operators: Box<[OperatorIndex]>,
}

/// Decision tree returning the ids of the operators applicable in a given
/// state. Callers resolve ids to `&Operator` via `task.operators()`.
///
/// All nodes live in two `Vec`s on the same heap allocation each, so
/// constructing the tree on tasks with hundreds of thousands of operators
/// performs O(n) allocations total instead of O(n) `Box::new` calls. The
/// tree holds no reference to the task: it borrows the task only during
/// construction and stores plain operator ids.
pub struct DecisionTree {
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

impl Debug for DecisionTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecisionTree")
            .field("branches", &self.branches.len())
            .field("leaves", &self.leaves.len())
            .field("root", &self.root)
            .finish()
    }
}

impl DecisionTree {
    pub fn new(task: &Task) -> Self {
        let mut builder = TreeBuilder::new(task);
        let mut queue: VecDeque<OperatorIndex> = (0..task.operators().len() as u32)
            .map(OperatorIndex::new)
            .collect();
        let root = builder
            .construct(0, &mut queue)
            .expect("successor-tree construction failed");
        DecisionTree {
            branches: builder.branches,
            leaves: builder.leaves,
            empty_leaf: builder.empty_leaf,
            root,
        }
    }

    /// Follows default children in a loop and recurses only into value
    /// children, so the depth is bounded by the operators' precondition
    /// counts, not by the number of variables (a default chain can pass
    /// through every variable).
    fn walk(&self, mut id: NodeId, state: &[ExplicitValueIndex], out: &mut Vec<OperatorIndex>) {
        loop {
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
            let value = state[branch.var_id.index()];
            if let Some(&child) = branch.value_children.get(value.index())
                && child.is_valid()
            {
                self.walk(child, state, out);
            }
            if !branch.default_child.is_valid() {
                return;
            }
            id = branch.default_child;
        }
    }
}

impl SuccessorGenerator for DecisionTree {
    /// Appends the operators applicable in `state` to `applicable` in the
    /// order the walk reaches them: a branch's shared immediate operators, then
    /// the child its tested variable's value selects, then the default child.
    ///
    /// That order is a function of the task alone, so it is reproducible, but it
    /// is not ascending operator id. It cannot be made so by ordering the tree at
    /// construction time: with operators `0: v=0`, `1: (no condition)`, `2: v=1`
    /// the root branches on `v` with `immediate = [1]` and `value_children =
    /// [[0], [2]]`, and one tree serves both states, so the immediate list is
    /// emitted either before or after both children while ascending id wants it
    /// last for `v=0` and first for `v=1`. Only sorting each expansion's output
    /// would give ascending id, and that costs about a percent of the operators
    /// emitted on the corpus's widest instance to buy nothing the search needs.
    ///
    /// Emission order is not free of consequence: the search inserts successors
    /// into the open list as they arrive, so it settles A* tie-breaking among
    /// nodes of equal f and therefore how many states are expanded. It follows
    /// that renumbering a task's variables reshapes the tree and moves expansion
    /// counts — in either direction, and by a lot. Plan cost and plan length are
    /// what a renumbering may not touch.
    fn applicable_operators(
        &self,
        state: &[ExplicitValueIndex],
        applicable: &mut Vec<OperatorIndex>,
    ) {
        self.walk(self.root, state, applicable);
    }
}

// -----------------------------------------------------------------------
// Construction
// -----------------------------------------------------------------------

struct TreeBuilder<'a> {
    task: &'a Task,
    branches: Vec<BranchEntry>,
    leaves: Vec<LeafEntry>,
    empty_leaf: NodeId,
    conditions: Vec<Condition<'a>>,
    next_condition_by_operator: Vec<usize>,
}

impl<'a> TreeBuilder<'a> {
    fn new(task: &'a Task) -> Self {
        let operators = task.operators();
        let mut conditions = Vec::with_capacity(operators.len());
        let mut next_condition_by_operator = Vec::with_capacity(operators.len());

        for operator in operators.iter() {
            let mut condition: Vec<&ExplicitFact> = operator.preconditions().iter().collect();
            // PARITY(numeric-fd): sort conditions by variable id (only).
            condition.sort_unstable_by_key(|f| f.var());
            conditions.push(condition);
            next_condition_by_operator.push(0);
        }

        let leaves: Vec<LeafEntry> = vec![LeafEntry {
            applicable_operators: Box::from([] as [OperatorIndex; 0]),
        }];
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

    fn push_leaf(&mut self, ops: Vec<OperatorIndex>) -> NodeId {
        if ops.is_empty() {
            return self.empty_leaf;
        }
        let id = NodeId::leaf(self.leaves.len());
        self.leaves.push(LeafEntry {
            applicable_operators: ops.into_boxed_slice(),
        });
        id
    }

    /// The subtree of the operators in `queue` (which it empties), branching
    /// on variables from `branch_var_id` on.
    ///
    /// A branch's default child (the operators without a condition on its
    /// variable) continues with the next variable, so default chains can pass
    /// through every variable of the task. They are built in a loop, each
    /// branch linked to its successor in the chain once that exists, and only
    /// value children recurse: the depth is bounded by the operators'
    /// precondition counts. The operators are moved from level to level, so
    /// no level keeps a drained buffer alive (with one buffer of nearly all
    /// operators per variable that was operators x variables memory, 8 GB on
    /// a grounded Sokoban task of 114k operators and 37k variables).
    fn construct(
        &mut self,
        mut branch_var_id: usize,
        queue: &mut VecDeque<OperatorIndex>,
    ) -> Result<NodeId, ConstructError> {
        let mut queue = std::mem::take(queue);
        // The first node of the chain and the branch whose default child is next.
        let mut root = None;
        let mut chain_end: Option<usize> = None;
        loop {
            let node = self.chain_node(&mut branch_var_id, &mut queue)?;
            match chain_end {
                Some(b) => self.branches[b].default_child = node,
                None => root = Some(node),
            }
            if !node.is_leaf() && !queue.is_empty() {
                chain_end = Some(node.index());
                branch_var_id += 1;
            } else {
                return Ok(root.expect("the chain has a node"));
            }
        }
    }

    /// The next node of a default chain: a leaf, or a branch on the first
    /// variable from `branch_var_id` some operator in `queue` has a condition
    /// on (`branch_var_id` is left at it). For a branch, `queue` is left
    /// holding its default operators and its default child is `empty_leaf`
    /// until the caller links the chain's next node.
    fn chain_node(
        &mut self,
        branch_var_id: &mut usize,
        queue: &mut VecDeque<OperatorIndex>,
    ) -> Result<NodeId, ConstructError> {
        if queue.is_empty() {
            return Ok(self.empty_leaf);
        }
        loop {
            let var = *branch_var_id;
            if var >= self.task.variables().len() {
                let ops: Vec<OperatorIndex> = std::mem::take(queue).into();
                return Ok(self.push_leaf(ops));
            }

            let num_children = self.task.variables()[var].domain_size();

            let mut operators_for_value: Vec<VecDeque<OperatorIndex>> =
                vec![VecDeque::new(); num_children];
            let mut default_operators: VecDeque<OperatorIndex> = VecDeque::new();
            let mut applicable_operators: Vec<OperatorIndex> = Vec::new();

            let mut all_ops_immediate = true;
            let mut var_interesting = false;

            while let Some(op_id) = queue.pop_front() {
                let op_idx = op_id;
                let condition_index = self.next_condition_by_operator[op_idx.index()];

                if condition_index >= self.conditions[op_idx.index()].len() {
                    var_interesting = true;
                    applicable_operators.push(op_id);
                } else {
                    all_ops_immediate = false;
                    let fact = &self.conditions[op_idx.index()][condition_index];
                    if fact.var() == var {
                        var_interesting = true;
                        let mut new_index = condition_index;
                        while new_index < self.conditions[op_idx.index()].len()
                            && self.conditions[op_idx.index()][new_index].var() == var
                        {
                            new_index += 1;
                        }
                        self.next_condition_by_operator[op_idx.index()] = new_index;
                        operators_for_value[fact.value()].push_back(op_id);
                    } else {
                        default_operators.push_back(op_id);
                    }
                }
            }
            // The drained buffer goes; the default operators continue.
            *queue = default_operators;

            if all_ops_immediate {
                return Ok(self.push_leaf(applicable_operators));
            } else if var_interesting {
                let mut value_children: Vec<NodeId> = Vec::with_capacity(operators_for_value.len());
                for mut ops in operators_for_value.into_iter() {
                    let child = self.construct(var + 1, &mut ops)?;
                    value_children.push(child);
                }
                let immediate = applicable_operators.into_boxed_slice();
                let var_id_u32: u32 = var
                    .try_into()
                    .expect("variable id overflows u32 — sas task has too many variables");
                return Ok(self.push_branch(BranchEntry {
                    var_id: VariableIndex::new(var_id_u32),
                    immediate_operators: immediate,
                    value_children: value_children.into_boxed_slice(),
                    default_child: self.empty_leaf,
                }));
            } else {
                *branch_var_id += 1;
            }
        }
    }
}
