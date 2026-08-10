//! Numeric conditions: the comparisons a task can test on its numeric state.
//!
//! A numeric condition is the semantic content of one [`ComparisonAxiom`]:
//! an arithmetic expression on the left, a comparison operator, and an
//! arithmetic expression on the right. The expressions are not stored as
//! syntax; they are *recovered* from the chain of [`AssignmentAxiom`]s that
//! define the derived numeric variables the comparison mentions, and kept in
//! a small arena DAG — a variable used by both sides is expanded once and
//! referenced twice.
//!
//! Conditions are a first-class part of a task: [`NumericConditions`] is
//! built once at task construction and reached through
//! [`AbstractNumericTask::numeric_conditions`]. Nothing else in the codebase
//! rebuilds the "propositional variable -> comparison axiom" mapping.
//!
//! # Evaluation
//!
//! Two orders, with different trade-offs:
//!
//! * **Bottom-up** ([`NumericCondition::evaluate`]) is the default: a
//!   post-order walk that computes children before parents. Successor
//!   generation, abstraction construction and flaw search all use it.
//! * **Lazy top-down** ([`LazyConditionEvaluator`]) is opt-in. It computes
//!   node values on demand and memoises them, so a heuristic that probes
//!   individual nodes of a large condition does not walk the sub-DAGs it
//!   never asks about. It is never the default.
//!
//! Both orders are generic over the value domain via [`ConditionDomain`], so
//! concrete states (`f64`) and abstract states ([`Interval`]) share one
//! implementation.

#[cfg(test)]
mod tests;

use crate::axioms::{AssignmentAxiom, CalOperator, ComparisonAxiom, ComparisonOperator};
use crate::numeric_task::{
    AbstractNumericTask, ExplicitFact, FactNamespace, NumericType, NumericVariable,
};
use crate::utils::interval::{EMPTY_INTERVAL, Interval};

/// The three values a propositional variable carrying a numeric condition's
/// truth value can take.
///
/// The discriminants *are* the SAS encoding — a condition variable's domain is
/// exactly these three values in this order — so [`Self::as_usize`] needs no
/// lookup table. This enum is the workspace's only definition of that
/// encoding; nothing else spells the literals out.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum ConditionValue {
    /// The comparison holds.
    True = 0,
    /// The comparison does not hold.
    False = 1,
    /// The comparison has not been derived from the numeric state yet.
    Unknown = 2,
}

impl ConditionValue {
    /// Domain size of a condition variable.
    pub const DOMAIN_SIZE: usize = 3;

    /// Domain size of a condition variable in a *concrete* packed state.
    ///
    /// A concrete state fixes every numeric variable, so every comparison has
    /// a verdict: only [`True`](Self::True) and [`False`](Self::False) occur
    /// there. [`Unknown`](Self::Unknown) is what an *interval* evaluates to
    /// and belongs to the abstract domain, which keeps all three values.
    pub const CONCRETE_DOMAIN_SIZE: usize = 2;

    /// The domain of a condition variable, in value order.
    pub const DOMAIN: [ConditionValue; Self::DOMAIN_SIZE] =
        [Self::True, Self::False, Self::Unknown];

    /// The SAS value this variant encodes.
    #[inline]
    pub const fn as_usize(self) -> usize {
        self as usize
    }

    /// The variant `value` encodes, or `None` when it is outside the domain.
    #[inline]
    pub const fn from_usize(value: usize) -> Option<Self> {
        match value {
            0 => Some(Self::True),
            1 => Some(Self::False),
            2 => Some(Self::Unknown),
            _ => None,
        }
    }
}

impl From<bool> for ConditionValue {
    #[inline]
    fn from(holds: bool) -> Self {
        if holds { Self::True } else { Self::False }
    }
}

/// A three-valued verdict, as produced by evaluating a condition over
/// intervals: `None` — both outcomes possible — is exactly [`Unknown`].
///
/// [`Unknown`]: ConditionValue::Unknown
impl From<Option<bool>> for ConditionValue {
    #[inline]
    fn from(verdict: Option<bool>) -> Self {
        match verdict {
            Some(holds) => Self::from(holds),
            None => Self::Unknown,
        }
    }
}

impl From<ConditionValue> for usize {
    #[inline]
    fn from(value: ConditionValue) -> Self {
        value.as_usize()
    }
}

/// Index of a [`NumericCondition`] inside [`NumericConditions`].
///
/// Identical to the id of the [`ComparisonAxiom`] it was derived from:
/// conditions are stored in comparison-axiom order, one per axiom.
pub type NumericConditionId = usize;

/// Index into a [`NumericCondition`]'s node arena.
pub type NodeId = usize;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl ArithOp {
    #[inline]
    pub fn apply(self, lhs: f64, rhs: f64) -> f64 {
        match self {
            ArithOp::Add => lhs + rhs,
            ArithOp::Sub => lhs - rhs,
            ArithOp::Mul => lhs * rhs,
            ArithOp::Div => lhs / rhs,
        }
    }

    #[inline]
    pub fn apply_interval(self, lhs: Interval, rhs: Interval) -> Interval {
        if lhs.is_empty() || rhs.is_empty() {
            return EMPTY_INTERVAL;
        }
        match self {
            ArithOp::Add => lhs + rhs,
            ArithOp::Sub => lhs - rhs,
            ArithOp::Mul => lhs * rhs,
            ArithOp::Div => lhs / rhs,
        }
    }
}

impl From<&CalOperator> for ArithOp {
    fn from(op: &CalOperator) -> Self {
        match op {
            CalOperator::Sum => ArithOp::Add,
            CalOperator::Difference => ArithOp::Sub,
            CalOperator::Product => ArithOp::Mul,
            CalOperator::Division => ArithOp::Div,
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
    pub fn apply(self, lhs: f64, rhs: f64) -> bool {
        match self {
            CompOp::Lt => lhs < rhs,
            CompOp::Le => lhs <= rhs,
            CompOp::Gt => lhs > rhs,
            CompOp::Ge => lhs >= rhs,
            CompOp::Eq => lhs == rhs,
            CompOp::Ne => lhs != rhs,
        }
    }

    /// Three-valued comparison of two intervals: `Some(b)` when every pair of
    /// concrete values agrees on `b`, `None` when both outcomes are possible.
    #[inline]
    pub fn apply_interval(self, lhs: Interval, rhs: Interval) -> Option<bool> {
        if lhs.is_empty() || rhs.is_empty() {
            return Some(false);
        }

        let (lmin, lmin_c) = lhs.min_bound();
        let (lmax, lmax_c) = lhs.max_bound();
        let (rmin, rmin_c) = rhs.min_bound();
        let (rmax, rmax_c) = rhs.max_bound();

        let max_lt_min = |amax: f64, amax_c: bool, bmin: f64, bmin_c: bool| -> bool {
            (amax < bmin) || (amax == bmin && (!amax_c || !bmin_c))
        };
        // "Every value in A is >= every value in B." When amin == bmax the
        // answer is yes in all four open/closed combinations: no x in A is
        // below any y in B.
        let min_ge_max = |amin: f64, bmax: f64| -> bool { amin >= bmax };
        let min_gt_max = |amin: f64, amin_c: bool, bmax: f64, bmax_c: bool| -> bool {
            (amin > bmax) || (amin == bmax && (!amin_c || !bmax_c))
        };
        let intervals_are_disjoint =
            || max_lt_min(lmax, lmax_c, rmin, rmin_c) || max_lt_min(rmax, rmax_c, lmin, lmin_c);

        match self {
            CompOp::Lt => {
                if max_lt_min(lmax, lmax_c, rmin, rmin_c) {
                    Some(true)
                } else if min_ge_max(lmin, rmax) {
                    Some(false)
                } else {
                    None
                }
            }
            CompOp::Le => {
                if lmax <= rmin {
                    Some(true)
                } else if min_gt_max(lmin, lmin_c, rmax, rmax_c) {
                    Some(false)
                } else {
                    None
                }
            }
            CompOp::Gt => CompOp::Lt.apply_interval(rhs, lhs),
            CompOp::Ge => CompOp::Le.apply_interval(rhs, lhs),
            CompOp::Eq => {
                if lhs.is_singleton() && rhs.is_singleton() && lmin == rmin {
                    Some(true)
                } else if intervals_are_disjoint() {
                    Some(false)
                } else {
                    None
                }
            }
            CompOp::Ne => {
                if lhs.is_singleton() && rhs.is_singleton() && lmin == rmin {
                    Some(false)
                } else if intervals_are_disjoint() {
                    Some(true)
                } else {
                    None
                }
            }
        }
    }
}

impl From<&ComparisonOperator> for CompOp {
    fn from(op: &ComparisonOperator) -> Self {
        match op {
            ComparisonOperator::LessThan => CompOp::Lt,
            ComparisonOperator::LessThanOrEqual => CompOp::Le,
            ComparisonOperator::Equal => CompOp::Eq,
            ComparisonOperator::GreaterThanOrEqual => CompOp::Ge,
            ComparisonOperator::GreaterThan => CompOp::Gt,
            ComparisonOperator::UnEqual => CompOp::Ne,
        }
    }
}

/// One node of a condition's expression DAG.
///
/// `Leaf` reads a numeric variable that no assignment axiom defines; `Arith`
/// recomputes the variable defined by `assignment_axiom_id`.
#[derive(Debug, Clone, PartialEq)]
pub enum ConditionNode {
    Leaf {
        numeric_var_id: usize,
    },
    Arith {
        result_numeric_var_id: usize,
        assignment_axiom_id: usize,
        op: ArithOp,
        left_numeric_var_id: usize,
        right_numeric_var_id: usize,
        left: NodeId,
        right: NodeId,
    },
}

impl ConditionNode {
    /// Numeric variable this node produces: the one it reads (`Leaf`) or the
    /// one the assignment axiom defines (`Arith`).
    #[inline]
    pub fn result_numeric_var_id(&self) -> usize {
        match self {
            ConditionNode::Leaf { numeric_var_id } => *numeric_var_id,
            ConditionNode::Arith {
                result_numeric_var_id,
                ..
            } => *result_numeric_var_id,
        }
    }
}

/// The value domain a condition can be evaluated over.
///
/// Implemented for `f64` (concrete states) and [`Interval`] (abstract
/// states); both evaluation orders are written once against this trait.
pub trait ConditionDomain: Copy {
    /// What comparing two values of this domain yields: `bool` for points,
    /// `Option<bool>` for intervals.
    type Verdict;

    /// Value of an `Arith` node given its evaluated children.
    ///
    /// `inputs` is the caller's per-numeric-variable table, passed so that
    /// domains which already know something about `result_numeric_var_id`
    /// can refine the computed value with it.
    fn combine(
        op: ArithOp,
        lhs: Self,
        rhs: Self,
        result_numeric_var_id: usize,
        inputs: &[Self],
    ) -> Self;

    fn compare(op: CompOp, lhs: Self, rhs: Self) -> Self::Verdict;
}

impl ConditionDomain for f64 {
    type Verdict = bool;

    /// A concrete state pins every derived variable exactly, so there is
    /// nothing in `inputs` that could sharpen the recomputed value.
    #[inline]
    fn combine(op: ArithOp, lhs: f64, rhs: f64, _result_numeric_var_id: usize, _: &[f64]) -> f64 {
        op.apply(lhs, rhs)
    }

    #[inline]
    fn compare(op: CompOp, lhs: f64, rhs: f64) -> bool {
        op.apply(lhs, rhs)
    }
}

impl ConditionDomain for Interval {
    type Verdict = Option<bool>;

    /// Intersect the recomputed value with whatever the caller already knows
    /// about the derived variable. An empty supplied interval means "nothing
    /// known", not "unsatisfiable".
    #[inline]
    fn combine(
        op: ArithOp,
        lhs: Interval,
        rhs: Interval,
        result_numeric_var_id: usize,
        inputs: &[Interval],
    ) -> Interval {
        let computed = op.apply_interval(lhs, rhs);
        let supplied = inputs[result_numeric_var_id];
        if supplied.is_empty() {
            computed
        } else {
            computed.intersection(&supplied)
        }
    }

    #[inline]
    fn compare(op: CompOp, lhs: Interval, rhs: Interval) -> Option<bool> {
        op.apply_interval(lhs, rhs)
    }
}

/// One comparison over the numeric state, with both sides expanded into an
/// expression DAG over the task's numeric variables.
///
/// Invariant, established by [`NumericConditions::build`]: a node's children
/// have strictly smaller [`NodeId`]s than the node itself, so the arena is
/// already in bottom-up order.
#[derive(Debug, Clone, PartialEq)]
pub struct NumericCondition {
    id: NumericConditionId,
    prop_var_id: usize,
    op: CompOp,
    left_numeric_var_id: usize,
    right_numeric_var_id: usize,
    nodes: Vec<ConditionNode>,
    left_root: NodeId,
    right_root: NodeId,
    regular_numeric_var_dependencies: Vec<usize>,
    required_numeric_len: usize,
}

impl NumericCondition {
    /// Id of this condition, equal to the id of the comparison axiom it was
    /// derived from.
    #[inline]
    pub fn id(&self) -> NumericConditionId {
        self.id
    }

    /// Propositional variable holding this condition's truth value, encoded as
    /// a [`ConditionValue`].
    #[inline]
    pub fn prop_var_id(&self) -> usize {
        self.prop_var_id
    }

    #[inline]
    pub fn op(&self) -> CompOp {
        self.op
    }

    /// Numeric variable the comparison axiom names on its left/right side,
    /// before expansion into the DAG.
    #[inline]
    pub fn left_numeric_var_id(&self) -> usize {
        self.left_numeric_var_id
    }

    #[inline]
    pub fn right_numeric_var_id(&self) -> usize {
        self.right_numeric_var_id
    }

    #[inline]
    pub fn nodes(&self) -> &[ConditionNode] {
        &self.nodes
    }

    #[inline]
    pub fn left_root(&self) -> NodeId {
        self.left_root
    }

    #[inline]
    pub fn right_root(&self) -> NodeId {
        self.right_root
    }

    #[inline]
    pub fn node(&self, node_id: NodeId) -> &ConditionNode {
        &self.nodes[node_id]
    }

    /// Sorted ids of the `Regular` numeric variables this condition reads.
    /// Derived variables are excluded: they are recomputed from these.
    #[inline]
    pub fn regular_numeric_var_dependencies(&self) -> &[usize] {
        &self.regular_numeric_var_dependencies
    }

    /// Smallest numeric-variable table length this condition can be evaluated
    /// against.
    #[inline]
    pub fn required_numeric_len(&self) -> usize {
        self.required_numeric_len
    }

    /// Bottom-up evaluation: a post-order walk that computes both operand
    /// DAGs children-first, then compares them. This is the default order.
    #[inline]
    pub fn evaluate<T: ConditionDomain>(&self, inputs: &[T]) -> T::Verdict {
        let lhs = self.evaluate_node(self.left_root, inputs);
        let rhs = self.evaluate_node(self.right_root, inputs);
        T::compare(self.op, lhs, rhs)
    }

    /// Bottom-up evaluation on a concrete numeric state.
    #[inline]
    pub fn evaluate_point(&self, values: &[f64]) -> bool {
        self.evaluate(values)
    }

    /// Bottom-up evaluation on an abstract numeric state: `None` when the
    /// intervals admit both outcomes.
    #[inline]
    pub fn evaluate_interval(&self, intervals: &[Interval]) -> Option<bool> {
        self.evaluate(intervals)
    }

    fn evaluate_node<T: ConditionDomain>(&self, node_id: NodeId, inputs: &[T]) -> T {
        match self.nodes[node_id] {
            ConditionNode::Leaf { numeric_var_id } => inputs[numeric_var_id],
            ConditionNode::Arith {
                op,
                left,
                right,
                result_numeric_var_id,
                ..
            } => {
                let lhs = self.evaluate_node(left, inputs);
                let rhs = self.evaluate_node(right, inputs);
                T::combine(op, lhs, rhs, result_numeric_var_id, inputs)
            }
        }
    }

    /// Like [`Self::evaluate_interval`], but also writes each derived
    /// variable's refined interval back into `intervals`, so the caller
    /// learns the bounds of the intermediate expressions.
    pub fn evaluate_interval_and_fill(&self, intervals: &mut [Interval]) -> Option<bool> {
        let lhs = self.evaluate_node_and_fill(self.left_root, intervals);
        let rhs = self.evaluate_node_and_fill(self.right_root, intervals);
        self.op.apply_interval(lhs, rhs)
    }

    fn evaluate_node_and_fill(&self, node_id: NodeId, intervals: &mut [Interval]) -> Interval {
        match self.nodes[node_id] {
            ConditionNode::Leaf { numeric_var_id } => intervals[numeric_var_id],
            ConditionNode::Arith {
                op,
                left,
                right,
                result_numeric_var_id,
                ..
            } => {
                let lhs = self.evaluate_node_and_fill(left, intervals);
                let rhs = self.evaluate_node_and_fill(right, intervals);
                let result = Interval::combine(op, lhs, rhs, result_numeric_var_id, intervals);
                intervals[result_numeric_var_id] = result;
                result
            }
        }
    }

    /// Interval of `lhs - rhs`.
    ///
    /// `lhs op rhs` is equivalent to `f op 0` for `f = lhs - rhs`, and a
    /// linear operator shifts `f` by a constant, so abstract-operator-variant
    /// filtering can reduce a joint constraint on source and target values to
    /// a one-dimensional check on `f`.
    pub fn lhs_minus_rhs_interval(&self, intervals: &[Interval]) -> Interval {
        let lhs = self.evaluate_node(self.left_root, intervals);
        let rhs = self.evaluate_node(self.right_root, intervals);
        ArithOp::Sub.apply_interval(lhs, rhs)
    }

    /// `true` iff *some* concrete assignment inside `intervals` satisfies the
    /// comparison — the optimistic predicate abstract-operator construction
    /// needs, since concrete values are recomputed per state during heuristic
    /// evaluation.
    #[inline]
    pub fn admits_true(&self, intervals: &[Interval]) -> bool {
        self.evaluate_interval(intervals) != Some(false)
    }

    /// Companion of [`Self::admits_true`]: `true` iff some concrete
    /// assignment inside `intervals` falsifies the comparison.
    #[inline]
    pub fn admits_false(&self, intervals: &[Interval]) -> bool {
        self.evaluate_interval(intervals) != Some(true)
    }

    /// Start a demand-driven evaluation of this condition; see
    /// [`LazyConditionEvaluator`].
    pub fn lazy_evaluator<T: ConditionDomain>(&self) -> LazyConditionEvaluator<'_, T> {
        LazyConditionEvaluator::new(self)
    }
}

/// Demand-driven evaluation with per-node memoisation.
///
/// Opt-in, for heuristics that ask about individual nodes of a condition: a
/// node is computed the first time it is requested and cached until
/// [`Self::reset`]. Successor generation and abstraction construction use the
/// bottom-up methods on [`NumericCondition`] instead — this is never the
/// default.
pub struct LazyConditionEvaluator<'a, T: ConditionDomain> {
    condition: &'a NumericCondition,
    memo: Vec<Option<T>>,
}

impl<'a, T: ConditionDomain> LazyConditionEvaluator<'a, T> {
    pub fn new(condition: &'a NumericCondition) -> Self {
        Self {
            condition,
            memo: vec![None; condition.nodes.len()],
        }
    }

    /// Drop all cached node values; call before evaluating over new inputs.
    pub fn reset(&mut self) {
        self.memo.fill(None);
    }

    pub fn condition(&self) -> &'a NumericCondition {
        self.condition
    }

    /// Value of one node, computing only the sub-DAG it depends on.
    pub fn node_value(&mut self, node_id: NodeId, inputs: &[T]) -> T {
        if let Some(value) = self.memo[node_id] {
            return value;
        }
        let value = match self.condition.nodes[node_id] {
            ConditionNode::Leaf { numeric_var_id } => inputs[numeric_var_id],
            ConditionNode::Arith {
                op,
                left,
                right,
                result_numeric_var_id,
                ..
            } => {
                let lhs = self.node_value(left, inputs);
                let rhs = self.node_value(right, inputs);
                T::combine(op, lhs, rhs, result_numeric_var_id, inputs)
            }
        };
        self.memo[node_id] = Some(value);
        value
    }

    pub fn evaluate(&mut self, inputs: &[T]) -> T::Verdict {
        let lhs = self.node_value(self.condition.left_root, inputs);
        let rhs = self.node_value(self.condition.right_root, inputs);
        T::compare(self.condition.op, lhs, rhs)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NumericConditionError {
    UnknownPropositionalVar {
        comparison_axiom_id: usize,
        provided: usize,
        num_propositional_vars: usize,
    },
    DuplicatePropositionalVar {
        prop_var_id: usize,
        first_comparison_axiom_id: usize,
        second_comparison_axiom_id: usize,
    },
    UnknownNumericVar {
        provided: usize,
        num_numeric_vars: usize,
    },
    InvalidAssignmentTarget {
        assignment_axiom_id: usize,
        provided: usize,
        num_numeric_vars: usize,
    },
    DuplicateAssignmentTarget {
        numeric_var_id: usize,
        first_assignment_axiom_id: usize,
        second_assignment_axiom_id: usize,
    },
    CycleDetected {
        numeric_var_id: usize,
    },
}

impl std::fmt::Display for NumericConditionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownPropositionalVar {
                comparison_axiom_id,
                provided,
                num_propositional_vars,
            } => write!(
                f,
                "comparison axiom {comparison_axiom_id} writes propositional variable {provided}, \
                 but the task has only {num_propositional_vars}"
            ),
            Self::DuplicatePropositionalVar {
                prop_var_id,
                first_comparison_axiom_id,
                second_comparison_axiom_id,
            } => write!(
                f,
                "comparison axioms {first_comparison_axiom_id} and {second_comparison_axiom_id} \
                 both write propositional variable {prop_var_id}"
            ),
            Self::UnknownNumericVar {
                provided,
                num_numeric_vars,
            } => write!(
                f,
                "numeric variable {provided} referenced, but the task has only {num_numeric_vars}"
            ),
            Self::InvalidAssignmentTarget {
                assignment_axiom_id,
                provided,
                num_numeric_vars,
            } => write!(
                f,
                "assignment axiom {assignment_axiom_id} writes numeric variable {provided}, \
                 but the task has only {num_numeric_vars}"
            ),
            Self::DuplicateAssignmentTarget {
                numeric_var_id,
                first_assignment_axiom_id,
                second_assignment_axiom_id,
            } => write!(
                f,
                "assignment axioms {first_assignment_axiom_id} and {second_assignment_axiom_id} \
                 both write numeric variable {numeric_var_id}"
            ),
            Self::CycleDetected { numeric_var_id } => write!(
                f,
                "assignment axioms define numeric variable {numeric_var_id} in terms of itself"
            ),
        }
    }
}

impl std::error::Error for NumericConditionError {}

/// All numeric conditions of a task, indexed both by condition id and by the
/// propositional variable that carries a condition's truth value.
///
/// Built once per task; the single source of truth for "is this variable a
/// comparison result?".
#[derive(Debug, Clone, PartialEq, Default)]
pub struct NumericConditions {
    conditions: Vec<NumericCondition>,
    by_prop_var: Vec<Option<NumericConditionId>>,
}

impl NumericConditions {
    pub fn from_task(task: &dyn AbstractNumericTask) -> Result<Self, NumericConditionError> {
        Self::build(
            task.variables().len(),
            task.numeric_variables(),
            task.comparison_axioms(),
            task.assignment_axioms(),
        )
    }

    pub fn build(
        num_propositional_vars: usize,
        numeric_variables: &[NumericVariable],
        comparison_axioms: &[ComparisonAxiom],
        assignment_axioms: &[AssignmentAxiom],
    ) -> Result<Self, NumericConditionError> {
        let num_numeric_vars = numeric_variables.len();
        let definitions = assignment_axiom_by_target(num_numeric_vars, assignment_axioms)?;

        let mut conditions = Vec::with_capacity(comparison_axioms.len());
        let mut by_prop_var = vec![None; num_propositional_vars];
        let mut builder = DagBuilder::new(num_numeric_vars);

        for (id, axiom) in comparison_axioms.iter().enumerate() {
            let prop_var_id = axiom.get_affected_var_id();
            let slot = by_prop_var.get_mut(prop_var_id).ok_or(
                NumericConditionError::UnknownPropositionalVar {
                    comparison_axiom_id: id,
                    provided: prop_var_id,
                    num_propositional_vars,
                },
            )?;
            if let Some(first_comparison_axiom_id) = slot.replace(id) {
                return Err(NumericConditionError::DuplicatePropositionalVar {
                    prop_var_id,
                    first_comparison_axiom_id,
                    second_comparison_axiom_id: id,
                });
            }

            let left_numeric_var_id = axiom.get_left_var_id();
            let right_numeric_var_id = axiom.get_right_var_id();
            builder.restart();
            let left_root = builder.expand(left_numeric_var_id, &definitions, assignment_axioms)?;
            let right_root =
                builder.expand(right_numeric_var_id, &definitions, assignment_axioms)?;
            let nodes = builder.take_nodes();

            conditions.push(NumericCondition {
                id,
                prop_var_id,
                op: CompOp::from(axiom.get_operator()),
                left_numeric_var_id,
                right_numeric_var_id,
                regular_numeric_var_dependencies: regular_leaf_dependencies(
                    &nodes,
                    numeric_variables,
                ),
                required_numeric_len: required_numeric_len(&nodes),
                nodes,
                left_root,
                right_root,
            });
        }

        Ok(Self {
            conditions,
            by_prop_var,
        })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.conditions.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.conditions.is_empty()
    }

    #[inline]
    pub fn all(&self) -> &[NumericCondition] {
        &self.conditions
    }

    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, NumericCondition> {
        self.conditions.iter()
    }

    #[inline]
    pub fn get(&self, id: NumericConditionId) -> Option<&NumericCondition> {
        self.conditions.get(id)
    }

    /// Is `prop_var_id` the truth value of a numeric condition rather than an
    /// ordinary propositional variable?
    #[inline]
    pub fn is_condition_var(&self, prop_var_id: usize) -> bool {
        matches!(self.by_prop_var.get(prop_var_id), Some(Some(_)))
    }

    /// The namespace variable `prop_var_id` belongs to.
    #[inline]
    pub fn namespace_of(&self, prop_var_id: usize) -> FactNamespace {
        if self.is_condition_var(prop_var_id) {
            FactNamespace::Condition
        } else {
            FactNamespace::Propositional
        }
    }

    /// Build a correctly tagged fact. This is the authoritative constructor
    /// for callers that hold a variable id and cannot themselves know which
    /// namespace it names.
    #[inline]
    pub fn fact(&self, prop_var_id: usize, value: usize) -> ExplicitFact {
        ExplicitFact::in_namespace(self.namespace_of(prop_var_id), prop_var_id, value)
    }

    #[inline]
    pub fn id_for_var(&self, prop_var_id: usize) -> Option<NumericConditionId> {
        *self.by_prop_var.get(prop_var_id)?
    }

    #[inline]
    pub fn for_var(&self, prop_var_id: usize) -> Option<&NumericCondition> {
        self.conditions.get(self.id_for_var(prop_var_id)?)
    }

    /// Propositional variables carrying condition truth values, ascending.
    pub fn condition_var_ids(&self) -> impl Iterator<Item = usize> + '_ {
        self.by_prop_var
            .iter()
            .enumerate()
            .filter_map(|(prop_var_id, condition)| condition.map(|_| prop_var_id))
    }

    /// Is `precondition` unsatisfiable for every concrete numeric assignment
    /// inside `numeric_intervals`?
    ///
    /// Optimistic, matching the rest of abstract-operator construction: a
    /// `TRUE` precondition is contradicted only when no value in the
    /// intervals makes the comparison true, and symmetrically for `FALSE`.
    /// Facts on ordinary propositional variables are never contradicted here.
    pub fn precondition_is_contradicted(
        &self,
        precondition: &ExplicitFact,
        numeric_intervals: &[Interval],
    ) -> bool {
        let Some(condition) = self.for_var(precondition.var()) else {
            return false;
        };
        match ConditionValue::from_usize(precondition.value()) {
            Some(ConditionValue::True) => !condition.admits_true(numeric_intervals),
            Some(ConditionValue::False) => !condition.admits_false(numeric_intervals),
            // `Unknown` asserts nothing about the numeric state, so no
            // assignment can contradict it. Abstracted values land outside the
            // concrete domain and are likewise not contradicted here.
            Some(ConditionValue::Unknown) | None => false,
        }
    }
}

/// Maps each numeric variable to the assignment axiom defining it, if any.
fn assignment_axiom_by_target(
    num_numeric_vars: usize,
    assignment_axioms: &[AssignmentAxiom],
) -> Result<Vec<Option<usize>>, NumericConditionError> {
    let mut by_target = vec![None; num_numeric_vars];
    for (assignment_axiom_id, axiom) in assignment_axioms.iter().enumerate() {
        let target = axiom.get_affected_var_id();
        let slot =
            by_target
                .get_mut(target)
                .ok_or(NumericConditionError::InvalidAssignmentTarget {
                    assignment_axiom_id,
                    provided: target,
                    num_numeric_vars,
                })?;
        if let Some(first_assignment_axiom_id) = slot.replace(assignment_axiom_id) {
            return Err(NumericConditionError::DuplicateAssignmentTarget {
                numeric_var_id: target,
                first_assignment_axiom_id,
                second_assignment_axiom_id: assignment_axiom_id,
            });
        }
    }
    Ok(by_target)
}

/// Expands numeric variables into a node arena, memoising per variable so
/// shared sub-expressions become shared nodes, and rejecting cyclic
/// definitions.
struct DagBuilder {
    nodes: Vec<ConditionNode>,
    memo: Vec<Option<NodeId>>,
    on_stack: Vec<bool>,
    memoised: Vec<usize>,
}

impl DagBuilder {
    fn new(num_numeric_vars: usize) -> Self {
        Self {
            nodes: Vec::new(),
            memo: vec![None; num_numeric_vars],
            on_stack: vec![false; num_numeric_vars],
            memoised: Vec::new(),
        }
    }

    /// Start a fresh arena, clearing only the memo slots actually used.
    fn restart(&mut self) {
        self.nodes.clear();
        for numeric_var_id in self.memoised.drain(..) {
            self.memo[numeric_var_id] = None;
        }
    }

    fn take_nodes(&mut self) -> Vec<ConditionNode> {
        std::mem::take(&mut self.nodes)
    }

    fn expand(
        &mut self,
        numeric_var_id: usize,
        definitions: &[Option<usize>],
        assignment_axioms: &[AssignmentAxiom],
    ) -> Result<NodeId, NumericConditionError> {
        let definition =
            *definitions
                .get(numeric_var_id)
                .ok_or(NumericConditionError::UnknownNumericVar {
                    provided: numeric_var_id,
                    num_numeric_vars: definitions.len(),
                })?;

        if let Some(node_id) = self.memo[numeric_var_id] {
            return Ok(node_id);
        }
        if self.on_stack[numeric_var_id] {
            return Err(NumericConditionError::CycleDetected { numeric_var_id });
        }
        self.on_stack[numeric_var_id] = true;

        let node = match definition {
            Some(assignment_axiom_id) => {
                let axiom = &assignment_axioms[assignment_axiom_id];
                let left_numeric_var_id = axiom.get_left_var_id();
                let right_numeric_var_id = axiom.get_right_var_id();
                let left = self.expand(left_numeric_var_id, definitions, assignment_axioms)?;
                let right = self.expand(right_numeric_var_id, definitions, assignment_axioms)?;
                ConditionNode::Arith {
                    result_numeric_var_id: numeric_var_id,
                    assignment_axiom_id,
                    op: ArithOp::from(axiom.get_operator()),
                    left_numeric_var_id,
                    right_numeric_var_id,
                    left,
                    right,
                }
            }
            None => ConditionNode::Leaf { numeric_var_id },
        };

        // Children are pushed by the recursive calls above, so a node's
        // `NodeId` always exceeds its children's.
        let node_id = self.nodes.len();
        self.nodes.push(node);

        self.on_stack[numeric_var_id] = false;
        self.memo[numeric_var_id] = Some(node_id);
        self.memoised.push(numeric_var_id);
        Ok(node_id)
    }
}

fn regular_leaf_dependencies(
    nodes: &[ConditionNode],
    numeric_variables: &[NumericVariable],
) -> Vec<usize> {
    let mut dependencies: Vec<usize> = nodes
        .iter()
        .filter_map(|node| match node {
            ConditionNode::Leaf { numeric_var_id } => numeric_variables
                .get(*numeric_var_id)
                .filter(|variable| variable.get_type() == &NumericType::Regular)
                .map(|_| *numeric_var_id),
            ConditionNode::Arith { .. } => None,
        })
        .collect();
    dependencies.sort_unstable();
    dependencies.dedup();
    dependencies
}

/// Every numeric variable the arena touches is either a leaf's source or an
/// `Arith` node's result, so the largest of those ids bounds the table.
fn required_numeric_len(nodes: &[ConditionNode]) -> usize {
    nodes
        .iter()
        .map(ConditionNode::result_numeric_var_id)
        .max()
        .map_or(0, |max_id| max_id + 1)
}
