//! The pieces of a numeric condition: truth values over a value domain, the
//! arithmetic and comparison operators, the DAG nodes, and the domains a
//! condition evaluates over (concrete values and intervals).

use super::*;

/// The two values a propositional variable carrying a numeric condition's
/// truth value can take.
///
/// The discriminants *are* the SAS encoding — a condition variable's domain is
/// exactly these two values in this order — so [`Self::as_usize`] needs no
/// lookup table. This enum is the workspace's only definition of that
/// encoding; nothing else spells the literals out.
///
/// A comparison is decided in every state a task can reach: the closure writes
/// a verdict for every condition variable before anything reads one. Whether an
/// *abstraction* can decide the comparison over a whole box of numeric values is
/// a property of that reasoning step, answered by
/// [`NumericCondition::evaluate_interval`]'s `Option<bool>`, and not a third
/// value a state can hold.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum ConditionValue {
    /// The comparison holds.
    True = 0,
    /// The comparison does not hold.
    False = 1,
}

impl ConditionValue {
    /// Domain size of a condition variable, concrete and abstract alike.
    pub const DOMAIN_SIZE: usize = 2;

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
            _ => None,
        }
    }

    /// The SAS value this variant encodes.
    #[inline]
    pub const fn as_u32(self) -> u32 {
        self as u32
    }

    /// The variant `value` encodes, or `None` when it is outside the domain.
    #[inline]
    pub const fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::True),
            1 => Some(Self::False),
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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl ArithOp {
    /// `lhs op rhs` on the canonical lattice, so that comparing it is exact.
    #[inline]
    pub fn apply(self, lhs: NumericValue, rhs: NumericValue) -> NumericValue {
        match self {
            ArithOp::Add => lhs + rhs,
            ArithOp::Sub => lhs - rhs,
            ArithOp::Mul => lhs * rhs,
            ArithOp::Div => lhs / rhs,
        }
    }

    /// [`Self::apply`] over intervals. Canonicalization is monotone, so the
    /// canonical result of every point pair stays inside the canonical image.
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
    pub fn apply(self, lhs: NumericValue, rhs: NumericValue) -> bool {
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

        let max_lt_min =
            |amax: NumericValue, amax_c: bool, bmin: NumericValue, bmin_c: bool| -> bool {
                (amax < bmin) || (amax == bmin && (!amax_c || !bmin_c))
            };
        // "Every value in A is >= every value in B." When amin == bmax the
        // answer is yes in all four open/closed combinations: no x in A is
        // below any y in B.
        let min_ge_max = |amin: NumericValue, bmax: NumericValue| -> bool { amin >= bmax };
        let min_gt_max =
            |amin: NumericValue, amin_c: bool, bmax: NumericValue, bmax_c: bool| -> bool {
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
    Linear {
        result_numeric_var_id: VariableIndex,
        terms: Vec<(VariableIndex, NumericValue)>,
    },
    Leaf {
        numeric_var_id: VariableIndex,
    },
    Arith {
        result_numeric_var_id: VariableIndex,
        assignment_axiom_id: AxiomIndex,
        op: ArithOp,
        left_numeric_var_id: VariableIndex,
        right_numeric_var_id: VariableIndex,
        left: NodeId,
        right: NodeId,
    },
}

impl ConditionNode {
    /// Numeric variable this node produces: the one it reads (`Leaf`) or the
    /// one the assignment axiom defines (`Arith`).
    #[inline]
    pub fn result_numeric_var_id(&self) -> VariableIndex {
        match self {
            ConditionNode::Leaf { numeric_var_id } => *numeric_var_id,
            ConditionNode::Linear {
                result_numeric_var_id,
                ..
            } => *result_numeric_var_id,
            ConditionNode::Arith {
                result_numeric_var_id,
                ..
            } => *result_numeric_var_id,
        }
    }
}

/// The value domain a condition can be evaluated over.
///
/// Implemented for [`NumericValue`] (concrete states) and [`Interval`] (abstract
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
        result_numeric_var_id: VariableIndex,
        inputs: &[Self],
    ) -> Self;

    fn linear(
        terms: &[(VariableIndex, NumericValue)],
        result: VariableIndex,
        inputs: &[Self],
    ) -> Self;

    fn compare(op: CompOp, lhs: Self, rhs: Self) -> Self::Verdict;
}

impl ConditionDomain for NumericValue {
    type Verdict = bool;

    /// A concrete state pins every derived variable exactly, so there is
    /// nothing in `inputs` that could sharpen the recomputed value.
    #[inline]
    fn combine(
        op: ArithOp,
        lhs: NumericValue,
        rhs: NumericValue,
        _result_numeric_var_id: VariableIndex,
        _: &[NumericValue],
    ) -> NumericValue {
        op.apply(lhs, rhs)
    }

    fn linear(terms: &[(VariableIndex, NumericValue)], _: VariableIndex, inputs: &[Self]) -> Self {
        NumericValue::affine(
            terms
                .iter()
                .map(|(var, coefficient)| (coefficient.value(), inputs[var.index()])),
            0.0,
        )
    }

    #[inline]
    fn compare(op: CompOp, lhs: NumericValue, rhs: NumericValue) -> bool {
        op.apply(lhs, rhs)
    }
}

impl ConditionDomain for Interval {
    type Verdict = Option<bool>;

    /// The recomputed value, refined by what the caller already knows about
    /// the derived variable.
    #[inline]
    fn combine(
        op: ArithOp,
        lhs: Interval,
        rhs: Interval,
        result_numeric_var_id: VariableIndex,
        inputs: &[Interval],
    ) -> Interval {
        refine_supplied(
            op.apply_interval(lhs, rhs),
            inputs[result_numeric_var_id.index()],
        )
    }

    fn linear(
        terms: &[(VariableIndex, NumericValue)],
        result: VariableIndex,
        inputs: &[Self],
    ) -> Self {
        refine_supplied(
            Interval::affine(
                terms
                    .iter()
                    .map(|(var, coefficient)| (coefficient.value(), inputs[var.index()])),
                0.0,
            ),
            inputs[result.index()],
        )
    }

    #[inline]
    fn compare(op: CompOp, lhs: Interval, rhs: Interval) -> Option<bool> {
        op.apply_interval(lhs, rhs)
    }
}

/// `computed` intersected with what the caller already knows about the
/// derived variable; an empty `supplied` interval means "nothing known".
fn refine_supplied(computed: Interval, supplied: Interval) -> Interval {
    if supplied.is_empty() {
        computed
    } else {
        computed.intersection(&supplied)
    }
}
