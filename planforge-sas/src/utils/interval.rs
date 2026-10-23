//! Closed/open real intervals with interval arithmetic.
//!
//! Used by every abstraction that reasons about sets of numeric values:
//! domain-abstraction partitions, CEGAR flaw search and the numeric
//! condition evaluator in [`crate::numeric_conditions`].

use crate::numeric_task::{
    AssignmentOperation, INF_VALUE, NEG_INF_VALUE, NumericValue, ONE_VALUE, ZERO_VALUE,
};

#[cfg(test)]
mod tests;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Interval {
    pub lower: NumericValue,
    pub upper: NumericValue,
    pub lower_closed: bool,
    pub upper_closed: bool,
}

pub const EMPTY_INTERVAL: Interval = Interval {
    lower: ONE_VALUE,
    upper: ZERO_VALUE,
    lower_closed: false,
    upper_closed: false,
};
pub const UNBOUNDED_INTERVAL: Interval = Interval {
    lower: NEG_INF_VALUE,
    upper: INF_VALUE,
    lower_closed: false,
    upper_closed: false,
};

impl Interval {
    #[inline]
    pub fn new(
        lower: NumericValue,
        upper: NumericValue,
        lower_closed: bool,
        upper_closed: bool,
    ) -> Self {
        Self {
            lower,
            upper,
            lower_closed,
            upper_closed,
        }
        .normalized()
    }

    #[inline]
    pub fn closed(lower: NumericValue, upper: NumericValue) -> Self {
        Self::new(lower, upper, true, true)
    }

    #[inline]
    pub fn open(lower: NumericValue, upper: NumericValue) -> Self {
        Self::new(lower, upper, false, false)
    }

    #[inline]
    pub fn singleton(value: NumericValue) -> Self {
        Self {
            lower: value,
            upper: value,
            lower_closed: true,
            upper_closed: true,
        }
    }

    #[inline]
    pub fn unbounded() -> Self {
        UNBOUNDED_INTERVAL
    }

    /// Image of an affine expression, rounding each resulting endpoint once.
    pub fn affine<I>(terms: I, constant: f64) -> Self
    where
        I: IntoIterator<Item = (f64, Self)>,
        I::IntoIter: Clone,
    {
        let terms = terms
            .into_iter()
            .filter(|(coefficient, _)| *coefficient != 0.0);
        let (mut lower_closed, mut upper_closed) = (true, true);
        for (coefficient, interval) in terms.clone() {
            assert!(!coefficient.is_nan(), "NaN affine coefficient");
            if interval.is_empty() {
                return EMPTY_INTERVAL;
            }
            let (lo_closed, hi_closed) = if coefficient > 0.0 {
                (interval.lower_closed, interval.upper_closed)
            } else {
                (interval.upper_closed, interval.lower_closed)
            };
            lower_closed &= lo_closed;
            upper_closed &= hi_closed;
        }
        let lower = NumericValue::affine(
            terms.clone().map(|(coefficient, interval)| {
                (
                    coefficient,
                    if coefficient > 0.0 {
                        interval.lower
                    } else {
                        interval.upper
                    },
                )
            }),
            constant,
        );
        let upper = NumericValue::affine(
            terms.map(|(coefficient, interval)| {
                (
                    coefficient,
                    if coefficient > 0.0 {
                        interval.upper
                    } else {
                        interval.lower
                    },
                )
            }),
            constant,
        );
        Self::new(lower, upper, lower_closed, upper_closed)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        if self.lower > self.upper {
            return true;
        }
        if self.lower == self.upper && !(self.lower_closed && self.upper_closed) {
            return true;
        }
        false
    }

    #[inline]
    pub fn is_constant(&self, constant: NumericValue) -> bool {
        self.lower_closed && self.upper_closed && self.lower == constant && self.upper == constant
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.is_constant(ZERO_VALUE)
    }

    #[inline]
    pub fn any_bound_is_zero(&self) -> bool {
        self.lower == ZERO_VALUE || self.upper == ZERO_VALUE
    }

    #[inline]
    pub fn contains(&self, value: NumericValue) -> bool {
        if self.is_empty() {
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
    pub fn intersects(&self, value: &Interval) -> bool {
        if value.is_empty() || self.is_empty() {
            return false;
        }

        // `value` is at right of `self`.
        if value.lower > self.upper
            || (value.lower == self.upper && (!value.lower_closed || !self.upper_closed))
        {
            return false;
        }

        // `value` is at left of `self`.
        if value.upper < self.lower
            || (value.upper == self.lower && (!value.upper_closed || !self.lower_closed))
        {
            return false;
        }

        true
    }

    #[inline]
    pub fn intersection(&self, other: &Interval) -> Interval {
        if self.is_empty() || other.is_empty() {
            return EMPTY_INTERVAL;
        }
        let (lower, lower_closed) = if self.lower > other.lower {
            (self.lower, self.lower_closed)
        } else if self.lower < other.lower {
            (other.lower, other.lower_closed)
        } else {
            (self.lower, self.lower_closed && other.lower_closed)
        };
        let (upper, upper_closed) = if self.upper < other.upper {
            (self.upper, self.upper_closed)
        } else if self.upper > other.upper {
            (other.upper, other.upper_closed)
        } else {
            (self.upper, self.upper_closed && other.upper_closed)
        };
        Interval::new(lower, upper, lower_closed, upper_closed)
    }

    /// True when `self`'s lower endpoint starts strictly before `other`'s.
    ///
    /// At equal values the closed endpoint starts first: `[a, ..]` begins
    /// before `(a, ..]`. A NaN bound compares below nothing, so the answer is
    /// `false`.
    #[inline]
    pub fn lower_is_lower(&self, other: &Self) -> bool {
        self.lower < other.lower
            || (self.lower == other.lower && self.lower_closed && !other.lower_closed)
    }

    /// True when `self`'s lower endpoint starts no later than `other`'s.
    ///
    /// At equal values the only way to start later is to exclude an endpoint
    /// the other includes, so `(a, ..]` is the sole case that fails.
    #[inline]
    pub fn lower_is_lower_or_equal(&self, other: &Self) -> bool {
        self.lower < other.lower
            || (self.lower == other.lower && (self.lower_closed || !other.lower_closed))
    }

    /// True when `self`'s upper endpoint reaches strictly beyond `other`'s.
    ///
    /// At equal values the closed endpoint reaches further: `[.., b]` ends
    /// after `[.., b)`. A NaN bound compares above nothing, so the answer is
    /// `false`.
    #[inline]
    pub fn upper_is_higher(&self, other: &Self) -> bool {
        self.upper > other.upper
            || (self.upper == other.upper && self.upper_closed && !other.upper_closed)
    }

    /// True when `self`'s upper endpoint reaches at least as far as `other`'s.
    ///
    /// At equal values the only way to fall short is to exclude an endpoint the
    /// other includes, so `[.., b)` against `[.., b]` is the sole failing case.
    #[inline]
    pub fn upper_is_higher_or_equal(&self, other: &Self) -> bool {
        self.upper > other.upper
            || (self.upper == other.upper && (self.upper_closed || !other.upper_closed))
    }

    #[inline]
    pub fn can_split_at(&self, value: NumericValue, include_in_lower: bool) -> bool {
        if self.is_empty() || value.is_infinite() {
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
        if self.lower == NEG_INF_VALUE {
            self.lower_closed = false;
        }
        if self.upper == INF_VALUE {
            self.upper_closed = false;
        }

        // TODO: Does not work at the moment because it is used in can_split(). Fix that in future releases cause assertions are our friend
        // debug_assert!(!self.is_empty());

        self
    }

    #[inline]
    pub(crate) fn min_bound(&self) -> (NumericValue, bool) {
        (self.lower, self.lower_closed)
    }

    #[inline]
    pub(crate) fn max_bound(&self) -> (NumericValue, bool) {
        (self.upper, self.upper_closed)
    }

    #[inline]
    pub fn is_singleton(&self) -> bool {
        self.lower == self.upper && self.lower_closed && self.upper_closed
    }

    #[inline]
    pub(crate) fn contains_zero(&self) -> bool {
        self.contains(ZERO_VALUE)
    }

    pub fn apply_op(&mut self, op: &AssignmentOperation, operand: &Interval) {
        match op {
            // Unknown previous value.
            AssignmentOperation::Assign => *self = *operand,
            AssignmentOperation::Plus => *self = *self + *operand,
            AssignmentOperation::Minus => *self = *self - *operand,
            AssignmentOperation::Times => *self = *self * *operand,
            AssignmentOperation::Divide => *self = *self / *operand,
        };
    }

    pub fn apply_reverse_op(&mut self, op: &AssignmentOperation, operand: &Interval) {
        match op {
            // Unknown previous value.
            AssignmentOperation::Assign => *self = UNBOUNDED_INTERVAL,
            AssignmentOperation::Plus => *self = *self - *operand,
            AssignmentOperation::Minus => *self = *self + *operand,
            AssignmentOperation::Times => {
                if operand.contains_zero() {
                    // Unknown previous value.
                    *self = UNBOUNDED_INTERVAL
                } else {
                    *self = *self / *operand
                }
            }
            AssignmentOperation::Divide => *self = *self * *operand,
        };
    }
}

impl std::ops::Add for Interval {
    type Output = Interval;

    #[inline]
    fn add(self, rhs: Interval) -> Interval {
        debug_assert!(!self.is_empty() && !rhs.is_empty());

        Interval {
            lower: self.lower + rhs.lower,
            upper: self.upper + rhs.upper,
            lower_closed: self.lower_closed && rhs.lower_closed,
            upper_closed: self.upper_closed && rhs.upper_closed,
        }
        .normalized()
    }
}

impl std::ops::Sub for Interval {
    type Output = Interval;

    #[inline]
    fn sub(self, rhs: Interval) -> Interval {
        debug_assert!(!self.is_empty() && !rhs.is_empty());

        Interval {
            lower: self.lower - rhs.upper,
            upper: self.upper - rhs.lower,
            lower_closed: self.lower_closed && rhs.upper_closed,
            upper_closed: self.upper_closed && rhs.lower_closed,
        }
        .normalized()
    }
}

impl std::ops::Mul for Interval {
    type Output = Interval;

    #[inline]
    fn mul(self, rhs: Interval) -> Interval {
        debug_assert!(!self.is_empty() && !rhs.is_empty());

        if self.is_zero() || rhs.is_zero() {
            return Interval::singleton(ZERO_VALUE);
        }

        self.endpoint_image(
            rhs,
            extended_product,
            self.contains_zero() || rhs.contains_zero(),
        )
    }
}

impl std::ops::Div for Interval {
    type Output = Interval;

    #[inline]
    fn div(self, rhs: Interval) -> Interval {
        debug_assert!(!self.is_empty() && !rhs.is_empty());
        if rhs.contains_zero() {
            return UNBOUNDED_INTERVAL;
        }

        self.endpoint_image(rhs, extended_quotient, self.contains_zero())
    }
}

impl Interval {
    fn endpoint_image(
        self,
        rhs: Self,
        operation: impl Fn(NumericValue, NumericValue) -> NumericValue,
        zero_attained: bool,
    ) -> Self {
        let left = [
            (self.lower, self.lower_closed),
            (self.upper, self.upper_closed),
        ];
        let right = [(rhs.lower, rhs.lower_closed), (rhs.upper, rhs.upper_closed)];
        let (mut lower, mut upper) = (INF_VALUE, NEG_INF_VALUE);
        let (mut lower_closed, mut upper_closed) = (false, false);
        for (left_value, left_closed) in left {
            for (right_value, right_closed) in right {
                let value = operation(left_value, right_value);
                // A finite attained corner remains attained if rounding makes
                // its nonzero real image zero.
                let attained = value.is_finite()
                    && ((left_closed && right_closed) || (value == ZERO_VALUE && zero_attained));
                if value < lower {
                    lower = value;
                    lower_closed = attained;
                } else if value == lower {
                    lower_closed |= attained;
                }
                if value > upper {
                    upper = value;
                    upper_closed = attained;
                } else if value == upper {
                    upper_closed |= attained;
                }
            }
        }
        Self::new(lower, upper, lower_closed, upper_closed)
    }
}

#[inline]
fn extended_quotient(left: NumericValue, right: NumericValue) -> NumericValue {
    // Indeterminate corners describe limits, not point computations. The
    // other corners already bound every finite quotient in these rectangles.
    if (left.is_infinite() && right.is_infinite()) || (left == ZERO_VALUE && right == ZERO_VALUE) {
        ZERO_VALUE
    } else {
        left / right
    }
}

#[inline]
fn extended_product(left: NumericValue, right: NumericValue) -> NumericValue {
    if (left == ZERO_VALUE && right.is_infinite()) || (right == ZERO_VALUE && left.is_infinite()) {
        ZERO_VALUE
    } else {
        left * right
    }
}
