//! Closed/open real intervals with interval arithmetic.
//!
//! Used by every abstraction that reasons about sets of numeric values:
//! domain-abstraction partitions, CEGAR flaw search and the numeric
//! condition evaluator in [`crate::numeric_conditions`].

use crate::numeric_task::AssignmentOperation;
use crate::utils::float_tolerance;

#[cfg(test)]
mod tests;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Interval {
    pub lower: f64,
    pub upper: f64,
    pub lower_closed: bool,
    pub upper_closed: bool,
}

pub const EMPTY_INTERVAL: Interval = Interval {
    lower: 1.0,
    upper: 0.0,
    lower_closed: false,
    upper_closed: false,
};
pub const UNBOUNDED_INTERVAL: Interval = Interval {
    lower: f64::NEG_INFINITY,
    upper: f64::INFINITY,
    lower_closed: false,
    upper_closed: false,
};

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
        UNBOUNDED_INTERVAL
    }

    /// Align interval boundaries with the numeric state registry's canonical
    /// lattice. Call this after state-transition arithmetic, not on queries.
    #[inline]
    pub fn canonicalized(self) -> Self {
        Self::new(
            float_tolerance::canonicalize(self.lower),
            float_tolerance::canonicalize(self.upper),
            self.lower_closed,
            self.upper_closed,
        )
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
    pub fn is_constant(&self, constant: f64) -> bool {
        self.lower_closed && self.upper_closed && self.lower == constant && self.upper == constant
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.is_constant(0.0)
    }

    #[inline]
    pub fn any_bound_is_zero(&self) -> bool {
        self.lower == 0.0 || self.upper == 0.0
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

        // TODO: Does not work at the moment because it is used in can_split(). Fix that in future releases cause assertions are our friend
        // debug_assert!(!self.is_empty());

        self
    }

    #[inline]
    pub(crate) fn min_bound(&self) -> (f64, bool) {
        (self.lower, self.lower_closed)
    }

    #[inline]
    pub(crate) fn max_bound(&self) -> (f64, bool) {
        (self.upper, self.upper_closed)
    }

    #[inline]
    pub fn is_singleton(&self) -> bool {
        self.lower == self.upper && self.lower_closed && self.upper_closed
    }

    #[inline]
    pub(crate) fn contains_zero(&self) -> bool {
        self.contains(0.0)
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
            return Interval::singleton(0.0);
        }

        let left = [
            (self.lower, self.lower_closed),
            (self.upper, self.upper_closed),
        ];
        let right = [(rhs.lower, rhs.lower_closed), (rhs.upper, rhs.upper_closed)];
        let mut lower = f64::INFINITY;
        let mut upper = f64::NEG_INFINITY;
        let mut lower_closed = false;
        let mut upper_closed = false;

        for (left_value, left_closed) in left {
            for (right_value, right_closed) in right {
                let value = extended_product(left_value, right_value);
                let attained = if value == 0.0 {
                    (left_value == 0.0 && self.contains(0.0))
                        || (right_value == 0.0 && rhs.contains(0.0))
                } else {
                    value.is_finite() && left_closed && right_closed
                };
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

        Interval::new(lower, upper, lower_closed, upper_closed)
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

        let reciprocal = Interval::new(
            1.0 / rhs.upper,
            1.0 / rhs.lower,
            rhs.upper_closed && rhs.upper.is_finite(),
            rhs.lower_closed && rhs.lower.is_finite(),
        );
        self * reciprocal
    }
}

#[inline]
fn extended_product(left: f64, right: f64) -> f64 {
    if (left == 0.0 && right.is_infinite()) || (right == 0.0 && left.is_infinite()) {
        0.0
    } else {
        left * right
    }
}
