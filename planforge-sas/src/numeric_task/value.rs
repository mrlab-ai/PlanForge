use crate::utils::float_tolerance;
use std::{
    fmt::{self, Display},
    hash::Hash,
};

pub const ZERO_VALUE: NumericValue = NumericValue(0.0);
pub const ONE_VALUE: NumericValue = NumericValue(1.0);
pub const INF_VALUE: NumericValue = NumericValue(f64::INFINITY);
pub const NEG_INF_VALUE: NumericValue = NumericValue(f64::NEG_INFINITY);

/// A canonical numeric state value, constant, threshold or interval endpoint.
///
/// Values represent reals approximated in f64. Equality means equality after
/// rounding to the 1e-12 grid; signed zero has a single representation.
/// Values at or above 2^53 / 1e12 in magnitude are retained unchanged because
/// the grid is finer than f64 resolution. Integers remain exact.
/// Construction and arithmetic round their result. An affine expression must
/// use [`Self::affine`]: its products and sum are computed in raw f64 and rounded
/// once, rather than rounding each intermediate operation. Costs and distances
/// use f64 and their own tolerances, not this type.
///
/// NaN is rejected at input boundaries with [`Self::try_new`] and asserted
/// impossible internally. Infinities represent unbounded interval endpoints.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NumericValue(f64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidNumericValue;

impl Display for InvalidNumericValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("numeric value must not be NaN")
    }
}

impl std::error::Error for InvalidNumericValue {}

impl Eq for NumericValue {}

impl PartialOrd for NumericValue {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for NumericValue {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl Hash for NumericValue {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

impl NumericValue {
    #[inline]
    pub fn new(value: f64) -> Self {
        assert!(!value.is_nan(), "numeric computation produced NaN");
        let value = float_tolerance::canonicalize(value);
        Self(if value == 0.0 { 0.0 } else { value })
    }

    pub fn try_new(value: f64) -> Result<Self, InvalidNumericValue> {
        if value.is_nan() {
            Err(InvalidNumericValue)
        } else {
            Ok(Self::new(value))
        }
    }

    /// Evaluate an affine expression with exactly one final rounding.
    #[inline]
    pub fn affine(terms: impl IntoIterator<Item = (f64, Self)>, constant: f64) -> Self {
        Self::new(
            constant
                + terms
                    .into_iter()
                    .map(|(coefficient, value)| coefficient * value.0)
                    .sum::<f64>(),
        )
    }

    /// Exact integer grid units while every unit is representable in f64.
    pub fn grid_units(self) -> Option<i64> {
        (self.0.abs() < float_tolerance::GRID_BOUND)
            .then(|| float_tolerance::rounded_grid_units(self.0) as i64)
    }

    #[inline(always)]
    pub fn value(&self) -> f64 {
        self.0
    }
}

/// Helpers preserve the canonical representation.
impl NumericValue {
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(self.0.abs())
    }

    #[inline(always)]
    pub fn is_finite(self) -> bool {
        self.0.is_finite()
    }

    #[inline(always)]
    pub fn is_infinite(self) -> bool {
        self.0.is_infinite()
    }

    #[inline(always)]
    pub fn to_bits(self) -> u64 {
        self.0.to_bits()
    }
}

impl Display for NumericValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Each arithmetic operation constructs a canonical result.
macro_rules! numeric_value_binary_op {
    ($trait:ident, $method:ident, $assign_trait:ident, $assign_method:ident, $op:tt) => {
        impl std::ops::$trait for NumericValue {
            type Output = NumericValue;

            #[inline(always)]
            fn $method(self, rhs: NumericValue) -> NumericValue {
                NumericValue::new(self.0 $op rhs.0)
            }
        }

        impl std::ops::$assign_trait for NumericValue {
            #[inline(always)]
            fn $assign_method(&mut self, rhs: NumericValue) {
                *self = *self $op rhs;
            }
        }
    };
}

numeric_value_binary_op!(Add, add, AddAssign, add_assign, +);
numeric_value_binary_op!(Sub, sub, SubAssign, sub_assign, -);
numeric_value_binary_op!(Mul, mul, MulAssign, mul_assign, *);
numeric_value_binary_op!(Div, div, DivAssign, div_assign, /);

impl std::ops::Neg for NumericValue {
    type Output = NumericValue;

    #[inline(always)]
    fn neg(self) -> NumericValue {
        NumericValue::new(-self.0)
    }
}

/// Sum raw values in iteration order, then round once.
impl std::iter::Sum for NumericValue {
    #[inline(always)]
    fn sum<I: Iterator<Item = NumericValue>>(iter: I) -> NumericValue {
        NumericValue::new(iter.map(|value| value.0).sum())
    }
}

#[cfg(test)]
mod numeric_value_tests {
    use super::*;
    use std::collections::{BTreeSet, HashSet};

    #[test]
    fn canonical_equality_hash_and_order_agree() {
        let values = [
            NumericValue::new(0.1 + 0.2),
            NumericValue::new(0.3),
            NumericValue::new(-0.0),
            ZERO_VALUE,
        ];
        assert_eq!(values.into_iter().collect::<HashSet<_>>().len(), 2);
        assert_eq!(values.into_iter().collect::<BTreeSet<_>>().len(), 2);
        assert_eq!((-ZERO_VALUE).to_bits(), ZERO_VALUE.to_bits());
    }

    #[test]
    fn affine_rounds_once() {
        let value = NumericValue::new(2e-12);
        assert_eq!(
            NumericValue::affine([(0.25, value), (0.25, value)], 0.0),
            NumericValue::new(1e-12)
        );
        assert_eq!(
            NumericValue::new(0.1) + NumericValue::new(0.2),
            NumericValue::new(0.3)
        );
    }

    #[test]
    fn grid_preserves_integers_and_large_values() {
        for value in [
            0.0,
            1.0,
            -1.0,
            9007.0,
            10_000_001.0,
            636_192_524.0,
            -636_192_524.0,
            f64::MAX,
            -f64::MAX,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ] {
            assert_eq!(NumericValue::new(value).value(), value);
        }
        for value in [
            9007.25,
            -9007.25,
            1e10 + 0.25,
            4480.672703054073,
            -4480.672703054073,
        ] {
            assert_eq!(NumericValue::new(value).value(), value);
        }
    }

    #[test]
    fn canonical_construction_is_idempotent_across_the_grid() {
        let mut bits = 1u64;
        for _ in 0..100_000 {
            bits = bits.wrapping_mul(6364136223846793005).wrapping_add(1);
            let raw = f64::from_bits(bits);
            if raw.is_nan() {
                continue;
            }
            let value = NumericValue::new(raw);
            assert_eq!(NumericValue::new(value.value()).to_bits(), value.to_bits());
        }
    }

    #[test]
    fn external_nan_is_an_error() {
        assert_eq!(NumericValue::try_new(f64::NAN), Err(InvalidNumericValue));
    }

    #[test]
    #[should_panic(expected = "numeric computation produced NaN")]
    fn internal_nan_is_an_invariant_failure() {
        let _ = INF_VALUE - INF_VALUE;
    }
}
