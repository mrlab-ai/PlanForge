use crate::numeric_task::{NumericValue, ZERO_VALUE};

pub const ABS_EPSILON: f64 = 1e-12;
pub const REL_EPSILON: f64 = 1e-12;
pub const DIJKSTRA_EPSILON: f64 = 1e-12;
pub const SEARCH_EPSILON: f64 = 1e-9;
/// How closely two computations of the same cost or value agree, as in a
/// concrete plan's cost and the abstract goal distance it realises.
pub const AGREEMENT_EPSILON: f64 = 1e-7;
/// How closely a value matches an LP optimum it should dominate or reach.
pub const LP_EPSILON: f64 = 1e-6;

const _: () = assert!(ABS_EPSILON > f64::EPSILON);
const _: () = assert!(REL_EPSILON > f64::EPSILON);
const _: () = assert!(ABS_EPSILON < 1.0);
const _: () = assert!(REL_EPSILON < 1.0);

#[inline]
pub fn tolerance(lhs: NumericValue, rhs: NumericValue) -> NumericValue {
    NumericValue::new(ABS_EPSILON.max(REL_EPSILON * lhs.abs().max(rhs.abs()).value()))
}

/// TODO(canonical values): multiplying by the inexact `ABS_EPSILON` moves
/// some integers by one ulp from about 1e7 on (636192524 becomes
/// 636192523.9999999), and from |v| >= 2^53 * 1e-12 (about 9007) the grid is
/// finer than f64 resolution, so no noise is removed there. Keeping integers
/// fixed (identity above that bound, `(v * 1e12).round() / 1e12` below) would
/// let comparisons of canonical values be exact; it changes search state
/// hashing, so it needs its own gated change.
#[inline]
pub fn canonicalize(value: NumericValue) -> NumericValue {
    if value.is_nan() || value.is_infinite() {
        return value;
    }
    NumericValue::new((value.value() / ABS_EPSILON).round() * ABS_EPSILON)
}

/// Bits of [`canonicalize`]d `value`, with `-0.0` folded into `0.0` so the
/// two zeros hash alike.
#[inline]
pub fn canonical_bits(value: NumericValue) -> u64 {
    let canonical = canonicalize(value);
    if canonical == ZERO_VALUE {
        ZERO_VALUE.to_bits()
    } else {
        canonical.to_bits()
    }
}

#[inline]
pub fn equal(lhs: NumericValue, rhs: NumericValue) -> bool {
    (lhs - rhs).abs() <= tolerance(lhs, rhs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nv(value: f64) -> NumericValue {
        NumericValue::new(value)
    }

    #[test]
    fn canonicalize_rounds_to_abs_epsilon_grid() {
        assert_eq!(canonicalize(nv(1.0 + 0.4e-12)), nv(1.0));
        assert_eq!(canonicalize(nv(1.0 + 1.4e-12)), nv(1.0 + 1e-12));
    }

    #[test]
    fn canonical_bits_deduplicate_close_values() {
        assert_eq!(canonical_bits(nv(0.1 + 0.2)), canonical_bits(nv(0.3)));
    }

    #[test]
    fn canonical_bits_deduplicate_negative_zero() {
        assert_eq!(canonical_bits(nv(-0.0)), canonical_bits(nv(0.0)));
    }

    #[test]
    fn nonfinite_values_are_preserved() {
        assert!(canonicalize(nv(f64::NAN)).is_nan());
        assert_eq!(canonicalize(nv(f64::INFINITY)), nv(f64::INFINITY));
    }
}
