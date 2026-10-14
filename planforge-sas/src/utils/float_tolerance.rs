use crate::numeric_task::{NumericValue, ZERO_VALUE};

pub const ABS_EPSILON: f64 = 1e-12;
pub const REL_EPSILON: f64 = 1e-12;
pub const DIJKSTRA_EPSILON: f64 = 1e-12;
pub const SEARCH_EPSILON: f64 = 1e-9;

const _: () = assert!(ABS_EPSILON > f64::EPSILON);
const _: () = assert!(REL_EPSILON > f64::EPSILON);
const _: () = assert!(ABS_EPSILON < 1.0);
const _: () = assert!(REL_EPSILON < 1.0);

#[inline]
pub fn tolerance(lhs: NumericValue, rhs: NumericValue) -> NumericValue {
    NumericValue::new(ABS_EPSILON.max(REL_EPSILON * lhs.abs().max(rhs.abs()).value()))
}

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
