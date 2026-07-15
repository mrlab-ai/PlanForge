pub const ABS_EPSILON: f64 = 1e-12;
pub const REL_EPSILON: f64 = 1e-12;

const _: () = assert!(ABS_EPSILON > f64::EPSILON);
const _: () = assert!(REL_EPSILON > f64::EPSILON);
const _: () = assert!(ABS_EPSILON < 1.0);
const _: () = assert!(REL_EPSILON < 1.0);

#[inline]
pub fn tolerance(lhs: f64, rhs: f64) -> f64 {
    ABS_EPSILON.max(REL_EPSILON * lhs.abs().max(rhs.abs()))
}

#[inline]
pub fn canonicalize(value: f64) -> f64 {
    if value.is_nan() || value.is_infinite() {
        return value;
    }
    (value / ABS_EPSILON).round() * ABS_EPSILON
}

#[inline]
pub fn canonical_bits(value: f64) -> u64 {
    let canonical = canonicalize(value);
    if canonical == 0.0 {
        0.0f64.to_bits()
    } else {
        canonical.to_bits()
    }
}

#[inline]
pub fn equal(lhs: f64, rhs: f64) -> bool {
    (lhs - rhs).abs() <= tolerance(lhs, rhs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonicalize_rounds_to_abs_epsilon_grid() {
        assert_eq!(canonicalize(1.0 + 0.4e-12), 1.0);
        assert_eq!(canonicalize(1.0 + 1.4e-12), 1.0 + 1e-12);
    }

    #[test]
    fn canonical_bits_deduplicate_close_values() {
        assert_eq!(canonical_bits(0.1 + 0.2), canonical_bits(0.3));
    }

    #[test]
    fn canonical_bits_deduplicate_negative_zero() {
        assert_eq!(canonical_bits(-0.0), canonical_bits(0.0));
    }

    #[test]
    fn nonfinite_values_are_preserved() {
        assert!(canonicalize(f64::NAN).is_nan());
        assert_eq!(canonicalize(f64::INFINITY), f64::INFINITY);
    }
}
