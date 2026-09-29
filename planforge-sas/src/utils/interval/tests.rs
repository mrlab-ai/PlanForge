use super::*;

use crate::{
    numeric_conditions::{ArithOp, CompOp},
    numeric_task::NAN_VALUE,
};

#[test]
fn interval_add_preserves_exact_bounds() {
    let bounded = Interval::closed(NumericValue::new(3.0), NumericValue::new(4.0));
    let unbounded = Interval::open(NumericValue::new(-3.0), INF_VALUE);

    assert_eq!(
        bounded + unbounded,
        Interval::new(ZERO_VALUE, INF_VALUE, false, false)
    );
    assert_eq!(
        unbounded + bounded,
        Interval::new(ZERO_VALUE, INF_VALUE, false, false)
    );
}

#[test]
fn interval_add_regular() {
    let a = Interval::closed(NumericValue::new(3.0), NumericValue::new(4.0));
    let b = Interval::closed(NumericValue::new(-3.0), NumericValue::new(10.0));
    assert_eq!(a + b, Interval::closed(ZERO_VALUE, NumericValue::new(14.0)));
}

#[test]
fn interval_subtract_uses_opposite_extrema() {
    let result = Interval::closed(NumericValue::new(3.0), NumericValue::new(4.0))
        - Interval::closed(NumericValue::new(-3.0), NumericValue::new(10.0));
    assert_eq!(
        result,
        Interval::closed(NumericValue::new(-7.0), NumericValue::new(7.0))
    );
}

#[test]
fn interval_comparison_definite_and_unknown() {
    // Always true: max(lhs) < min(rhs).
    assert_eq!(
        CompOp::Lt.apply_interval(
            Interval::closed(ZERO_VALUE, ONE_VALUE),
            Interval::closed(NumericValue::new(2.0), NumericValue::new(3.0))
        ),
        Some(true)
    );

    // Always false: min(lhs) >= max(rhs).
    assert_eq!(
        CompOp::Lt.apply_interval(
            Interval::closed(NumericValue::new(2.0), NumericValue::new(3.0)),
            Interval::closed(ZERO_VALUE, ONE_VALUE)
        ),
        Some(false)
    );

    // Unknown: intervals overlap.
    assert_eq!(
        CompOp::Lt.apply_interval(
            Interval::closed(ZERO_VALUE, NumericValue::new(3.0)),
            Interval::closed(NumericValue::new(2.0), NumericValue::new(4.0))
        ),
        None
    );
}

#[test]
fn interval_eq_and_ne() {
    // Singletons equal => definitely true.
    assert_eq!(
        CompOp::Eq.apply_interval(
            Interval::singleton(NumericValue::new(2.0)),
            Interval::singleton(NumericValue::new(2.0))
        ),
        Some(true)
    );

    // Disjoint => definitely false.
    assert_eq!(
        CompOp::Eq.apply_interval(
            Interval::closed(ZERO_VALUE, ONE_VALUE),
            Interval::closed(NumericValue::new(2.0), NumericValue::new(3.0))
        ),
        Some(false)
    );

    // Overlap => unknown.
    assert_eq!(
        CompOp::Eq.apply_interval(
            Interval::closed(ZERO_VALUE, NumericValue::new(2.0)),
            Interval::closed(NumericValue::new(2.0), NumericValue::new(3.0))
        ),
        None
    );

    // Ne: disjoint => definitely true.
    assert_eq!(
        CompOp::Ne.apply_interval(
            Interval::closed(ZERO_VALUE, ONE_VALUE),
            Interval::closed(NumericValue::new(2.0), NumericValue::new(3.0))
        ),
        Some(true)
    );
}

#[test]
fn interval_mul_preserves_closed_extrema() {
    let result = ArithOp::Mul.apply_interval(
        Interval::singleton(NumericValue::new(2.0)),
        Interval::closed(NumericValue::new(3.0), NumericValue::new(4.0)),
    );
    assert_eq!(
        result,
        Interval::closed(NumericValue::new(6.0), NumericValue::new(8.0))
    );
}

#[test]
fn interval_mul_handles_mixed_signs_and_unbounded_zero() {
    assert_eq!(
        Interval::closed(NumericValue::new(-1.0), NumericValue::new(2.0))
            * Interval::closed(NumericValue::new(-3.0), NumericValue::new(4.0)),
        Interval::closed(NumericValue::new(-6.0), NumericValue::new(8.0))
    );
    assert_eq!(
        Interval::singleton(ZERO_VALUE) * UNBOUNDED_INTERVAL,
        Interval::singleton(ZERO_VALUE)
    );
    assert_eq!(
        Interval::new(ZERO_VALUE, ONE_VALUE, false, true)
            * Interval::closed(NumericValue::new(2.0), NumericValue::new(3.0)),
        Interval::new(ZERO_VALUE, NumericValue::new(3.0), false, true)
    );
}

#[test]
fn interval_div_handles_signs_and_zero_crossings() {
    assert_eq!(
        Interval::closed(NumericValue::new(2.0), NumericValue::new(4.0))
            / Interval::closed(NumericValue::new(-2.0), NumericValue::new(-1.0)),
        Interval::closed(NumericValue::new(-4.0), NumericValue::new(-1.0))
    );
    assert_eq!(
        Interval::closed(NumericValue::new(2.0), NumericValue::new(4.0))
            / Interval::new(ZERO_VALUE, NumericValue::new(2.0), false, true),
        Interval::new(ONE_VALUE, INF_VALUE, true, false)
    );
    assert_eq!(
        Interval::closed(NumericValue::new(2.0), NumericValue::new(4.0))
            / Interval::closed(NumericValue::new(-1.0), ONE_VALUE),
        UNBOUNDED_INTERVAL
    );
}

#[test]
fn interval_le_handles_open_touching_bounds() {
    assert_eq!(
        CompOp::Le.apply_interval(
            Interval::open(ONE_VALUE, NumericValue::new(2.0)),
            Interval::closed(NumericValue::new(2.0), NumericValue::new(3.0))
        ),
        Some(true)
    );
    assert_eq!(
        CompOp::Le.apply_interval(
            Interval::closed(NumericValue::new(2.0), NumericValue::new(3.0)),
            Interval::open(ONE_VALUE, NumericValue::new(2.0))
        ),
        Some(false)
    );
}

#[test]
fn interval_intersections() {
    let smaller = Interval::new(NumericValue::new(2.0), NumericValue::new(4.0), false, false);
    let closed_smaller = Interval::new(NumericValue::new(2.0), NumericValue::new(4.0), true, true);
    let larger = Interval::new(NumericValue::new(2.0), NumericValue::new(8.0), true, true);
    let lefter = Interval::new(NumericValue::new(-2.0), NumericValue::new(2.0), true, true);
    let righter = Interval::new(NumericValue::new(4.0), NumericValue::new(6.0), true, false);
    let very_lefter = Interval::new(NumericValue::new(-2.0), ZERO_VALUE, true, true);
    let very_righter = Interval::new(NumericValue::new(8.0), INF_VALUE, true, false);
    let empty = Interval::new(NumericValue::new(4.0), ZERO_VALUE, true, true);

    assert!(smaller.intersects(&larger));
    assert!(larger.intersects(&smaller));
    assert!(!smaller.intersects(&lefter));
    assert!(closed_smaller.intersects(&lefter));
    assert!(!smaller.intersects(&righter));
    assert!(closed_smaller.intersects(&righter));
    assert!(!smaller.intersects(&very_lefter));
    assert!(!smaller.intersects(&very_righter));
    assert!(!smaller.intersects(&empty));
    assert!(UNBOUNDED_INTERVAL.intersects(&very_righter));
    assert!(UNBOUNDED_INTERVAL.intersects(&smaller));
}

#[test]
fn canonicalized_interval_preserves_open_transition_boundaries() {
    let target = Interval::new(
        NumericValue::new(-74.7),
        NumericValue::new(-74.0),
        false,
        true,
    );
    let preimage = Interval::new(
        NumericValue::new(target.lower.value() + 7.35),
        NumericValue::new(target.upper.value() + 7.35),
        target.lower_closed,
        target.upper_closed,
    )
    .canonicalized();

    assert_eq!(preimage.lower, NumericValue::new(-67.35));
    assert!(!preimage.lower_closed);
    assert!(!preimage.contains(NumericValue::new(-67.35)));
}

/// The endpoint-ordering predicates are the whole truth table, including the
/// one asymmetric case each: at an equal bound value, closedness decides.
#[test]
fn lower_endpoint_ordering_is_decided_by_closedness_at_equal_bounds() {
    let closed = Interval::closed(ONE_VALUE, NumericValue::new(5.0));
    let open = Interval::open(ONE_VALUE, NumericValue::new(5.0));
    let higher = Interval::closed(NumericValue::new(2.0), NumericValue::new(5.0));

    assert!(closed.lower_is_lower(&higher));
    assert!(!higher.lower_is_lower(&closed));
    assert!(closed.lower_is_lower(&open));
    assert!(!open.lower_is_lower(&closed));
    assert!(!closed.lower_is_lower(&closed));
    assert!(!open.lower_is_lower(&open));

    assert!(closed.lower_is_lower_or_equal(&higher));
    assert!(!higher.lower_is_lower_or_equal(&closed));
    assert!(closed.lower_is_lower_or_equal(&open));
    assert!(!open.lower_is_lower_or_equal(&closed));
    assert!(closed.lower_is_lower_or_equal(&closed));
    assert!(open.lower_is_lower_or_equal(&open));
}

#[test]
fn upper_endpoint_ordering_is_decided_by_closedness_at_equal_bounds() {
    let closed = Interval::closed(ONE_VALUE, NumericValue::new(5.0));
    let open = Interval::open(ONE_VALUE, NumericValue::new(5.0));
    let lower = Interval::closed(ONE_VALUE, NumericValue::new(4.0));

    assert!(closed.upper_is_higher(&lower));
    assert!(!lower.upper_is_higher(&closed));
    assert!(closed.upper_is_higher(&open));
    assert!(!open.upper_is_higher(&closed));
    assert!(!closed.upper_is_higher(&closed));
    assert!(!open.upper_is_higher(&open));

    assert!(closed.upper_is_higher_or_equal(&lower));
    assert!(!lower.upper_is_higher_or_equal(&closed));
    assert!(closed.upper_is_higher_or_equal(&open));
    assert!(!open.upper_is_higher_or_equal(&closed));
    assert!(closed.upper_is_higher_or_equal(&closed));
    assert!(open.upper_is_higher_or_equal(&open));
}

#[test]
fn endpoint_ordering_rejects_nan_bounds() {
    let nan = Interval::closed(NAN_VALUE, NAN_VALUE);
    let unit = Interval::closed(ZERO_VALUE, ONE_VALUE);

    assert!(!nan.lower_is_lower(&unit));
    assert!(!unit.lower_is_lower(&nan));
    assert!(!nan.lower_is_lower_or_equal(&unit));
    assert!(!unit.lower_is_lower_or_equal(&nan));
    assert!(!nan.upper_is_higher(&unit));
    assert!(!unit.upper_is_higher(&nan));
    assert!(!nan.upper_is_higher_or_equal(&unit));
    assert!(!unit.upper_is_higher_or_equal(&nan));
}
