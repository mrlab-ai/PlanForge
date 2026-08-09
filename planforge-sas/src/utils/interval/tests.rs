use super::*;

use crate::numeric_conditions::{ArithOp, CompOp};

#[test]
fn interval_add_preserves_exact_bounds() {
    let bounded = Interval::closed(3.0, 4.0);
    let unbounded = Interval::open(-3.0, f64::INFINITY);

    assert_eq!(
        bounded + unbounded,
        Interval::new(0.0, f64::INFINITY, false, false)
    );
    assert_eq!(
        unbounded + bounded,
        Interval::new(0.0, f64::INFINITY, false, false)
    );
}

#[test]
fn interval_add_regular() {
    let a = Interval::closed(3.0, 4.0);
    let b = Interval::closed(-3.0, 10.0);
    assert_eq!(a + b, Interval::closed(0.0, 14.0));
}

#[test]
fn interval_subtract_uses_opposite_extrema() {
    let result = Interval::closed(3.0, 4.0) - Interval::closed(-3.0, 10.0);
    assert_eq!(result, Interval::closed(-7.0, 7.0));
}

#[test]
fn interval_comparison_definite_and_unknown() {
    // Always true: max(lhs) < min(rhs).
    assert_eq!(
        CompOp::Lt.apply_interval(Interval::closed(0.0, 1.0), Interval::closed(2.0, 3.0)),
        Some(true)
    );

    // Always false: min(lhs) >= max(rhs).
    assert_eq!(
        CompOp::Lt.apply_interval(Interval::closed(2.0, 3.0), Interval::closed(0.0, 1.0)),
        Some(false)
    );

    // Unknown: intervals overlap.
    assert_eq!(
        CompOp::Lt.apply_interval(Interval::closed(0.0, 3.0), Interval::closed(2.0, 4.0)),
        None
    );
}

#[test]
fn interval_eq_and_ne() {
    // Singletons equal => definitely true.
    assert_eq!(
        CompOp::Eq.apply_interval(Interval::singleton(2.0), Interval::singleton(2.0)),
        Some(true)
    );

    // Disjoint => definitely false.
    assert_eq!(
        CompOp::Eq.apply_interval(Interval::closed(0.0, 1.0), Interval::closed(2.0, 3.0)),
        Some(false)
    );

    // Overlap => unknown.
    assert_eq!(
        CompOp::Eq.apply_interval(Interval::closed(0.0, 2.0), Interval::closed(2.0, 3.0)),
        None
    );

    // Ne: disjoint => definitely true.
    assert_eq!(
        CompOp::Ne.apply_interval(Interval::closed(0.0, 1.0), Interval::closed(2.0, 3.0)),
        Some(true)
    );
}

#[test]
fn interval_mul_preserves_closed_extrema() {
    let result = ArithOp::Mul.apply_interval(Interval::singleton(2.0), Interval::closed(3.0, 4.0));
    assert_eq!(result, Interval::closed(6.0, 8.0));
}

#[test]
fn interval_mul_handles_mixed_signs_and_unbounded_zero() {
    assert_eq!(
        Interval::closed(-1.0, 2.0) * Interval::closed(-3.0, 4.0),
        Interval::closed(-6.0, 8.0)
    );
    assert_eq!(
        Interval::singleton(0.0) * UNBOUNDED_INTERVAL,
        Interval::singleton(0.0)
    );
    assert_eq!(
        Interval::new(0.0, 1.0, false, true) * Interval::closed(2.0, 3.0),
        Interval::new(0.0, 3.0, false, true)
    );
}

#[test]
fn interval_div_handles_signs_and_zero_crossings() {
    assert_eq!(
        Interval::closed(2.0, 4.0) / Interval::closed(-2.0, -1.0),
        Interval::closed(-4.0, -1.0)
    );
    assert_eq!(
        Interval::closed(2.0, 4.0) / Interval::new(0.0, 2.0, false, true),
        Interval::new(1.0, f64::INFINITY, true, false)
    );
    assert_eq!(
        Interval::closed(2.0, 4.0) / Interval::closed(-1.0, 1.0),
        UNBOUNDED_INTERVAL
    );
}

#[test]
fn interval_le_handles_open_touching_bounds() {
    assert_eq!(
        CompOp::Le.apply_interval(Interval::open(1.0, 2.0), Interval::closed(2.0, 3.0)),
        Some(true)
    );
    assert_eq!(
        CompOp::Le.apply_interval(Interval::closed(2.0, 3.0), Interval::open(1.0, 2.0)),
        Some(false)
    );
}

#[test]
fn interval_intersections() {
    let smaller = Interval::new(2.0, 4.0, false, false);
    let closed_smaller = Interval::new(2.0, 4.0, true, true);
    let larger = Interval::new(-2.0, 8.0, true, true);
    let lefter = Interval::new(-2.0, 2.0, true, true);
    let righter = Interval::new(4.0, 6.0, true, false);
    let very_lefter = Interval::new(-2.0, 0.0, true, true);
    let very_righter = Interval::new(8.0, f64::INFINITY, true, false);
    let empty = Interval::new(4.0, 0.0, true, true);

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
    let target = Interval::new(-74.7, -74.0, false, true);
    let preimage = Interval::new(
        target.lower + 7.35,
        target.upper + 7.35,
        target.lower_closed,
        target.upper_closed,
    )
    .canonicalized();

    assert_eq!(preimage.lower, -67.35);
    assert!(!preimage.lower_closed);
    assert!(!preimage.contains(-67.35));
}
