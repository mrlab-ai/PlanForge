//! Preimages of canonical scalar effects, including floating rounding plateaus.

use planforge_sas::numeric_task::{
    AssignmentOperation, INF_VALUE, NEG_INF_VALUE, NumericValue, ONE_VALUE, ZERO_VALUE,
};
use planforge_sas::utils::{interval::Interval, value_lattice::ValueLattice};

pub(super) fn scalar_preimage(
    target: Interval,
    operation: &AssignmentOperation,
    operand: NumericValue,
) -> Option<Interval> {
    assert!(operand.is_finite(), "a scalar effect operand is finite");
    match operation {
        AssignmentOperation::Assign => {
            return target.contains(operand).then(Interval::unbounded);
        }
        AssignmentOperation::Times if operand == ZERO_VALUE => {
            return target.contains(ZERO_VALUE).then(Interval::unbounded);
        }
        AssignmentOperation::Divide => assert!(operand != ZERO_VALUE, "division by zero"),
        _ => {}
    }
    if matches!(
        operation,
        AssignmentOperation::Plus | AssignmentOperation::Minus
    ) && operand == ZERO_VALUE
        || matches!(
            operation,
            AssignmentOperation::Times | AssignmentOperation::Divide
        ) && operand == ONE_VALUE
    {
        return Some(target);
    }
    let increasing = matches!(
        operation,
        AssignmentOperation::Plus | AssignmentOperation::Minus
    ) || operand > ZERO_VALUE;
    let image = |value| AssignmentOperation::apply(value, operation, operand);
    let above_lower = |value| value > target.lower || value == target.lower && target.lower_closed;
    let below_upper = |value| value < target.upper || value == target.upper && target.upper_closed;
    // Algebra gives a useful guess, not an inverse of rounded arithmetic.
    let inverse_guess = |bound: NumericValue| {
        NumericValue::new(match operation {
            AssignmentOperation::Plus => bound.value() - operand.value(),
            AssignmentOperation::Minus => bound.value() + operand.value(),
            AssignmentOperation::Times => bound.value() / operand.value(),
            AssignmentOperation::Divide => bound.value() * operand.value(),
            AssignmentOperation::Assign => unreachable!("assignment handled above"),
        })
    };
    let (lower_target, upper_target) = if increasing {
        (target.lower, target.upper)
    } else {
        (target.upper, target.lower)
    };
    let lower = first_satisfying(inverse_guess(lower_target), |value| {
        if increasing {
            above_lower(image(value))
        } else {
            below_upper(image(value))
        }
    })?;
    // Reflection turns the upper-bound search into the same monotone problem.
    let upper = -first_satisfying(-inverse_guess(upper_target), |value| {
        if increasing {
            below_upper(image(-value))
        } else {
            above_lower(image(-value))
        }
    })?;
    // A jump over the target leaves crossed bounds. Retain that empty interval:
    // its separating cut still lets CEGAR refine a convex image's spurious edge.
    Some(Interval::new(
        if lower.value() == -f64::MAX {
            NEG_INF_VALUE
        } else {
            lower
        },
        if upper.value() == f64::MAX {
            INF_VALUE
        } else {
            upper
        },
        true,
        true,
    ))
}

/// First finite canonical value satisfying a monotone false-to-true predicate.
/// Verify the algebraic guess locally; otherwise bisect the finite f64 order.
/// Termination follows from the shrinking integer rank interval, without a cap.
fn first_satisfying(
    guess: NumericValue,
    predicate: impl Fn(NumericValue) -> bool,
) -> Option<NumericValue> {
    if guess.is_finite() {
        if predicate(guess) {
            let (previous, _) = ValueLattice::GRID
                .split_bounds(guess, false)
                .expect("finite bound");
            if !previous.is_finite() || !predicate(previous) {
                return Some(guess);
            }
        } else {
            let (_, next) = ValueLattice::GRID
                .split_bounds(guess, true)
                .expect("finite bound");
            if !next.is_finite() {
                return None;
            }
            if predicate(next) {
                return Some(next);
            }
        }
    }
    let minimum = NumericValue::new(-f64::MAX);
    let maximum = NumericValue::new(f64::MAX);
    if predicate(minimum) {
        return Some(minimum);
    }
    if !predicate(maximum) {
        return None;
    }
    const SIGN: u64 = 1 << 63;
    // Ordered IEEE ranks exclude NaN and infinity; constructing each midpoint
    // folds raw grid cells onto their canonical value and preserves monotonicity.
    let mut lower = !(-f64::MAX).to_bits();
    let mut upper = f64::MAX.to_bits() ^ SIGN;
    let value_at = |rank: u64| {
        NumericValue::new(f64::from_bits(if rank & SIGN == 0 {
            !rank
        } else {
            rank ^ SIGN
        }))
    };
    while upper - lower > 1 {
        let middle = lower + (upper - lower) / 2;
        if predicate(value_at(middle)) {
            upper = middle;
        } else {
            lower = middle;
        }
    }
    Some(value_at(upper))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn nv(value: f64) -> NumericValue {
        NumericValue::new(value)
    }

    #[test]
    fn inverse_rounding_does_not_admit_the_zenotravel_deviation() {
        let target = Interval::new(nv(6073.000000000001), INF_VALUE, true, false);
        let preimage = scalar_preimage(target, &AssignmentOperation::Minus, nv(3764.0)).unwrap();
        assert!(!preimage.contains(nv(9837.0)));
        assert!(preimage.contains(nv(9837.0_f64.next_up())));
    }

    #[test]
    fn an_empty_preimage_retains_the_cut_across_a_rounding_gap() {
        let preimage = scalar_preimage(
            Interval::singleton(nv(1e-12)),
            &AssignmentOperation::Times,
            nv(2.0),
        )
        .unwrap();
        assert!(preimage.is_empty());
        assert_eq!(preimage.lower, nv(1e-12));
        assert_eq!(preimage.upper, ZERO_VALUE);
    }

    #[test]
    fn cancellation_plateaus_include_every_canonical_predecessor() {
        let operand = nv(1e16);
        let target = Interval::singleton(operand);
        let preimage = scalar_preimage(target, &AssignmentOperation::Plus, operand).unwrap();
        for value in [-1.0, -0.5, 0.0, 0.5, 1.0] {
            assert!(preimage.contains(nv(value)));
            assert!(target.contains(nv(value) + operand));
        }
        assert!(!preimage.contains(nv(-1.000000000001)));
        assert!(!preimage.contains(nv(1.000000000001)));
    }

    #[test]
    fn preimages_match_forward_arithmetic_at_their_neighbours() {
        for operation in [
            AssignmentOperation::Plus,
            AssignmentOperation::Minus,
            AssignmentOperation::Times,
            AssignmentOperation::Divide,
        ] {
            for operand in [-3764.0, -0.3, 0.3, 3764.0] {
                for source in [-9837.0, -3.0, 0.0, 3.0, 9837.0] {
                    let source = nv(source);
                    let target = Interval::singleton(AssignmentOperation::apply(
                        source,
                        &operation,
                        nv(operand),
                    ));
                    let preimage = scalar_preimage(target, &operation, nv(operand)).unwrap();
                    assert!(preimage.contains(source));
                    for bound in [preimage.lower, preimage.upper] {
                        if !bound.is_finite() {
                            continue;
                        }
                        for point in [
                            ValueLattice::GRID.split_bounds(bound, false).unwrap().0,
                            bound,
                            ValueLattice::GRID.split_bounds(bound, true).unwrap().1,
                        ] {
                            assert_eq!(
                                preimage.contains(point),
                                target.contains(AssignmentOperation::apply(
                                    point,
                                    &operation,
                                    nv(operand)
                                )),
                                "{operation:?} {operand} {source} {point}"
                            );
                        }
                    }
                }
            }
        }
    }
}
