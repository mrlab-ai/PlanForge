//! Reachable numeric values in exact canonical grid units.

use crate::numeric_task::NumericValue;
use crate::utils::float_tolerance::GRID_SCALE;

/// 5^12: since 1e12 = 2^12 * 5^12, a number of grid units divisible by it is
/// an exact binary fraction.
pub const FIVE_TO_TWELVE: u64 = 244_140_625;

/// An offset and gcd lattice in canonical grid units.
/// Callers infer reachable-value lattices only after proving that rounding
/// preserves these generators, or taking their arithmetic closure.
/// The offset and step are integers in units of the canonical grid, so gcd
/// and boundary snapping never depend on a floating integer tolerance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ValueLattice {
    offset: i64,
    step: u64,
}

impl ValueLattice {
    /// The full canonical grid: valid even when no coarser lattice is known.
    pub const GRID: Self = Self { offset: 0, step: 1 };

    /// Infer a lattice from an initial value and constant additive generators.
    /// Assigned constants contribute their difference from the initial value.
    /// Unknown effects and values outside the grid-resolution bound use GRID.
    pub fn from_generators(
        initial: NumericValue,
        generators: impl IntoIterator<Item = Option<i64>>,
    ) -> Self {
        let Some(offset) = initial.grid_units() else {
            return Self::GRID;
        };
        let mut step = 0;
        for generator in generators {
            let Some(generator) = generator else {
                return Self::GRID;
            };
            step = gcd(step, generator.unsigned_abs());
        }
        // With no nonzero generator, the coordinate is fixed. The full grid
        // still contains it and lets the ordinary abstraction machinery refine
        // its initially unbounded interval without a constant-coordinate path.
        if step == 0 {
            Self::GRID
        } else {
            Self { offset, step }
        }
    }

    /// A containing lattice closed under floating addition and rounding,
    /// without assuming any reachable magnitude bound. Only binary fractions
    /// aligned to the decimal grid can retain a nontrivial global quantum.
    pub fn arithmetic_closure(self) -> Self {
        // Only binary fractions keep a quantum under floating arithmetic. The
        // phase can disappear at large magnitudes, so include the offset when
        // finding the common power-of-two quantum.
        let divisor = gcd(self.offset.unsigned_abs(), self.step);
        if !divisor.is_multiple_of(FIVE_TO_TWELVE) {
            return Self::GRID;
        }
        let binary_units = divisor / FIVE_TO_TWELVE;
        assert!(binary_units > 0, "a lattice has a positive step");
        let step = (1u64 << binary_units.trailing_zeros()) * FIVE_TO_TWELVE;
        Self { offset: 0, step }
    }

    pub fn step(self) -> NumericValue {
        NumericValue::new(self.step as f64 / GRID_SCALE)
    }

    /// Greatest lattice value <= boundary (or < when `include` is false),
    /// followed by the least lattice value on the other side of the split.
    pub fn split_bounds(
        self,
        boundary: NumericValue,
        include: bool,
    ) -> Option<(NumericValue, NumericValue)> {
        if !boundary.is_finite() {
            return None;
        }
        let Some(units) = boundary.grid_units() else {
            // Here the grid is finer than f64: every representable value is
            // canonical, and neighbouring representable values close the split.
            return Some(if include {
                (boundary, NumericValue::new(boundary.value().next_up()))
            } else {
                (NumericValue::new(boundary.value().next_down()), boundary)
            });
        };
        let step = i64::try_from(self.step).expect("lattice generators fit signed grid units");
        let mut lower = self.offset + (units - self.offset).div_euclid(step) * step;
        if !include && lower == units {
            lower -= step;
        }
        let upper = lower + step;
        let lower = NumericValue::new(lower as f64 / GRID_SCALE);
        let upper = NumericValue::new(upper as f64 / GRID_SCALE);
        if lower == upper {
            // Near the resolution bound two neighbouring grid units can map
            // to the same f64. Advance to its neighbouring representable point.
            Some(if include {
                (boundary, NumericValue::new(boundary.value().next_up()))
            } else {
                (NumericValue::new(boundary.value().next_down()), boundary)
            })
        } else {
            Some((lower, upper))
        }
    }
}

fn gcd(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        (left, right) = (right, left % right);
    }
    left
}

#[cfg(test)]
mod tests {
    use super::*;
    fn nv(value: f64) -> NumericValue {
        NumericValue::new(value)
    }
    fn lattice(offset: f64, increments: &[f64]) -> ValueLattice {
        ValueLattice::from_generators(
            nv(offset),
            increments.iter().map(|&value| nv(value).grid_units()),
        )
    }

    #[test]
    fn integer_and_decimal_lattices_snap_strict_bounds() {
        assert_eq!(
            lattice(0.0, &[2.0, 3.0]).split_bounds(nv(3.0), true),
            Some((nv(3.0), nv(4.0)))
        );
        assert_eq!(
            lattice(0.0, &[0.2, 0.3]).split_bounds(nv(140.0), true),
            Some((nv(140.0), nv(140.1)))
        );
        assert_eq!(
            lattice(0.0, &[0.2, 0.3]).split_bounds(nv(140.0), false),
            Some((nv(139.9), nv(140.0)))
        );
    }

    #[test]
    fn offsets_negative_values_and_off_lattice_thresholds_are_exact() {
        let lattice = lattice(0.25, &[0.2, 0.3]);
        assert_eq!(
            lattice.split_bounds(nv(3.0), true),
            Some((nv(2.95), nv(3.05)))
        );
        assert_eq!(
            lattice.split_bounds(nv(-3.0), false),
            Some((nv(-3.05), nv(-2.95)))
        );
        assert_eq!(
            lattice.split_bounds(nv(-3.05), false),
            Some((nv(-3.15), nv(-3.05)))
        );
    }

    #[test]
    fn grid_splits_are_disjoint_near_f64_resolution() {
        for value in [4480.672703054073, 8192.0, 9007.0, -9007.0] {
            let boundary = nv(value);
            for include in [false, true] {
                let (lower, upper) = ValueLattice::GRID.split_bounds(boundary, include).unwrap();
                assert!(lower < upper, "{value} {include}");
                assert_eq!(lower <= boundary && boundary < upper, include);
                assert_eq!(lower < boundary && boundary <= upper, !include);
            }
        }
    }

    #[test]
    fn unknown_and_large_generators_use_the_full_grid() {
        assert_eq!(
            ValueLattice::from_generators(nv(0.0), [None]),
            ValueLattice::GRID
        );
        assert_eq!(lattice(1e8, &[1.0]), ValueLattice::GRID);
        let (lower, upper) = ValueLattice::GRID.split_bounds(nv(1e8), true).unwrap();
        assert!(lower < upper);
        assert_eq!(upper.value(), lower.value().next_up());
    }
}
