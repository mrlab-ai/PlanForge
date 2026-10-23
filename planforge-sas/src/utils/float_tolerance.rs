use crate::numeric_task::NumericValue;

pub const ABS_EPSILON: f64 = 1e-12;
/// Exact decimal scale of the canonical grid.
pub const GRID_SCALE: f64 = 1e12;
/// Above this magnitude, adjacent grid units cannot all be represented in f64.
pub const GRID_BOUND: f64 = (1u64 << 53) as f64 / GRID_SCALE;
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

/// Round raw arithmetic to the numeric state grid. Only NumericValue's
/// constructor owns this operation.
#[inline]
pub(crate) fn canonicalize(value: f64) -> f64 {
    if value.abs() >= GRID_BOUND {
        value
    } else {
        let scaled = value * GRID_SCALE;
        let mut units = scaled.round();
        // Multiplication can round a value just below a half-grid point onto
        // that point. Resolve those ties using the exact product residual;
        // otherwise canonicalizing the resulting f64 a second time can move it.
        if (scaled - units).abs() == 0.5
            && value.mul_add(GRID_SCALE, -scaled) * (units - scaled) < 0.0
        {
            units -= scaled.signum();
        }
        units / GRID_SCALE
    }
}

#[inline]
pub fn equal(lhs: NumericValue, rhs: NumericValue) -> bool {
    (lhs.value() - rhs.value()).abs()
        <= ABS_EPSILON.max(REL_EPSILON * lhs.value().abs().max(rhs.value().abs()))
}
