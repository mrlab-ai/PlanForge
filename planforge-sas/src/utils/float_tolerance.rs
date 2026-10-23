use crate::numeric_task::NumericValue;

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

/// Round raw arithmetic to the numeric state grid. Only NumericValue's
/// constructor owns this operation.
#[inline]
pub(crate) fn canonicalize(value: f64) -> f64 {
    if value.is_infinite() {
        return value;
    }
    (value / ABS_EPSILON).round() * ABS_EPSILON
}

#[inline]
pub fn equal(lhs: NumericValue, rhs: NumericValue) -> bool {
    (lhs.value() - rhs.value()).abs()
        <= ABS_EPSILON.max(REL_EPSILON * lhs.value().abs().max(rhs.value().abs()))
}
