use planforge_sas::numeric_task::NumericValue;
use planforge_sas::utils::float_tolerance;
use planforge_sas::utils::interval::Interval;

pub(crate) fn fmt_interval(iv: Interval) -> String {
    let l = if iv.lower_closed { '[' } else { '(' };
    let r = if iv.upper_closed { ']' } else { ')' };
    let lo = fmt_f64_compact(iv.lower.value());
    let hi = fmt_f64_compact(iv.upper.value());
    format!("{l}{lo}, {hi}{r}")
}

pub(crate) fn fmt_f64_compact(v: f64) -> String {
    if v.is_nan() {
        return "NaN".to_string();
    }
    if v.is_infinite() {
        return if v.is_sign_negative() {
            "-inf".to_string()
        } else {
            "inf".to_string()
        };
    }
    let mut s = format!("{v}");
    let is_scientific = s.contains('e') || s.contains('E');
    if !is_scientific && let Some(dot) = s.find('.') {
        let (head, tail) = s.split_at(dot + 1);
        let trimmed_tail = tail.trim_end_matches('0');
        s = if trimmed_tail.is_empty() {
            head.trim_end_matches('.').to_string()
        } else {
            format!("{head}{trimmed_tail}")
        };
    }
    if s == "-0" { "0".to_string() } else { s }
}

/// The partition of `partitions` that contains `value`.
///
/// This is the one value-to-partition mapping of domain abstractions: CEGAR's
/// flaw search and splits, the initial-state hash of the distance tables, and
/// the heuristic's state lookup all go through it, so a value is assigned to
/// the same abstract value everywhere.
///
/// The mapping is exact. Numeric values are not integral in general (rational
/// constants, metric costs, derived linear views), so instead of comparing
/// with a tolerance the value is first rounded to the canonical
/// `float_tolerance::ABS_EPSILON` lattice that states and partition
/// boundaries already live on (`split_at` and the state registry canonicalize
/// both), and then compared exactly against the boundaries, honoring their
/// open/closed flags. The result is the same whatever the number of
/// partitions.
///
/// `partitions` are the disjoint, ascending intervals of one numeric variable.
/// `None` means no partition contains the value, which happens only for a
/// value outside a bounded layout (a constant that changed) or NaN.
pub(crate) fn partition_for_value(partitions: &[Interval], value: NumericValue) -> Option<usize> {
    let value = float_tolerance::canonicalize(value);
    if value.is_nan() {
        return None;
    }
    // The first partition that does not lie entirely below the value.
    let index = partitions
        .partition_point(|iv| iv.upper < value || (iv.upper == value && !iv.upper_closed));
    let found = partitions
        .get(index)
        .filter(|iv| iv.contains(value))
        .map(|_| index);
    debug_assert_eq!(
        found,
        partitions.iter().position(|iv| iv.contains(value)),
        "partitions {partitions:?} are not disjoint and ascending"
    );
    found
}

pub(crate) fn partitions_for_interval(partitions: &[Interval], value: &Interval) -> Vec<usize> {
    partitions
        .iter()
        .enumerate()
        .filter_map(|(i, iv)| if iv.intersects(value) { Some(i) } else { None })
        .collect()
}

#[cfg(test)]
mod tests;
