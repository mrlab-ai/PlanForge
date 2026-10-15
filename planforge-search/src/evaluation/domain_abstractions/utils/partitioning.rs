use planforge_sas::numeric_task::{INF_VALUE, NEG_INF_VALUE, NumericValue};
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

#[inline]
fn interval_contains_value_tolerant(iv: &Interval, value: NumericValue) -> bool {
    if value.value().is_nan() || iv.is_empty() {
        return false;
    }

    // Parity-over-quality: exact match at partition boundaries, matching
    // C++ numeric-FD's `get_partition_index`. Tolerant comparison drifted
    // boundary-aligned values into the wrong partition relative to C++.
    let lower_ok = if iv.lower == NEG_INF_VALUE {
        true
    } else if iv.lower_closed {
        value >= iv.lower
    } else {
        value > iv.lower
    };

    let upper_ok = if iv.upper == INF_VALUE {
        true
    } else if iv.upper_closed {
        value <= iv.upper
    } else {
        value < iv.upper
    };

    lower_ok && upper_ok
}

pub(crate) fn partition_for_value(partitions: &[Interval], value: NumericValue) -> Option<usize> {
    if partitions.len() <= 8 {
        return partitions
            .iter()
            .position(|iv| interval_contains_value_tolerant(iv, value));
    }

    let mut low = 0;
    let mut high = partitions.len();
    while low < high {
        let mid = low + (high - low) / 2;
        let iv = &partitions[mid];
        let below_lower = if iv.lower.value().is_finite() {
            let tolerance = float_tolerance::tolerance_nv(value, iv.lower);
            value.value() < iv.lower.value() - tolerance.value()
                || (value.value() - iv.lower.value()).abs() <= tolerance.value() && !iv.lower_closed
        } else {
            false
        };
        if below_lower {
            high = mid;
            continue;
        }

        let above_upper = if iv.upper.value().is_finite() {
            let tolerance = float_tolerance::tolerance_nv(value, iv.upper);
            value.value() > iv.upper.value() + tolerance.value()
                || (value.value() - iv.upper.value()).abs() <= tolerance.value() && !iv.upper_closed
        } else {
            false
        };
        if above_upper {
            low = mid + 1;
            continue;
        }

        let mut first = mid;
        while first > 0 && interval_contains_value_tolerant(&partitions[first - 1], value) {
            first -= 1;
        }
        return Some(first);
    }
    None
}

/// O(1) descriptor for a partition layout that is contiguous, uniform-width,
/// and surrounded by at most one unbounded interval on each side.
///
/// Examples that fit (sailing's typical CEGAR layout):
///   `(-∞, 0.5)  [0.5, 1.0)  [1.0, 1.5)  …  [N-0.5, N)  [N, +∞)`
///
/// When this pattern holds we can skip `partition_for_value`'s binary search
/// and compute the partition index directly. With ~200 partitions per
/// numeric var, the saving is ~log₂(200) tolerant float comparisons per
/// hash, per state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct EquispacedPartitioning {
    /// Lower bound of the first fully-finite partition.
    base: NumericValue,
    /// Width of one finite partition.
    step: NumericValue,
    /// Number of fully-finite partitions covering `[base, base+step*finite_count)`.
    finite_count: usize,
    /// Index of the first finite partition (0 if no lower unbounded tail, 1 otherwise).
    finite_offset: usize,
    /// Total number of partitions in the underlying slice.
    total: usize,
}

impl EquispacedPartitioning {
    pub(crate) fn detect(partitions: &[Interval]) -> Option<Self> {
        if partitions.len() < 2 {
            return None;
        }

        let mut finite_offset = 0;
        if !partitions[0].lower.value().is_finite() {
            finite_offset = 1;
        }

        let mut finite_end = partitions.len();
        if !partitions[finite_end - 1].upper.value().is_finite() {
            finite_end -= 1;
        }

        // Anything non-finite must be exactly one trailing or leading partition.
        if finite_end <= finite_offset {
            return None;
        }
        for iv in &partitions[finite_offset..finite_end] {
            if !iv.lower.value().is_finite() || !iv.upper.value().is_finite() {
                return None;
            }
        }

        let first = &partitions[finite_offset];
        let step = first.upper.value() - first.lower.value();
        if !(step.is_finite() && step > 0.0) {
            return None;
        }
        let base = first.lower;
        let lower_closed = first.lower_closed;
        let upper_closed = first.upper_closed;

        let step_tol = step.abs() * 1e-9 + 1e-12;
        let mut expected_lower = base;
        let mut count = 0;
        for iv in &partitions[finite_offset..finite_end] {
            if iv.lower_closed != lower_closed || iv.upper_closed != upper_closed {
                return None;
            }
            if (iv.lower.value() - expected_lower.value()).abs() > step_tol {
                return None;
            }
            if (iv.upper.value() - iv.lower.value() - step).abs() > step_tol {
                return None;
            }
            count += 1;
            expected_lower = NumericValue::new(iv.lower.value() + step);
        }
        if count < 2 {
            return None;
        }

        // Unbounded-lower partition (if present) must abut the finite region's
        // base; same for an unbounded-upper partition.
        if finite_offset == 1 {
            let head = &partitions[0];
            if head.lower != NEG_INF_VALUE || (head.upper.value() - base.value()).abs() > step_tol {
                return None;
            }
        }
        if finite_end < partitions.len() {
            let tail = &partitions[finite_end];
            let last_upper = base.value() + step * count as f64;
            if tail.upper != INF_VALUE || (tail.lower.value() - last_upper).abs() > step_tol {
                return None;
            }
        }

        Some(Self {
            base,
            step: NumericValue::new(step),
            finite_count: count,
            finite_offset,
            total: partitions.len(),
        })
    }

    /// O(1) partition lookup. Returns `None` when the value falls outside the
    /// covered range (which can only happen when there is no unbounded tail
    /// on the relevant side, e.g., a constant-pinned numeric var that drifted
    /// out of range — that is a real error, the same one `partition_for_value`
    /// reports by returning `None`).
    ///
    /// Currently unused: the cast-based body lookup does not respect
    /// per-interval closed/open boundary flags, so values that land exactly
    /// on a partition boundary can disagree with the tolerant
    /// `partition_for_value` and produce a different abstract hash than
    /// CEGAR's `compute_initial_state_hash_determined`. See the note on
    /// `NumericPartitions::equispaced` for context.
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn lookup(&self, value: NumericValue) -> Option<usize> {
        if !value.value().is_finite() {
            return None;
        }

        // Lower unbounded tail.
        if value < self.base {
            return if self.finite_offset == 1 {
                Some(0)
            } else {
                // Value is below the first finite partition with no head tail.
                // Tolerate values that round to `base` (matches the legacy
                // tolerant binary search).
                let tol = float_tolerance::tolerance_nv(value, self.base);
                ((self.base.value() - value.value()) <= tol.value()).then_some(0)
            };
        }

        let last_upper =
            NumericValue::new(self.base.value() + self.step.value() * self.finite_count as f64);
        let upper_tail_present = self.finite_offset + self.finite_count < self.total;
        if value >= last_upper {
            let tol = float_tolerance::tolerance_nv(value, last_upper);
            if !upper_tail_present {
                return ((value.value() - last_upper.value()) <= tol.value())
                    .then_some(self.total - 1);
            }
            // If value is *exactly* at the boundary, the lower-closed
            // convention puts it in the finite partition; otherwise the tail.
            if (value.value() - last_upper.value()).abs() <= tol.value() {
                return Some(self.finite_offset + self.finite_count - 1);
            }
            return Some(self.finite_offset + self.finite_count);
        }

        let raw = (value.value() - self.base.value()) / self.step.value();
        let mut idx = raw as usize;
        // Defensive clamp against tiny rounding edge cases at the right edge.
        if idx >= self.finite_count {
            idx = self.finite_count - 1;
        }
        Some(self.finite_offset + idx)
    }
}

#[allow(unused)]
pub(crate) fn partitions_for_interval(partitions: &[Interval], value: &Interval) -> Vec<usize> {
    partitions
        .iter()
        .enumerate()
        .filter_map(|(i, iv)| if iv.intersects(value) { Some(i) } else { None })
        .collect()
}
