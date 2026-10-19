use super::*;

pub(super) fn optimization_deadline(max_time: f64) -> Option<Instant> {
    (max_time.is_finite() && max_time.is_sign_positive())
        .then(|| Instant::now() + Duration::from_secs_f64(max_time))
}

pub(super) fn earliest_deadline(left: Option<Instant>, right: Option<Instant>) -> Option<Instant> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.min(right)),
        (Some(deadline), None) | (None, Some(deadline)) => Some(deadline),
        (None, None) => None,
    }
}
