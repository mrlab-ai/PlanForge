use std::time::{Duration, Instant};

use super::*;

/// The deadline `limit` from now, for the time limit `option`.
pub(super) fn deadline_from_now(
    option: &str,
    limit: Option<Duration>,
) -> Result<Deadline, EvaluationError> {
    Deadline::after(Instant::now(), limit)
        .map_err(|error| EvaluationError::ComputationFailed(format!("scp: {option}: {error}")))
}
