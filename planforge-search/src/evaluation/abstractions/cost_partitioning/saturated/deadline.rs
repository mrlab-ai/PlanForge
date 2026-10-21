use super::*;

/// The deadline `seconds` from now, for the time limit `option` that
/// [`ScpConfig::validate`] accepted.
pub(super) fn deadline_from_now(option: &str, seconds: f64) -> Result<Deadline, EvaluationError> {
    Deadline::from_now_secs(seconds)
        .map_err(|error| EvaluationError::ComputationFailed(format!("scp: {option}: {error}")))
}
