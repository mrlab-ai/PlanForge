use std::time::Duration;

use super::{ConfigValue, FromOptionValue};

/// A time budget in seconds; `infinity` means unlimited.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TimeLimit(pub Option<Duration>);

impl TimeLimit {
    pub const UNLIMITED: Self = Self(None);

    fn from_seconds(seconds: f64) -> Result<Self, String> {
        crate::resource_limits::time_limit_from_secs(seconds)
            .map(Self)
            .map_err(|error| error.to_string())
    }

    /// A budget that must be explicitly bounded and positive. Validate seconds
    /// before rounding to Duration, preserving sub-nanosecond inputs.
    pub(crate) fn positive_bounded(value: &ConfigValue) -> Result<Self, String> {
        let seconds = f64::from_option_value(value)?;
        if !seconds.is_finite() || seconds <= 0.0 {
            return Err(format!("time limit must be finite and > 0, got {seconds}"));
        }
        Self::from_seconds(seconds)
    }

    pub fn seconds(self) -> f64 {
        self.0
            .map_or(f64::INFINITY, |duration| duration.as_secs_f64())
    }
}

impl From<Option<Duration>> for TimeLimit {
    fn from(duration: Option<Duration>) -> Self {
        Self(duration)
    }
}

impl FromOptionValue for TimeLimit {
    fn from_option_value(value: &ConfigValue) -> Result<Self, String> {
        Self::from_seconds(f64::from_option_value(value)?)
    }
}
