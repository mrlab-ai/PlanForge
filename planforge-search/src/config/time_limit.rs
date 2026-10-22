use super::{ConfigValue, FromOptionValue};
use std::time::Duration;

/// A time budget in seconds; positive infinity means unlimited.
/// Keeping seconds also preserves budgets beyond the platform clock range for
/// algorithms that compare elapsed seconds directly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeLimit(pub f64);

impl Default for TimeLimit {
    fn default() -> Self {
        Self::UNLIMITED
    }
}

impl TimeLimit {
    pub const UNLIMITED: Self = Self(f64::INFINITY);
    pub const fn seconds(self) -> f64 {
        self.0
    }
    pub fn duration(self) -> Result<Option<Duration>, String> {
        crate::resource_limits::time_limit_from_secs(self.0).map_err(|error| error.to_string())
    }
    fn from_seconds(seconds: f64) -> Result<Self, String> {
        if seconds.is_nan() || seconds < 0.0 {
            return Err(format!("time limit must be non-negative, got {seconds}"));
        }
        Ok(Self(seconds))
    }
    pub(crate) fn positive_bounded(value: &ConfigValue) -> Result<Self, String> {
        let limit = Self::from_option_value(value)?;
        if !limit.0.is_finite() || limit.duration()? == Some(Duration::ZERO) {
            return Err(format!(
                "time limit must be finite and at least one nanosecond, got {}",
                limit.0
            ));
        }
        // Construction budgets become deadlines before any parent deadline is
        // applied, so a clock overflow is unconditionally a configuration error.
        crate::resource_limits::Deadline::after(std::time::Instant::now(), limit.duration()?)
            .map_err(|error| error.to_string())?;
        Ok(limit)
    }
}
impl From<Option<Duration>> for TimeLimit {
    fn from(value: Option<Duration>) -> Self {
        Self(value.map_or(f64::INFINITY, |duration| duration.as_secs_f64()))
    }
}
impl From<Duration> for TimeLimit {
    fn from(value: Duration) -> Self {
        Self(value.as_secs_f64())
    }
}
impl FromOptionValue for TimeLimit {
    fn from_option_value(value: &ConfigValue) -> Result<Self, crate::config::ConfigError> {
        Ok(Self::from_seconds(f64::from_option_value(value)?)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_limits_preserve_seconds_and_convert_only_when_a_clock_is_needed() {
        let parse = |text: &str| TimeLimit::from_option_value(&ConfigValue::Atom(text.into()));
        assert_eq!(parse("infinity").unwrap().duration().unwrap(), None);
        assert_eq!(
            parse("2.5").unwrap().duration().unwrap(),
            Some(Duration::from_millis(2500))
        );
        let huge = parse("1e300").unwrap();
        assert_eq!(huge.seconds(), 1e300);
        assert!(huge.duration().is_err());
        assert!(parse("NaN").is_err());
        assert!(parse("-1").is_err());
        assert_eq!(
            parse("1e-100").unwrap().duration().unwrap(),
            Some(Duration::ZERO)
        );
    }
}
