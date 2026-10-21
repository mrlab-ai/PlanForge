//! Process-local memory headroom used by preprocessing and search, and the
//! deadlines of bounded operations.

use std::fmt;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use anyhow::Result;

static PADDING: OnceLock<Mutex<Option<Vec<u8>>>> = OnceLock::new();
static LIMIT_BYTES: OnceLock<u64> = OnceLock::new();

const MIB: u64 = 1024 * 1024;
const DEFAULT_PADDING_MIB: u64 = 75;
const MAX_HEADROOM_MIB: u64 = 128;

#[derive(Debug)]
pub struct DeadlineExceeded {
    operation: &'static str,
}

impl fmt::Display for DeadlineExceeded {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} deadline exceeded", self.operation)
    }
}

impl std::error::Error for DeadlineExceeded {}

pub fn is_deadline_exceeded(error: &anyhow::Error) -> bool {
    error.is::<DeadlineExceeded>()
}

/// A time limit no deadline can follow.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InvalidTimeLimit {
    /// Seconds that are negative or not a number.
    Seconds(f64),
    /// A limit past the end of the platform clock.
    TooLarge,
}

impl fmt::Display for InvalidTimeLimit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Seconds(seconds) => write!(
                formatter,
                "a time limit must be a non-negative number of seconds, got {seconds}"
            ),
            Self::TooLarge => {
                formatter.write_str("the time limit is too large for the platform clock")
            }
        }
    }
}

impl std::error::Error for InvalidTimeLimit {}

/// The time limit of `seconds`; unbounded (`None`) for infinity.
pub fn time_limit_from_secs(seconds: f64) -> Result<Option<Duration>, InvalidTimeLimit> {
    if seconds == f64::INFINITY {
        return Ok(None);
    }
    if seconds.is_nan() || seconds < 0.0 {
        return Err(InvalidTimeLimit::Seconds(seconds));
    }
    Duration::try_from_secs_f64(seconds)
        .map(Some)
        .map_err(|_| InvalidTimeLimit::TooLarge)
}

/// The tighter of two time limits, where `None` is unbounded.
pub fn min_time_limit(left: Option<Duration>, right: Option<Duration>) -> Option<Duration> {
    left.into_iter().chain(right).min()
}

/// The instant a bounded operation must stop at, or none.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Deadline(Option<Instant>);

impl Deadline {
    /// No deadline: it never passes.
    pub const NONE: Self = Self(None);

    pub fn at(instant: Instant) -> Self {
        Self(Some(instant))
    }

    /// `limit` after `start`; none when `limit` is `None`.
    pub fn after(start: Instant, limit: Option<Duration>) -> Result<Self, InvalidTimeLimit> {
        let Some(limit) = limit else {
            return Ok(Self::NONE);
        };
        start
            .checked_add(limit)
            .map(Self::at)
            .ok_or(InvalidTimeLimit::TooLarge)
    }

    /// `seconds` from now; none for infinity.
    pub fn from_now_secs(seconds: f64) -> Result<Self, InvalidTimeLimit> {
        Self::after(Instant::now(), time_limit_from_secs(seconds)?)
    }

    pub fn passed(self) -> bool {
        self.0.is_some_and(|deadline| Instant::now() >= deadline)
    }

    /// A [`DeadlineExceeded`] error for `operation` once the deadline passed.
    pub fn ensure(self, operation: &'static str) -> Result<()> {
        if self.passed() {
            return Err(DeadlineExceeded { operation }.into());
        }
        Ok(())
    }

    /// The time left, zero once passed; `None` without a deadline.
    pub fn remaining(self) -> Option<Duration> {
        self.0
            .map(|deadline| deadline.saturating_duration_since(Instant::now()))
    }

    /// The deadline that passes first.
    pub fn earliest(self, other: Self) -> Self {
        match (self.0, other.0) {
            (Some(left), Some(right)) => Self::at(left.min(right)),
            (Some(_), None) => self,
            (None, _) => other,
        }
    }
}

fn padding_cell() -> &'static Mutex<Option<Vec<u8>>> {
    PADDING.get_or_init(|| Mutex::new(None))
}

fn cli_release_threshold(limit: u64) -> u64 {
    let headroom = (MAX_HEADROOM_MIB * MIB).min(limit / 4);
    limit.saturating_sub(headroom)
}

fn bounded_padding_size(desired: u64, release_threshold: u64, current_rss: u64) -> u64 {
    desired.min(release_threshold.saturating_sub(current_rss))
}

fn parse_mib_env(name: &str) -> std::result::Result<Option<u64>, String> {
    match std::env::var(name) {
        Ok(value) => value
            .parse::<u64>()
            .map(Some)
            .map_err(|error| format!("invalid {name} value `{value}`: {error}")),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(format!("{name} must contain valid UTF-8")),
    }
}

fn mib_to_bytes(name: &str, value: u64) -> std::result::Result<u64, String> {
    value
        .checked_mul(MIB)
        .ok_or_else(|| format!("{name} value {value} MiB is too large"))
}

pub fn reserve_memory_padding(
    cli_max_memory_bytes: Option<u64>,
) -> std::result::Result<(), String> {
    let padding_mib = parse_mib_env("DA_MEMORY_PADDING_MB")?.unwrap_or(DEFAULT_PADDING_MIB);
    if padding_mib == 0 {
        return Ok(());
    }

    let env_limit_bytes = parse_mib_env("DA_MEMORY_LIMIT_MB")?
        .map(|mib| mib_to_bytes("DA_MEMORY_LIMIT_MB", mib))
        .transpose()?;
    let release_threshold =
        env_limit_bytes.or_else(|| cli_max_memory_bytes.map(cli_release_threshold));
    let Some(release_threshold) = release_threshold else {
        return Ok(());
    };

    let mut padding = padding_cell()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if padding.is_some() {
        return Ok(());
    }
    let current_rss = current_rss_bytes().unwrap_or(0);
    let desired_bytes = mib_to_bytes("DA_MEMORY_PADDING_MB", padding_mib)?;
    let padding_bytes = bounded_padding_size(desired_bytes, release_threshold, current_rss);
    let Ok(padding_bytes) = usize::try_from(padding_bytes) else {
        tracing::warn!("memory padding size does not fit usize; continuing without padding");
        return Ok(());
    };
    if padding_bytes == 0 {
        tracing::warn!(
            "memory usage already reaches the release threshold; continuing without padding"
        );
        return Ok(());
    }

    let mut reserve = Vec::new();
    if let Err(error) = reserve.try_reserve_exact(padding_bytes) {
        tracing::warn!("failed to reserve memory padding: {error}");
        return Ok(());
    }
    reserve.resize(padding_bytes, 0xA5);
    LIMIT_BYTES
        .set(release_threshold)
        .expect("memory padding release threshold must be initialized once");
    *padding = Some(reserve);
    Ok(())
}

pub fn padding_is_reserved() -> bool {
    padding_cell()
        .lock()
        .map(|padding| padding.is_some())
        .unwrap_or(false)
}

pub fn release_padding() {
    if let Ok(mut padding) = padding_cell().lock() {
        *padding = None;
    }
}

/// Releases process-local emergency memory after an allocation failure.
///
/// The allocator retries the failed allocation once. CEGAR observes the
/// missing padding at its next resource poll and finalizes its partial result.
pub fn release_padding_for_oom() -> bool {
    let Ok(mut padding) = padding_cell().lock() else {
        return false;
    };
    let was_reserved = padding.is_some();
    *padding = None;
    was_reserved
}

fn current_rss_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        status.lines().find_map(|line| {
            line.strip_prefix("VmRSS:").and_then(|value| {
                value
                    .split_whitespace()
                    .next()
                    .and_then(|kilobytes| kilobytes.parse::<u64>().ok())
                    .map(|kilobytes| kilobytes * 1024)
            })
        })
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Returns false after releasing the padding because the configured RSS
/// threshold was reached. An unavailable RSS reading is not a memory failure.
pub fn poll_and_release_if_exceeded() -> bool {
    let Some(&limit_bytes) = LIMIT_BYTES.get() else {
        return true;
    };
    if !padding_is_reserved() {
        return false;
    }
    if let Some(rss) = current_rss_bytes()
        && rss >= limit_bytes
    {
        // Routine: the padding exists so a memory-out is reported cleanly.
        // Warnings go to stderr, which experiment tooling reads as errors.
        tracing::info!(
            "memory limit threshold reached (RSS={} MiB, threshold={} MiB); releasing memory padding",
            rss / MIB,
            limit_bytes / MIB
        );
        release_padding();
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_limit_keeps_fixed_headroom() {
        assert_eq!(
            cli_release_threshold(8 * 1024 * MIB),
            8 * 1024 * MIB - 128 * MIB
        );
        assert_eq!(
            bounded_padding_size(75 * MIB, cli_release_threshold(8 * 1024 * MIB), 100 * MIB),
            75 * MIB
        );
    }

    #[test]
    fn low_limit_does_not_inflate_the_padding() {
        let threshold = cli_release_threshold(600 * MIB);
        assert_eq!(threshold, 472 * MIB);
        assert_eq!(
            bounded_padding_size(75 * MIB, threshold, 140 * MIB),
            75 * MIB
        );
    }

    #[test]
    fn tiny_limit_bounds_both_headroom_and_padding() {
        let threshold = cli_release_threshold(100 * MIB);
        assert_eq!(threshold, 75 * MIB);
        assert_eq!(
            bounded_padding_size(75 * MIB, threshold, 10 * MIB),
            65 * MIB
        );
    }

    #[test]
    fn time_limits_reject_negative_and_nan_seconds() {
        assert_eq!(time_limit_from_secs(f64::INFINITY), Ok(None));
        assert_eq!(time_limit_from_secs(0.0), Ok(Some(Duration::ZERO)));
        assert_eq!(
            time_limit_from_secs(1.5),
            Ok(Some(Duration::from_millis(1500)))
        );
        assert_eq!(
            time_limit_from_secs(-1.0),
            Err(InvalidTimeLimit::Seconds(-1.0))
        );
        assert_eq!(
            time_limit_from_secs(f64::NEG_INFINITY),
            Err(InvalidTimeLimit::Seconds(f64::NEG_INFINITY))
        );
        assert!(matches!(
            time_limit_from_secs(f64::NAN),
            Err(InvalidTimeLimit::Seconds(seconds)) if seconds.is_nan()
        ));
        assert_eq!(time_limit_from_secs(1e300), Err(InvalidTimeLimit::TooLarge));
    }

    #[test]
    fn deadlines_pass_ensure_and_combine() {
        let now = Instant::now();
        let passed = Deadline::at(now);
        let later = Deadline::after(now, Some(Duration::from_secs(3600))).unwrap();

        assert!(passed.passed());
        assert!(is_deadline_exceeded(&passed.ensure("test").unwrap_err()));
        assert_eq!(passed.remaining(), Some(Duration::ZERO));
        assert!(!later.passed());
        later.ensure("test").unwrap();
        assert!(later.remaining().is_some_and(|left| left > Duration::ZERO));

        assert!(!Deadline::NONE.passed());
        assert_eq!(Deadline::NONE.remaining(), None);
        assert_eq!(Deadline::after(now, None), Ok(Deadline::NONE));
        assert_eq!(Deadline::from_now_secs(f64::INFINITY), Ok(Deadline::NONE));
        assert!(Deadline::from_now_secs(-1.0).is_err());

        assert_eq!(passed.earliest(later), passed);
        assert_eq!(later.earliest(passed), passed);
        assert_eq!(later.earliest(Deadline::NONE), later);
        assert_eq!(Deadline::NONE.earliest(later), later);
        assert_eq!(Deadline::NONE.earliest(Deadline::NONE), Deadline::NONE);
    }

    #[test]
    fn the_tighter_time_limit_wins() {
        let second = Some(Duration::from_secs(1));
        assert_eq!(min_time_limit(second, None), second);
        assert_eq!(min_time_limit(None, second), second);
        assert_eq!(min_time_limit(Some(Duration::from_secs(2)), second), second);
        assert_eq!(min_time_limit(None, None), None);
    }
}
