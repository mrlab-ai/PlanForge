//! Task input a heuristic does not support, as opposed to a failure.
//!
//! A run on such a task ends with its own exit code, so that experiments can
//! tell "this configuration cannot handle the task" from a bug.

use std::fmt;

#[derive(Debug)]
pub struct UnsupportedTask {
    reason: String,
}

impl UnsupportedTask {
    pub fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }
}

impl fmt::Display for UnsupportedTask {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "unsupported task: {}", self.reason)
    }
}

impl std::error::Error for UnsupportedTask {}

/// Whether `error` or one of its causes is an [`UnsupportedTask`].
pub fn is_unsupported_task(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| cause.is::<UnsupportedTask>())
}

/// `Err(UnsupportedTask)` with the formatted reason unless `condition` holds.
#[macro_export]
macro_rules! ensure_supported {
    ($condition:expr, $($reason:tt)+) => {
        if !$condition {
            return Err($crate::unsupported::UnsupportedTask::new(format!($($reason)+)).into());
        }
    };
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};

    use super::*;

    fn check(supported: bool) -> Result<()> {
        crate::ensure_supported!(supported, "feature {} is missing", 7);
        Ok(())
    }

    #[test]
    fn an_unsupported_task_is_recognized_through_its_context() {
        assert!(check(true).is_ok());
        let error = check(false).context("building source 0").unwrap_err();

        assert!(is_unsupported_task(&error));
        assert_eq!(
            format!("{error:#}"),
            "building source 0: unsupported task: feature 7 is missing"
        );
        assert!(!is_unsupported_task(&anyhow::anyhow!("a bug")));
    }

    #[test]
    fn the_heuristic_factory_reports_an_unsupported_task_as_such() {
        use crate::heuristic_factory::HeuristicBuildError;
        let unsupported =
            HeuristicBuildError::from_source(check(false).unwrap_err(), "source 0".into());
        let failure = HeuristicBuildError::from_source(anyhow::anyhow!("a bug"), "source 0".into());

        assert_eq!(
            unsupported.into_io_error().kind(),
            std::io::ErrorKind::Unsupported
        );
        assert_eq!(failure.into_io_error().kind(), std::io::ErrorKind::Other);
    }
}
