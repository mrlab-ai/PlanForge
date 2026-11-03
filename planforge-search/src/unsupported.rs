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

/// `Err(UnsupportedTask)` with the formatted reason.
#[macro_export]
macro_rules! unsupported {
    ($($reason:tt)+) => {
        return Err($crate::unsupported::UnsupportedTask::new(format!($($reason)+)).into())
    };
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
    use crate::heuristic_features::HeuristicBuildError;
    use crate::task_transformations::TransformError;

    fn check(supported: bool) -> Result<(), TransformError> {
        crate::ensure_supported!(supported, "feature {} is missing", 7);
        Ok(())
    }

    #[test]
    fn an_unsupported_task_converts_into_each_error_that_has_the_kind() {
        assert!(check(true).is_ok());
        let error = check(false).unwrap_err();
        assert!(matches!(error, TransformError::Unsupported(_)));
        assert_eq!(error.to_string(), "unsupported task: feature 7 is missing");

        let error = HeuristicBuildError::from(super::UnsupportedTask::new("no axioms"));
        assert!(matches!(error, HeuristicBuildError::Unsupported(_)));
    }
}
