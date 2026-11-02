//! Abstraction heuristics for numeric planning.
//!
//! The representations -- Cartesian abstractions, domain abstractions and
//! numeric pattern databases -- live in their own modules; what they share,
//! such as the CEGAR loop and its flaws, lives in [`common`]. Cost
//! partitioning and collection generation build on them; [`sources`] builds
//! the abstractions a heuristic names, and [`features`] registers the
//! abstraction heuristics.

pub mod cartesian;
pub mod collections;
pub mod common;
pub mod component;
pub mod cost_partitioning;
pub mod domain;
pub mod features;
pub mod icaps26;
pub mod pdb;
pub mod sources;

use planforge_search::heuristic_factory::HeuristicBuildError;
use planforge_search::resource_limits::DeadlineExceeded;
use planforge_search::task_transformations::TransformError;
use planforge_search::unsupported::UnsupportedTask;

/// The error the abstraction machinery's internal `anyhow` chain `error`
/// ends a task transformation in: unsupported when a cause in the chain
/// says so, a failure otherwise. The message is the whole chain.
pub(crate) fn transform_error(error: anyhow::Error) -> TransformError {
    if error.chain().any(|cause| cause.is::<UnsupportedTask>()) {
        TransformError::Unsupported(format!("{error:#}"))
    } else {
        TransformError::Failed(format!("{error:#}"))
    }
}

/// The error the abstraction machinery's internal `anyhow` chain `error`
/// ends a construction in: unsupported or timed out when a cause in the
/// chain says so, a failure otherwise. The message is the whole chain.
pub(crate) fn build_error(error: anyhow::Error) -> HeuristicBuildError {
    let message = format!("{error:#}");
    for cause in error.chain() {
        if cause.is::<UnsupportedTask>()
            || matches!(
                cause.downcast_ref(),
                Some(HeuristicBuildError::Unsupported(_))
            )
        {
            return HeuristicBuildError::Unsupported(message);
        }
        if cause.is::<DeadlineExceeded>()
            || matches!(cause.downcast_ref(), Some(HeuristicBuildError::Timeout(_)))
        {
            return HeuristicBuildError::Timeout(message);
        }
    }
    HeuristicBuildError::Failure(message)
}
