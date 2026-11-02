//! What the heuristics of this crate need of the task, as
//! [`TaskCheck`]s, and of the build: an unrestricted CPLEX for the LP-based
//! ones.

use planforge_sas::numeric_task::Task;
use planforge_search::heuristic_factory::{HeuristicBuildError, TaskCheck};
use planforge_search::task_transformations::projection::validate_abstractable_goal;
use planforge_search::task_transformations::restriction::validate_restricted_task;

/// The task is restricted (`--transform restrict`): every numeric
/// comparison and the metric read state variables directly. A task that is
/// not is a usage error, since the restriction would restrict it.
pub const RESTRICTED: TaskCheck = TaskCheck {
    name: "restricted",
    check: |task: &Task| validate_restricted_task(task).map_err(HeuristicBuildError::Failure),
};

/// The goal is a conjunction of facts some operator can bring about, which
/// an abstraction can express; a task whose goal is not is unsupported.
pub const ABSTRACTABLE_GOAL: TaskCheck = TaskCheck {
    name: "abstractable_goal",
    check: |task: &Task| validate_abstractable_goal(task).map_err(HeuristicBuildError::Unsupported),
};

/// The preflight check of an LP-based heuristic: this build has an
/// unrestricted CPLEX.
pub fn require_cplex() -> std::io::Result<()> {
    #[cfg(feature = "cplex")]
    {
        planforge_cplex::assert_unrestricted_license().map_err(|error| {
            std::io::Error::other(format!(
                "an unrestricted CPLEX installation is required: {error}"
            ))
        })
    }
    #[cfg(not(feature = "cplex"))]
    {
        Err(std::io::Error::other(
            "the requested LP-backed heuristic requires unrestricted CPLEX, which is not compiled into this build; rebuild with `--features cplex` and set CPLEX_ROOT",
        ))
    }
}
