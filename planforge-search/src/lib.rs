//! PlanForge's search core over [`planforge_sas::numeric_task::Task`]s:
//! state-space search, the heuristic interface, successor generation, task
//! transformations, and the registries through which the `--search`
//! configuration language names features. Blind search is its one
//! heuristic; the others live in `planforge_heuristics`, which depends on
//! this crate and on nothing above it.
//!
//! # Where to start
//!
//! Each extension point is a trait; a feature the configuration language
//! names is that trait plus a typed configuration, registered in the
//! feature's registry (a binary registers through `planforge::Extensions`).
//!
//! | To add | Implement | Registry |
//! |---|---|---|
//! | a heuristic | [`evaluation::Heuristic`], [`heuristic_factory::HeuristicFeature`] | [`heuristic_factory::HEURISTICS`] |
//! | a search algorithm | [`search::SearchFeature`], composing [`search::SearchSpace`] and [`search::SearchProgress`] | [`search::SEARCH_ALGORITHMS`] |
//! | a best-first node ordering | [`search::best_first::NodeOrdering`] | -- (a search algorithm instantiates it) |
//! | an open list | [`search::best_first::OpenList`], [`search::best_first::OpenListFeature`] | [`search::best_first::OPEN_LISTS`] |
//! | a successor generator | [`successor_generator::SuccessorGenerator`], [`successor_generator::SuccessorGeneratorFeature`] | [`successor_generator::SUCCESSOR_GENERATORS`] |
//! | a task transformation | [`task_transformations::TransformationFeature`] | [`task_transformations::TRANSFORMATIONS`] |
//! | a requirement on the task | [`heuristic_factory::TaskCheck`] in [`heuristic_factory::TaskRequirements`] | -- (a heuristic feature returns it) |
//!
//! The tutorials in the repository show each path end to end.
//!
//! # Example
//!
//! The goal-count heuristic, run with A*:
//!
//! ```
//! use std::sync::Arc;
//!
//! use planforge_sas::numeric_task::Task;
//! use planforge_search::evaluation::{EvaluationError, EvaluationState, Heuristic};
//! use planforge_search::search::SearchOutcome;
//! use planforge_search::search::best_first::{AStarSearch, SearchParams};
//!
//! /// The number of unsatisfied goal facts.
//! struct GoalCount;
//!
//! impl Heuristic for GoalCount {
//!     fn evaluate(&mut self, state: &EvaluationState<'_, '_>) -> Result<f64, EvaluationError> {
//!         let view = state.state_registry().view(state.state());
//!         let unsatisfied = state.task().goals().iter().filter(|goal| !goal.is_held(view));
//!         Ok(unsatisfied.count() as f64)
//!     }
//!
//!     fn name(&self) -> &str {
//!         "goal_count"
//!     }
//! }
//!
//! let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../tests/assets/numeric_sas/example5.sas");
//! let task = Arc::new(Task::from_file(path));
//! let result = AStarSearch::new(task, SearchParams::new(Box::new(GoalCount)))
//!     .run()
//!     .expect("goal count evaluates every state");
//! assert!(matches!(result.outcome, SearchOutcome::Solved { cost: 12.0, .. }));
//! ```

// Make `::planforge_search::…` paths resolve inside this crate too, so that
// `#[derive(ApplyOptions)]` (which emits absolute paths) works both here and
// in downstream crates that depend on `planforge_search`.
extern crate self as planforge_search;

pub mod config;
pub mod evaluation;
pub mod graph;
pub mod heuristic_factory;
pub mod registry;
pub mod resource_limits;
pub mod search;
pub mod state_space;
pub mod successor_generator;
pub mod task_transformations;
#[cfg(any(test, feature = "test-tasks"))]
pub mod test_tasks;
pub mod unsupported;
