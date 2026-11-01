//! PlanForge's search core over [`planforge_sas::numeric_task::Task`]s:
//! state-space search, the heuristic interface and registry, successor
//! generation, the exact state space and task transformations. Blind search
//! is its one heuristic; the others live in `planforge_heuristics`, which
//! depends on this crate and on nothing above it.
//!
//! There are three supported Rust extension paths:
//!
//! - implement [`evaluation::Heuristic`] for a new state evaluator;
//! - implement [`search::best_first::NodeOrdering`] for a new best-first
//!   node ordering;
//! - register entries with [`heuristic_factory::HEURISTICS`] to make external
//!   heuristic factories visible to the standard configuration grammar.
//!
//! The repository tutorials contain complete examples for all three paths. A
//! built-in heuristic specification can be parsed directly:
//!
//! ```
//! let spec = planforge_search::config::parse_call("blind()")
//!     .expect("blind is a built-in heuristic");
//! assert_eq!(spec.name, "blind");
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
