//! Search algorithms and heuristics over PlanForge SAS+ tasks.
//!
//! This crate is the final stage of the PDDL translation → SAS+ task → search
//! pipeline. It consumes [`planforge_sas::numeric_task::AbstractNumericTask`]
//! values and provides state-space search, heuristic evaluation, task
//! restriction, and abstraction machinery.
//!
//! There are three supported Rust extension paths:
//!
//! - implement [`evaluation::Heuristic`] for a new state evaluator;
//! - implement [`search::SearchAlgorithm`] for a new best-first priority
//!   policy;
//! - call [`heuristic_factory::register_external_heuristics`] to make external
//!   heuristic factories visible to the standard configuration grammar.
//!
//! The repository tutorials contain complete examples for all three paths. A
//! built-in heuristic specification can be parsed directly:
//!
//! ```
//! let spec = planforge_search::config::parse_heuristic_spec("blind()")
//!     .expect("blind is a built-in heuristic");
//! assert_eq!(spec.name, "blind");
//! ```

// Make `::planforge_search::…` paths resolve inside this crate too, so that
// `#[derive(ApplyOptions)]` (which emits absolute paths) works both here and
// in downstream crates that depend on `planforge_search`.
extern crate self as planforge_search;

pub mod causal_graph;
pub mod config;
pub mod evaluation;
pub mod heuristic_factory;
pub mod resource_limits;
pub mod search;
pub mod state_space;
pub mod successor_generator;
pub mod task_restriction;
