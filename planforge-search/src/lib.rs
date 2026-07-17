// Make `::planforge_search::…` paths resolve inside this crate too, so that
// `#[derive(ApplyOptions)]` (which emits absolute paths) works both here and
// in downstream crates that depend on `planforge_search`.
extern crate self as planforge_search;

pub mod causal_graph;
pub mod config;
pub mod evaluation;
pub mod open_lists;
pub mod resource_limits;
pub mod search;
pub mod successor_generator;
pub mod task_restriction;
