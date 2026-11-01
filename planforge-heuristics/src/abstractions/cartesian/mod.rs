//! Numeric Cartesian abstractions refined by concrete counterexamples.
//!
//! Unlike the factorized domain abstraction, splitting one Cartesian state
//! adds exactly one state. Every abstract transition is a may-transition of a
//! grounded concrete operator. CEGAR replays a deterministic optimal abstract
//! trace. The default refines its first witnessed flaw; the explicit
//! whole-plan mode ranks all sound witnesses on an acyclic adaptive trace.
//! Only a successfully replayed concrete plan may set `solved_by_self`.

mod abstraction;
mod collection;
mod config;
mod distances;
mod finalize;
mod generator;
mod hierarchy;
mod initial_splits;
mod refinement;
mod semantics;
#[cfg(test)]
mod tests;
mod view;
mod working;

pub use abstraction::*;
pub use collection::*;
pub use config::*;
pub use generator::*;
pub use hierarchy::*;
