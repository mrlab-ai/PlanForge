//! What Cartesian abstractions, domain abstractions and pattern databases share.
//!
//! That is the CEGAR loop and its flaws, the coordinates regression works
//! over, concrete states and Cartesian sets of them, explicit abstract
//! transition systems with their goal distances, and the task views
//! abstractions are built for.

pub mod abstraction;
pub mod cartesian_set;
pub mod cegar;
pub(crate) mod coordinates;
pub(crate) mod operator_support;
pub(crate) mod regions;
pub mod transition_system;

/// How many items pass between two checks of a deadline in a loop over
/// abstract states, operators or transitions: often enough to stop soon after
/// the deadline, rarely enough to cost nothing.
pub(crate) const DEADLINE_POLL_STRIDE: usize = 1024;
