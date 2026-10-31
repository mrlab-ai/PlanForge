//! Transformations of a planning task into another task that heuristics
//! read instead of the original.
//!
//! - [`restriction`]: the restricted task (`--restrict-task`), in which every
//!   numeric comparison and metric reads state variables directly: derived
//!   arithmetic is folded into the comparisons, which the PDB and ICAPS 2026
//!   abstractions require.
//! - [`projection`]: the task projected to a pattern of variables, primarily
//!   the abstract task of a pattern database.
//! - `single_goal`: the task with one of its goal facts as the whole goal,
//!   which goal-focused abstraction collections refine for.
pub mod projection;
pub mod restriction;
pub(crate) mod single_goal;
