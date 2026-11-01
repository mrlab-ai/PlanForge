//! Transformations of a planning task into another task that heuristics
//! read instead of the original.
//!
//! - [`restriction`]: the restricted task (`--restrict-task`), in which every
//!   numeric comparison and metric reads state variables directly: derived
//!   arithmetic is folded into the comparisons, which the PDB and ICAPS 2026
//!   abstractions require.
//! - [`projection`]: the task projected to a pattern of variables, primarily
//!   the abstract task of a pattern database.
//!
//! A task with another goal, which goal-focused abstraction collections
//! refine for, needs no transformation: [`Task::with_goals`] shares
//! everything else.
//!
//! [`Task::with_goals`]: planforge_sas::numeric_task::Task::with_goals
pub mod projection;
pub mod restriction;
