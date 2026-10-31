//! Transformations of a planning task into another task that heuristics
//! read instead of the original.
//!
//! - [`restriction`]: the restricted task (`--restrict-task`), in which every
//!   numeric comparison and metric reads state variables directly: derived
//!   arithmetic is folded into the comparisons, which the PDB and ICAPS 2026
//!   abstractions require.
pub mod restriction;
