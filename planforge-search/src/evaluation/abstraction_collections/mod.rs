//! Heuristics and cost-partitioning machinery over collections of abstractions.
//!
//! Backends such as domain abstractions, Cartesian abstractions, and pattern
//! databases live in sibling modules. This module owns only the algorithms
//! that combine them.

pub mod canonical_heuristic;
pub mod component;
pub mod cost_partitioning;
pub mod max_heuristic;
pub mod portfolio;
pub mod saturated_cost_partitioning_online_heuristic;
