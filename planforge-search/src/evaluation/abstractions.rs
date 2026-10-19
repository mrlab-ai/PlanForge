//! Abstraction heuristics for numeric planning.
//!
//! The representations -- Cartesian abstractions, domain abstractions and
//! numeric pattern databases -- live in their own modules. CEGAR, cost
//! partitioning and collection generation build on them.

pub mod cartesian;
pub mod cartesian_set;
pub mod cegar;
pub mod collections;
pub mod component;
pub mod cost_partitioning;
pub mod domain;
pub mod pdb;
pub(crate) mod task;
