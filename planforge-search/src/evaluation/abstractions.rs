//! Abstraction heuristics for numeric planning.
//!
//! The representations -- Cartesian abstractions, domain abstractions and
//! numeric pattern databases -- live in their own modules; what they share,
//! such as the CEGAR loop and its flaws, lives in [`common`]. Cost
//! partitioning and collection generation build on them.

pub mod cartesian;
pub mod collections;
pub mod common;
pub mod component;
pub mod cost_partitioning;
pub mod domain;
pub mod pdb;
