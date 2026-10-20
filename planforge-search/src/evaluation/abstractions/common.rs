//! What Cartesian abstractions, domain abstractions and pattern databases
//! share: the CEGAR loop and its flaws, the coordinates regression works
//! over, concrete states and Cartesian sets of them, explicit abstract
//! transition systems with their goal distances, and the task views
//! abstractions are built for.

pub mod cartesian_set;
pub mod cegar;
pub(crate) mod concrete_state;
pub(crate) mod coordinates;
pub(crate) mod regions;
pub(crate) mod task;
pub mod transition_system;
