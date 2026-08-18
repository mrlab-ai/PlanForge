//! SAS+ task representation and execution semantics for PlanForge.
//!
//! This crate is the middle of the PDDL translation → SAS+ task → search
//! pipeline. `planforge-translate` produces [`numeric_task::NumericRootTask`]
//! values; `planforge-search` consumes the [`numeric_task::AbstractNumericTask`]
//! interface, while this crate owns parsing, axioms, state transitions, state
//! registration, and exact plan replay.
//!
//! Facts carry their semantic namespace explicitly:
//!
//! ```
//! use planforge_sas::numeric_task::ExplicitFact;
//!
//! let fact = ExplicitFact::propositional(2, 1);
//! assert_eq!((fact.var(), fact.value()), (2, 1));
//! ```

pub mod axioms;
pub mod default_value_axioms;
pub mod numeric_conditions;
pub mod numeric_parser;
pub mod numeric_task;
pub mod plan_verification;
pub mod sas_format;
pub mod sas_writer;
pub mod state_registry;
pub mod utils;

#[cfg(test)]
pub(crate) mod simultaneous_effects_tests;
#[cfg(test)]
pub(crate) mod tests;
