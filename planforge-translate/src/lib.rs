//! Numeric PDDL to SAS+ translation.
//!
//! [`translate_to_task`], [`translate_to_sas_to_path`],
//! [`translate_to_sas_to_path_fast`] and [`translate_to_sas_string`] are the
//! whole public surface. The pipeline stages behind them are private: they only
//! run in this order.
//!
//! This is the first stage of PlanForge's PDDL translation → SAS+ task → search
//! pipeline. The usual in-memory entry point returns the task consumed by
//! `planforge-search`:
//!
//! ```no_run
//! # fn main() -> Result<(), planforge_translate::TranslateError> {
//! let task = planforge_translate::translate_to_task("domain.pddl", "problem.pddl")?;
//! let _ = task;
//! # Ok(())
//! # }
//! ```

mod api;
mod axiom_rules;
mod build_model;
mod constraints;
mod exploration_rules;
mod fact_groups;
mod greedy_join;
mod grounding;
mod instantiate;
mod invariant_finder;
mod invariants;
mod normalize;
mod numeric_axiom_rules;
mod options;
mod pddl;
mod pddl_parser;
mod pddl_to_prolog;
mod preprocess;
mod sas_tasks;
mod simplify;
mod split_rules;
mod symbols;
mod tools;
mod translate;

pub use api::{
    TranslateError, translate_to_sas_string, translate_to_sas_to_path,
    translate_to_sas_to_path_fast, translate_to_sas_to_path_with_limits, translate_to_task,
    translate_to_task_with_limits,
};
pub use grounding::{
    DEFAULT_MAX_GROUND_ACTIONS, DEFAULT_MAX_GROUND_ATOMS, DEFAULT_MAX_GROUNDING_MEMORY,
    GroundingLimitError, GroundingLimitKind, GroundingLimits,
};
pub use options::LayerStrategy;
pub use pddl_parser::lisp_parser::{ParseError, ParseErrorKind};
