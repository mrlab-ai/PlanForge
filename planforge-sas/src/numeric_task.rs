//! The immutable root planning task and its four variable-id spaces.
//!
//! Task propositional ids index [`NumericRootTask::variables`]. Numeric-condition
//! ids use that same index range but are tagged as [`FactNamespace::Condition`]
//! so callers cannot confuse a comparison verdict with an ordinary proposition.
//! Task numeric ids independently index [`NumericRootTask::numeric_variables`].
//! Domain abstractions have a fourth, private combined space, tagged
//! [`FactNamespace::NumericVariable`], whose values are partition ids rather
//! than task-domain values.
//!
//! Axiom layers are stratified across those spaces: derived numeric assignment
//! axioms run first, all numeric comparisons occupy the next layer, and
//! propositional axioms occupy higher layers. [`NumericRootTask::new`] checks
//! these representation and layer invariants once before publishing a task.

use crate::axioms::{AssignmentAxiom, AxiomEvaluator, ComparisonAxiom, PropositionalAxiom};
use crate::numeric_conditions::{
    ConditionValue, NumericConditionError, NumericConditions, assignment_axiom_lookup,
};
use crate::numeric_parser::parse_numeric_sas_output;
use crate::utils::errors::AssignmentAxiomError;
use crate::utils::linear_effects::{
    LinearNumericEffect, LinearizationError, linearize_numeric_var,
    linearize_operator_assignment_effects,
};
use crate::utils::state_packer::StatePacker;
use std::{collections::HashSet, fmt, sync::Arc};

mod task_api;
pub(crate) mod value_types;

pub use task_api::*;
pub use value_types::*;
