//! The Cartesian abstractions of Schindler, Speck, and Helmert (ICAPS 2026),
//! reproduced from their artifact as the ICAPS baseline of the AAMAS27
//! experiments (`icaps26_cartesian(...)`).
//!
//! A frozen copy of the Cartesian construction as it stood when the AAMAS27
//! experiments ran, so the native Cartesian abstractions can change without
//! moving the baseline. It builds the native artifact type
//! [`CartesianAbstraction`]. Delete this module once AAMAS27 is accepted.

mod artifact;
mod concrete_state;
mod config;
mod distances;
mod finalize;
mod flaws;
mod generator;
mod refinement;
mod semantics;
mod split;
mod split_selection;

/// The concrete refinement root maps to an abstract dead end. The ICAPS26
/// reproduction reports it as an error, as the published code did.
#[derive(Debug)]
pub(crate) struct RefinementRootDeadEnd {
    detail: String,
}

impl RefinementRootDeadEnd {
    pub(crate) fn new(detail: String) -> Self {
        Self { detail }
    }
}

impl std::fmt::Display for RefinementRootDeadEnd {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "concrete refinement root maps to an abstract dead end: {}",
            self.detail
        )
    }
}

impl std::error::Error for RefinementRootDeadEnd {}

#[cfg(test)]
mod tests;
mod working;

pub use artifact::Icaps26SplitSelection;
use concrete_state::*;
pub use config::*;
use distances::*;
use finalize::*;
use flaws::*;
pub use generator::*;
use semantics::*;
use split::*;
use split_selection::*;
use working::*;

use super::cartesian::{
    CartesianAbstraction, CartesianRefinementHierarchy, NumericSplit, RefinementNode,
};
use super::common::abstraction::{AbstractionMetadata, AbstractionUse};
use super::common::cegar::SplitKind;

use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use ordered_float::NotNan;
use planforge_sas::axioms::{AxiomEvaluator, ComparisonOperator};
use planforge_sas::numeric_task::{
    AbstractNumericTask, AssignmentOperation, ExplicitFact, ExplicitValueIndex, NumericType,
    NumericValue, OperatorIndex, VariableIndex, ZERO_VALUE,
    metric_operator_cost_from_initial_values,
};
use planforge_sas::utils::float_tolerance;
use planforge_sas::utils::state_packer::StatePacker;
use tracing::info;

use crate::evaluation::validate_abstractable_goal;

use super::common::cegar::{CegarDriver, CegarIterationResult, CegarStopReason};
use super::common::coordinates::Coordinates;
use crate::evaluation::abstractions::common::cartesian_set::{
    CartesianSet, sorted_value_sets_overlap,
};
use crate::evaluation::abstractions::common::coordinates::AdditiveNumericView;
use crate::evaluation::abstractions::common::task::validate_abstraction_operator;
use crate::evaluation::abstractions::common::transition_system::{
    AbstractDistanceTable, AbstractOperatorSourceSets, AbstractTransition,
    AbstractTransitionSystem, CostedTransitionSystem, LabelSourceSet, backward_goal_distances,
};
use artifact::*;
use planforge_sas::utils::interval::Interval;

#[inline]
fn fact_is_held(fact: &ExplicitFact, packer: &StatePacker, buffer: &[u64]) -> bool {
    fact.is_held(
        planforge_sas::state_registry::ConcreteStateView::from_decoded(packer, buffer, &[]),
    )
}
