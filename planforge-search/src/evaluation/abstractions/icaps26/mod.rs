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
    CartesianAbstraction, CartesianRefinementHierarchy, CartesianTransitions, NumericSplit,
    RefinementNode,
};
use super::common::abstraction::{AbstractionCore, AbstractionMetadata, AbstractionUse};
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
    NumericValue, Operator, OperatorIndex, VariableIndex, ZERO_VALUE, evaluate_metric_from_values,
    propagate_assignment_axiom_values,
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
    AbstractOperatorSourceSets, AbstractTransition, AbstractTransitionSystem,
    CostedTransitionSystem, LabelSourceSet, backward_goal_distances,
};
use artifact::*;
use planforge_sas::utils::interval::Interval;

#[inline]
fn fact_is_held(fact: &ExplicitFact, packer: &StatePacker, buffer: &[u64]) -> bool {
    fact.is_held(
        planforge_sas::state_registry::ConcreteStateView::from_decoded(packer, buffer, &[]),
    )
}

/// The metric cost of `operator` as the ICAPS 2026 artifact computed it: its
/// assignment effects applied to the initial values unconditionally, the
/// last write to a variable winning. Kept verbatim so the frozen
/// configurations reproduce their published results.
fn metric_operator_cost_from_initial_values<T: AbstractNumericTask + ?Sized>(
    task: &T,
    operator: &Operator,
) -> NumericValue {
    if !task.metric().use_metric() {
        return NumericValue::new(operator.cost().value() as f64);
    }

    let initial_numeric_values = task.get_initial_numeric_state_values();
    let mut numeric_values = initial_numeric_values.to_vec();
    let old_metric = evaluate_metric_from_values(task, &numeric_values);

    // Effects of one operator apply simultaneously, so every read must see the
    // pre-application values. Collect first, publish second; see
    // `StateRegistry::apply_numeric_effects_inner` for the same reasoning on
    // the search path.
    let mut results = Vec::with_capacity(operator.assignment_effects().len());
    for effect in operator.assignment_effects() {
        let assignment_var_id = effect.var_id();
        let affected_var_id = effect.affected_var_id();
        assert!(
            (assignment_var_id.index()) < numeric_values.len(),
            "assignment variable {} of operator {} is out of bounds for {} numeric variables",
            assignment_var_id.index(),
            operator.name(),
            numeric_values.len(),
        );
        assert!(
            (affected_var_id.index()) < numeric_values.len(),
            "affected variable {} of operator {} is out of bounds for {} numeric variables",
            affected_var_id.index(),
            operator.name(),
            numeric_values.len(),
        );

        let result = AssignmentOperation::apply(
            numeric_values[affected_var_id.index()],
            effect.operation(),
            numeric_values[assignment_var_id.index()],
        );
        results.push((affected_var_id, result));
    }
    for (affected_var_id, result) in results {
        numeric_values[affected_var_id.index()] = result;
    }

    propagate_assignment_axiom_values(task, &mut numeric_values).unwrap_or_else(|error| {
        panic!(
            "operator {} cannot evaluate assignment axioms while computing its metric cost: \
             {error:?}",
            operator.name()
        )
    });
    let new_metric = evaluate_metric_from_values(task, &numeric_values);
    let delta = if task.metric().is_min() {
        new_metric - old_metric
    } else {
        old_metric - new_metric
    };
    assert!(
        delta >= ZERO_VALUE,
        "operator {} has negative metric cost {delta}, which search does not support",
        operator.name()
    );
    delta
}
