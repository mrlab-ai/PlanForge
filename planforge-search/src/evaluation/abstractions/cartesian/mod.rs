//! Numeric Cartesian abstractions refined by concrete counterexamples.
//!
//! Unlike the factorized domain abstraction, splitting one Cartesian state
//! adds exactly one state. Every abstract transition is a may-transition of a
//! grounded concrete operator. CEGAR replays a deterministic optimal abstract
//! trace. The default refines its first witnessed flaw; the explicit
//! whole-plan mode ranks all sound witnesses on an acyclic adaptive trace.
//! Only a successfully replayed concrete plan may set `solved_by_self`.

mod abstraction;
mod collection;
mod config;
mod finalize;
mod flaw_splits;
mod generator;
mod heuristic;
mod hierarchy;
pub mod icaps26;
mod plan_replay;
mod refinement;
mod regression;
mod semantics;
mod shortest_paths;
mod split;
mod split_generation;
mod split_selector;
#[cfg(test)]
mod tests;
mod working;

pub use abstraction::*;
pub use collection::*;
pub use config::*;
use finalize::*;
use flaw_splits::*;
pub use generator::*;
pub use heuristic::*;
pub use hierarchy::*;
use plan_replay::*;
use regression::*;
use semantics::*;
use shortest_paths::*;
use split::*;
use split_generation::*;
use split_selector::*;
use working::*;

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
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use tracing::info;

use crate::evaluation::evaluator::{EvaluationError, EvaluationState};
use crate::evaluation::heuristic::Heuristic;
use crate::evaluation::validate_abstractable_goal;

use super::cegar::regression::ensure_regressable;
use super::cegar::{
    CegarDriver, CegarIterationResult, CegarStopReason, FlawKind, progress_concrete_state,
};
use super::coordinates::Coordinates;
use crate::evaluation::abstractions::collections::goal_focused::{
    BuiltReport, CollectionMembership, ConcreteState, FULL_TASK_PROGRESSIVE_ROOTS_ERROR,
    GoalFocusedBackend, GoalFocusedSchedule, GoalVariant, MemberBudget, MemberDirection,
    RefinementRootDeadEnd, build_goal_focused_collection,
};
use crate::evaluation::abstractions::collections::portfolio::{
    CollectionStrategy, mix_seed, stable_text_seed,
};
use crate::evaluation::abstractions::coordinates::{
    AdditiveNumericView, comparison_refinement_dimensions, numeric_dimension_delta_for_operator,
};
use crate::evaluation::abstractions::cost_partitioning::costs::{
    AbstractOperatorSourceSets, AbstractTransition, AbstractTransitionSystem, CartesianSet,
    LabelSourceSet, build_explicit_goal_distances, sorted_value_sets_overlap,
};
use crate::evaluation::abstractions::domain::factory::AbstractDistanceTable;
use crate::evaluation::abstractions::domain::utils::{get_initial_state, make_prop_state_packer};
use crate::evaluation::abstractions::task::{AbstractionUse, validate_abstraction_operator};
use icaps26::*;
use planforge_sas::utils::interval::Interval;

#[inline]
fn fact_is_held(fact: &ExplicitFact, packer: &StatePacker, buffer: &[u64]) -> bool {
    fact.is_held(
        planforge_sas::state_registry::ConcreteStateView::from_decoded(packer, buffer, &[]),
    )
}
