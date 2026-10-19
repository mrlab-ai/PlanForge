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
mod distances;
mod finalize;
mod flaws;
mod generator;
mod heuristic;
mod hierarchy;
mod refinement;
mod semantics;
mod split;
mod split_selection;
#[cfg(test)]
mod tests;
mod working;

pub use abstraction::*;
pub use collection::*;
pub use config::*;
use distances::*;
use finalize::*;
use flaws::*;
pub use generator::*;
pub use heuristic::*;
pub use hierarchy::*;
use semantics::*;
use split::*;
use split_selection::*;
use working::*;

use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use ordered_float::NotNan;
use planforge_sas::axioms::AxiomEvaluator;
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

use super::common::cegar::regression::ensure_regressable;
use super::common::cegar::{
    CegarDriver, CegarIterationResult, CegarStopReason, FlawKind, progress_concrete_state,
};
use super::common::coordinates::Coordinates;
use super::common::regions::RegionSemantics;
use crate::evaluation::abstractions::collections::goal_focused::{
    BuiltReport, CollectionMembership, FULL_TASK_PROGRESSIVE_ROOTS_ERROR, GoalFocusedBackend,
    GoalFocusedSchedule, GoalVariant, MemberBudget, MemberDirection, RefinementRootDeadEnd,
    build_goal_focused_collection,
};
use crate::evaluation::abstractions::collections::portfolio::{
    CollectionStrategy, mix_seed, stable_text_seed,
};
use crate::evaluation::abstractions::common::cartesian_set::{
    CartesianSet, sorted_value_sets_overlap,
};
use crate::evaluation::abstractions::common::concrete_state::{
    ConcreteState, get_initial_state, make_prop_state_packer,
};
use crate::evaluation::abstractions::common::coordinates::comparison_refinement_dimensions;
use crate::evaluation::abstractions::common::task::{
    AbstractionUse, validate_abstraction_operator,
};
use crate::evaluation::abstractions::common::transition_system::{
    AbstractDistanceTable, AbstractOperatorSourceSets, AbstractTransition,
    AbstractTransitionSystem, LabelSourceSet, backward_goal_distances,
};
use planforge_sas::utils::interval::Interval;

#[inline]
fn fact_is_held(fact: &ExplicitFact, packer: &StatePacker, buffer: &[u64]) -> bool {
    fact.is_held(
        planforge_sas::state_registry::ConcreteStateView::from_decoded(packer, buffer, &[]),
    )
}
