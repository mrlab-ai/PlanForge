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
mod generator;
mod heuristic;
mod hierarchy;
mod refinement;
mod semantics;
#[cfg(test)]
mod tests;
mod view;
mod working;

use super::common::cegar::progression::*;
use super::common::cegar::refine::{CegarRun, Refinable, refine};
use super::common::cegar::semantics::{CegarSemantics, SplitOptions};
use super::common::cegar::split::*;
use super::common::cegar::view::{AbstractionView, TransitionKey};
use super::common::cegar::{SplitKind, SplitSelection};
pub use abstraction::*;
pub use collection::*;
pub use config::*;
use distances::*;
use finalize::*;
pub use generator::*;
pub use heuristic::*;
pub use hierarchy::*;
use semantics::*;
use view::*;
use working::*;

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
use ordered_float::NotNan;
use planforge_sas::axioms::AxiomEvaluator;
use planforge_sas::numeric_task::{
    AbstractNumericTask, ExplicitFact, ExplicitValueIndex, NumericType, NumericValue,
    OperatorIndex, VariableIndex,
};
use planforge_sas::utils::float_tolerance;
use tracing::info;

use crate::evaluation::evaluator::{EvaluationError, EvaluationState};
use crate::evaluation::heuristic::Heuristic;

use super::common::cegar::regression::ensure_regressable;
use super::common::cegar::{CegarDriver, CegarStopReason, FlawKind, FlawScope};
use super::common::regions::RegionSemantics;
use crate::evaluation::abstractions::collections::goal_focused::{
    BuiltReport, CollectionMembership, FULL_TASK_PROGRESS_STATES_ERROR, GoalFocusedBackend,
    GoalFocusedSchedule, GoalVariant, MemberBudget, build_goal_focused_collection,
};
use crate::evaluation::abstractions::collections::portfolio::CollectionStrategy;
use crate::evaluation::abstractions::common::cartesian_set::{
    CartesianSet, sorted_value_sets_overlap,
};
use crate::evaluation::abstractions::common::concrete_state::{
    ConcreteState, concrete_prop_values, get_initial_state, make_prop_state_packer,
};
use crate::evaluation::abstractions::common::coordinates::comparison_refinement_dimensions;
use crate::evaluation::abstractions::common::task::AbstractionUse;
use crate::evaluation::abstractions::common::transition_system::{
    AbstractDistanceTable, AbstractOperatorSourceSets, AbstractTransition,
    AbstractTransitionSystem, LabelSourceSet, backward_goal_distances,
};
