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
mod hierarchy;
mod initial_splits;
mod refinement;
mod semantics;
#[cfg(test)]
mod tests;
mod view;
mod working;

use super::common::cegar::refine::{CegarRun, Refinable, refine};
use super::common::cegar::semantics::CegarSemantics;
use super::common::cegar::split::*;
use super::common::cegar::view::{AbstractionView, RefinementLimits, TransitionKey};
use super::common::cegar::{SplitKind, SplitSelection};
pub use abstraction::*;
pub use collection::*;
pub use config::*;
use distances::*;
use finalize::*;
pub use generator::*;
pub use hierarchy::*;
use initial_splits::*;
use semantics::*;
use view::*;
use working::*;

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use ordered_float::NotNan;
use planforge_sas::explicit_state::ExplicitState;
use planforge_sas::numeric_task::{
    AbstractNumericTask, ExplicitFact, ExplicitValueIndex, NumericType, NumericValue,
    OperatorIndex, VariableIndex,
};
use planforge_sas::utils::float_tolerance;
use tracing::info;

use super::common::cegar::{CegarDriver, FlawKind, FlawScope};
use super::common::regions::RegionSemantics;
use crate::evaluation::abstractions::collections::generator::AbstractionGenerator;
use crate::evaluation::abstractions::collections::goal_focused::{GoalVariant, MemberBudget};
use crate::evaluation::abstractions::collections::standard::StandardMember;
use crate::evaluation::abstractions::collections::strategy::CollectionStrategy;
use crate::evaluation::abstractions::common::cartesian_set::{
    CartesianSet, sorted_value_sets_overlap,
};
use crate::evaluation::abstractions::common::transition_system::{
    AbstractDistanceTable, AbstractOperatorSourceSets, AbstractTransition,
    AbstractTransitionSystem, BackwardGraph, CostedTransitionSystem, LabelSourceSet,
    backward_goal_distances,
};
