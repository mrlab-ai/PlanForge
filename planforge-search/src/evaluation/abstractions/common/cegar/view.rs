//! What CEGAR's flaw search and split selection read from an abstraction
//! under refinement.

use std::borrow::Cow;

use anyhow::Result;
use planforge_sas::numeric_task::{ExplicitValueIndex, NumericValue, OperatorIndex};

use super::split::Split;
use crate::evaluation::abstractions::common::cartesian_set::CartesianSet;

/// An abstract transition: `concrete_op_id` leads from abstract state
/// `source` to `target`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct TransitionKey {
    pub(crate) source: usize,
    pub(crate) concrete_op_id: OperatorIndex,
    pub(crate) target: usize,
}

/// An abstraction under refinement, as the flaw search sees it: abstract
/// states with their regions and goal distances, and an optimal transition
/// out of every non-goal state with finite distance.
pub(crate) trait AbstractionView {
    /// How much a split grows the abstraction; smaller is better.
    type Growth: Ord;

    fn num_states(&self) -> usize;

    /// The abstract state containing the concrete state with these values.
    fn abstract_state(
        &self,
        propositions: &[ExplicitValueIndex],
        numeric: &[NumericValue],
    ) -> Result<usize>;

    /// The concrete states abstract state `state` stands for.
    fn region(&self, state: usize) -> Cow<'_, CartesianSet>;

    fn goal_distance(&self, state: usize) -> f64;

    fn is_goal(&self, state: usize) -> Result<bool>;

    /// The first transition of an optimal plan from the non-goal `state`.
    fn optimal_transition(&self, state: usize) -> Result<TransitionKey>;

    fn growth(&self, split: &Split) -> Result<Self::Growth>;

    /// The concrete operators a plan step along `transition` may execute,
    /// tried in this order; just the transition's own by default.
    fn step_operators(&self, transition: TransitionKey) -> Vec<OperatorIndex> {
        vec![transition.concrete_op_id]
    }

    /// Whether the abstraction can take `split` within its limits. Split
    /// selection prefers such splits; CEGAR stops when the chosen one is not.
    fn refinable(&self, _split: &Split) -> bool {
        true
    }
}
