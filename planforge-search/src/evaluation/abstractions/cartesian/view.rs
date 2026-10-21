//! The working abstraction as CEGAR's flaw search sees it.

use std::borrow::Cow;
use std::collections::HashSet;

use anyhow::{Context, Result, ensure};
use planforge_sas::numeric_task::{ExplicitValueIndex, NumericValue};

use super::distances::{ShortestPaths, update_shortest_paths_after_split};
use super::refinement::{SplitStates, replacement_transitions};
use super::semantics::CartesianSemantics;
use super::working::WorkingAbstraction;
use crate::evaluation::abstractions::common::cartesian_set::CartesianSet;
use crate::evaluation::abstractions::common::cegar::refine::Refinable;
use crate::evaluation::abstractions::common::cegar::split::{Split, split_child_regions};
use crate::evaluation::abstractions::common::cegar::view::{
    AbstractionView, RefinementLimits, TransitionKey,
};
use crate::evaluation::abstractions::common::coordinates::coordinate_value;
use crate::resource_limits::Deadline;

pub(super) struct CartesianView<'a> {
    pub(super) working: &'a WorkingAbstraction,
    pub(super) semantics: &'a CartesianSemantics<'a>,
    pub(super) shortest_paths: &'a ShortestPaths,
    pub(super) limits: RefinementLimits<'a>,
}

/// The working abstraction under refinement, with the shortest paths CEGAR
/// keeps up to date across splits.
pub(super) struct CartesianRefinement<'a> {
    pub(super) working: WorkingAbstraction,
    pub(super) shortest_paths: ShortestPaths,
    pub(super) semantics: &'a CartesianSemantics<'a>,
    pub(super) limits: RefinementLimits<'a>,
}

impl Refinable for CartesianRefinement<'_> {
    type View<'b>
        = CartesianView<'b>
    where
        Self: 'b;

    fn view(&mut self, _deadline: Deadline) -> Result<CartesianView<'_>> {
        Ok(CartesianView {
            working: &self.working,
            semantics: self.semantics,
            shortest_paths: &self.shortest_paths,
            limits: self.limits,
        })
    }

    fn apply_split(&mut self, split: Split) -> Result<()> {
        let old_state_id = split.state_id();
        let new_state_id = self.working.apply_split(self.semantics, split)?;
        update_shortest_paths_after_split(
            &self.working,
            self.semantics,
            &mut self.shortest_paths,
            old_state_id,
            new_state_id,
        )
    }
}

impl AbstractionView for CartesianView<'_> {
    /// The number of abstract transitions after the split.
    type Growth = usize;

    fn num_states(&self) -> usize {
        self.working.states().len()
    }

    fn abstract_state(
        &self,
        propositions: &[ExplicitValueIndex],
        numeric: &[NumericValue],
    ) -> Result<usize> {
        let views = self.semantics.cegar().coordinates().views();
        self.working
            .hierarchy()
            .map_state_with(propositions, |var_id| {
                coordinate_value(views, numeric, var_id)
            })
    }

    fn region(&self, state: usize) -> Cow<'_, CartesianSet> {
        Cow::Borrowed(&self.working.states()[state])
    }

    fn goal_distance(&self, state: usize) -> f64 {
        self.shortest_paths.distance(state)
    }

    fn is_goal(&self, state: usize) -> Result<bool> {
        Ok(self.shortest_paths.is_goal(state))
    }

    fn optimal_transition(&self, state: usize) -> Result<TransitionKey> {
        let transition = self.shortest_paths.generating_transition(state).context(
            "non-goal Cartesian state with finite distance has no generating transition",
        )?;
        ensure!(
            self.working.contains_transition(transition),
            "Cartesian shortest path references missing transition {transition:?}"
        );
        Ok(transition)
    }

    fn growth(&self, split: &Split) -> Result<usize> {
        projected_transition_count(self.working, self.semantics, split)
    }

    fn limits(&self) -> RefinementLimits<'_> {
        self.limits
    }

    /// Every split adds one state.
    fn states_after(&self, _split: &Split) -> u128 {
        self.working.states().len() as u128 + 1
    }
}

/// The number of abstract transitions after `split`, computed without
/// applying it.
fn projected_transition_count(
    working: &WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    split: &Split,
) -> Result<usize> {
    let split_state_id = split.state_id();
    let children = split_child_regions(&working.states()[split_state_id], split)?;
    let incident = working.incident_transition_ids(split_state_id);
    let unaffected = working
        .transition_ids_by_key
        .len()
        .checked_sub(incident.len())
        .expect("incident Cartesian transition count exceeds active transition count");
    let incident: Vec<TransitionKey> = incident
        .into_iter()
        .map(|transition_id| *working.transition(transition_id))
        .collect();
    let replacements: HashSet<TransitionKey> = replacement_transitions(
        semantics,
        SplitStates {
            split_state_id,
            new_state_id: working.states().len(),
            children: [&children.witness, &children.other],
            split_dimension: split.dimension(),
        },
        |state| &working.states()[state],
        &incident,
        &working.self_loop_operator_ids()[split_state_id],
    )?
    .into_iter()
    .filter(|transition| transition.source != transition.target)
    .collect();
    Ok(unaffected + replacements.len())
}
