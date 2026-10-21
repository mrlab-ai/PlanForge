//! The working abstraction as CEGAR's flaw search sees it.

use std::borrow::Cow;

use super::*;

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

    fn view(&mut self, _deadline: Option<Instant>) -> Result<CartesianView<'_>> {
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
        self.working.hierarchy().map_state(propositions, numeric)
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
    let split_dimension = split.dimension();
    let new_state_id = working.states().len();
    let (old_child, new_child) = split_child_regions(&working.states()[split.state_id()], split)?;
    let mut incident = working.outgoing()[split_state_id].clone();
    incident.extend(working.incoming()[split_state_id].iter().copied());
    incident.sort_unstable();
    incident.dedup();

    let unaffected = working
        .transition_ids_by_key
        .len()
        .checked_sub(incident.len())
        .expect("incident Cartesian transition count exceeds active transition count");
    let mut replacements = HashSet::new();
    for transition_id in incident {
        let transition = working.transition(transition_id);
        debug_assert!(
            transition.source != transition.target,
            "Cartesian non-loop storage contains a self loop"
        );
        let sources: &[usize] = if transition.source == split_state_id {
            &[split_state_id, new_state_id]
        } else {
            std::slice::from_ref(&transition.source)
        };
        let targets: &[usize] = if transition.target == split_state_id {
            &[split_state_id, new_state_id]
        } else {
            std::slice::from_ref(&transition.target)
        };
        for &source in sources {
            let source_region = if source == split_state_id {
                &old_child
            } else if source == new_state_id {
                &new_child
            } else {
                &working.states()[source]
            };
            for &target in targets {
                let target_region = if target == split_state_id {
                    &old_child
                } else if target == new_state_id {
                    &new_child
                } else {
                    &working.states()[target]
                };
                let may_transition = if semantics
                    .operator_depends_on_split(transition.concrete_op_id, split_dimension)
                {
                    semantics.regions().may_transition(
                        source_region,
                        transition.concrete_op_id,
                        target_region,
                    )?
                } else {
                    semantics.may_transition_after_independent_split(
                        source_region,
                        transition.concrete_op_id,
                        target_region,
                        split_dimension,
                    )?
                };
                if may_transition && source != target {
                    replacements.insert(TransitionKey {
                        source,
                        concrete_op_id: transition.concrete_op_id,
                        target,
                    });
                }
            }
        }
    }
    let split_dependent_operators = semantics.split_dependent_operators(split_dimension);
    for concrete_op_id in working.self_loop_operator_ids()[split_state_id]
        .intersection_iter(split_dependent_operators)
    {
        for (source, source_region) in [(split_state_id, &old_child), (new_state_id, &new_child)] {
            let targets = [(split_state_id, &old_child), (new_state_id, &new_child)];
            let may_targets = semantics.parent_loop_source_to_split_children(
                source_region,
                concrete_op_id,
                [targets[0].1, targets[1].1],
                split_dimension,
            )?;
            for ((target, _), may_transition) in targets.into_iter().zip(may_targets) {
                if source != target && may_transition {
                    replacements.insert(TransitionKey {
                        source,
                        concrete_op_id,
                        target,
                    });
                }
            }
        }
    }
    unaffected
        .checked_add(replacements.len())
        .context("projected Cartesian transition count overflow")
}
