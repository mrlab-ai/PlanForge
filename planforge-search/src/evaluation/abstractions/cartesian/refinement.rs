use super::*;

impl WorkingAbstraction {
    pub(super) fn apply_split(
        &mut self,
        semantics: &CartesianSemantics<'_>,
        split: Split,
    ) -> Result<usize> {
        let working = self;
        let old_state_id = split.state_id();
        let split_dimension = split.dimension();
        let old_region = working
            .states
            .get(old_state_id)
            .with_context(|| format!("missing split state {old_state_id}"))?
            .clone();
        let leaf_node_id = working.leaf_node_ids[old_state_id];
        let new_state_id = working.states.len();
        let (old_child, new_child) = match split {
            Split::Propositional {
                var_id,
                wanted,
                witness_value,
                ..
            } => {
                let current = old_region
                    .propositions()
                    .get(var_id.index())
                    .with_context(|| format!("split references missing prop var {var_id}"))?;
                let wanted_child_values: Vec<_> = current
                    .iter()
                    .copied()
                    .filter(|value| wanted.binary_search(value).is_ok())
                    .collect();
                let other_child_values: Vec<_> = current
                    .iter()
                    .copied()
                    .filter(|value| wanted.binary_search(value).is_err())
                    .collect();
                ensure!(
                    !wanted_child_values.is_empty() && !other_child_values.is_empty(),
                    "non-strict propositional Cartesian split on var {var_id}: current={current:?}, wanted={wanted:?}"
                );
                let witness_is_wanted = wanted_child_values.binary_search(&witness_value).is_ok();
                let mut wanted_region = old_region.clone();
                wanted_region.narrow_prop(var_id, wanted_child_values);
                let mut other_region = old_region.clone();
                other_region.narrow_prop(var_id, other_child_values);
                working.hierarchy.split_propositional(
                    leaf_node_id,
                    old_state_id,
                    new_state_id,
                    var_id,
                    wanted,
                    witness_is_wanted,
                )?;
                if witness_is_wanted {
                    (wanted_region, other_region)
                } else {
                    (other_region, wanted_region)
                }
            }
            Split::Numeric {
                var_id,
                boundary,
                lower_includes_boundary,
                witness_value,
                integer_lattice,
                ..
            } => {
                let parent = old_region.numeric[var_id.index()];
                let (lower, upper) = numeric_split_intervals(
                    parent,
                    boundary,
                    lower_includes_boundary,
                    integer_lattice,
                )
                .with_context(|| {
                    format!(
                        "non-strict numeric split on var {var_id} at {boundary}: parent={parent:?}"
                    )
                })?;
                let witness_is_lower = lower.contains(witness_value);
                ensure!(
                    witness_is_lower ^ upper.contains(witness_value),
                    "numeric split does not place witness {witness_value} in exactly one child"
                );
                let mut lower_region = old_region.clone();
                Arc::make_mut(&mut lower_region.numeric)[var_id.index()] = lower;
                let mut upper_region = old_region.clone();
                Arc::make_mut(&mut upper_region.numeric)[var_id.index()] = upper;
                working.hierarchy.split_numeric(
                    leaf_node_id,
                    old_state_id,
                    new_state_id,
                    NumericSplit {
                        var_id,
                        boundary,
                        lower_includes_boundary,
                        old_state_is_lower: witness_is_lower,
                    },
                )?;
                if witness_is_lower {
                    (lower_region, upper_region)
                } else {
                    (upper_region, lower_region)
                }
            }
        };

        working.states[old_state_id] = old_child;
        working.states.push(new_child);
        working.outgoing.push(Vec::new());
        working.incoming.push(Vec::new());
        let operator_count = semantics.task().get_operators().len();
        let old_self_loops = std::mem::replace(
            &mut working.self_loop_operator_ids[old_state_id],
            OperatorBitSet::empty(operator_count),
        );
        let split_dependent_operators = semantics.split_dependent_operators(split_dimension);
        working.self_loop_operator_ids[old_state_id] =
            old_self_loops.clone_without(split_dependent_operators);
        working
            .self_loop_operator_ids
            .push(old_self_loops.clone_without(split_dependent_operators));
        let new_leaf_nodes = match &working.hierarchy.nodes[leaf_node_id] {
            RefinementNode::Propositional {
                wanted_child,
                other_child,
                ..
            } => (*wanted_child, *other_child),
            RefinementNode::Numeric {
                lower_child,
                upper_child,
                ..
            } => (*lower_child, *upper_child),
            RefinementNode::Leaf { .. } => unreachable!(),
        };
        let old_leaf_node = if matches!(working.hierarchy.nodes[new_leaf_nodes.0], RefinementNode::Leaf { state_id } if state_id == old_state_id)
        {
            new_leaf_nodes.0
        } else {
            new_leaf_nodes.1
        };
        let new_leaf_node = if old_leaf_node == new_leaf_nodes.0 {
            new_leaf_nodes.1
        } else {
            new_leaf_nodes.0
        };
        working.leaf_node_ids[old_state_id] = old_leaf_node;
        working.leaf_node_ids.push(new_leaf_node);

        let old_transitions = working.remove_incident_transitions(old_state_id);
        for transition in old_transitions {
            debug_assert!(
                transition.source != transition.target,
                "Cartesian non-loop storage contains a self loop"
            );
            let sources: &[usize] = if transition.source == old_state_id {
                &[old_state_id, new_state_id]
            } else {
                std::slice::from_ref(&transition.source)
            };
            let targets: &[usize] = if transition.target == old_state_id {
                &[old_state_id, new_state_id]
            } else {
                std::slice::from_ref(&transition.target)
            };
            for &source in sources {
                for &target in targets {
                    let may_transition = if semantics
                        .operator_depends_on_split(transition.concrete_op_id, split_dimension)
                    {
                        semantics.regions().may_transition(
                            &working.states[source],
                            transition.concrete_op_id,
                            &working.states[target],
                        )?
                    } else {
                        semantics.may_transition_after_independent_split(
                            &working.states[source],
                            transition.concrete_op_id,
                            &working.states[target],
                            split_dimension,
                        )?
                    };
                    if may_transition {
                        working.add_transition(source, transition.concrete_op_id, target);
                    }
                }
            }
        }
        for concrete_op_id in old_self_loops.intersection_iter(split_dependent_operators) {
            for source in [old_state_id, new_state_id] {
                let targets = [old_state_id, new_state_id];
                let may_targets = semantics.parent_loop_source_to_split_children(
                    &working.states[source],
                    concrete_op_id,
                    [&working.states[old_state_id], &working.states[new_state_id]],
                    split_dimension,
                )?;
                for (target, may_transition) in targets.into_iter().zip(may_targets) {
                    if may_transition {
                        working.add_transition(source, concrete_op_id, target);
                    }
                }
            }
        }
        Ok(new_state_id)
    }
}
