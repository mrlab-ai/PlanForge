use super::*;

pub(super) fn numeric_split_choice_key(
    variable_name: &str,
    boundary: NumericValue,
    lower_closed: bool,
) -> u64 {
    mix_seed(stable_text_seed(variable_name) ^ boundary.to_bits()) ^ (u64::from(lower_closed) << 63)
}

pub(super) fn split_choice_key(semantics: &CartesianSemantics<'_>, split: &Split) -> u64 {
    match split {
        Split::Propositional { var_id, wanted, .. } => {
            wanted.iter().fold(var_id.index() as u64, |key, value| {
                mix_seed(key ^ (value.index() as u64))
            })
        }
        Split::Numeric {
            var_id,
            boundary,
            lower_includes_boundary,
            ..
        } => {
            let variable_name = semantics.task().numeric_variables()[var_id.index()].name();
            numeric_split_choice_key(variable_name, *boundary, *lower_includes_boundary)
        }
    }
}

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

pub(super) fn retain_min_growth_splits<T>(
    working: &WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    candidates: &mut Vec<T>,
    split: impl Fn(&T) -> &Split,
) -> Result<()> {
    let projected_transition_counts = candidates
        .iter()
        .map(|candidate| projected_transition_count(working, semantics, split(candidate)))
        .collect::<Result<Vec<_>>>()?;
    let minimum = projected_transition_counts
        .iter()
        .copied()
        .min()
        .context("cannot rank an empty split candidate set by growth")?;
    let mut index = 0;
    candidates.retain(|_| {
        let retain = projected_transition_counts[index] == minimum;
        index += 1;
        retain
    });
    Ok(())
}

fn select_min_growth_split(
    working: &WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    mut candidates: Vec<Split>,
    tag: u64,
) -> Result<Split> {
    ensure!(
        !candidates.is_empty(),
        "cannot select a Cartesian refinement from an empty candidate set"
    );
    retain_min_growth_splits(working, semantics, &mut candidates, |split| split)?;
    let index = semantics.choose_split_index(&candidates, tag);
    Ok(candidates.swap_remove(index))
}

pub(super) fn select_refinement_split(
    working: &WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    mut candidates: Vec<Split>,
    tag: u64,
) -> Result<Split> {
    match semantics.split_selection() {
        CartesianSplitSelection::MinTransitionGrowth => {
            select_min_growth_split(working, semantics, candidates, tag)
        }
        CartesianSplitSelection::Random => {
            ensure!(
                !candidates.is_empty(),
                "cannot select a Cartesian refinement from an empty candidate set"
            );
            let index = semantics.choose_random_split_index(candidates.len());
            Ok(candidates.swap_remove(index))
        }
    }
}

pub(super) fn select_refinement(
    working: &WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    candidates: Vec<Split>,
) -> Result<PlanCheck> {
    Ok(PlanCheck::Refine(select_refinement_split(
        working,
        semantics,
        candidates,
        0x454E_5449,
    )?))
}

pub(super) fn push_unique_split(
    candidates: &mut Vec<Split>,
    identities: &mut HashSet<SplitIdentity>,
    split: Split,
) {
    if identities.insert(SplitIdentity::from(&split)) {
        candidates.push(split);
    }
}
