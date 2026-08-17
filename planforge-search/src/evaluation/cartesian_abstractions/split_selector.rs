use super::*;

pub(super) fn numeric_split_choice_key(
    variable_name: &str,
    boundary: f64,
    lower_closed: bool,
) -> u64 {
    mix_seed(stable_text_seed(variable_name) ^ boundary.to_bits()) ^ (u64::from(lower_closed) << 63)
}

pub(super) fn split_choice_key(semantics: &CartesianSemantics<'_>, split: &Split) -> u64 {
    match split {
        Split::Propositional { var_id, wanted, .. } => {
            let var_id = u64::try_from(*var_id).expect("split variable id does not fit u64");
            wanted
                .iter()
                .fold(var_id, |key, value| mix_seed(key ^ u64::from(*value)))
        }
        Split::Numeric {
            var_id,
            boundary,
            lower_includes_boundary,
            ..
        } => {
            let variable_name = semantics.task().numeric_variables()[*var_id].name();
            numeric_split_choice_key(variable_name, *boundary, *lower_includes_boundary)
        }
    }
}

fn split_child_regions(
    working: &WorkingAbstraction,
    split: &Split,
) -> Result<(StateRegion, StateRegion)> {
    let parent = working
        .states
        .get(split.state_id())
        .with_context(|| format!("missing split state {}", split.state_id()))?;
    match split {
        Split::Propositional {
            var_id,
            wanted,
            witness_value,
            ..
        } => {
            let current = parent
                .propositions
                .get(*var_id)
                .with_context(|| format!("split references missing prop var {var_id}"))?;
            ensure!(
                wanted.windows(2).all(|values| values[0] < values[1]),
                "propositional Cartesian split values must be sorted and unique: {wanted:?}"
            );
            let wanted_values = current
                .iter()
                .copied()
                .filter(|value| wanted.binary_search(value).is_ok())
                .collect::<Vec<_>>();
            let other_values = current
                .iter()
                .copied()
                .filter(|value| wanted.binary_search(value).is_err())
                .collect::<Vec<_>>();
            ensure!(
                !wanted_values.is_empty() && !other_values.is_empty(),
                "non-strict propositional Cartesian split on var {var_id}: current={current:?}, wanted={wanted:?}"
            );
            let witness_is_wanted = wanted_values.binary_search(witness_value).is_ok();
            let mut wanted_region = parent.clone();
            Arc::make_mut(&mut wanted_region.propositions)[*var_id] = wanted_values;
            let mut other_region = parent.clone();
            Arc::make_mut(&mut other_region.propositions)[*var_id] = other_values;
            Ok(if witness_is_wanted {
                (wanted_region, other_region)
            } else {
                (other_region, wanted_region)
            })
        }
        Split::Numeric {
            var_id,
            boundary,
            lower_includes_boundary,
            witness_value,
            integer_lattice,
            ..
        } => {
            let current = *parent
                .numeric
                .get(*var_id)
                .with_context(|| format!("split references missing numeric var {var_id}"))?;
            ensure!(
                current.can_split_at(*boundary, *lower_includes_boundary),
                "non-strict numeric Cartesian split on var {var_id} at {boundary}: parent={current:?}, include_lower={lower_includes_boundary}"
            );
            let (lower, upper) = numeric_split_intervals(
                current,
                *boundary,
                *lower_includes_boundary,
                *integer_lattice,
            )?;
            let witness_is_lower = lower.contains(*witness_value);
            ensure!(
                witness_is_lower ^ upper.contains(*witness_value),
                "numeric split does not place witness {witness_value} in exactly one child"
            );
            let mut lower_region = parent.clone();
            Arc::make_mut(&mut lower_region.numeric)[*var_id] = lower;
            let mut upper_region = parent.clone();
            Arc::make_mut(&mut upper_region.numeric)[*var_id] = upper;
            Ok(if witness_is_lower {
                (lower_region, upper_region)
            } else {
                (upper_region, lower_region)
            })
        }
    }
}

pub(super) fn numeric_split_intervals(
    parent: Interval,
    boundary: f64,
    lower_includes_boundary: bool,
    integer_lattice: bool,
) -> Result<(Interval, Interval)> {
    let (lower_bound, upper_bound, lower_closed, upper_closed) = if integer_lattice {
        ensure!(
            boundary.is_finite() && approximately_equal(boundary, boundary.round()),
            "integer Cartesian split has non-integer boundary {boundary}"
        );
        if lower_includes_boundary {
            (boundary, boundary + 1.0, true, true)
        } else {
            (boundary - 1.0, boundary, true, true)
        }
    } else {
        (
            boundary,
            boundary,
            lower_includes_boundary,
            !lower_includes_boundary,
        )
    };
    let lower = parent.intersection(&Interval::new(
        f64::NEG_INFINITY,
        lower_bound,
        false,
        lower_closed,
    ));
    let upper = parent.intersection(&Interval::new(
        upper_bound,
        f64::INFINITY,
        upper_closed,
        false,
    ));
    ensure!(
        !lower.is_empty() && !upper.is_empty(),
        "non-strict numeric Cartesian split at {boundary}: parent={parent:?}, include_lower={lower_includes_boundary}, integer_lattice={integer_lattice}"
    );
    Ok((lower, upper))
}

fn projected_transition_count(
    working: &WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    split: &Split,
) -> Result<usize> {
    let split_state_id = split.state_id();
    let split_dimension = split.dimension();
    let new_state_id = working.states().len();
    let (old_child, new_child) = split_child_regions(working, split)?;
    let mut incident = working.outgoing()[split_state_id].clone();
    incident.extend(working.incoming()[split_state_id].iter().copied());
    incident.sort_unstable();
    incident.dedup();

    let unaffected = working
        .transition_ids_by_key
        .as_ref()
        .expect("projected transition growth requires the native transition index")
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
                    semantics.may_transition(
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

fn select_least_refined_split(
    working: &WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    mut candidates: Vec<Split>,
    tag: u64,
) -> Result<Split> {
    ensure!(
        !candidates.is_empty(),
        "cannot select a Cartesian refinement from an empty candidate set"
    );
    let minimum = candidates
        .iter()
        .map(|split| match split.dimension() {
            SplitDimension::Propositional(var_id) => {
                working.propositional_refinement_counts()[var_id]
            }
            SplitDimension::Numeric(var_id) => working.numeric_refinement_counts()[var_id],
        })
        .min()
        .expect("nonempty Cartesian candidate set has no minimum");
    candidates.retain(|split| {
        let count = match split.dimension() {
            SplitDimension::Propositional(var_id) => {
                working.propositional_refinement_counts()[var_id]
            }
            SplitDimension::Numeric(var_id) => working.numeric_refinement_counts()[var_id],
        };
        count == minimum
    });
    let index = semantics.choose_split_index(&candidates, tag);
    Ok(candidates.swap_remove(index))
}

fn additive_step_distance(
    working: &WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    split: &Split,
) -> Result<Option<f64>> {
    let Split::Numeric {
        var_id,
        witness_value,
        desired_contains_witness,
        ..
    } = split
    else {
        return Ok(None);
    };
    if *desired_contains_witness {
        return Ok(None);
    }
    let (_, desired_region) = split_child_regions(working, split)?;
    let desired = desired_region.numeric[*var_id];
    let (distance, positive_direction) = if *witness_value < desired.lower
        || (*witness_value == desired.lower && !desired.lower_closed)
    {
        (desired.lower - *witness_value, true)
    } else if *witness_value > desired.upper
        || (*witness_value == desired.upper && !desired.upper_closed)
    {
        (*witness_value - desired.upper, false)
    } else {
        bail!(
            "numeric refinement witness {witness_value} is inside its purported desired child {desired:?}"
        );
    };
    ensure!(
        distance.is_finite() && distance >= 0.0,
        "invalid additive refinement distance {distance}"
    );
    let maximum_progress = semantics.additive_effect_deltas()[*var_id]
        .iter()
        .copied()
        .filter(|delta| {
            if positive_direction {
                *delta > float_tolerance::SEARCH_EPSILON
            } else {
                *delta < -float_tolerance::SEARCH_EPSILON
            }
        })
        .map(f64::abs)
        .max_by(f64::total_cmp);
    Ok(maximum_progress.map(|progress| (distance / progress).max(1.0)))
}

fn select_max_additive_steps_split(
    working: &WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    mut candidates: Vec<Split>,
    tag: u64,
) -> Result<Split> {
    ensure!(
        !candidates.is_empty(),
        "cannot select a Cartesian refinement from an empty candidate set"
    );
    let scores = candidates
        .iter()
        .map(|candidate| additive_step_distance(working, semantics, candidate))
        .collect::<Result<Vec<_>>>()?;
    let Some(maximum) = scores.iter().flatten().copied().max_by(f64::total_cmp) else {
        return select_min_growth_split(working, semantics, candidates, tag);
    };
    let mut index = 0;
    candidates.retain(|_| {
        let retain = scores[index].is_some_and(|score| approximately_equal(score, maximum));
        index += 1;
        retain
    });
    select_min_growth_split(working, semantics, candidates, tag)
}

pub(super) fn artifact_unwanted_score(working: &WorkingAbstraction, split: &Split) -> Result<f64> {
    let parent = working
        .states
        .get(split.state_id())
        .with_context(|| format!("missing split state {}", split.state_id()))?;
    match split {
        Split::Propositional { var_id, wanted, .. } => {
            let current = parent
                .propositions
                .get(*var_id)
                .with_context(|| format!("split references missing prop var {var_id}"))?;
            let wanted_count = current
                .iter()
                .filter(|value| wanted.binary_search(value).is_ok())
                .count();
            ensure!(
                wanted_count > 0 && wanted_count < current.len(),
                "ICAPS 2026 selector received a non-strict propositional split"
            );
            Ok((current.len() - wanted_count) as f64)
        }
        Split::Numeric {
            var_id,
            desired_contains_witness,
            ..
        } => {
            let (witness_region, other_region) = split_child_regions(working, split)?;
            let desired_region = if *desired_contains_witness {
                witness_region
            } else {
                other_region
            };
            let desired = desired_region.numeric[*var_id];
            if !desired.lower.is_finite() || !desired.upper.is_finite() {
                return Ok(f64::INFINITY);
            }
            let current = parent.numeric[*var_id];
            if !current.lower.is_finite() || !current.upper.is_finite() {
                return Ok(f64::INFINITY);
            }
            let current_values = integer_interval_cardinality(current);
            let desired_values = integer_interval_cardinality(desired);
            let unwanted_values = current_values - desired_values;
            ensure!(
                unwanted_values >= 0.0,
                "ICAPS 2026 desired interval contains more integer values than its parent"
            );
            let unwanted_width = (current.upper - current.lower) - (desired.upper - desired.lower);
            ensure!(
                unwanted_width >= 0.0,
                "ICAPS 2026 desired interval is wider than its parent"
            );
            // Preserve the artifact's integer-task ordering whenever the
            // excluded child contains an integer value. Strict fractional
            // splits can exclude no integer while still having positive
            // width; only that previously unsupported case uses the width.
            Ok(if unwanted_values > 0.0 {
                unwanted_values
            } else {
                unwanted_width
            })
        }
    }
}

fn integer_interval_cardinality(interval: Interval) -> f64 {
    debug_assert!(interval.lower.is_finite() && interval.upper.is_finite());
    let first = if interval.lower_closed {
        interval.lower.ceil()
    } else {
        interval.lower.floor() + 1.0
    };
    let last = if interval.upper_closed {
        interval.upper.floor()
    } else {
        interval.upper.ceil() - 1.0
    };
    (last - first + 1.0).max(0.0)
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
        CartesianSplitSelection::MaxAdditiveSteps => {
            select_max_additive_steps_split(working, semantics, candidates, tag)
        }
        CartesianSplitSelection::Random => {
            ensure!(
                !candidates.is_empty(),
                "cannot select a Cartesian refinement from an empty candidate set"
            );
            let index = semantics.choose_random_split_index(candidates.len());
            Ok(candidates.swap_remove(index))
        }
        CartesianSplitSelection::LeastRefined => {
            select_least_refined_split(working, semantics, candidates, tag)
        }
        CartesianSplitSelection::Icaps26(Icaps26SplitSelection::Random) => {
            ensure!(
                !candidates.is_empty(),
                "cannot select a Cartesian refinement from an empty candidate set"
            );
            if candidates.len() == 1 {
                return Ok(candidates.pop().expect("checked nonempty split set"));
            }
            let index = semantics.choose_icaps_random_index(candidates.len());
            Ok(candidates.swap_remove(index))
        }
        CartesianSplitSelection::Icaps26(policy) => {
            ensure!(
                !candidates.is_empty(),
                "cannot select a Cartesian refinement from an empty candidate set"
            );
            let mut selected = 0;
            let mut selected_score = artifact_unwanted_score(working, &candidates[0])?;
            for (index, candidate) in candidates.iter().enumerate().skip(1) {
                let score = artifact_unwanted_score(working, candidate)?;
                let better = match policy {
                    Icaps26SplitSelection::MinUnwanted => score < selected_score,
                    Icaps26SplitSelection::MaxUnwanted => score > selected_score,
                    Icaps26SplitSelection::Random => unreachable!(),
                };
                if better {
                    selected = index;
                    selected_score = score;
                }
            }
            Ok(candidates.swap_remove(selected))
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
