use super::*;

pub(super) fn split_failed_fact(
    working: &WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    state_id: usize,
    fact: &ExplicitFact,
    prop_values: &[usize],
    numeric_values: &[f64],
    description: String,
) -> Result<Split> {
    if let Some(tree_id) = semantics.task().numeric_conditions().id_for_var(fact.var()) {
        return comparison_refinement(
            working,
            semantics,
            state_id,
            tree_id,
            numeric_values,
            ComparisonRefinementGoal::exclude(fact.value())?,
            description,
        );
    }
    if !semantics.propositional_axioms_by_prop_var()[fact.var()].is_empty() {
        let default_value = semantics.propositional_axiom_default(fact.var())?;
        if fact.value() == default_value {
            let concrete_value = *prop_values
                .get(fact.var())
                .with_context(|| format!("missing concrete prop var {}", fact.var()))?;
            ensure!(
                concrete_value != default_value,
                "failed default-valued derived fact unexpectedly holds for variable {}",
                fact.var()
            );
            for &axiom_id in &semantics.propositional_axioms_by_prop_var()[fact.var()] {
                let axiom = &semantics.task().axioms()[axiom_id];
                if axiom.effect_value() != concrete_value
                    || !conditions_hold_concretely(axiom.conditions(), prop_values)?
                {
                    continue;
                }
                for condition in axiom.conditions() {
                    if !semantics.region_guarantees_fact(&working.states()[state_id], condition)? {
                        return split_to_guarantee_fact(
                            working,
                            semantics,
                            state_id,
                            condition,
                            prop_values,
                            numeric_values,
                            format!(
                                "{description} via concrete axiom {axiom_id} condition {condition:?}"
                            ),
                        );
                    }
                }
                bail!(
                    "derived default fact {fact:?} is abstractly admitted although concrete axiom {axiom_id} is guaranteed"
                );
            }
            bail!(
                "concrete derived value {concrete_value} for variable {} has no supporting axiom",
                fact.var()
            );
        }
        for &axiom_id in &semantics.propositional_axioms_by_prop_var()[fact.var()] {
            let axiom = &semantics.task().axioms()[axiom_id];
            if axiom.effect_value() != fact.value()
                || !all_conditions_admitted(
                    semantics,
                    &working.states()[state_id],
                    axiom.conditions(),
                )?
            {
                continue;
            }
            for condition in axiom.conditions() {
                let value = *prop_values
                    .get(condition.var())
                    .with_context(|| format!("missing concrete prop var {}", condition.var()))?;
                if value != condition.value() {
                    return split_failed_fact(
                        working,
                        semantics,
                        state_id,
                        condition,
                        prop_values,
                        numeric_values,
                        format!("{description} via axiom {axiom_id} condition {condition:?}"),
                    );
                }
            }
        }
        bail!(
            "derived fact {fact:?} is false in the concrete state, but every supporting axiom condition holds"
        );
    }
    let witness_value = *prop_values
        .get(fact.var())
        .with_context(|| format!("missing concrete prop var {}", fact.var()))?
        as PropValueId;
    ensure!(
        witness_value != fact.value() as PropValueId,
        "failed fact split witness unexpectedly satisfies {fact:?}"
    );
    Ok(Split::Propositional {
        state_id,
        var_id: fact.var(),
        wanted: vec![fact.value() as PropValueId],
        witness_value,
        description,
    })
}

fn split_to_guarantee_fact(
    working: &WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    state_id: usize,
    fact: &ExplicitFact,
    prop_values: &[usize],
    numeric_values: &[f64],
    description: String,
) -> Result<Split> {
    let concrete_value = *prop_values
        .get(fact.var())
        .with_context(|| format!("missing concrete prop var {}", fact.var()))?;
    ensure!(
        concrete_value == fact.value(),
        "cannot guarantee fact {fact:?}: concrete value is {concrete_value}"
    );
    if let Some(tree_id) = semantics.task().numeric_conditions().id_for_var(fact.var()) {
        return comparison_refinement(
            working,
            semantics,
            state_id,
            tree_id,
            numeric_values,
            ComparisonRefinementGoal::guarantee(fact.value())?,
            description,
        );
    }
    if !semantics.propositional_axioms_by_prop_var()[fact.var()].is_empty() {
        let default_value = semantics.propositional_axiom_default(fact.var())?;
        if fact.value() == default_value {
            for &axiom_id in &semantics.propositional_axioms_by_prop_var()[fact.var()] {
                let axiom = &semantics.task().axioms()[axiom_id];
                if !all_conditions_admitted(
                    semantics,
                    &working.states()[state_id],
                    axiom.conditions(),
                )? {
                    continue;
                }
                let condition = axiom
                    .conditions()
                    .iter()
                    .find(|condition| {
                        prop_values
                            .get(condition.var())
                            .is_some_and(|&value| value != condition.value())
                    })
                    .with_context(|| {
                        format!(
                            "concrete default value for derived variable {} conflicts with firing axiom {axiom_id}",
                            fact.var()
                        )
                    })?;
                let witness_value = prop_values[condition.var()];
                let witness_fact = ExplicitFact::propositional(condition.var(), witness_value);
                return split_to_guarantee_fact(
                    working,
                    semantics,
                    state_id,
                    &witness_fact,
                    prop_values,
                    numeric_values,
                    format!("{description} by disabling axiom {axiom_id} condition {condition:?}"),
                );
            }
            bail!(
                "derived default fact {fact:?} is not guaranteed although no competing axiom is admitted"
            );
        }

        for &axiom_id in &semantics.propositional_axioms_by_prop_var()[fact.var()] {
            let axiom = &semantics.task().axioms()[axiom_id];
            if axiom.effect_value() != fact.value()
                || !conditions_hold_concretely(axiom.conditions(), prop_values)?
            {
                continue;
            }
            for condition in axiom.conditions() {
                if !semantics.region_guarantees_fact(&working.states()[state_id], condition)? {
                    return split_to_guarantee_fact(
                        working,
                        semantics,
                        state_id,
                        condition,
                        prop_values,
                        numeric_values,
                        format!("{description} via axiom {axiom_id} condition {condition:?}"),
                    );
                }
            }
            bail!(
                "derived fact {fact:?} is not guaranteed although supporting axiom {axiom_id} is guaranteed"
            );
        }
        bail!("concrete derived fact {fact:?} has no supporting axiom");
    }

    let witness_value = concrete_value as PropValueId;
    let allowed = working
        .states
        .get(state_id)
        .and_then(|state| state.propositions.get(fact.var()))
        .with_context(|| format!("missing Cartesian state {state_id} prop var {}", fact.var()))?;
    ensure!(
        allowed.binary_search(&witness_value).is_ok() && allowed.len() > 1,
        "fact {fact:?} is already guaranteed in Cartesian state {state_id}"
    );
    Ok(Split::Propositional {
        state_id,
        var_id: fact.var(),
        wanted: vec![witness_value],
        witness_value,
        description,
    })
}

#[derive(Debug, Clone, Copy)]
enum ComparisonRefinementGoal {
    ExcludeDesired(bool),
    GuaranteeDesired(bool),
}

impl ComparisonRefinementGoal {
    fn exclude(prop_value: usize) -> Result<Self> {
        Ok(Self::ExcludeDesired(comparison_truth(prop_value)?))
    }

    fn guarantee(prop_value: usize) -> Result<Self> {
        Ok(Self::GuaranteeDesired(comparison_truth(prop_value)?))
    }

    fn desired_truth(self) -> bool {
        match self {
            Self::ExcludeDesired(truth) | Self::GuaranteeDesired(truth) => truth,
        }
    }
}

fn comparison_refinement(
    working: &WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    state_id: usize,
    tree_id: usize,
    numeric_values: &[f64],
    goal: ComparisonRefinementGoal,
    description: String,
) -> Result<Split> {
    let desired_truth = goal.desired_truth();
    let tree = semantics
        .task
        .numeric_conditions()
        .get(tree_id)
        .with_context(|| format!("missing comparison tree {tree_id}"))?;
    let concrete_truth = tree.evaluate_point(numeric_values);
    ensure!(
        matches!(goal, ComparisonRefinementGoal::ExcludeDesired(_))
            == (concrete_truth != desired_truth),
        "comparison refinement goal disagrees with concrete truth for tree {tree_id}"
    );
    let state = working
        .states
        .get(state_id)
        .with_context(|| format!("missing Cartesian state {state_id}"))?;
    let mut candidates = Vec::new();
    for var_id in tree.regular_numeric_var_dependencies().iter().copied() {
        let witness_value = float_tolerance::canonicalize(
            *numeric_values
                .get(var_id)
                .with_context(|| format!("missing concrete numeric var {var_id}"))?,
        );
        ensure!(
            witness_value.is_finite(),
            "comparison split witness for numeric var {var_id} is non-finite: {witness_value}"
        );
        let parent = *state
            .numeric
            .get(var_id)
            .with_context(|| format!("missing Cartesian numeric var {var_id}"))?;
        let mut boundaries = Vec::new();
        if semantics.refinement_direction() == CartesianRefinementDirection::Regression {
            boundaries.extend(semantics.target_split_boundaries().iter().copied());
        }
        boundaries.push(witness_value);
        boundaries.sort_by(f64::total_cmp);
        boundaries.dedup_by(|left, right| left.to_bits() == right.to_bits());

        let integer_lattice = semantics.numeric_integer_lattice()[var_id];
        for boundary in boundaries {
            for lower_includes_boundary in [true, false] {
                let Ok((lower, upper)) = numeric_split_intervals(
                    parent,
                    boundary,
                    lower_includes_boundary,
                    integer_lattice,
                ) else {
                    continue;
                };
                let (witness_child, other_child) = if lower.contains(witness_value) {
                    (lower, upper)
                } else {
                    ensure!(
                        upper.contains(witness_value),
                        "comparison split at {boundary} loses witness {witness_value} for numeric var {var_id}"
                    );
                    (upper, lower)
                };
                let mut child_numeric = state.numeric.clone();
                Arc::make_mut(&mut child_numeric)[var_id] = witness_child;
                let witness_result = tree.evaluate_interval(&child_numeric);
                ensure!(
                    witness_result != Some(!concrete_truth),
                    "comparison interval for tree {tree_id} excludes its concrete witness after splitting numeric var {var_id}"
                );
                Arc::make_mut(&mut child_numeric)[var_id] = other_child;
                let other_result = tree.evaluate_interval(&child_numeric);
                let achieved = match goal {
                    ComparisonRefinementGoal::ExcludeDesired(_) => {
                        witness_result == Some(!desired_truth)
                    }
                    ComparisonRefinementGoal::GuaranteeDesired(_) => {
                        witness_result == Some(desired_truth)
                    }
                };
                let separates_truth = achieved && other_result == Some(!concrete_truth);
                let candidate = Split::Numeric {
                    state_id,
                    var_id,
                    boundary,
                    lower_includes_boundary,
                    witness_value,
                    desired_contains_witness: matches!(
                        goal,
                        ComparisonRefinementGoal::GuaranteeDesired(_)
                    ),
                    integer_lattice,
                    description: description.clone(),
                };
                candidates.push((separates_truth, achieved, candidate));
            }
        }
    }
    ensure!(
        !candidates.is_empty(),
        "comparison tree {tree_id} has no strict regular-variable split in Cartesian state {state_id}"
    );
    if semantics.split_selection() == CartesianSplitSelection::MinTransitionGrowth {
        retain_min_growth_splits(working, semantics, &mut candidates, |(_, _, split)| split)?;
    }
    let has_target_centered_candidate = semantics.refinement_direction()
        == CartesianRefinementDirection::Regression
        && candidates
            .iter()
            .any(|(separates_truth, _, _)| *separates_truth);
    if has_target_centered_candidate {
        candidates.retain(|(separates_truth, _, _)| *separates_truth);
    }
    let has_achieving_candidate = candidates.iter().any(|(_, achieved, _)| *achieved);
    if has_achieving_candidate {
        candidates.retain(|(_, achieved, _)| *achieved);
    }
    select_refinement_split(
        working,
        semantics,
        candidates.into_iter().map(|(_, _, split)| split).collect(),
        0x434F_4D50,
    )
}

pub(super) fn comparison_truth(prop_value: usize) -> Result<bool> {
    match prop_value {
        0 => Ok(true),
        1 => Ok(false),
        _ => bail!("invalid comparison fact value {prop_value}"),
    }
}

fn conditions_hold_concretely(conditions: &[ExplicitFact], prop_values: &[usize]) -> Result<bool> {
    for condition in conditions {
        let value = *prop_values
            .get(condition.var())
            .with_context(|| format!("missing concrete prop var {}", condition.var()))?;
        if value != condition.value() {
            return Ok(false);
        }
    }
    Ok(true)
}

fn all_conditions_admitted(
    semantics: &CartesianSemantics<'_>,
    region: &StateRegion,
    conditions: &[ExplicitFact],
) -> Result<bool> {
    for condition in conditions {
        if !semantics.region_admits_fact(region, condition)? {
            return Ok(false);
        }
    }
    Ok(true)
}

#[derive(Clone, Copy)]
pub(super) struct DeviationWitness<'a> {
    source_state_id: usize,
    target_state_id: usize,
    op_id: usize,
    successor_prop: &'a [usize],
    source_numeric: &'a [f64],
    successor_numeric: &'a [f64],
}

impl<'a> DeviationWitness<'a> {
    pub(super) fn new(
        source_state_id: usize,
        target_state_id: usize,
        op_id: usize,
        successor_prop: &'a [usize],
        source_numeric: &'a [f64],
        successor_numeric: &'a [f64],
    ) -> Self {
        Self {
            source_state_id,
            target_state_id,
            op_id,
            successor_prop,
            source_numeric,
            successor_numeric,
        }
    }
}

pub(super) fn split_deviation(
    working: &WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    witness: DeviationWitness<'_>,
) -> Result<Split> {
    let candidates = split_deviation_candidates(working, semantics, witness)?;
    select_refinement_split(working, semantics, candidates, 0x4445_5649)
}

pub(super) fn split_deviation_candidates(
    working: &WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    witness: DeviationWitness<'_>,
) -> Result<Vec<Split>> {
    let DeviationWitness {
        source_state_id,
        target_state_id,
        op_id,
        successor_prop,
        source_numeric,
        successor_numeric,
    } = witness;
    let target = &working.states()[target_state_id];
    let mut candidates = Vec::new();
    let mut rejected_numeric_splits = Vec::new();
    for (var_id, allowed) in target.propositions.iter().enumerate() {
        if semantics
            .task()
            .numeric_conditions()
            .is_condition_var(var_id)
            || !semantics.propositional_axioms_by_prop_var()[var_id].is_empty()
        {
            continue;
        }
        let value = successor_prop[var_id] as PropValueId;
        if allowed.binary_search(&value).is_err() {
            let op = &semantics.task().get_operators()[op_id];
            let unaffected = !op.effects().iter().any(|effect| effect.var_id() == var_id);
            ensure!(
                unaffected,
                "operator {op_id} effect image admitted wrong target prop region for var {var_id}"
            );
            candidates.push(Split::Propositional {
                state_id: source_state_id,
                var_id,
                wanted: allowed.clone(),
                witness_value: value,
                description: format!(
                    "operator {op_id} successor prop var {var_id}={value} outside target {allowed:?}"
                ),
            });
        }
    }

    for (var_id, target_interval) in target.numeric.iter().copied().enumerate() {
        let successor = successor_numeric[var_id];
        if target_interval.contains(successor) {
            continue;
        }
        let preimage = semantics
            .numeric_effect_preimage(target_interval, op_id, var_id)?
            .with_context(|| {
                format!(
                    "Cartesian transition for operator {op_id} has no numeric preimage for var {var_id} and target {target_interval:?}"
                )
            })?;
        let source = source_numeric[var_id];
        if preimage.contains(source) {
            rejected_numeric_splits.push(format!(
                "var {var_id}: source={source}, successor={successor}, target={target_interval:?}, preimage={preimage:?} contains source"
            ));
            continue;
        }
        let (boundary, lower_includes_boundary) =
            if source < preimage.lower || (source == preimage.lower && !preimage.lower_closed) {
                (preimage.lower, !preimage.lower_closed)
            } else {
                ensure!(
                    source > preimage.upper || (source == preimage.upper && !preimage.upper_closed),
                    "numeric successor mismatch has no separating preimage boundary"
                );
                (preimage.upper, preimage.upper_closed)
            };
        let parent = working.states()[source_state_id].numeric[var_id];
        ensure!(
            parent.contains(source),
            "Cartesian source state {source_state_id} interval {parent:?} does not contain concrete numeric var {var_id}={source}"
        );
        if !boundary.is_finite() {
            rejected_numeric_splits.push(format!(
                "var {var_id}: source={source}, successor={successor}, target={target_interval:?}, preimage={preimage:?}, parent={parent:?} has only infinite separating boundary"
            ));
            continue;
        }
        let integer_lattice = semantics.numeric_integer_lattice()[var_id];
        if numeric_split_intervals(parent, boundary, lower_includes_boundary, integer_lattice)
            .is_err()
        {
            rejected_numeric_splits.push(format!(
                "var {var_id}: source={source}, successor={successor}, target={target_interval:?}, preimage={preimage:?}, parent={parent:?}, boundary={boundary}, lower_includes_boundary={lower_includes_boundary} is not strict"
            ));
            continue;
        }
        candidates.push(Split::Numeric {
            state_id: source_state_id,
            var_id,
            boundary,
            lower_includes_boundary,
            witness_value: source,
            desired_contains_witness: false,
            integer_lattice,
            description: format!(
                "operator {op_id} successor numeric var {var_id}={successor} outside target {target_interval:?}"
            ),
        });
    }
    ensure!(
        !candidates.is_empty(),
        "concrete successor maps from Cartesian state {source_state_id} to a state other than abstract target {target_state_id}, but no sound strict split exists for operator {op_id} ({}); numeric split rejections: [{}]",
        semantics.task().get_operators()[op_id].name(),
        rejected_numeric_splits.join("; ")
    );
    Ok(candidates)
}
