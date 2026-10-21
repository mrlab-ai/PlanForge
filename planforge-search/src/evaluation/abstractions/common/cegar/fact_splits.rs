//! The splits that refine a flaw: separate a concrete witness from the facts
//! it violates, or from the abstract successor it misses.

use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
use planforge_sas::numeric_task::{
    ExplicitFact, ExplicitValueIndex, NumericValue, OperatorIndex, VariableIndex,
};
use planforge_sas::utils::float_tolerance;

use super::SplitKind;
use super::SplitSelection;
use super::selection::{retain_min_growth_splits, select_refinement_split};
use super::semantics::CegarSemantics;
use super::split::{Split, numeric_split_intervals};
use super::view::AbstractionView;
use crate::evaluation::abstractions::common::cartesian_set::CartesianSet;

pub(crate) fn split_failed_fact(
    view: &impl AbstractionView,
    semantics: &CegarSemantics<'_>,
    state_id: usize,
    fact: &ExplicitFact,
    prop_values: &[ExplicitValueIndex],
    numeric_values: &[NumericValue],
) -> Result<Split> {
    if let Some(tree_id) = semantics
        .task()
        .numeric_conditions()
        .id_for_var(fact.var_index())
    {
        return comparison_refinement(
            view,
            semantics,
            state_id,
            tree_id,
            numeric_values,
            ComparisonRefinementGoal::exclude(fact.value_index())?,
        );
    }
    if !semantics.regions().axioms_for(fact.var_index()).is_empty() {
        let default_value = semantics.regions().axiom_default(fact.var_index())?;
        if fact.value_index() == default_value {
            let concrete_value = *prop_values
                .get(fact.var())
                .with_context(|| format!("missing concrete prop var {}", fact.var()))?;
            ensure!(
                concrete_value != default_value,
                "failed default-valued derived fact unexpectedly holds for variable {}",
                fact.var()
            );
            for &axiom_id in semantics.regions().axioms_for(fact.var_index()) {
                let axiom = &semantics.task().axioms()[axiom_id];
                if axiom.effect_value() != concrete_value
                    || !conditions_hold_concretely(axiom.conditions(), prop_values)?
                {
                    continue;
                }
                for condition in axiom.conditions() {
                    if !semantics
                        .regions()
                        .guarantees(&view.region(state_id), condition)?
                    {
                        return split_to_guarantee_fact(
                            view,
                            semantics,
                            state_id,
                            condition,
                            prop_values,
                            numeric_values,
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
        for &axiom_id in semantics.regions().axioms_for(fact.var_index()) {
            let axiom = &semantics.task().axioms()[axiom_id];
            if axiom.effect_value() != fact.value_index()
                || !all_conditions_admitted(semantics, &view.region(state_id), axiom.conditions())?
            {
                continue;
            }
            for condition in axiom.conditions() {
                let value = *prop_values
                    .get(condition.var())
                    .with_context(|| format!("missing concrete prop var {}", condition.var()))?;
                if value != condition.value_index() {
                    return split_failed_fact(
                        view,
                        semantics,
                        state_id,
                        condition,
                        prop_values,
                        numeric_values,
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
        as ExplicitValueIndex;
    ensure!(
        witness_value != fact.value_index(),
        "failed fact split witness unexpectedly satisfies {fact:?}"
    );
    Ok(Split::Propositional {
        state_id,
        var_id: fact.var_index(),
        wanted: vec![fact.value_index()],
        witness_value,
    })
}

fn split_to_guarantee_fact(
    view: &impl AbstractionView,
    semantics: &CegarSemantics<'_>,
    state_id: usize,
    fact: &ExplicitFact,
    prop_values: &[ExplicitValueIndex],
    numeric_values: &[NumericValue],
) -> Result<Split> {
    let concrete_value = *prop_values
        .get(fact.var())
        .with_context(|| format!("missing concrete prop var {}", fact.var()))?;
    ensure!(
        concrete_value == fact.value_index(),
        "cannot guarantee fact {fact:?}: concrete value is {concrete_value}"
    );
    if let Some(tree_id) = semantics
        .task()
        .numeric_conditions()
        .id_for_var(fact.var_index())
    {
        return comparison_refinement(
            view,
            semantics,
            state_id,
            tree_id,
            numeric_values,
            ComparisonRefinementGoal::guarantee(fact.value_index())?,
        );
    }
    if !semantics.regions().axioms_for(fact.var_index()).is_empty() {
        let default_value = semantics.regions().axiom_default(fact.var_index())?;
        if fact.value_index() == default_value {
            for &axiom_id in semantics.regions().axioms_for(fact.var_index()) {
                let axiom = &semantics.task().axioms()[axiom_id];
                if !all_conditions_admitted(semantics, &view.region(state_id), axiom.conditions())?
                {
                    continue;
                }
                let condition = axiom
                    .conditions()
                    .iter()
                    .find(|condition| {
                        prop_values
                            .get(condition.var())
                            .is_some_and(|&value| value != condition.value_index())
                    })
                    .with_context(|| {
                        format!(
                            "concrete default value for derived variable {} conflicts with firing axiom {axiom_id}",
                            fact.var()
                        )
                    })?;
                let witness_value = prop_values[condition.var()];
                let witness_fact =
                    ExplicitFact::propositional(condition.var(), witness_value.index());
                return split_to_guarantee_fact(
                    view,
                    semantics,
                    state_id,
                    &witness_fact,
                    prop_values,
                    numeric_values,
                );
            }
            bail!(
                "derived default fact {fact:?} is not guaranteed although no competing axiom is admitted"
            );
        }

        for &axiom_id in semantics.regions().axioms_for(fact.var_index()) {
            let axiom = &semantics.task().axioms()[axiom_id];
            if axiom.effect_value() != fact.value_index()
                || !conditions_hold_concretely(axiom.conditions(), prop_values)?
            {
                continue;
            }
            for condition in axiom.conditions() {
                if !semantics
                    .regions()
                    .guarantees(&view.region(state_id), condition)?
                {
                    return split_to_guarantee_fact(
                        view,
                        semantics,
                        state_id,
                        condition,
                        prop_values,
                        numeric_values,
                    );
                }
            }
            bail!(
                "derived fact {fact:?} is not guaranteed although supporting axiom {axiom_id} is guaranteed"
            );
        }
        bail!("concrete derived fact {fact:?} has no supporting axiom");
    }

    let witness_value = concrete_value as ExplicitValueIndex;
    let region = view.region(state_id);
    let allowed = &region.propositions()[fact.var()];
    ensure!(
        allowed.binary_search(&witness_value).is_ok() && allowed.len() > 1,
        "fact {fact:?} is already guaranteed in abstract state {state_id}"
    );
    Ok(Split::Propositional {
        state_id,
        var_id: fact.var_index(),
        wanted: vec![witness_value],
        witness_value,
    })
}

#[derive(Debug, Clone, Copy)]
enum ComparisonRefinementGoal {
    ExcludeDesired(bool),
    GuaranteeDesired(bool),
}

impl ComparisonRefinementGoal {
    fn exclude(prop_value: ExplicitValueIndex) -> Result<Self> {
        Ok(Self::ExcludeDesired(comparison_truth(prop_value)?))
    }

    fn guarantee(prop_value: ExplicitValueIndex) -> Result<Self> {
        Ok(Self::GuaranteeDesired(comparison_truth(prop_value)?))
    }

    fn desired_truth(self) -> bool {
        match self {
            Self::ExcludeDesired(truth) | Self::GuaranteeDesired(truth) => truth,
        }
    }
}

fn comparison_refinement(
    view: &impl AbstractionView,
    semantics: &CegarSemantics<'_>,
    state_id: usize,
    tree_id: usize,
    numeric_values: &[NumericValue],
    goal: ComparisonRefinementGoal,
) -> Result<Split> {
    let desired_truth = goal.desired_truth();
    let tree = semantics
        .task()
        .numeric_conditions()
        .get(tree_id)
        .with_context(|| format!("missing comparison tree {tree_id}"))?;
    let concrete_truth = tree.evaluate_point(numeric_values);
    ensure!(
        matches!(goal, ComparisonRefinementGoal::ExcludeDesired(_))
            == (concrete_truth != desired_truth),
        "comparison refinement goal disagrees with concrete truth for tree {tree_id}"
    );
    let state = view.region(state_id);
    let mut candidates = Vec::new();
    // The coordinates the condition is over, or its regular operands when it
    // is over no coordinate.
    for var_id in semantics.coordinates().refinement_dimensions(tree) {
        let witness_value =
            float_tolerance::canonicalize(match &semantics.coordinates().views()[var_id.index()] {
                Some(view) => view.evaluate(numeric_values),
                None => *numeric_values
                    .get(var_id.index())
                    .with_context(|| format!("missing concrete numeric var {var_id}"))?,
            });
        ensure!(
            witness_value.is_finite(),
            "comparison split witness for numeric var {var_id} is non-finite: {witness_value}"
        );
        let parent = *state
            .numeric
            .get(var_id.index())
            .with_context(|| format!("missing numeric var {var_id} in the abstract state"))?;
        let mut boundaries = Vec::new();
        if semantics.split_kind() == SplitKind::Wanted {
            boundaries.extend(semantics.wanted_split_boundaries().iter().copied());
        }
        boundaries.push(witness_value);
        boundaries.sort_by(NumericValue::total_cmp);
        boundaries.dedup_by(|left, right| {
            float_tolerance::canonical_bits(*left) == float_tolerance::canonical_bits(*right)
        });

        let integer_lattice = semantics.numeric_integer_lattice()[var_id.index()];
        for boundary in boundaries {
            for lower_includes_boundary in [true, false] {
                let Some((lower, upper)) = numeric_split_intervals(
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
                Arc::make_mut(&mut child_numeric)[var_id.index()] = witness_child;
                let witness_result = tree.evaluate_interval(&child_numeric);
                ensure!(
                    witness_result != Some(!concrete_truth),
                    "comparison interval for tree {tree_id} excludes its concrete witness after splitting numeric var {var_id}"
                );
                Arc::make_mut(&mut child_numeric)[var_id.index()] = other_child;
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
                    integer_lattice,
                };
                candidates.push((separates_truth, achieved, candidate));
            }
        }
    }
    ensure!(
        !candidates.is_empty(),
        "comparison tree {tree_id} has no strict split in abstract state {state_id}"
    );
    if semantics.split_selection() == SplitSelection::MinGrowth {
        retain_min_growth_splits(view, &mut candidates, |(_, _, split)| split)?;
    }
    let has_wanted_candidate = semantics.split_kind() == SplitKind::Wanted
        && candidates
            .iter()
            .any(|(separates_truth, _, _)| *separates_truth);
    if has_wanted_candidate {
        candidates.retain(|(separates_truth, _, _)| *separates_truth);
    }
    let has_achieving_candidate = candidates.iter().any(|(_, achieved, _)| *achieved);
    if has_achieving_candidate {
        candidates.retain(|(_, achieved, _)| *achieved);
    }
    select_refinement_split(
        view,
        semantics,
        candidates.into_iter().map(|(_, _, split)| split).collect(),
        0x434F_4D50,
    )
}

pub(crate) fn comparison_truth(prop_value: ExplicitValueIndex) -> Result<bool> {
    match prop_value.index() {
        0 => Ok(true),
        1 => Ok(false),
        _ => bail!("invalid comparison fact value {prop_value}"),
    }
}

fn conditions_hold_concretely(
    conditions: &[ExplicitFact],
    prop_values: &[ExplicitValueIndex],
) -> Result<bool> {
    for condition in conditions {
        let value = *prop_values
            .get(condition.var())
            .with_context(|| format!("missing concrete prop var {}", condition.var()))?;
        if value != condition.value_index() {
            return Ok(false);
        }
    }
    Ok(true)
}

fn all_conditions_admitted(
    semantics: &CegarSemantics<'_>,
    region: &CartesianSet,
    conditions: &[ExplicitFact],
) -> Result<bool> {
    for condition in conditions {
        if !semantics.regions().admits(region, condition)? {
            return Ok(false);
        }
    }
    Ok(true)
}

#[derive(Clone, Copy)]
pub(crate) struct DeviationWitness<'a> {
    source_state_id: usize,
    target_state_id: usize,
    op_id: OperatorIndex,
    successor_prop: &'a [ExplicitValueIndex],
    source_numeric: &'a [NumericValue],
    successor_numeric: &'a [NumericValue],
}

impl<'a> DeviationWitness<'a> {
    pub(crate) fn new(
        source_state_id: usize,
        target_state_id: usize,
        op_id: OperatorIndex,
        successor_prop: &'a [ExplicitValueIndex],
        source_numeric: &'a [NumericValue],
        successor_numeric: &'a [NumericValue],
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

pub(crate) fn split_deviation_candidates(
    view: &impl AbstractionView,
    semantics: &CegarSemantics<'_>,
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
    let target = &*view.region(target_state_id);
    let mut candidates = Vec::new();
    let mut rejected_numeric_splits = Vec::new();
    for (var_id, allowed) in target.propositions().iter().enumerate() {
        let var_id = VariableIndex::from_usize(var_id);
        if semantics
            .task()
            .numeric_conditions()
            .is_condition_var(var_id)
            || !semantics.regions().axioms_for(var_id).is_empty()
        {
            continue;
        }
        let value = successor_prop[var_id.index()];
        if allowed.binary_search(&value).is_err() {
            let op = &semantics.task().get_operators()[op_id.index()];
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
            });
        }
    }

    for (var_id, target_interval) in target.numeric.iter().copied().enumerate() {
        let var_id = VariableIndex::from_usize(var_id);
        let successor = successor_numeric[var_id.index()];
        if target_interval.contains(successor) {
            continue;
        }
        let preimage = semantics
            .coordinates()
            .preimage(target_interval, op_id, var_id)?
            .with_context(|| {
                format!(
                    "abstract transition for operator {op_id} has no numeric preimage for var {var_id} and target {target_interval:?}"
                )
            })?;
        let source = source_numeric[var_id.index()];
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
        let parent = view.region(source_state_id).numeric[var_id.index()];
        ensure!(
            parent.contains(source),
            "abstract source state {source_state_id} interval {parent:?} does not contain concrete numeric var {var_id}={source}"
        );
        if !boundary.is_finite() {
            rejected_numeric_splits.push(format!(
                "var {var_id}: source={source}, successor={successor}, target={target_interval:?}, preimage={preimage:?}, parent={parent:?} has only infinite separating boundary"
            ));
            continue;
        }
        let integer_lattice = semantics.numeric_integer_lattice()[var_id.index()];
        if numeric_split_intervals(parent, boundary, lower_includes_boundary, integer_lattice)
            .is_none()
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
            integer_lattice,
        });
    }
    ensure!(
        !candidates.is_empty(),
        "concrete successor maps from abstract state {source_state_id} to a state other than abstract target {target_state_id}, but no sound strict split exists for operator {op_id} ({}); numeric split rejections: [{}]",
        semantics.task().get_operators()[op_id.index()].name(),
        rejected_numeric_splits.join("; ")
    );
    Ok(candidates)
}
