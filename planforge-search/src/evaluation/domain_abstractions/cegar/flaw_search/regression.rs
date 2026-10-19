//! Regression flaw search for domain abstractions; see
//! [`crate::evaluation::cegar::regression`] for the flaw definition.

#[cfg(test)]
mod tests;

use std::cell::OnceCell;
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
use planforge_sas::axioms::AxiomEvaluator;
use planforge_sas::numeric_conditions::{ConditionValue, NumericCondition};
use planforge_sas::numeric_task::{
    AbstractNumericTask, ExplicitFact, NumericValue, OperatorIndex, VariableIndex,
};
use planforge_sas::utils::float_tolerance;
use planforge_sas::utils::interval::Interval;
use planforge_sas::utils::state_packer::StatePacker;

use super::progression::{ExecutedTrace, trace_progression_flaws};
use super::target_centered::split_for_missing_numeric_requirement;
use super::{
    Flaw, FlawSearchOutcome, NumericFlaw, PropFlaw, SplitDirection, can_split_numeric_var,
    numeric_requirement_for_comparison_fact,
};
use crate::evaluation::cegar::regression::{
    FailedConditions, failed_conditions_on_relaxed_suffix, progress_numeric_box,
    regress_numeric_interval, regression_candidates, writes_propositional_variable,
};
use crate::evaluation::domain_abstractions::{
    abstract_operator_generator::DomainMapping,
    additive_numeric_views::{
        AdditiveNumericView, analyze_additive_numeric_view, comparison_refinement_dimensions,
    },
    domain_abstraction::NumericPartitions,
    domain_abstraction_factory::WildcardPlanResult,
    utils::{make_prop_state_packer, partition_for_value},
};

/// Regression flaw search: the failed conditions of the wildcard plan's
/// relaxed suffix, regressed to the latest prefix state whose abstract state
/// a split can decide them in. The pair `(k, k)` -- the conditions `s_k`
/// itself violates -- is the progression flaw, so the search returns the
/// progression flaws when no later condition regresses to a split.
///
/// The relaxed suffix applies the first operator of every remaining wildcard
/// step.
pub fn get_regression_flaws(
    task: &dyn AbstractNumericTask,
    partitions: &NumericPartitions,
    domain_mapping: &DomainMapping,
    wildcard_plan: &WildcardPlanResult,
    direction: SplitDirection,
) -> Result<FlawSearchOutcome> {
    let mut trace = ExecutedTrace::default();
    let progression =
        trace_progression_flaws(task, partitions, wildcard_plan, direction, Some(&mut trace))?;
    if progression.flaws.is_empty() {
        return Ok(progression);
    }
    let last_anchor = trace.operator_ids.len();
    ensure!(
        trace.states.len() == last_anchor + 1,
        "a trace with {last_anchor} operators has {} states",
        trace.states.len()
    );
    let mut plan_operator_ids = trace.operator_ids.clone();
    for (offset, step_operator_ids) in wildcard_plan.wildcard_plan[last_anchor..]
        .iter()
        .enumerate()
    {
        plan_operator_ids.push(*step_operator_ids.first().with_context(|| {
            format!(
                "wildcard plan step {} has no operator",
                last_anchor + offset + 1
            )
        })?);
    }

    let packer = Arc::new(make_prop_state_packer(task));
    let axiom_evaluator = AxiomEvaluator::new(Arc::new(task), packer.clone());
    let (propositions, numeric) = trace.states[last_anchor].clone();
    let failed = failed_conditions_on_relaxed_suffix(
        task,
        &axiom_evaluator,
        &packer,
        propositions,
        numeric,
        last_anchor,
        &plan_operator_ids[last_anchor..],
    )?;
    let search = RegressionFlawSearch {
        task,
        partitions,
        domain_mapping,
        packer: &packer,
        direction,
        additive_views: OnceCell::new(),
    };
    for (anchor, conditions) in regression_candidates(&failed, last_anchor) {
        if anchor == last_anchor && conditions.step == last_anchor {
            break;
        }
        let flaws = search.flaws(
            anchor,
            &trace.states[anchor],
            conditions,
            &plan_operator_ids[anchor..conditions.step],
        )?;
        if !flaws.is_empty() {
            return Ok(FlawSearchOutcome {
                flaws,
                concrete_plan: None,
            });
        }
    }
    Ok(progression)
}

struct RegressionFlawSearch<'a> {
    task: &'a dyn AbstractNumericTask,
    partitions: &'a NumericPartitions,
    domain_mapping: &'a DomainMapping,
    packer: &'a StatePacker,
    direction: SplitDirection,
    /// Computed on the first regression through an operator.
    additive_views: OnceCell<Vec<Option<AdditiveNumericView>>>,
}

impl RegressionFlawSearch<'_> {
    fn additive_views(&self) -> &[Option<AdditiveNumericView>] {
        self.additive_views.get_or_init(|| {
            (0..self.task.numeric_variables().len())
                .map(|var_id| {
                    analyze_additive_numeric_view(self.task, VariableIndex::from_usize(var_id))
                })
                .collect()
        })
    }

    /// The flaws that split the abstract state of the anchor `s_anchor` so
    /// that it decides `conditions`, regressed through `ops`, as violated.
    /// They carry the step of the operator after the anchor, like the
    /// precondition flaws progression finds in `s_anchor`. A propositional
    /// condition on a variable that `ops` writes regresses to a constant,
    /// which no split decides.
    fn flaws(
        &self,
        anchor: usize,
        (propositions, numeric): &(Vec<u64>, Vec<NumericValue>),
        conditions: &FailedConditions,
        ops: &[OperatorIndex],
    ) -> Result<Vec<Flaw>> {
        let step = anchor + 1;
        let mut flaws = Vec::new();
        for fact in &conditions.facts {
            if let Some(tree) = self.task.numeric_conditions().for_var(fact.var_index()) {
                let dependent_numeric_flaws = match self.direction {
                    SplitDirection::Forward | SplitDirection::ForwardPartitionDeviation => {
                        self.concrete_value_splits(tree, fact, numeric, ops, step)?
                    }
                    SplitDirection::Backward => {
                        self.wanted_value_splits(fact, numeric, ops, step)?
                    }
                };
                if !dependent_numeric_flaws.is_empty() {
                    flaws.push(Flaw::Propositional(PropFlaw {
                        fact: *fact,
                        dependent_numeric_flaws,
                        step,
                    }));
                }
                continue;
            }
            if writes_propositional_variable(self.task, ops, fact.var_index()) {
                continue;
            }
            let value = self.packer.get(propositions, fact.var()) as usize;
            ensure!(
                value != fact.value(),
                "regressed fact {fact:?} holds in the anchor state although no operator writes it"
            );
            let mapping = &self.domain_mapping[fact.var()];
            if mapping[fact.value()] == mapping[value] {
                flaws.push(Flaw::Propositional(PropFlaw {
                    fact: *fact,
                    dependent_numeric_flaws: Vec::new(),
                    step,
                }));
            }
        }
        Ok(flaws)
    }

    /// The partition of every numeric variable that contains its value in
    /// `numeric`: the numeric part of the anchor's abstract state.
    fn abstract_box(&self, numeric: &[NumericValue]) -> Result<Vec<Interval>> {
        numeric
            .iter()
            .enumerate()
            .map(|(var_id, &value)| {
                let parts = self
                    .partitions
                    .partitions(VariableIndex::from_usize(var_id))
                    .with_context(|| format!("numeric variable {var_id} has no partitions"))?;
                let part_id = partition_for_value(parts, value).with_context(|| {
                    format!("no partition of numeric variable {var_id} contains {value}")
                })?;
                Ok(parts[part_id])
            })
            .collect()
    }

    /// One split per refinable coordinate of `tree`, at the anchor's value,
    /// that makes the anchor's partition decide the regressed comparison as
    /// violated.
    fn concrete_value_splits(
        &self,
        tree: &NumericCondition,
        fact: &ExplicitFact,
        numeric: &[NumericValue],
        ops: &[OperatorIndex],
        step: usize,
    ) -> Result<Vec<NumericFlaw>> {
        let desired = fact.value() == ConditionValue::True.as_usize();
        let mut abstract_box = self.abstract_box(numeric)?;
        let mut progressed = Vec::with_capacity(abstract_box.len());
        let mut violated_throughout = |intervals: &[Interval]| -> Result<bool> {
            progressed.clear();
            progressed.extend_from_slice(intervals);
            for &op_id in ops {
                progress_numeric_box(self.task, self.additive_views(), op_id, &mut progressed)?;
            }
            Ok(tree.evaluate_interval(&progressed) == Some(!desired))
        };
        if violated_throughout(&abstract_box)? {
            return Ok(Vec::new());
        }
        let mut flaws = Vec::new();
        for numeric_var_id in comparison_refinement_dimensions(self.task, tree) {
            let value = float_tolerance::canonicalize(numeric[numeric_var_id.index()]);
            let partition = abstract_box[numeric_var_id.index()];
            for include_in_lower in [true, false] {
                if !can_split_numeric_var(self.partitions, numeric_var_id, value, include_in_lower)
                {
                    continue;
                }
                abstract_box[numeric_var_id.index()] = if include_in_lower {
                    Interval::new(partition.lower, value, partition.lower_closed, true)
                } else {
                    Interval::new(value, partition.upper, true, partition.upper_closed)
                };
                let decides = violated_throughout(&abstract_box)?;
                abstract_box[numeric_var_id.index()] = partition;
                if decides {
                    flaws.push(NumericFlaw {
                        numeric_var_id,
                        value,
                        include_in_lower,
                        step,
                    });
                    break;
                }
            }
        }
        Ok(flaws)
    }

    /// The split at the boundary of the comparison's wanted values, regressed
    /// through `ops`, that faces the anchor's value.
    fn wanted_value_splits(
        &self,
        fact: &ExplicitFact,
        numeric: &[NumericValue],
        ops: &[OperatorIndex],
        step: usize,
    ) -> Result<Vec<NumericFlaw>> {
        let Some((numeric_var_id, wanted)) =
            numeric_requirement_for_comparison_fact(self.task, fact)
        else {
            bail!(
                "wanted-value regression splits need a comparison of one coordinate with a constant, but {fact:?} is not one"
            );
        };
        let numeric_var_id = VariableIndex::from_usize(numeric_var_id);
        let Some(regressed) = regress_numeric_interval(
            self.task,
            self.additive_views(),
            wanted,
            ops,
            numeric_var_id,
        )?
        else {
            return Ok(Vec::new());
        };
        let value = float_tolerance::canonicalize(numeric[numeric_var_id.index()]);
        let parts = self
            .partitions
            .partitions(numeric_var_id)
            .with_context(|| format!("numeric variable {numeric_var_id} has no partitions"))?;
        let partition = parts[partition_for_value(parts, value).with_context(|| {
            format!("no partition of numeric variable {numeric_var_id} contains {value}")
        })?];
        if !partition.intersects(&regressed) {
            return Ok(Vec::new());
        }
        ensure!(
            !regressed.contains(value),
            "the anchor value {value} of numeric variable {numeric_var_id} satisfies the regressed requirement {regressed:?} of {fact:?}"
        );
        let (value, include_in_lower) = split_for_missing_numeric_requirement(regressed, value)
            .with_context(|| {
                format!("regressed requirement {regressed:?} of {fact:?} has no finite boundary facing {value}")
            })?;
        Ok(
            can_split_numeric_var(self.partitions, numeric_var_id, value, include_in_lower)
                .then_some(NumericFlaw {
                    numeric_var_id,
                    value,
                    include_in_lower,
                    step,
                })
                .into_iter()
                .collect(),
        )
    }
}
