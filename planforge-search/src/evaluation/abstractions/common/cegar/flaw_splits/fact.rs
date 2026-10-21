//! Fact splits: separate a concrete witness from a fact it violates. A fact
//! over a comparison is split on the comparison's operands, and a derived
//! fact through the axioms that derive its variable, down to a fact of an
//! operator variable.

use anyhow::{Context, Result, bail, ensure};
use planforge_sas::numeric_task::ExplicitFact;

use super::ConcreteWitness;
use super::comparison::split_comparison;
use crate::evaluation::abstractions::common::cegar::semantics::CegarSemantics;
use crate::evaluation::abstractions::common::cegar::split::Split;
use crate::evaluation::abstractions::common::cegar::view::AbstractionView;

/// What a split does with a fact in the child that keeps the witness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FactGoal {
    /// The witness violates the fact, and its child excludes it.
    Exclude,
    /// The witness satisfies the fact, and its child guarantees it.
    Guarantee,
}

/// The split of abstract state `state_id` whose child with the concrete
/// `witness` excludes `fact`, which the witness violates.
pub(crate) fn split_failed_fact(
    view: &impl AbstractionView,
    semantics: &CegarSemantics<'_>,
    state_id: usize,
    fact: &ExplicitFact,
    witness: ConcreteWitness<'_>,
) -> Result<Split> {
    split_on_fact(view, semantics, state_id, fact, FactGoal::Exclude, witness)
}

/// The split of abstract state `state_id` whose child with the concrete
/// `witness` meets `goal` for `fact`.
fn split_on_fact(
    view: &impl AbstractionView,
    semantics: &CegarSemantics<'_>,
    state_id: usize,
    fact: &ExplicitFact,
    goal: FactGoal,
    witness: ConcreteWitness<'_>,
) -> Result<Split> {
    let witness_value = witness.value(fact);
    ensure!(
        (witness_value == fact.value_index()) == (goal == FactGoal::Guarantee),
        "cannot {goal:?} {fact:?} for a witness whose value is {witness_value}"
    );
    let var_id = fact.var_index();
    if let Some(tree_id) = semantics.task().numeric_conditions().id_for_var(var_id) {
        return split_comparison(
            view,
            semantics,
            state_id,
            tree_id,
            witness_value,
            witness.numeric,
        );
    }
    let axiom_ids = semantics.regions().axioms_for(var_id);
    if axiom_ids.is_empty() {
        let wanted = match goal {
            FactGoal::Exclude => fact.value_index(),
            FactGoal::Guarantee => {
                let region = view.region(state_id);
                let allowed = &region.propositions()[var_id.index()];
                ensure!(
                    allowed.binary_search(&witness_value).is_ok() && allowed.len() > 1,
                    "fact {fact:?} is already guaranteed in abstract state {state_id}"
                );
                witness_value
            }
        };
        return Ok(Split::Propositional {
            state_id,
            var_id,
            wanted: vec![wanted],
            witness_value,
        });
    }

    let is_default = fact.value_index() == semantics.regions().axiom_default(var_id)?;
    if goal == FactGoal::Exclude && is_default {
        // The witness's value is derived by an axiom whose conditions it
        // satisfies, so guaranteeing that value excludes the default.
        let derived = ExplicitFact::propositional(var_id.index(), witness_value.index());
        return split_on_fact(
            view,
            semantics,
            state_id,
            &derived,
            FactGoal::Guarantee,
            witness,
        );
    }
    let axioms = semantics.task().axioms();
    let region = view.region(state_id);
    if goal == FactGoal::Guarantee && !is_default {
        // Guarantee every condition of an axiom that derives the fact for
        // the witness.
        let &axiom_id = axiom_ids
            .iter()
            .find(|&&axiom_id| {
                axioms[axiom_id].effect_value() == fact.value_index()
                    && witness.satisfies_all(axioms[axiom_id].conditions())
            })
            .with_context(|| format!("concrete derived fact {fact:?} has no supporting axiom"))?;
        for condition in axioms[axiom_id].conditions() {
            if !semantics.regions().guarantees(&region, condition)? {
                return split_on_fact(
                    view,
                    semantics,
                    state_id,
                    condition,
                    FactGoal::Guarantee,
                    witness,
                );
            }
        }
        bail!(
            "derived fact {fact:?} is not guaranteed although supporting axiom {axiom_id} is guaranteed"
        );
    }
    // The fact fails for the witness and is not the default, or it is the
    // default and holds: an axiom the abstract state admits does not fire
    // for the witness. Split off the witness on a condition it violates.
    for &axiom_id in axiom_ids {
        let axiom = &axioms[axiom_id];
        if (goal == FactGoal::Exclude && axiom.effect_value() != fact.value_index())
            || !semantics
                .regions()
                .admits_all(&region, axiom.conditions())?
        {
            continue;
        }
        let condition = axiom
            .conditions()
            .iter()
            .find(|condition| witness.value(condition) != condition.value_index())
            .with_context(|| {
                format!("axiom {axiom_id} fires for a witness that violates {fact:?}")
            })?;
        return match goal {
            FactGoal::Exclude => split_on_fact(view, semantics, state_id, condition, goal, witness),
            FactGoal::Guarantee => {
                let violated =
                    ExplicitFact::propositional(condition.var(), witness.value(condition).index());
                split_on_fact(view, semantics, state_id, &violated, goal, witness)
            }
        };
    }
    bail!("abstract state {state_id} meets {goal:?} for {fact:?} on no admitted axiom")
}
