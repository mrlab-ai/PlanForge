//! Regression flaws for domain abstractions; see
//! [`crate::evaluation::abstractions::common::cegar::regression`] for the definition.

#[cfg(test)]
mod tests;

use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use planforge_sas::axioms::AxiomEvaluator;
use planforge_sas::numeric_task::{
    AbstractNumericTask, ExplicitFact, ExplicitValueIndex, OperatorIndex, VariableIndex,
};

use super::{Flaw, FlawSearchOutcome, NumericFlaw, PropFlaw, SplitDirection};
use crate::evaluation::abstractions::common::cartesian_set::CartesianSet;
use crate::evaluation::abstractions::common::cegar::regression::{
    Cut, InitialCut, InitialState, find_regression_flaw,
};
use crate::evaluation::abstractions::common::concrete_state::{
    get_initial_state, make_prop_state_packer,
};
use crate::evaluation::abstractions::common::coordinates::Coordinates;
use crate::evaluation::abstractions::domain::{
    abstract_operators::DomainMapping, factory::WildcardPlanResult, partitions::NumericPartitions,
};

/// Regression flaw search on the wildcard plan, regressing through the first
/// operator of every step. Without a flaw, those operators are a concrete
/// plan. Concrete-value splits cut next to the initial value in case 3;
/// wanted-value splits cut at the regressed bound.
pub fn get_regression_flaws(
    task: &dyn AbstractNumericTask,
    partitions: &NumericPartitions,
    domain_mapping: &DomainMapping,
    wildcard_plan: &WildcardPlanResult,
    direction: SplitDirection,
) -> Result<FlawSearchOutcome> {
    let operators = wildcard_plan
        .wildcard_plan
        .iter()
        .enumerate()
        .map(|(step, operators)| {
            operators
                .first()
                .copied()
                .with_context(|| format!("wildcard plan step {} has no operator", step + 1))
        })
        .collect::<Result<Vec<OperatorIndex>>>()?;
    let states = (0..=operators.len())
        .map(|index| abstract_state_set(task, partitions, domain_mapping, wildcard_plan, index))
        .collect::<Result<Vec<_>>>()?;

    let packer = Arc::new(make_prop_state_packer(task));
    let axiom_evaluator = AxiomEvaluator::new(Arc::new(task), packer.clone());
    let (packed, numeric) = get_initial_state(task, &packer, &axiom_evaluator)?;
    let propositions = (0..task.get_num_variables())
        .map(|var| ExplicitValueIndex::from_usize(packer.get(&packed, var) as usize))
        .collect::<Vec<_>>();
    let coordinates = Coordinates::new(task);
    let initial_cut = match direction {
        SplitDirection::Forward | SplitDirection::ForwardPartitionDeviation => InitialCut::Concrete,
        SplitDirection::Backward => InitialCut::Wanted,
    };
    let Some(flaw) = find_regression_flaw(
        task,
        &coordinates,
        &states,
        &operators,
        InitialState {
            propositions: &propositions,
            numeric: &numeric,
        },
        initial_cut,
    )?
    else {
        return Ok(FlawSearchOutcome {
            flaws: Vec::new(),
            concrete_plan: Some(operators),
        });
    };
    // Flaws in the abstract state before step `i` belong to step `i`, like
    // the precondition flaws progression finds there.
    let step = flaw.state + 1;
    let flaw = match flaw.cut {
        Cut::Numeric {
            var,
            boundary,
            lower_includes_boundary,
            ..
        } => Flaw::Numeric(NumericFlaw {
            numeric_var_id: var,
            value: boundary,
            include_in_lower: lower_includes_boundary,
            step,
        }),
        Cut::Propositional { var, wanted } => Flaw::Propositional(PropFlaw {
            fact: ExplicitFact::propositional(var.index(), wanted.index()),
            dependent_numeric_flaws: Vec::new(),
            step,
        }),
    };
    Ok(FlawSearchOutcome {
        flaws: vec![flaw],
        concrete_plan: None,
    })
}

/// The concrete states of the `index`-th abstract state of the wildcard
/// plan: per variable, the values its abstract value maps from, and per
/// numeric variable, its partition.
fn abstract_state_set(
    task: &dyn AbstractNumericTask,
    partitions: &NumericPartitions,
    domain_mapping: &DomainMapping,
    wildcard_plan: &WildcardPlanResult,
    index: usize,
) -> Result<CartesianSet> {
    let abstract_props = wildcard_plan
        .abstract_prop_states
        .get(index)
        .with_context(|| format!("wildcard plan has no abstract propositional state {index}"))?;
    let abstract_numeric = wildcard_plan
        .abstract_numeric_states
        .get(index)
        .with_context(|| format!("wildcard plan has no abstract numeric state {index}"))?;
    ensure!(
        abstract_props.len() == task.get_num_variables()
            && abstract_numeric.len() == task.numeric_variables().len(),
        "abstract state {index} of the wildcard plan does not cover the task's variables"
    );
    let propositions = domain_mapping
        .iter()
        .zip(abstract_props)
        .map(|(mapping, &abstract_value)| {
            mapping
                .iter()
                .enumerate()
                .filter(|&(_, mapped)| mapped.index() == abstract_value)
                .map(|(value, _)| ExplicitValueIndex::from_usize(value))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let numeric = abstract_numeric
        .iter()
        .enumerate()
        .map(|(var, &part)| {
            partitions
                .partition_interval(VariableIndex::from_usize(var), part)
                .with_context(|| format!("numeric variable {var} has no partition {part}"))
        })
        .collect::<Result<Vec<_>>>()?;
    let domain_sizes = domain_mapping.iter().map(Vec::len).collect::<Vec<_>>();
    Ok(CartesianSet::new(propositions, numeric, &domain_sizes))
}
