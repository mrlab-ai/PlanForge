//! The splits a domain abstraction starts CEGAR from: the trivial abstraction
//! refined on the goal variables.

use std::collections::HashSet;

use anyhow::{Context, Result, ensure};
use planforge_sas::numeric_task::{
    AbstractNumericTask, ExplicitFact, ExplicitValueIndex, VariableIndex,
};
use rand::Rng;
use rand::rngs::SmallRng;

use super::abstract_operators::DomainMapping;
use super::config::CegarConfig;
use planforge_sas::numeric_conditions::ConditionValue;

use super::refinement::{
    RefinementState, can_refine_numeric_variable, can_refine_propositional_variable,
};

use crate::evaluation::abstractions::common::coordinates::{
    comparison_refinement_dimensions, initial_numeric_values_with_additive_views,
    is_refinable_numeric_dimension,
};

/// The task's goal facts, sorted and deduplicated.
///
/// Every goal fact is one the abstraction reasons about directly:
/// `validate_abstractable_goal` refuses a goal on a derived variable, which is
/// what this used to substitute a rule body for. Sorting makes the split-candidate
/// order below a function of the task rather than of the order the goals were
/// written in.
pub(super) fn sorted_goal_facts(task: &dyn AbstractNumericTask) -> Vec<ExplicitFact> {
    let mut goals: Vec<ExplicitFact> = (0..task.get_num_goals())
        .map(|goal_id| *task.get_goal_fact(goal_id))
        .collect();
    goals.sort_unstable();
    goals.dedup();
    goals
}

/// The initial split of propositional variable `var_id`: the value it is
/// split off on in one class, every other value in the other. That value is
/// the initial value, or for a condition variable the value that establishes
/// the condition.
pub(super) fn initial_split_mapping(
    task: &dyn AbstractNumericTask,
    var_id: VariableIndex,
) -> Result<Vec<ExplicitValueIndex>> {
    let domain_size = task
        .get_variable_domain_size(var_id)
        .map_err(|error| anyhow::anyhow!(error.to_string()))
        .with_context(|| format!("domain size of variable {var_id}"))?;
    let separated = if task.numeric_conditions().is_condition_var(var_id) {
        ExplicitValueIndex::from_usize(ConditionValue::True.as_usize())
    } else {
        *task
            .get_initial_propositional_state_values()
            .get(var_id.index())
            .with_context(|| format!("variable {var_id} has no initial value"))?
    };
    ensure!(
        separated.index() < domain_size,
        "initial split value {separated} of variable {var_id} is outside its domain of size {domain_size}"
    );
    let mut mapping = vec![ExplicitValueIndex::new(0); domain_size];
    mapping[separated.index()] = ExplicitValueIndex::new(1);
    Ok(mapping)
}

/// Split the trivial abstraction on the initial-split variables: the
/// configured ones, or the goal variables. Propositional variables are split
/// by [`initial_split_mapping`]; a selected numeric variable gets no split of
/// its own. The numeric operands of goal conditions on selected variables are
/// split at their initial values, so an operator can change the condition.
pub(super) fn apply_initial_goal_splits(
    task: &dyn AbstractNumericTask,
    config: &CegarConfig,
    rng: &mut SmallRng,
    state: RefinementState<'_>,
) -> Result<()> {
    let RefinementState {
        domain_mapping,
        domain_sizes,
        partitions,
        numeric_domain_sizes,
        blacklisted_prop_var_ids,
        blacklisted_numeric_var_ids,
    } = state;
    let goals = sorted_goal_facts(task);
    let goal_var_ids: HashSet<VariableIndex> = goals.iter().map(|goal| goal.var_index()).collect();
    let num_prop_vars = task.variables().len();
    let mut candidate_var_ids: Vec<VariableIndex> = match &config.init_split_var_ids {
        Some(var_ids) => var_ids.iter().copied().collect(),
        None => goal_var_ids.iter().copied().collect(),
    };
    candidate_var_ids.sort_by_key(|var_id| {
        let is_goal = var_id.index() < num_prop_vars && goal_var_ids.contains(var_id);
        (!is_goal, *var_id)
    });
    candidate_var_ids.dedup();
    let initial_max_abstraction_size = if config.init_split_var_ids.is_some() {
        (config.max_abstraction_size / 2).max(1)
    } else {
        config.max_abstraction_size
    };

    for var_id in candidate_var_ids {
        // Numeric variables get no initial split of their own, and condition
        // variables none at all: conditions are derived from the numeric
        // partitions, whose operands the loop below seeds.
        if var_id.index() >= num_prop_vars
            || task.numeric_conditions().is_condition_var(var_id)
            || blacklisted_prop_var_ids.contains(&var_id)
        {
            continue;
        }
        let mapping = initial_split_mapping(task, var_id)?;
        if !can_refine_propositional_variable(
            domain_sizes,
            numeric_domain_sizes,
            var_id,
            2,
            initial_max_abstraction_size,
        ) {
            continue;
        }
        domain_mapping[var_id.index()] = mapping;
        domain_sizes[var_id.index()] = 2;
    }

    // Seed the numeric operands of every goal condition whose variable this
    // run splits initially. A condition variable split without its operands
    // has no operator that flips it, so the abstract initial state cannot
    // reach the goal. Seeding only the selected conditions, rather than every
    // goal condition, lets the members of a collection focus on different
    // conditions; seeding them all made every member refine the same
    // conditions, which left canonical heuristics no additive subsets.
    let initial_numeric = initial_numeric_values_with_additive_views(task);
    for goal in &goals {
        if let Some(selected) = &config.init_split_var_ids
            && !selected.contains(&goal.var_index())
        {
            continue;
        }
        let Some(condition) = task.numeric_conditions().for_var(goal.var_index()) else {
            continue;
        };
        for numeric_var_id in comparison_refinement_dimensions(task, condition) {
            if blacklisted_numeric_var_ids.contains(&numeric_var_id)
                || !is_refinable_numeric_dimension(task, numeric_var_id)
                || !can_refine_numeric_variable(
                    domain_sizes,
                    numeric_domain_sizes,
                    numeric_var_id,
                    initial_max_abstraction_size,
                )
            {
                continue;
            }
            let include_in_lower = rng.gen_range(0..2) == 0;
            if partitions.split_at(
                numeric_var_id,
                initial_numeric[numeric_var_id.index()],
                include_in_lower,
            ) {
                numeric_domain_sizes[numeric_var_id.index()] = partitions
                    .partitions(numeric_var_id)
                    .expect("a split numeric variable has partitions")
                    .len();
            }
        }
    }

    Ok(())
}

pub(super) fn trivial_domain_mapping_and_sizes(
    task: &dyn AbstractNumericTask,
) -> Result<(DomainMapping, Vec<usize>)> {
    let num_vars = task.get_num_variables();

    let domain_sizes: Vec<usize> = vec![1; num_vars];
    let mut domain_mapping: DomainMapping = Vec::with_capacity(num_vars);

    for var in 0..num_vars {
        let size = task
            .get_variable_domain_size(VariableIndex::from_usize(var))
            .map_err(|e| anyhow::anyhow!(e.to_string()))
            .with_context(|| format!("get_variable_domain_size({var}) failed"))?;
        ensure!(size > 0, "non-positive domain size for var {var}: {size}");
        domain_mapping.push(vec![ExplicitValueIndex::new(0); size]);
    }

    Ok((domain_mapping, domain_sizes))
}

/// The abstraction that keeps every propositional value apart.
#[cfg(test)]
pub(crate) fn identity_domain_mapping_and_sizes(
    task: &dyn AbstractNumericTask,
) -> Result<(DomainMapping, Vec<usize>)> {
    let num_vars = task.get_num_variables();
    let mut domain_mapping: DomainMapping = Vec::with_capacity(num_vars);
    let mut domain_sizes: Vec<usize> = Vec::with_capacity(num_vars);
    for var_id in 0..num_vars {
        let size = task
            .get_variable_domain_size(VariableIndex::from_usize(var_id))
            .map_err(|e| anyhow::anyhow!(e.to_string()))
            .with_context(|| format!("failed to get domain size for variable {var_id}"))?;
        domain_mapping.push((0..size).map(ExplicitValueIndex::from_usize).collect());
        domain_sizes.push(size);
    }

    Ok((domain_mapping, domain_sizes))
}
