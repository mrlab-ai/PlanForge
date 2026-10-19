//! The splits a domain abstraction starts CEGAR from: the trivial abstraction
//! refined on the goal variables.

use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result, ensure};
use planforge_sas::numeric_task::{
    AbstractNumericTask, ExplicitFact, ExplicitValueIndex, VariableIndex,
};
use rand::Rng;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;

use super::abstract_operators::DomainMapping;
use super::config::CegarConfig;
use super::refinement::{
    RefinementState, can_refine_numeric_variable, can_refine_propositional_variable,
};
use super::split_selection::InitSplitMethod;
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

fn choose_random_domain_value(domain_size: usize, rng: &mut SmallRng) -> ExplicitValueIndex {
    if domain_size <= 1 {
        ExplicitValueIndex::new(0)
    } else {
        ExplicitValueIndex::from_usize(rng.gen_range(0..domain_size))
    }
}

pub(super) fn compute_initial_split_mapping(
    task: &dyn AbstractNumericTask,
    config: &CegarConfig,
    var_id: VariableIndex,
    goal_value: Option<ExplicitValueIndex>,
    rng: &mut SmallRng,
) -> Option<(usize, Vec<ExplicitValueIndex>)> {
    let concrete_domain_size = task.get_variable_domain_size(var_id).unwrap_or(0);
    if concrete_domain_size == 0 {
        return None;
    }

    let initial_value = task
        .get_initial_propositional_state_values()
        .get(var_id.index())
        .copied()
        .unwrap_or(ExplicitValueIndex::new(0));
    let is_comparison_var = task.numeric_conditions().is_condition_var(var_id);

    match config.init_split_method {
        InitSplitMethod::GoalValue => {
            let goal = goal_value?;
            let mut mapping = vec![ExplicitValueIndex::new(0); concrete_domain_size];
            mapping[goal.index()] = ExplicitValueIndex::new(1);
            Some((2, mapping))
        }
        InitSplitMethod::GoalValueOrRandomIfNonGoal => {
            let chosen =
                goal_value.unwrap_or_else(|| choose_random_domain_value(concrete_domain_size, rng));
            let mut mapping = vec![ExplicitValueIndex::new(0); concrete_domain_size];
            mapping[chosen.index()] = ExplicitValueIndex::new(1);
            Some((2, mapping))
        }
        InitSplitMethod::InitValue => {
            let mut mapping = vec![ExplicitValueIndex::new(0); concrete_domain_size];
            let chosen = if is_comparison_var {
                ExplicitValueIndex::new(0)
            } else {
                initial_value
            };
            if chosen.index() < mapping.len() {
                mapping[chosen.index()] = ExplicitValueIndex::new(1);
            }
            Some((2, mapping))
        }
        InitSplitMethod::RandomValue => {
            let chosen = choose_random_domain_value(concrete_domain_size, rng);
            let mut mapping = vec![ExplicitValueIndex::new(0); concrete_domain_size];
            mapping[chosen.index()] = ExplicitValueIndex::new(1);
            Some((2, mapping))
        }
        InitSplitMethod::RandomPartition => {
            let mut order: Vec<usize> = (0..concrete_domain_size).collect();
            order.shuffle(rng);
            let max_partition = choose_random_domain_value(concrete_domain_size, rng)
                .max(ExplicitValueIndex::new(1));
            let mut mapping = vec![ExplicitValueIndex::new(0); concrete_domain_size];
            for (index, concrete_value) in order.into_iter().enumerate() {
                mapping[concrete_value] =
                    ExplicitValueIndex::from_usize(index % (max_partition.index() + 1));
            }
            let abstract_domain_size = mapping
                .iter()
                .copied()
                .max()
                .unwrap_or(ExplicitValueIndex::new(0))
                .index()
                + 1;
            Some((abstract_domain_size, mapping))
        }
        InitSplitMethod::RandomBinaryPartitionSeparatingInitGoal => {
            let mut mapping: Vec<ExplicitValueIndex> = (0..concrete_domain_size)
                .map(|_| choose_random_domain_value(2, rng))
                .collect();
            if let Some(goal) = goal_value
                && initial_value != goal
                && initial_value.index() < mapping.len()
                && goal.index() < mapping.len()
            {
                mapping[initial_value.index()] = ExplicitValueIndex::new(0);
                mapping[goal.index()] = ExplicitValueIndex::new(1);
            }
            if mapping.iter().all(|&value| value == mapping[0]) {
                None
            } else {
                Some((2, mapping))
            }
        }
        InitSplitMethod::Identity => Some((
            concrete_domain_size,
            (0..concrete_domain_size)
                .map(ExplicitValueIndex::from_usize)
                .collect(),
        )),
    }
}

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
    let goal_values: HashMap<VariableIndex, ExplicitValueIndex> = sorted_goal_facts(task)
        .into_iter()
        .map(|v| (v.var_index(), v.value_index()))
        .collect();
    let num_prop_vars = task.variables().len();
    let mut candidate_var_ids: Vec<VariableIndex> = config
        .init_split_var_ids
        .as_ref()
        .map(|var_ids| var_ids.iter().copied().collect())
        .unwrap_or_else(|| goal_values.keys().copied().collect());
    candidate_var_ids.sort_by_key(|var_id| {
        let is_goal = var_id.index() < num_prop_vars && goal_values.contains_key(var_id);
        (!is_goal, *var_id)
    });
    candidate_var_ids.dedup();
    let initial_max_abstraction_size = if config.init_split_var_ids.is_some() {
        (config.max_abstraction_size / 2).max(1)
    } else {
        config.max_abstraction_size
    };
    let initial_numeric = initial_numeric_values_with_additive_views(task);

    for encoded_var_id in candidate_var_ids {
        if encoded_var_id.index() >= num_prop_vars {
            let numeric_var_id = VariableIndex::from_usize(encoded_var_id.index() - num_prop_vars);
            if blacklisted_numeric_var_ids.contains(&numeric_var_id) {
                continue;
            }
            let Some(_) = task.numeric_variables().get(numeric_var_id.index()) else {
                continue;
            };
            if !is_refinable_numeric_dimension(task, numeric_var_id) {
                continue;
            }
            if !matches!(
                config.init_split_method,
                InitSplitMethod::Identity | InitSplitMethod::GoalValueOrRandomIfNonGoal
            ) {
                continue;
            }
            let Some(&init_value) = initial_numeric.get(numeric_var_id.index()) else {
                continue;
            };
            if !can_refine_numeric_variable(
                domain_sizes,
                numeric_domain_sizes,
                numeric_var_id,
                initial_max_abstraction_size,
            ) {
                continue;
            }
            let include_in_lower = rng.gen_range(0..2) == 0;
            if partitions.split_at(numeric_var_id, init_value, include_in_lower)
                && let Some(parts) = partitions.partitions(numeric_var_id)
                && let Some(slot) = numeric_domain_sizes.get_mut(numeric_var_id.index())
            {
                *slot = parts.len();
            }
            continue;
        }

        let var_id = encoded_var_id;
        if blacklisted_prop_var_ids.contains(&var_id) {
            continue;
        }
        let Some((new_domain_size, mapping)) = compute_initial_split_mapping(
            task,
            config,
            var_id,
            goal_values.get(&var_id).copied(),
            rng,
        ) else {
            continue;
        };
        if new_domain_size <= 1 {
            continue;
        }
        if !can_refine_propositional_variable(
            domain_sizes,
            numeric_domain_sizes,
            var_id,
            new_domain_size,
            initial_max_abstraction_size,
        ) {
            continue;
        }
        if let Some(slot) = domain_mapping.get_mut(var_id.index()) {
            *slot = mapping;
        }
        if let Some(slot) = domain_sizes.get_mut(var_id.index()) {
            *slot = new_domain_size;
        }
    }

    // For every comparison-axiom prop var actually being refined by this
    // CEGAR run (the intersection of `init_split_var_ids` with goal-side
    // comparison vars; if `init_split_var_ids` is unset, fall back to all
    // goal-side comparison vars), seed numeric splits at the initial
    // concrete numeric value of each of the axiom's regular numeric
    // dependencies. Without this, the initial abstraction contains the
    // comparison-axiom prop vars at binary resolution but the underlying
    // numerics are unrefined — so no operator can flip the comparison bit,
    // and the abstract initial state cannot reach the goal. CEGAR then
    // bails with "abstract dead end at iteration 1".
    //
    // The previous version of this loop seeded numerics for *every* goal
    // comparison axiom regardless of which prop var the collection
    // generator chose for this CEGAR iteration. With `init_split_quantity
    // = Single` (the canonical-DA default), one iteration selects one
    // comparison var as its init split, but the unconditional seeding
    // still drove all numeric deps of all goal comparisons into the
    // partition — so every CEGAR run started with the same fully-seeded
    // numeric universe and refined the remaining comparison vars
    // identically. Result: every abstraction's pattern ended up with all
    // goal comparison vars refined and cascade-relevance covering every
    // op, so Bron-Kerbosch returned singleton additive subsets and
    // canonical degenerated to `max h_i`. Scoping the seeding to the
    // selected init split lets different CEGAR iterations focus on
    // different comparison axioms, producing pattern diversity (and
    // hence additivity) in the resulting collection.
    let init_split_filter: Option<&HashSet<VariableIndex>> = config.init_split_var_ids.as_ref();
    for fact in sorted_goal_facts(task) {
        if let Some(allowed) = init_split_filter
            && !allowed.contains(&fact.var_index())
        {
            continue;
        }
        let Some(tree) = task.numeric_conditions().for_var(fact.var_index()) else {
            continue;
        };
        for numeric_var_id in comparison_refinement_dimensions(task, tree) {
            if blacklisted_numeric_var_ids.contains(&numeric_var_id) {
                continue;
            }
            let Some(_) = task.numeric_variables().get(numeric_var_id.index()) else {
                continue;
            };
            if !is_refinable_numeric_dimension(task, numeric_var_id) {
                continue;
            }
            if !can_refine_numeric_variable(
                domain_sizes,
                numeric_domain_sizes,
                numeric_var_id,
                initial_max_abstraction_size,
            ) {
                continue;
            }
            let Some(&init_value) = initial_numeric.get(numeric_var_id.index()) else {
                continue;
            };
            let include_in_lower = rng.gen_range(0..2) == 0;
            if partitions.split_at(numeric_var_id, init_value, include_in_lower)
                && let Some(parts) = partitions.partitions(numeric_var_id)
                && let Some(slot) = numeric_domain_sizes.get_mut(numeric_var_id.index())
            {
                *slot = parts.len();
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
