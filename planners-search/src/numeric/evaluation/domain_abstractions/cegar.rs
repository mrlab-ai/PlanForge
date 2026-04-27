#[cfg(test)]
mod tests;

pub mod flaw_search;

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
use log::debug;
use rand::Rng;
use rand::seq::SliceRandom;
use rand::{SeedableRng, rngs::SmallRng};

use planners_sas::numeric::numeric_task::{AbstractNumericTask, NumericType};

use flaw_search::{DependentNumericRefinement, Flaw, NumericFlaw, get_flaws};

pub use flaw_search::flaw_selection::{FlawTreatment, FlawTreatmentVariants, InitSplitMethod};
pub use flaw_search::{ExecEntirePlanMode, FlawKind};

use crate::numeric::evaluation::domain_abstractions::cegar::flaw_search::state::FlawSearchState;

use super::abstract_operator_generator::DomainMapping;
use super::comparison_expression::Interval;
use super::domain_abstraction::NumericPartitions;
use super::domain_abstraction_factory::{DomainAbstractionFactory, WildcardPlanResult};
use super::domain_abstraction_heuristic::{
    COMPARISON_FALSE_VAL, COMPARISON_TRUE_VAL, COMPARISON_UNKNOWN_VAL,
};
use super::utils::{compute_abstraction_size_u128, debug_print_refinement_summary};

#[derive(Debug, Clone)]
pub struct CegarConfig {
    pub max_abstraction_size: usize,
    pub max_iterations: usize,
    pub max_time: Option<Duration>,
    pub use_wildcard_plans: bool,
    pub combine_labels: bool,
    pub debug: bool,
    pub random_seed: Option<u64>,
    pub flaw_kind: FlawKind,
    pub flaw_treatment: FlawTreatmentVariants,
    pub init_split_method: InitSplitMethod,
    pub exec_entire_plan: ExecEntirePlanMode,
    pub init_split_var_ids: Option<HashSet<usize>>,
    pub blacklisted_prop_var_ids: HashSet<usize>,
    pub blacklisted_numeric_var_ids: HashSet<usize>,
}

impl Default for CegarConfig {
    fn default() -> Self {
        Self {
            max_abstraction_size: usize::MAX,
            max_iterations: 10_000,
            max_time: None,
            use_wildcard_plans: true,
            combine_labels: true,
            debug: false,
            random_seed: None,
            flaw_kind: FlawKind::Progression,
            flaw_treatment: FlawTreatmentVariants::RandomSingleAtom,
            init_split_method: InitSplitMethod::InitValue,
            exec_entire_plan: ExecEntirePlanMode::StopAtFirstFlaw,
            init_split_var_ids: None,
            blacklisted_prop_var_ids: HashSet::new(),
            blacklisted_numeric_var_ids: HashSet::new(),
        }
    }
}

#[derive(Clone)]
pub struct FlawCandidate {
    idx: usize,
    score: usize,
    restricted_dep: Option<Vec<NumericFlaw>>,
}
pub type ChosenFlaws = Vec<FlawCandidate>;

#[derive(Debug, Clone)]
pub struct CegarState {
    pub factory: DomainAbstractionFactory,
    pub iteration: usize,
}

impl CegarState {
    pub fn new(factory: DomainAbstractionFactory, iteration: usize) -> CegarState {
        CegarState { factory, iteration }
    }
}

#[derive(Debug, Clone)]
pub struct CegarStep {
    pub wildcard_plan: Option<WildcardPlanResult>,
}

#[derive(Debug, Clone)]
pub struct CegarOutcome {
    pub final_state: CegarState,
    pub last_step: CegarStep,
}

#[derive(Debug, Clone)]
pub struct Cegar {
    config: CegarConfig,
}

impl Cegar {
    pub fn new(config: CegarConfig) -> Result<Self> {
        ensure!(
            config.max_abstraction_size > 0,
            "max_abstraction_size must be > 0"
        );
        ensure!(config.max_iterations > 0, "max_iterations must be > 0");
        Ok(Self { config })
    }

    pub fn build_abstraction(&self, task: &dyn AbstractNumericTask) -> Result<CegarOutcome> {
        self.run_cegar(task)
    }

    fn run_cegar(&self, task: &dyn AbstractNumericTask) -> Result<CegarOutcome> {
        let config = &self.config;
        ensure!(
            config.max_abstraction_size > 0,
            "max_abstraction_size must be > 0"
        );
        ensure!(config.max_iterations > 0, "max_iterations must be > 0");

        let start = Instant::now();
        let mut rng = SmallRng::seed_from_u64(config.random_seed.unwrap_or_else(current_time_seed));

        let (mut domain_mapping, mut domain_sizes) = trivial_domain_mapping_and_sizes(task)
            .context("failed to build trivial domain mapping")?;

        let mut partitions = NumericPartitions::trivial(task);
        let mut numeric_domain_sizes: Vec<usize> = vec![1; task.numeric_variables().len()];
        let mut blacklisted_prop_var_ids = config.blacklisted_prop_var_ids.clone();
        let mut blacklisted_numeric_var_ids = config.blacklisted_numeric_var_ids.clone();
        let execute_entire_plan = match config.exec_entire_plan {
            ExecEntirePlanMode::StopAtFirstFlaw => false,
            ExecEntirePlanMode::ExecuteEntirePlan => true,
        };

        apply_initial_goal_splits(
            task,
            config,
            &mut rng,
            &blacklisted_prop_var_ids,
            &blacklisted_numeric_var_ids,
            &mut domain_mapping,
            &mut domain_sizes,
            &mut partitions,
            &mut numeric_domain_sizes,
        );

        let mut iteration: usize = 1;

        let mut factory = DomainAbstractionFactory::new(
            task,
            domain_mapping,
            domain_sizes,
            partitions,
            numeric_domain_sizes,
        )
        .with_context(|| {
            format!("failed to construct DomainAbstractionFactory (iteration {iteration})")
        })?;
        let mut wildcard_plan = None;

        while iteration <= config.max_iterations {
            if let Some(max_time) = config.max_time
                && start.elapsed() >= max_time
            {
                break;
            }

            if config.debug {
                super::utils::debug_print_abstraction_stats(
                    iteration,
                    &factory.domain_sizes,
                    &factory.numeric_domain_sizes,
                );
            }

            wildcard_plan = factory
                .compute_plan_with_rng(
                    task,
                    config.combine_labels,
                    config.debug,
                    config.use_wildcard_plans,
                    if config.use_wildcard_plans {
                        None
                    } else {
                        Some(&mut rng)
                    },
                )
                .with_context(|| {
                    format!("failed to compute abstract plan (iteration {iteration})")
                })?;
            if config.debug {
                match wildcard_plan.as_ref() {
                    Some(plan) => super::utils::debug_print_wildcard_plan(
                        task,
                        plan,
                        &factory.domain_sizes,
                        &factory.numeric_domain_sizes,
                        &factory.partitions,
                    ),
                    None => debug!("[Abstract Plan] <none>"),
                }
            }

            let Some(plan) = wildcard_plan.as_ref() else {
                break;
            };

            let flaws = get_flaws(
                task,
                &factory.partitions,
                &factory.domain_mapping,
                plan,
                execute_entire_plan,
                self.config.flaw_kind,
            )
            .with_context(|| format!("failed to collect flaws (iteration {iteration})"))?;
            if config.debug {
                super::utils::debug_print_flaws(&flaws);
            }
            if flaws.is_empty() {
                break;
            }

            let before_size = if config.debug {
                compute_abstraction_size_u128(&factory.domain_sizes, &factory.numeric_domain_sizes)
            } else {
                None
            };
            let refined = fix_flaws(
                &self.config,
                task,
                &flaws,
                &mut factory.domain_mapping,
                &mut factory.domain_sizes,
                &mut factory.partitions,
                &mut factory.numeric_domain_sizes,
                &mut rng,
                &mut blacklisted_prop_var_ids,
                &mut blacklisted_numeric_var_ids,
            )
            .with_context(|| format!("failed to fix flaws (iteration {iteration})"))?;
            if config.debug {
                let after_size = compute_abstraction_size_u128(
                    &factory.domain_sizes,
                    &factory.numeric_domain_sizes,
                );
                debug_print_refinement_summary(
                    before_size,
                    after_size,
                    &factory.domain_sizes,
                    &factory.numeric_domain_sizes,
                    refined,
                );
            }
            if !refined {
                break;
            }

            iteration += 1;
        }

        let last_step = CegarStep { wildcard_plan };
        Ok(CegarOutcome {
            final_state: CegarState::new(factory, iteration),
            last_step,
        })
    }
}

fn current_time_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};

    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(0x9e37_79b9_7f4a_7c15)
}

fn shuffle_indices_with_rng<R: rand::Rng + ?Sized>(indices: &mut [usize], rng: &mut R) {
    indices.shuffle(rng);
}

fn abstraction_size_u128(domain_sizes: &[usize], numeric_domain_sizes: &[usize]) -> Option<u128> {
    compute_abstraction_size_u128(domain_sizes, numeric_domain_sizes)
}

fn can_refine_propositional_variable(
    domain_sizes: &[usize],
    numeric_domain_sizes: &[usize],
    var_id: usize,
    new_domain_size: usize,
    max_abstraction_size: usize,
) -> bool {
    let Some(total_size) = abstraction_size_u128(domain_sizes, numeric_domain_sizes) else {
        return false;
    };
    let Some(&old_domain_size) = domain_sizes.get(var_id) else {
        return false;
    };
    if old_domain_size == 0 || new_domain_size == 0 {
        return false;
    }
    let reduced = total_size / (old_domain_size as u128);
    reduced
        .checked_mul(new_domain_size as u128)
        .map(|candidate| candidate <= max_abstraction_size as u128)
        .unwrap_or(false)
}

fn can_refine_numeric_variable(
    domain_sizes: &[usize],
    numeric_domain_sizes: &[usize],
    numeric_var_id: usize,
    max_abstraction_size: usize,
) -> bool {
    let Some(total_size) = abstraction_size_u128(domain_sizes, numeric_domain_sizes) else {
        return false;
    };
    let Some(&old_partition_count) = numeric_domain_sizes.get(numeric_var_id) else {
        return false;
    };
    if old_partition_count == 0 {
        return false;
    }
    let reduced = total_size / (old_partition_count as u128);
    reduced
        .checked_mul((old_partition_count as u128) + 1)
        .map(|candidate| candidate <= max_abstraction_size as u128)
        .unwrap_or(false)
}

fn can_refine_propositional_variable_with_blacklist(
    domain_sizes: &[usize],
    numeric_domain_sizes: &[usize],
    var_id: usize,
    new_domain_size: usize,
    max_abstraction_size: usize,
    comparison_var_ids: &HashSet<usize>,
    blacklisted_prop_var_ids: &mut HashSet<usize>,
) -> bool {
    if blacklisted_prop_var_ids.contains(&var_id) {
        return false;
    }
    if comparison_var_ids.contains(&var_id) && domain_sizes.get(var_id).copied().unwrap_or(0) >= 2 {
        return true;
    }
    if can_refine_propositional_variable(
        domain_sizes,
        numeric_domain_sizes,
        var_id,
        new_domain_size,
        max_abstraction_size,
    ) {
        true
    } else {
        blacklisted_prop_var_ids.insert(var_id);
        false
    }
}

fn can_refine_numeric_variable_with_blacklist(
    domain_sizes: &[usize],
    numeric_domain_sizes: &[usize],
    numeric_var_id: usize,
    max_abstraction_size: usize,
    blacklisted_numeric_var_ids: &mut HashSet<usize>,
) -> bool {
    if blacklisted_numeric_var_ids.contains(&numeric_var_id) {
        return false;
    }
    if can_refine_numeric_variable(
        domain_sizes,
        numeric_domain_sizes,
        numeric_var_id,
        max_abstraction_size,
    ) {
        true
    } else {
        blacklisted_numeric_var_ids.insert(numeric_var_id);
        false
    }
}

/// Port of numeric-FD's refinement step (`fix_flaws`).
///
/// Return `true` if any refinement was applied.
#[allow(clippy::too_many_arguments)]
pub fn fix_flaws(
    config: &CegarConfig,
    task: &dyn AbstractNumericTask,
    flaws: &[Flaw],
    domain_mapping: &mut DomainMapping,
    domain_sizes: &mut [usize],
    partitions: &mut NumericPartitions,
    numeric_domain_sizes: &mut [usize],
    rng: &mut SmallRng,
    blacklisted_prop_var_ids: &mut HashSet<usize>,
    blacklisted_numeric_var_ids: &mut HashSet<usize>,
) -> Result<bool> {
    let comparison_var_ids: HashSet<usize> = task
        .comparison_axioms()
        .iter()
        .map(|ax| ax.get_affected_var_id())
        .collect();

    let chosen_flaws: ChosenFlaws = config.flaw_treatment.choose_flaws(
        task,
        flaws,
        config,
        &comparison_var_ids,
        rng,
        blacklisted_prop_var_ids,
        blacklisted_numeric_var_ids,
        domain_mapping,
        domain_sizes,
        partitions,
        numeric_domain_sizes,
    );

    let mut any_refined = false;
    let mut last_refined = None;
    for cand in chosen_flaws {
        let mut chosen = flaws[cand.idx].clone();
        if let (Flaw::Propositional(pf), Some(restricted)) = (&mut chosen, cand.restricted_dep) {
            pf.dependent_numeric_flaws = restricted;
        }

        if !any_refined
            || config
                .flaw_treatment
                .should_be_refined(&chosen, last_refined.unwrap())
        {
            let flaw_refined = try_refine_from_flaw(
                task,
                &chosen,
                config,
                &comparison_var_ids,
                blacklisted_prop_var_ids,
                blacklisted_numeric_var_ids,
                domain_mapping,
                domain_sizes,
                partitions,
                numeric_domain_sizes,
                DependentNumericRefinement::One,
            )?;

            any_refined = any_refined || flaw_refined;
            if flaw_refined {
                if !config.flaw_treatment.refine_all() {
                    return Ok(true);
                }
                last_refined = Some(&flaws[cand.idx]);
            }
        }
    }

    Ok(any_refined)
}

#[allow(clippy::too_many_arguments)]
fn try_refine_from_flaw(
    task: &dyn AbstractNumericTask,
    flaw: &Flaw,
    config: &CegarConfig,
    comparison_var_ids: &HashSet<usize>,
    blacklisted_prop_var_ids: &mut HashSet<usize>,
    blacklisted_numeric_var_ids: &mut HashSet<usize>,
    domain_mapping: &mut DomainMapping,
    domain_sizes: &mut [usize],
    partitions: &mut NumericPartitions,
    numeric_domain_sizes: &mut [usize],
    dependent_numeric_refinement: DependentNumericRefinement,
) -> Result<bool> {
    match flaw {
        Flaw::Numeric(nf) => {
            let var_id = nf.numeric_var_id;
            if !can_refine_numeric_variable_with_blacklist(
                domain_sizes,
                numeric_domain_sizes,
                var_id,
                config.max_abstraction_size,
                blacklisted_numeric_var_ids,
            ) {
                return Ok(false);
            }
            if partitions.split_at(var_id, nf.value, nf.include_in_lower) {
                if let Some(parts) = partitions.partitions(var_id)
                    && let Some(slot) = numeric_domain_sizes.get_mut(var_id)
                {
                    *slot = parts.len();
                }
                return Ok(true);
            }
            Ok(false)
        }
        Flaw::Propositional(pf) => {
            let var_id = pf.fact.var;
            let value = pf.fact.value;

            // Bounds and conversion checks: these should hold in normal operation;
            // surface violations during debug builds but keep release behavior.
            if var_id >= domain_mapping.len() || var_id >= domain_sizes.len() {
                debug_assert!(
                    false,
                    "try_refine_from_flaw: var_id out of bounds: {} (mapping.len={}, domain_sizes.len={})",
                    var_id,
                    domain_mapping.len(),
                    domain_sizes.len()
                );
                return Ok(false);
            }

            let concrete_size = match task.get_variable_domain_size(var_id) {
                Ok(s) => s,
                Err(e) => {
                    debug_assert!(
                        false,
                        "try_refine_from_flaw: get_variable_domain_size({}) failed: {}",
                        var_id, e
                    );
                    return Ok(false);
                }
            };

            if value >= concrete_size {
                debug_assert!(
                    false,
                    "try_refine_from_flaw: fact value {} out of range (concrete size {}) for var {}",
                    value, concrete_size, var_id
                );
                return Ok(false);
            }

            let mut changed = false;

            if comparison_var_ids.contains(&var_id) {
                if !can_refine_propositional_variable_with_blacklist(
                    domain_sizes,
                    numeric_domain_sizes,
                    var_id,
                    2,
                    config.max_abstraction_size,
                    comparison_var_ids,
                    blacklisted_prop_var_ids,
                ) {
                    return Ok(false);
                }
                // Comparison axiom vars: split into {false/unknown} vs {true} like numeric-fd.
                let old_size = domain_sizes[var_id];
                if domain_sizes[var_id] < 2 {
                    domain_sizes[var_id] = 2;
                    changed = true;
                }
                // Ensure mapping values are within the new abstract size.
                if !domain_mapping[var_id].is_empty() && domain_mapping[var_id][0] != 1 {
                    domain_mapping[var_id][0] = 1;
                    changed = true;
                }
                if domain_mapping[var_id].len() >= 2 && domain_mapping[var_id][1] != 0 {
                    domain_mapping[var_id][1] = 0;
                    changed = true;
                }
                if domain_mapping[var_id].len() >= 3 && domain_mapping[var_id][2] != 0 {
                    domain_mapping[var_id][2] = 0;
                    changed = true;
                }
                let _ = old_size; // Keep structure similar to numeric-fd; size tracking handled elsewhere.
            } else {
                let abs_size = domain_sizes[var_id];
                // If we've already fully refined this variable, nothing to do.
                if abs_size >= concrete_size {
                    return Ok(false);
                }
                // Only refine if the value is still mapped to the default class (0).
                if domain_mapping[var_id].get(value).copied().unwrap_or(0) != 0 {
                    return Ok(false);
                }
                if !can_refine_propositional_variable_with_blacklist(
                    domain_sizes,
                    numeric_domain_sizes,
                    var_id,
                    abs_size + 1,
                    config.max_abstraction_size,
                    comparison_var_ids,
                    blacklisted_prop_var_ids,
                ) {
                    return Ok(false);
                }

                domain_mapping[var_id][value] = abs_size;
                domain_sizes[var_id] = abs_size + 1;
                changed = true;
            }

            // Optional dependent numeric refinements (currently produced only for comparison vars).
            if dependent_numeric_refinement != DependentNumericRefinement::None
                && !pf.dependent_numeric_flaws.is_empty()
            {
                let mut any_numeric_changed = false;
                let iter: Box<dyn Iterator<Item = &NumericFlaw>> =
                    match dependent_numeric_refinement {
                        DependentNumericRefinement::None => Box::new(std::iter::empty()),
                        DependentNumericRefinement::All => {
                            Box::new(pf.dependent_numeric_flaws.iter())
                        }
                        DependentNumericRefinement::One => {
                            Box::new(pf.dependent_numeric_flaws.iter())
                        }
                    };

                for dep in iter {
                    let num_id = dep.numeric_var_id;

                    if !can_refine_numeric_variable_with_blacklist(
                        domain_sizes,
                        numeric_domain_sizes,
                        num_id,
                        config.max_abstraction_size,
                        blacklisted_numeric_var_ids,
                    ) {
                        continue;
                    }

                    if partitions.split_at(num_id, dep.value, dep.include_in_lower) {
                        if let Some(parts) = partitions.partitions(num_id)
                            && let Some(slot) = numeric_domain_sizes.get_mut(num_id)
                        {
                            *slot = parts.len();
                        }
                        any_numeric_changed = true;
                        if dependent_numeric_refinement == DependentNumericRefinement::One {
                            break;
                        }
                    }
                }
                return Ok(any_numeric_changed || changed);
            }

            Ok(changed)
        }
    }
}

fn goal_variable_values(task: &dyn AbstractNumericTask) -> Vec<(usize, usize)> {
    let mut goal_axiom_map: HashMap<usize, usize> = HashMap::new();
    for (axiom_idx, axiom) in task.axioms().iter().enumerate() {
        if !axiom.conditions().is_empty() {
            goal_axiom_map.insert(axiom.var_id(), axiom_idx);
        }
    }

    let num_goals = task.get_num_goals();
    let mut goals = Vec::with_capacity(num_goals);
    for goal_idx in 0..num_goals {
        let goal = task.get_goal_fact(goal_idx);
        if let Some(&axiom_idx) = goal_axiom_map.get(&goal.var) {
            let axiom = &task.axioms()[axiom_idx];
            for condition in axiom.conditions() {
                goals.push((condition.var, condition.value));
            }
        } else {
            goals.push((goal.var, goal.value));
        }
    }
    goals.sort_unstable();
    goals.dedup();
    goals
}

fn choose_random_domain_value(domain_size: usize, rng: &mut SmallRng) -> usize {
    if domain_size <= 1 {
        0
    } else {
        rng.gen_range(0..domain_size)
    }
}

fn compute_initial_split_mapping(
    task: &dyn AbstractNumericTask,
    config: &CegarConfig,
    var_id: usize,
    goal_value: Option<usize>,
    rng: &mut SmallRng,
) -> Option<(usize, Vec<usize>)> {
    let concrete_domain_size = task.get_variable_domain_size(var_id).unwrap_or(0);
    if concrete_domain_size == 0 {
        return None;
    }

    let initial_value = task
        .get_initial_propositional_state_values()
        .get(var_id)
        .copied()
        .unwrap_or(0);
    let comparison_var_ids: HashSet<usize> = task
        .comparison_axioms()
        .iter()
        .map(|axiom| axiom.get_affected_var_id())
        .collect();
    let is_comparison_var = comparison_var_ids.contains(&var_id);

    match config.init_split_method {
        InitSplitMethod::GoalValue => {
            let goal = goal_value?;
            let mut mapping = vec![0; concrete_domain_size];
            mapping[goal] = 1;
            Some((2, mapping))
        }
        InitSplitMethod::GoalValueOrRandomIfNonGoal => {
            let chosen =
                goal_value.unwrap_or_else(|| choose_random_domain_value(concrete_domain_size, rng));
            let mut mapping = vec![0; concrete_domain_size];
            mapping[chosen] = 1;
            Some((2, mapping))
        }
        InitSplitMethod::InitValue => {
            let mut mapping = vec![0; concrete_domain_size];
            let chosen = if is_comparison_var { 0 } else { initial_value };
            if chosen < mapping.len() {
                mapping[chosen] = 1;
            }
            Some((2, mapping))
        }
        InitSplitMethod::RandomValue => {
            let chosen = choose_random_domain_value(concrete_domain_size, rng);
            let mut mapping = vec![0; concrete_domain_size];
            mapping[chosen] = 1;
            Some((2, mapping))
        }
        InitSplitMethod::RandomPartition => {
            let mut order: Vec<usize> = (0..concrete_domain_size).collect();
            shuffle_indices_with_rng(&mut order, rng);
            let max_partition = choose_random_domain_value(concrete_domain_size, rng).max(1);
            let mut mapping = vec![0; concrete_domain_size];
            for (index, concrete_value) in order.into_iter().enumerate() {
                mapping[concrete_value] = index % (max_partition + 1);
            }
            let abstract_domain_size = mapping.iter().copied().max().unwrap_or(0) + 1;
            Some((abstract_domain_size, mapping))
        }
        InitSplitMethod::RandomBinaryPartitionSeparatingInitGoal => {
            let mut mapping: Vec<usize> = (0..concrete_domain_size)
                .map(|_| choose_random_domain_value(2, rng))
                .collect();
            if let Some(goal) = goal_value
                && initial_value != goal
                && initial_value < mapping.len()
                && goal < mapping.len()
            {
                mapping[initial_value] = 0;
                mapping[goal] = 1;
            }
            if mapping.iter().all(|&value| value == mapping[0]) {
                None
            } else {
                Some((2, mapping))
            }
        }
        InitSplitMethod::Identity => {
            Some((concrete_domain_size, (0..concrete_domain_size).collect()))
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_initial_goal_splits(
    task: &dyn AbstractNumericTask,
    config: &CegarConfig,
    rng: &mut SmallRng,
    blacklisted_prop_var_ids: &HashSet<usize>,
    blacklisted_numeric_var_ids: &HashSet<usize>,
    domain_mapping: &mut DomainMapping,
    domain_sizes: &mut [usize],
    partitions: &mut NumericPartitions,
    numeric_domain_sizes: &mut [usize],
) {
    let goal_values: HashMap<usize, usize> = goal_variable_values(task).into_iter().collect();
    let num_prop_vars = task.variables().len();
    let mut candidate_var_ids: Vec<usize> = config
        .init_split_var_ids
        .as_ref()
        .map(|var_ids| var_ids.iter().copied().collect())
        .unwrap_or_else(|| goal_values.keys().copied().collect());
    candidate_var_ids.sort_unstable();
    candidate_var_ids.dedup();

    for encoded_var_id in candidate_var_ids {
        if encoded_var_id >= num_prop_vars {
            let numeric_var_id = encoded_var_id - num_prop_vars;
            if blacklisted_numeric_var_ids.contains(&numeric_var_id) {
                continue;
            }
            let Some(numeric_var) = task.numeric_variables().get(numeric_var_id) else {
                continue;
            };
            if numeric_var.get_type() != &NumericType::Regular {
                continue;
            }
            if !matches!(
                config.init_split_method,
                InitSplitMethod::Identity | InitSplitMethod::GoalValueOrRandomIfNonGoal
            ) {
                continue;
            }
            let Some(&init_value) = task.get_initial_numeric_state_values().get(numeric_var_id)
            else {
                continue;
            };
            let include_in_lower = rng.gen_range(0..2) == 0;
            if partitions.split_at(numeric_var_id, init_value, include_in_lower)
                && let Some(parts) = partitions.partitions(numeric_var_id)
                && let Some(slot) = numeric_domain_sizes.get_mut(numeric_var_id)
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
            config.max_abstraction_size,
        ) {
            continue;
        }
        if let Some(slot) = domain_mapping.get_mut(var_id) {
            *slot = mapping;
        }
        if let Some(slot) = domain_sizes.get_mut(var_id) {
            *slot = new_domain_size;
        }
    }
}

fn trivial_domain_mapping_and_sizes(
    task: &dyn AbstractNumericTask,
) -> Result<(DomainMapping, Vec<usize>)> {
    let num_vars = task.get_num_variables();

    let domain_sizes: Vec<usize> = vec![1; num_vars];
    let mut domain_mapping: DomainMapping = Vec::with_capacity(num_vars);

    for var in 0..num_vars {
        let size = task
            .get_variable_domain_size(var)
            .map_err(|e| anyhow::anyhow!(e.to_string()))
            .with_context(|| format!("get_variable_domain_size({var}) failed"))?;
        ensure!(size > 0, "non-positive domain size for var {var}: {size}");
        domain_mapping.push(vec![0; size]);
    }

    Ok((domain_mapping, domain_sizes))
}

fn comparison_eval_code(v: Option<bool>) -> usize {
    match v {
        Some(true) => COMPARISON_TRUE_VAL,
        Some(false) => COMPARISON_FALSE_VAL,
        None => COMPARISON_UNKNOWN_VAL,
    }
}

#[allow(clippy::if_same_then_else, clippy::needless_bool)]
fn determine_include_in_lower(
    tree: &super::comparison_expression::ComparisonTree,
    split_var_id: usize,
    split_value: f64,
    concrete_values: &[f64],
) -> bool {
    let mut lower_inputs: Vec<Interval> = concrete_values
        .iter()
        .copied()
        .map(Interval::singleton)
        .collect();
    let mut upper_inputs = lower_inputs.clone();

    if split_var_id < lower_inputs.len() {
        // If the split point is included in the lower interval, the current concrete
        // value belongs to (-inf, split_value].
        lower_inputs[split_var_id] = Interval::new(f64::NEG_INFINITY, split_value, false, true);
    }
    if split_var_id < upper_inputs.len() {
        // If the split point is included in the upper interval, the current concrete
        // value belongs to [split_value, inf).
        upper_inputs[split_var_id] = Interval::new(split_value, f64::INFINITY, true, false);
    }

    let eval_lower = comparison_eval_code(tree.evaluate_interval(&lower_inputs));
    let eval_upper = comparison_eval_code(tree.evaluate_interval(&upper_inputs));

    // Mirror numeric-FD's preference: FALSE (=1) over UNKNOWN (=2) over TRUE (=0).
    if eval_lower == COMPARISON_FALSE_VAL && eval_upper != COMPARISON_FALSE_VAL {
        true
    } else if eval_upper == COMPARISON_FALSE_VAL && eval_lower != COMPARISON_FALSE_VAL {
        false
    } else if eval_lower == COMPARISON_FALSE_VAL && eval_upper == COMPARISON_FALSE_VAL {
        false
    } else if eval_lower == COMPARISON_UNKNOWN_VAL && eval_upper == COMPARISON_UNKNOWN_VAL {
        false
    } else if eval_lower == COMPARISON_UNKNOWN_VAL {
        true
    } else if eval_upper == COMPARISON_UNKNOWN_VAL {
        false
    } else {
        false
    }
}

#[allow(clippy::if_same_then_else, clippy::needless_bool)]
fn determine_include_in_lower_for_flaw_search_state(
    tree: &super::comparison_expression::ComparisonTree,
    state: &FlawSearchState,
) -> bool {
    let lower_inputs: Vec<Interval> = state
        .numeric
        .iter()
        .map(|v| Interval::singleton(v.lower))
        .collect();
    let upper_inputs: Vec<Interval> = state
        .numeric
        .iter()
        .map(|v| Interval::singleton(v.upper))
        .collect();

    let eval_lower = comparison_eval_code(tree.evaluate_interval(&lower_inputs));
    let eval_upper = comparison_eval_code(tree.evaluate_interval(&upper_inputs));

    // Mirror numeric-FD's preference: FALSE (=1) over UNKNOWN (=2) over TRUE (=0).
    if eval_lower == COMPARISON_FALSE_VAL && eval_upper != COMPARISON_FALSE_VAL {
        true
    } else if eval_upper == COMPARISON_FALSE_VAL && eval_lower != COMPARISON_FALSE_VAL {
        false
    } else if eval_lower == COMPARISON_FALSE_VAL && eval_upper == COMPARISON_FALSE_VAL {
        false
    } else if eval_lower == COMPARISON_UNKNOWN_VAL && eval_upper == COMPARISON_UNKNOWN_VAL {
        false
    } else if eval_lower == COMPARISON_UNKNOWN_VAL {
        true
    } else if eval_upper == COMPARISON_UNKNOWN_VAL {
        false
    } else {
        false
    }
}
