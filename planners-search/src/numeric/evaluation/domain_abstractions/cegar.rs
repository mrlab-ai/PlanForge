#[cfg(test)]
mod tests;

pub mod flaw_search;

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
use planners_sas::numeric::axioms::AxiomEvaluator;
use planners_sas::numeric::utils::int_packer::IntDoublePacker;
use rand::Rng;
use rand::seq::SliceRandom;
use rand::{SeedableRng, rngs::SmallRng};
use tracing::debug;

use planners_sas::numeric::numeric_task::{
    AbstractNumericTask, ExplicitFact, NumericType, Operator,
};

use flaw_search::{DependentNumericRefinement, Flaw, NumericFlaw, get_flaws};

pub use flaw_search::FlawKind;
pub use flaw_search::flaw_selection::{FlawTreatment, FlawTreatmentVariants, InitSplitMethod};

use crate::numeric::evaluation::domain_abstractions::cegar::flaw_search::state::{
    FlawSearchState, progress,
};
use crate::numeric::evaluation::domain_abstractions::transition_system::TransitionSystem;
use crate::numeric::evaluation::domain_abstractions::utils::{
    fact_is_hold, get_initial_state, make_prop_state_packer,
};

use super::abstract_operator_generator::{DomainMapping, IncrementalAbstractOperatorCache};
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

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RefinementSummary {
    pub refined_propositional_vars: HashSet<usize>,
    pub refined_numeric_vars: HashSet<usize>,
}

impl RefinementSummary {
    pub fn is_empty(&self) -> bool {
        self.refined_propositional_vars.is_empty() && self.refined_numeric_vars.is_empty()
    }

    pub fn mark_propositional(&mut self, var_id: usize) {
        self.refined_propositional_vars.insert(var_id);
    }

    pub fn mark_numeric(&mut self, var_id: usize) {
        self.refined_numeric_vars.insert(var_id);
    }

    pub fn merge(&mut self, other: Self) {
        self.refined_propositional_vars
            .extend(other.refined_propositional_vars);
        self.refined_numeric_vars.extend(other.refined_numeric_vars);
    }
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
        let mut operator_cache = IncrementalAbstractOperatorCache::default();
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
                    factory.transition_system.domain_sizes(),
                    factory.transition_system.numeric_domain_sizes(),
                );
            }

            wildcard_plan = factory
                .compute_plan_with_rng_and_cache(
                    task,
                    config.combine_labels,
                    config.debug,
                    config.use_wildcard_plans,
                    Some(&mut operator_cache),
                    Some(&mut rng),
                )
                .with_context(|| {
                    format!("failed to compute abstract plan (iteration {iteration})")
                })?;
            if config.debug {
                match wildcard_plan.as_ref() {
                    Some(plan) => super::utils::debug_print_wildcard_plan(
                        task,
                        plan,
                        factory.transition_system.domain_sizes(),
                        factory.transition_system.numeric_domain_sizes(),
                        factory.transition_system.partitions(),
                    ),
                    None => debug!("[Abstract Plan] <none>"),
                }
            }

            let Some(plan) = wildcard_plan.as_ref() else {
                break;
            };
            if wildcard_plan_is_real(task, plan)? {
                // There is a real plan in the abstract plan, perfect heuristic.
                break;
            }

            let flaws = get_flaws(
                task,
                factory.transition_system.partitions(),
                factory.transition_system.domain_mapping(),
                plan,
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
                compute_abstraction_size_u128(
                    factory.transition_system.domain_sizes(),
                    factory.transition_system.numeric_domain_sizes(),
                )
            } else {
                None
            };
            let refined = fix_flaws(
                &self.config,
                task,
                &flaws,
                &mut factory.transition_system,
                &mut rng,
                &mut blacklisted_prop_var_ids,
                &mut blacklisted_numeric_var_ids,
                plan.wildcard_plan.len(),
            )
            .with_context(|| format!("failed to fix flaws (iteration {iteration})"))?;
            operator_cache.mark_refined(&refined);
            if config.debug {
                let after_size = compute_abstraction_size_u128(
                    factory.transition_system.domain_sizes(),
                    factory.transition_system.numeric_domain_sizes(),
                );
                debug_print_refinement_summary(
                    before_size,
                    after_size,
                    factory.transition_system.domain_sizes(),
                    factory.transition_system.numeric_domain_sizes(),
                    !refined.is_empty(),
                );
            }
            if refined.is_empty() {
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

fn wildcard_plan_is_real(
    task: &dyn AbstractNumericTask,
    wildcard_plan: &WildcardPlanResult,
) -> Result<bool> {
    let state_packer = make_prop_state_packer(task);
    let axiom_evaluator = AxiomEvaluator::new(task, &state_packer);

    let plan_length = wildcard_plan.wildcard_plan.len();
    let (mut prop_state, mut numeric_state) =
        get_initial_state(task, &state_packer, &axiom_evaluator)?;
    if plan_length == 0 {
        return Ok(is_goal(task, &prop_state, &state_packer));
    }
    let mut last_state_per_layer = vec![None; plan_length];
    last_state_per_layer[0] = Some((prop_state.clone(), numeric_state.clone()));

    let mut real_plan_exists = false;
    // TODO: A set of dead ends could be used if `f64` could be hashed.
    // let mut dead_ends: HashSet<ConcreteState> = HashSet::new();
    let mut current_step: usize = 0;
    let mut equiv_op_iterators = Vec::with_capacity(plan_length);
    // Equivalent operators of the first layer.
    equiv_op_iterators.push(wildcard_plan.wildcard_plan[0].iter());

    loop {
        if let Some(op_id) = equiv_op_iterators[current_step].next() {
            let Some(op) = task.get_operators().get(*op_id) else {
                continue;
            };
            if is_applicable(&prop_state, &state_packer, op) {
                progress(
                    op,
                    &axiom_evaluator,
                    &state_packer,
                    &mut prop_state,
                    &mut numeric_state,
                )?;
                // if dead_ends.contains((prop_state, numeric_state)) { // Go to the previous layer. }
                current_step += 1;
                if current_step == plan_length {
                    if is_goal(task, &prop_state, &state_packer) {
                        real_plan_exists = true;
                        break;
                    } else {
                        // Go to the previous layer.
                        current_step -= 1;
                        (prop_state, numeric_state) =
                            last_state_per_layer[current_step].clone().unwrap();
                        continue;
                    }
                }
                last_state_per_layer[current_step] =
                    Some((prop_state.clone(), numeric_state.clone()));
                // All operators must be tried again from this state.
                equiv_op_iterators.push(wildcard_plan.wildcard_plan[current_step].iter());
            } else {
                // dead_ends.insert((prop_state, numeric_state));
                continue;
            }
        } else {
            if current_step == 0 {
                // All operators tried.
                break;
            } else {
                // Go to the previous layer.
                current_step -= 1;
                equiv_op_iterators.pop();
                (prop_state, numeric_state) = last_state_per_layer[current_step].clone().unwrap();
                continue;
            }
        }
    }

    Ok(real_plan_exists)
}

fn is_applicable(buffer: &[u64], packer: &IntDoublePacker, op: &Operator) -> bool {
    for pre in op.preconditions().iter() {
        if !fact_is_hold(pre, packer, buffer) {
            return false;
        }
    }

    true
}

fn is_goal(task: &dyn AbstractNumericTask, buffer: &[u64], packer: &IntDoublePacker) -> bool {
    for goal_fact in goal_variable_values(task) {
        if !fact_is_hold(&goal_fact, packer, buffer) {
            return false;
        }
    }

    true
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

/// Port of numeric-FD's refinement step (`fix_flaws`).
///
/// Return the refined variable IDs.
#[allow(clippy::too_many_arguments)]
pub fn fix_flaws(
    config: &CegarConfig,
    task: &dyn AbstractNumericTask,
    flaws: &[Flaw],
    transition_system: &mut TransitionSystem,
    rng: &mut SmallRng,
    blacklisted_prop_var_ids: &mut HashSet<usize>,
    blacklisted_numeric_var_ids: &mut HashSet<usize>,
    plan_length: usize,
) -> Result<RefinementSummary> {
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
        transition_system.domain_mapping(),
        transition_system.domain_sizes(),
        transition_system.partitions(),
        transition_system.numeric_domain_sizes(),
        plan_length,
    );

    let mut refined_summary = RefinementSummary::default();
    let mut last_refined = None;
    for cand in chosen_flaws {
        let mut chosen = flaws[cand.idx].clone();
        if let (Flaw::Propositional(pf), Some(restricted)) = (&mut chosen, cand.restricted_dep) {
            pf.dependent_numeric_flaws = restricted;
        }

        if refined_summary.is_empty()
            || config
                .flaw_treatment
                .should_be_refined(&chosen, last_refined.unwrap())
        {
            let flaw_refined = transition_system.try_refine_from_flaw(
                task,
                &chosen,
                config,
                &comparison_var_ids,
                blacklisted_prop_var_ids,
                blacklisted_numeric_var_ids,
                DependentNumericRefinement::One,
            )?;

            if let Some(flaw_refined) = flaw_refined {
                refined_summary.merge(flaw_refined);
                if !config.flaw_treatment.refine_all() {
                    return Ok(refined_summary);
                }
                last_refined = Some(&flaws[cand.idx]);
            }
        }
    }

    Ok(refined_summary)
}

fn goal_variable_values(task: &dyn AbstractNumericTask) -> Vec<ExplicitFact> {
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
                goals.push(ExplicitFact::new(condition.var, condition.value));
            }
        } else {
            goals.push(ExplicitFact::new(goal.var, goal.value));
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
    let goal_values: HashMap<usize, usize> = goal_variable_values(task)
        .into_iter()
        .map(|v| (v.var, v.value))
        .collect();
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
        if !TransitionSystem::can_refine_propositional_variable(
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

pub fn run_cegar(task: &dyn AbstractNumericTask, config: CegarConfig) -> Result<CegarOutcome> {
    Cegar::new(config)?.build_abstraction(task)
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

#[allow(unused, clippy::if_same_then_else, clippy::needless_bool)]
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
