#[cfg(test)]
mod tests;

pub mod flaw_search;

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use planners_sas::numeric::axioms::AxiomEvaluator;
use planners_sas::numeric::utils::int_packer::IntDoublePacker;
use rand::Rng;
use rand::seq::SliceRandom;
use rand::{SeedableRng, rngs::SmallRng};
use tracing::{debug, info};

use planners_sas::numeric::numeric_task::{
    AbstractNumericTask, ExplicitFact, NumericType, Operator,
};

use flaw_search::{DependentNumericRefinement, Flaw, NumericFlaw};

pub use flaw_search::FlawKind;
pub use flaw_search::SplitDirection;
pub use flaw_search::flaw_selection::{FlawTreatment, FlawTreatmentVariants, InitSplitMethod};

use crate::numeric::evaluation::domain_abstractions::cegar::flaw_search::state::{
    FlawSearchState, progress,
};
use crate::numeric::evaluation::domain_abstractions::utils::{
    fact_is_hold, get_initial_state, make_prop_state_packer,
};

use super::abstract_operator_generator::DomainMapping;
use super::comparison_expression::Interval;
use super::domain_abstraction::{ComparisonAxiomIndex, NumericPartitions};
use super::domain_abstraction_factory::{DomainAbstractionFactory, WildcardPlanResult};
use super::domain_abstraction_heuristic::{
    COMPARISON_FALSE_VAL, COMPARISON_TRUE_VAL, COMPARISON_UNKNOWN_VAL,
};
use super::transition_cost_partitioning::FiniteSupportConfig;
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
    pub transform_linear_task: bool,
    pub initial_seed_splits: Vec<InitialSeedSplit>,
    /// Width threshold for the finite-support transition-cost-partitioning
    /// gate applied when the abstraction's operator footprints are built. The
    /// default reproduces the legacy finite-vs-infinite behavior.
    pub finite_support: FiniteSupportConfig,
    /// How numeric flaw split values are chosen: `Forward` keeps the legacy
    /// concrete-value split; `Backward` places splits at the boundary derived
    /// from the regressed-target / required interval. When `None`, the flaw
    /// kind's default ([`FlawKind::default_split_direction`]) is used.
    pub split_direction: Option<SplitDirection>,
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
            transform_linear_task: false,
            initial_seed_splits: Vec::new(),
            finite_support: FiniteSupportConfig::default(),
            split_direction: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum InitialSeedSplit {
    Propositional {
        var_id: usize,
        value: usize,
    },
    Numeric {
        numeric_var_id: usize,
        value: f64,
        include_in_lower: bool,
    },
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
    /// True iff CEGAR's loop exited because no flaws remained: the
    /// abstract wildcard plan is therefore already a real concrete plan
    /// (or the initial state is itself the goal, plan empty). When set,
    /// `h_DA(init)` is exact for the optimal cost — admissible *and*
    /// tight — so subsequent abstractions in the collection cannot
    /// improve the canonical (max) heuristic at the initial state, and
    /// the collection generator can stop early to save memory.
    pub solved_by_self: bool,
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

    fn mark_propositional(&mut self, var_id: usize) {
        self.refined_propositional_vars.insert(var_id);
    }

    fn mark_numeric(&mut self, var_id: usize) {
        self.refined_numeric_vars.insert(var_id);
    }

    fn merge(&mut self, other: Self) {
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

        if config.initial_seed_splits.is_empty() {
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
        } else {
            apply_initial_seed_splits(
                task,
                config,
                &blacklisted_prop_var_ids,
                &blacklisted_numeric_var_ids,
                &mut domain_mapping,
                &mut domain_sizes,
                &mut partitions,
                &mut numeric_domain_sizes,
            );
        }

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
        let mut solved_by_self = false;

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

            let iteration_start = Instant::now();
            let plan_start = Instant::now();
            wildcard_plan = factory
                .compute_plan_with_rng(
                    task,
                    config.combine_labels,
                    config.debug,
                    config.use_wildcard_plans,
                    Some(&mut rng),
                )
                .with_context(|| {
                    format!("failed to compute abstract plan (iteration {iteration})")
                })?;
            let plan_time = plan_start.elapsed();
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
                let abstraction_size = compute_abstraction_size_u128(
                    &factory.domain_sizes,
                    &factory.numeric_domain_sizes,
                )
                .unwrap_or(u128::MAX);
                bail!(
                    "CEGAR produced an abstract dead end for the concrete initial state at iteration {iteration}; abstraction_size={abstraction_size}, prop_domains={:?}, numeric_domains={:?}",
                    factory.domain_sizes,
                    factory.numeric_domain_sizes
                );
            };
            let real_check_time = Duration::ZERO;

            let flaw_start = Instant::now();
            let direction = self
                .config
                .split_direction
                .unwrap_or_else(|| self.config.flaw_kind.default_split_direction());
            let flaws = self
                .config
                .flaw_kind
                .get_flaws_with_direction(
                    task,
                    &factory.partitions,
                    &factory.domain_mapping,
                    plan,
                    direction,
                )
                .with_context(|| format!("failed to collect flaws (iteration {iteration})"))?;
            let flaw_time = flaw_start.elapsed();
            if config.debug {
                super::utils::debug_print_flaws(&flaws);
            }
            if flaws.is_empty() {
                // Plan has no flaws → it is a real concrete plan; flag for
                // collection-level early exit.
                solved_by_self = true;
                break;
            }

            let before_size = if config.debug {
                compute_abstraction_size_u128(&factory.domain_sizes, &factory.numeric_domain_sizes)
            } else {
                None
            };
            let refine_start = Instant::now();
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
                plan.wildcard_plan.len(),
            )
            .with_context(|| format!("failed to fix flaws (iteration {iteration})"))?;
            let refine_time = refine_start.elapsed();
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
                    !refined.is_empty(),
                );
            }
            if refined.is_empty() {
                break;
            }
            if config.debug {
                let abstraction_size = compute_abstraction_size_u128(
                    &factory.domain_sizes,
                    &factory.numeric_domain_sizes,
                )
                .unwrap_or(u128::MAX);
                info!(
                    "CEGAR iteration {iteration}: plan_len={}, flaws={}, refined={:?}, size={}, elapsed={:.3}s, plan={:.3}s, real_check={:.3}s, flaws_time={:.3}s, refine={:.3}s",
                    plan.wildcard_plan.len(),
                    flaws.len(),
                    refined,
                    abstraction_size,
                    iteration_start.elapsed().as_secs_f64(),
                    plan_time.as_secs_f64(),
                    real_check_time.as_secs_f64(),
                    flaw_time.as_secs_f64(),
                    refine_time.as_secs_f64()
                );
            }

            iteration += 1;
        }

        if config.debug || std::env::var_os("DA_DUMP_FINAL_ABSTRACTION").is_some() {
            log_final_target_centered_abstraction(task, &factory);
        }

        let last_step = CegarStep { wildcard_plan };
        Ok(CegarOutcome {
            final_state: CegarState::new(factory, iteration),
            last_step,
            solved_by_self,
        })
    }
}

fn log_final_target_centered_abstraction(
    task: &dyn AbstractNumericTask,
    factory: &DomainAbstractionFactory,
) {
    info!("target-centered domain abstraction final domains:");
    for (numeric_var_id, &size) in factory
        .numeric_domain_sizes()
        .iter()
        .enumerate()
        .filter(|(_, size)| **size > 1)
    {
        let name = task
            .numeric_variables()
            .get(numeric_var_id)
            .map(|variable| variable.name())
            .unwrap_or("<unknown>");
        let Some(parts) = factory.partitions().partitions(numeric_var_id) else {
            continue;
        };
        let intervals = parts
            .iter()
            .enumerate()
            .map(|(part_id, interval)| format!("p{part_id}:{interval:?}"))
            .collect::<Vec<_>>()
            .join(" ");
        info!("  n{numeric_var_id}={name}, size={size}, {intervals}");
    }
    for (var_id, &size) in factory
        .domain_sizes()
        .iter()
        .enumerate()
        .filter(|(_, size)| **size > 1)
    {
        let name = task.get_variable_name(var_id).unwrap_or("<unknown>");
        info!("  p{var_id}={name}, size={size}");
    }
}

#[allow(dead_code)]
fn wildcard_plan_is_real(
    task: &dyn AbstractNumericTask,
    wildcard_plan: &WildcardPlanResult,
) -> Result<bool> {
    // Note for Martin:
    // This verifies whether *any* concrete operator choice through a wildcard
    // abstract plan is a real plan. For long wildcard plans the search is
    // exponential in the number of equivalent concrete operators per step. On
    // sailing it cost several seconds per CEGAR iteration once plans became
    // long, so the CEGAR loop intentionally does not call this hot-path check.
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
    let mut current_step: usize = 0;
    let mut equiv_op_iterators = Vec::with_capacity(plan_length);
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
                current_step += 1;
                if current_step == plan_length {
                    if is_goal(task, &prop_state, &state_packer) {
                        real_plan_exists = true;
                        break;
                    }
                    current_step -= 1;
                    (prop_state, numeric_state) =
                        last_state_per_layer[current_step].clone().unwrap();
                    continue;
                }
                last_state_per_layer[current_step] =
                    Some((prop_state.clone(), numeric_state.clone()));
                equiv_op_iterators.push(wildcard_plan.wildcard_plan[current_step].iter());
            }
        } else if current_step == 0 {
            break;
        } else {
            current_step -= 1;
            equiv_op_iterators.pop();
            (prop_state, numeric_state) = last_state_per_layer[current_step].clone().unwrap();
        }
    }

    Ok(real_plan_exists)
}

#[allow(dead_code)]
fn is_applicable(buffer: &[u64], packer: &IntDoublePacker, op: &Operator) -> bool {
    op.preconditions()
        .iter()
        .all(|pre| fact_is_hold(pre, packer, buffer))
}

#[allow(dead_code)]
fn is_goal(task: &dyn AbstractNumericTask, buffer: &[u64], packer: &IntDoublePacker) -> bool {
    goal_variable_values(task)
        .iter()
        .all(|goal_fact| fact_is_hold(goal_fact, packer, buffer))
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
/// Return the refined variable IDs.
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
        domain_mapping,
        domain_sizes,
        partitions,
        numeric_domain_sizes,
        plan_length,
    );

    let mut refined_summary = RefinementSummary::default();
    let mut last_refined = None;
    for cand in chosen_flaws {
        let dependent_numeric_refinement = if matches!(config.flaw_kind, FlawKind::TargetCentered) {
            DependentNumericRefinement::All
        } else {
            DependentNumericRefinement::One
        };
        let mut chosen = flaws[cand.idx].clone();
        if dependent_numeric_refinement != DependentNumericRefinement::All
            && let (Flaw::Propositional(pf), Some(restricted)) = (&mut chosen, cand.restricted_dep)
        {
            pf.dependent_numeric_flaws = restricted;
        }

        if refined_summary.is_empty()
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
                dependent_numeric_refinement,
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
) -> Result<Option<RefinementSummary>> {
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
                return Ok(None);
            }
            if partitions.split_at(var_id, nf.value, nf.include_in_lower) {
                if let Some(parts) = partitions.partitions(var_id)
                    && let Some(slot) = numeric_domain_sizes.get_mut(var_id)
                {
                    *slot = parts.len();
                }
                let mut refined = RefinementSummary::default();
                refined.mark_numeric(var_id);
                return Ok(Some(refined));
            }
            Ok(None)
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
                return Ok(None);
            }

            let concrete_size = match task.get_variable_domain_size(var_id) {
                Ok(s) => s,
                Err(e) => {
                    debug_assert!(
                        false,
                        "try_refine_from_flaw: get_variable_domain_size({}) failed: {}",
                        var_id, e
                    );
                    return Ok(None);
                }
            };

            if value >= concrete_size {
                debug_assert!(
                    false,
                    "try_refine_from_flaw: fact value {} out of range (concrete size {}) for var {}",
                    value, concrete_size, var_id
                );
                return Ok(None);
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
                    return Ok(None);
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
                    return Ok(None);
                }
                // Only refine if the value is still mapped to the default class (0).
                if domain_mapping[var_id].get(value).copied().unwrap_or(0) != 0 {
                    return Ok(None);
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
                    return Ok(None);
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
                let mut refined = RefinementSummary::default();
                if changed {
                    refined.mark_propositional(var_id);
                }
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
                        refined.mark_numeric(num_id);
                        if dependent_numeric_refinement == DependentNumericRefinement::One {
                            break;
                        }
                    }
                }
                return Ok((any_numeric_changed || changed).then_some(refined));
            }

            if changed {
                let mut refined = RefinementSummary::default();
                refined.mark_propositional(var_id);
                Ok(Some(refined))
            } else {
                Ok(None)
            }
        }
    }
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
    candidate_var_ids.sort_by_key(|var_id| {
        let is_goal = *var_id < num_prop_vars && goal_values.contains_key(var_id);
        (!is_goal, *var_id)
    });
    candidate_var_ids.dedup();
    let initial_max_abstraction_size = if config.init_split_var_ids.is_some() {
        (config.max_abstraction_size / 2).max(1)
    } else {
        config.max_abstraction_size
    };

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
            initial_max_abstraction_size,
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

    // For every goal-side comparison-axiom prop var (the conditions of the
    // goal axiom, expanded by `goal_variable_values`), also seed numeric
    // splits at the initial concrete numeric value of each of the axiom's
    // regular numeric dependencies. Without this, the initial abstraction
    // contains the comparison-axiom prop vars at binary resolution but the
    // underlying numerics are unrefined — so no operator can flip the
    // comparison bit, and the abstract initial state cannot reach the goal.
    // CEGAR then bails with "abstract dead end at iteration 1" before it
    // could refine the numeric to enable reachability.
    if let Ok(index) = ComparisonAxiomIndex::from_task(task) {
        let init_numeric = task.get_initial_numeric_state_values();
        for fact in goal_variable_values(task) {
            let Some(tree) = index.comparison_tree(fact.var) else {
                continue;
            };
            for numeric_var_id in tree.regular_numeric_var_dependencies(task) {
                if blacklisted_numeric_var_ids.contains(&numeric_var_id) {
                    continue;
                }
                let Some(numeric_var) = task.numeric_variables().get(numeric_var_id) else {
                    continue;
                };
                if numeric_var.get_type() != &NumericType::Regular {
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
                let Some(&init_value) = init_numeric.get(numeric_var_id) else {
                    continue;
                };
                let include_in_lower = rng.gen_range(0..2) == 0;
                if partitions.split_at(numeric_var_id, init_value, include_in_lower)
                    && let Some(parts) = partitions.partitions(numeric_var_id)
                    && let Some(slot) = numeric_domain_sizes.get_mut(numeric_var_id)
                {
                    *slot = parts.len();
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_initial_seed_splits(
    task: &dyn AbstractNumericTask,
    config: &CegarConfig,
    blacklisted_prop_var_ids: &HashSet<usize>,
    blacklisted_numeric_var_ids: &HashSet<usize>,
    domain_mapping: &mut DomainMapping,
    domain_sizes: &mut [usize],
    partitions: &mut NumericPartitions,
    numeric_domain_sizes: &mut [usize],
) {
    let comparison_var_ids: HashSet<usize> = task
        .comparison_axioms()
        .iter()
        .map(|axiom| axiom.get_affected_var_id())
        .collect();

    for seed in &config.initial_seed_splits {
        match *seed {
            InitialSeedSplit::Numeric {
                numeric_var_id,
                value,
                include_in_lower,
            } => {
                if blacklisted_numeric_var_ids.contains(&numeric_var_id) {
                    continue;
                }
                let Some(numeric_var) = task.numeric_variables().get(numeric_var_id) else {
                    continue;
                };
                if numeric_var.get_type() != &NumericType::Regular {
                    continue;
                }
                if !can_refine_numeric_variable(
                    domain_sizes,
                    numeric_domain_sizes,
                    numeric_var_id,
                    config.max_abstraction_size,
                ) {
                    continue;
                }
                if partitions.split_at(numeric_var_id, value, include_in_lower)
                    && let Some(parts) = partitions.partitions(numeric_var_id)
                    && let Some(slot) = numeric_domain_sizes.get_mut(numeric_var_id)
                {
                    *slot = parts.len();
                }
            }
            InitialSeedSplit::Propositional { var_id, value } => {
                if blacklisted_prop_var_ids.contains(&var_id) {
                    continue;
                }
                let Ok(concrete_size) = task.get_variable_domain_size(var_id) else {
                    continue;
                };
                if value >= concrete_size {
                    continue;
                }
                let (new_domain_size, mapping) = if comparison_var_ids.contains(&var_id) {
                    let mut mapping = vec![0; concrete_size];
                    if !mapping.is_empty() {
                        mapping[0] = 1;
                    }
                    (2, mapping)
                } else {
                    let mut mapping = vec![0; concrete_size];
                    mapping[value] = 1;
                    (2, mapping)
                };
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
