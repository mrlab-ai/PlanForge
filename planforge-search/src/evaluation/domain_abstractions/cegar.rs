#[cfg(test)]
mod tests;

pub mod flaw_search;

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use planforge_sas::axioms::AxiomEvaluator;
use planforge_sas::utils::int_packer::IntDoublePacker;
use rand::Rng;
use rand::seq::SliceRandom;
use rand::{SeedableRng, rngs::SmallRng};
use tracing::{debug, info};

use planforge_sas::numeric_task::{AbstractNumericTask, ExplicitFact, Operator};

use flaw_search::{DependentNumericRefinement, Flaw, NumericFlaw, PropFlaw, can_split_numeric_var};

pub use flaw_search::FlawKind;
pub use flaw_search::SplitDirection;
pub use flaw_search::flaw_selection::{FlawTreatment, FlawTreatmentVariants, InitSplitMethod};

use crate::evaluation::domain_abstractions::cegar::flaw_search::state::{
    FlawSearchState, progress,
};
use crate::evaluation::domain_abstractions::utils::{
    fact_is_hold, get_initial_state, make_prop_state_packer,
};

use super::abstract_operator_generator::DomainMapping;
use super::additive_numeric_views::{
    comparison_refinement_dimensions, initial_numeric_values_with_additive_views,
    is_refinable_numeric_dimension,
};
use super::comparison_expression::Interval;
use super::domain_abstraction::{ComparisonAxiomIndex, NumericPartitions};
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
    pub initial_seed_splits: Vec<InitialSeedSplit>,
    /// When false, `DomainAbstractionGenerator::generate` skips building the
    /// `Vec<AbstractOperatorFootprint>`. Footprints are only
    /// consumed by abstract-operator transition-cost partitioning
    /// (SCP / fillSCP); for canonical-max and other heuristics that read
    /// only the distance table they are pure memory bloat — on
    /// minecraft-sword-advanced/prob_30x30_5 they account for ~12 GB of
    /// per-concrete-op `StateRegion` storage. Default `true` for
    /// backward-compat; the canonical/max wrappers flip this off.
    pub compute_operator_footprints: bool,
    /// How numeric flaw split values are chosen: `Forward` keeps the legacy
    /// concrete-value split; `Backward` places splits at the boundary derived
    /// from the regressed-target / required interval. When `None`, the flaw
    /// kind's default ([`FlawKind::default_split_direction`]) is used.
    pub split_direction: Option<SplitDirection>,
    /// Maximum number of comparison-axiom propositional vars this CEGAR run
    /// may refine into its pattern. `None` = unbounded. When the cap is
    /// reached, the refinement loop refuses to introduce additional
    /// comparison-axiom prop vars: `max_refined_single_atom` (and its
    /// siblings) fall through to alternative split candidates (typically
    /// a numeric split on a comparison-axiom dependency). If no eligible
    /// candidates remain, the CEGAR run terminates and the collection
    /// generator starts the next iteration with a different init seed.
    /// Exists to preserve canonical-DA additive-subset diversity (see
    /// `DomainAbstractionCollectionGeneratorMultipleCegarConfig::max_refined_comparison_vars_per_abstraction`
    /// for the longer rationale).
    pub max_refined_comparison_vars_per_abstraction: Option<usize>,
}

impl Default for CegarConfig {
    fn default() -> Self {
        Self {
            max_abstraction_size: 100_000,
            max_iterations: usize::MAX,
            max_time: None,
            use_wildcard_plans: false,
            combine_labels: true,
            debug: false,
            random_seed: Some(2011),
            flaw_kind: FlawKind::Progression,
            flaw_treatment: FlawTreatmentVariants::RandomSingleAtom,
            init_split_method: InitSplitMethod::InitValue,
            init_split_var_ids: None,
            blacklisted_prop_var_ids: HashSet::new(),
            blacklisted_numeric_var_ids: HashSet::new(),
            initial_seed_splits: Vec::new(),
            split_direction: None,
            compute_operator_footprints: true,
            max_refined_comparison_vars_per_abstraction: None,
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
    pub stop_reason: CegarStopReason,
    /// True iff CEGAR's loop exited because no flaws remained. For a
    /// standalone full-task abstraction, the abstract plan is then a real
    /// concrete plan and `h(init)` is optimal.
    pub solved_by_self: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CegarStopReason {
    ConcretePlan,
    TimeLimit,
    MemoryLimit,
    IterationLimit,
    NoRefinableFlaw,
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
        // Per-CEGAR-invocation seed diversification. The collection-generator hands every
        // CEGAR call the same `config.random_seed`, so without this counter each abstraction
        // would explore the identical RNG trajectory — defeating diversity. A process-wide
        // counter xor'd (via splitmix-style mixing) into the seed gives every CEGAR a distinct
        // initial state while remaining fully reproducible when `random_seed` is set.
        static CEGAR_INVOCATION_COUNTER: AtomicU64 = AtomicU64::new(0);
        let invocation = CEGAR_INVOCATION_COUNTER.fetch_add(1, Ordering::Relaxed);
        let base_seed = config.random_seed.unwrap_or_else(current_time_seed);
        let mut rng = SmallRng::seed_from_u64(base_seed ^ splitmix64(invocation));

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
            )?;
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
        let mut stop_reason = CegarStopReason::IterationLimit;

        while iteration <= config.max_iterations {
            if iteration.is_multiple_of(64)
                && !crate::resource_limits::poll_and_release_if_exceeded()
            {
                stop_reason = CegarStopReason::MemoryLimit;
                break;
            }
            if let Some(max_time) = config.max_time
                && start.elapsed() >= max_time
            {
                stop_reason = CegarStopReason::TimeLimit;
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
            let deadline = config.max_time.map(|max_time| start + max_time);
            match factory.compute_plan_with_rng_and_deadline(
                task,
                config.combine_labels,
                config.debug,
                config.use_wildcard_plans,
                Some(&mut rng),
                deadline,
            ) {
                Ok(plan) => wildcard_plan = plan,
                Err(error) if is_deadline_error(&error) => {
                    info!(
                        "CEGAR: deadline expired while computing abstract plan at iteration {}; stopping refinement",
                        iteration
                    );
                    stop_reason = CegarStopReason::TimeLimit;
                    break;
                }
                Err(error) => {
                    return Err(error).with_context(|| {
                        format!("failed to compute abstract plan (iteration {iteration})")
                    });
                }
            }
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
                solved_by_self = true;
                stop_reason = CegarStopReason::ConcretePlan;
                break;
            }
            let eligible_flaws =
                filter_eligible_flaws(task, &factory.partitions, &factory.domain_sizes, &flaws);
            if eligible_flaws.filtered_stale > 0 {
                debug!(
                    "filtered {} stale flaws (already-split values / fully refined vars)",
                    eligible_flaws.filtered_stale
                );
            }
            if eligible_flaws.flaws.is_empty() {
                info!(
                    "CEGAR: {} flaws remain but none are refinable (stale boundaries / fully refined vars); stopping refinement at iteration {}",
                    flaws.len(),
                    iteration
                );
                stop_reason = CegarStopReason::NoRefinableFlaw;
                break;
            }

            let before_size = if config.debug {
                compute_abstraction_size_u128(&factory.domain_sizes, &factory.numeric_domain_sizes)
            } else {
                None
            };
            // Overflow-free progress signal: every landed split increments
            // exactly one entry of `domain_sizes` or `numeric_domain_sizes`,
            // so the sum strictly increases iff any refinement landed. The
            // u128 size product cannot serve here: it saturates to `None` on
            // overflow, which would make progress undetectable.
            let before_partition_count = factory.domain_sizes.iter().sum::<usize>()
                + factory.numeric_domain_sizes.iter().sum::<usize>();
            let refine_start = Instant::now();
            let refined = fix_flaws(
                &self.config,
                task,
                &eligible_flaws.flaws,
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
            let after_partition_count = factory.domain_sizes.iter().sum::<usize>()
                + factory.numeric_domain_sizes.iter().sum::<usize>();
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
                stop_reason = CegarStopReason::NoRefinableFlaw;
                break;
            }
            let split_values: Vec<_> = eligible_flaws
                .flaws
                .iter()
                .filter_map(|flaw| match flaw {
                    Flaw::Numeric(numeric) => Some((
                        numeric.numeric_var_id,
                        numeric.value,
                        numeric.include_in_lower,
                    )),
                    Flaw::Propositional(_) => None,
                })
                .collect();
            ensure!(
                after_partition_count > before_partition_count,
                "CEGAR refinement made no progress at iteration {iteration}: flaws={:?}, refined={:?}, split_values={:?}",
                eligible_flaws.flaws,
                refined,
                split_values
            );
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
            stop_reason,
            solved_by_self,
        })
    }
}

fn is_deadline_error(error: &anyhow::Error) -> bool {
    crate::resource_limits::is_deadline_exceeded(error)
}

#[derive(Debug, Clone)]
struct EligibleFlaws {
    flaws: Vec<Flaw>,
    filtered_stale: usize,
}

fn filter_eligible_flaws(
    task: &dyn AbstractNumericTask,
    partitions: &NumericPartitions,
    domain_sizes: &[usize],
    flaws: &[Flaw],
) -> EligibleFlaws {
    let mut eligible = Vec::with_capacity(flaws.len());
    let mut filtered_stale = 0usize;
    for flaw in flaws {
        match flaw {
            Flaw::Numeric(numeric) => {
                if numeric_flaw_is_refinable(partitions, numeric) {
                    eligible.push(Flaw::Numeric(numeric.clone()));
                } else {
                    filtered_stale += 1;
                }
            }
            Flaw::Propositional(prop) => {
                if !propositional_flaw_is_refinable(task, domain_sizes, prop) {
                    filtered_stale += 1 + prop.dependent_numeric_flaws.len();
                    continue;
                }
                let mut filtered_prop = prop.clone();
                let old_dep_count = filtered_prop.dependent_numeric_flaws.len();
                filtered_prop
                    .dependent_numeric_flaws
                    .retain(|numeric| numeric_flaw_is_refinable(partitions, numeric));
                filtered_stale += old_dep_count - filtered_prop.dependent_numeric_flaws.len();
                eligible.push(Flaw::Propositional(filtered_prop));
            }
        }
    }
    EligibleFlaws {
        flaws: eligible,
        filtered_stale,
    }
}

fn numeric_flaw_is_refinable(partitions: &NumericPartitions, flaw: &NumericFlaw) -> bool {
    can_split_numeric_var(
        partitions,
        flaw.numeric_var_id,
        flaw.value,
        flaw.include_in_lower,
    )
}

fn propositional_flaw_is_refinable(
    task: &dyn AbstractNumericTask,
    domain_sizes: &[usize],
    flaw: &PropFlaw,
) -> bool {
    let var_id = flaw.fact.var();
    let current_size = *domain_sizes
        .get(var_id)
        .unwrap_or_else(|| panic!("flaw variable {var_id} is outside domain_sizes"));
    let true_size = task
        .get_variable_domain_size(var_id)
        .unwrap_or_else(|error| panic!("failed to get true domain size for var {var_id}: {error}"));
    assert!(
        flaw.fact.value() < true_size,
        "flaw fact value {} outside true domain size {} for var {}",
        flaw.fact.value(),
        true_size,
        var_id
    );
    current_size < true_size
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
    let state_packer = std::sync::Arc::new(make_prop_state_packer(task));
    let axiom_evaluator = AxiomEvaluator::new(std::sync::Arc::new(task), state_packer.clone());

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

/// SplitMix64 bit-mixer — turns a low-entropy counter into a well-spread `u64` so xor'ing it
/// into a base seed produces independent SmallRng streams across CEGAR invocations.
#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
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
    let eligible_flaws = filter_eligible_flaws(task, partitions, domain_sizes, flaws);
    if eligible_flaws.filtered_stale > 0 {
        debug!(
            "filtered {} stale flaws (already-split values / fully refined vars)",
            eligible_flaws.filtered_stale
        );
    }
    let flaws = eligible_flaws.flaws;
    if flaws.is_empty() {
        return Ok(RefinementSummary::default());
    }

    let comparison_var_ids: HashSet<usize> = task
        .comparison_axioms()
        .iter()
        .map(|ax| ax.get_affected_var_id())
        .collect();

    let chosen_flaws: ChosenFlaws = config.flaw_treatment.choose_flaws(
        task,
        &flaws,
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
            let var_id = pf.fact.var();
            let value = pf.fact.value();

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
            let mut prop_domain_size_changed = false;

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
                    prop_domain_size_changed = true;
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
                debug_assert!(domain_sizes[var_id] >= old_size);
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
                prop_domain_size_changed = true;
            }

            // Optional dependent numeric refinements (currently produced only for comparison vars).
            if dependent_numeric_refinement != DependentNumericRefinement::None
                && !pf.dependent_numeric_flaws.is_empty()
            {
                let mut any_numeric_changed = false;
                let mut refined = RefinementSummary::default();
                if prop_domain_size_changed {
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
                if prop_domain_size_changed {
                    refined.mark_propositional(var_id);
                }
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
        if let Some(&axiom_idx) = goal_axiom_map.get(&goal.var()) {
            let axiom = &task.axioms()[axiom_idx];
            for condition in axiom.conditions() {
                goals.push(ExplicitFact::new(condition.var(), condition.value()));
            }
        } else {
            goals.push(ExplicitFact::new(goal.var(), goal.value()));
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
        .map(|v| (v.var(), v.value()))
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
    let initial_numeric = initial_numeric_values_with_additive_views(task);

    for encoded_var_id in candidate_var_ids {
        if encoded_var_id >= num_prop_vars {
            let numeric_var_id = encoded_var_id - num_prop_vars;
            if blacklisted_numeric_var_ids.contains(&numeric_var_id) {
                continue;
            }
            let Some(_) = task.numeric_variables().get(numeric_var_id) else {
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
            let Some(&init_value) = initial_numeric.get(numeric_var_id) else {
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
    if let Ok(index) = ComparisonAxiomIndex::from_task(task) {
        let init_split_filter: Option<&HashSet<usize>> = config.init_split_var_ids.as_ref();
        for fact in goal_variable_values(task) {
            if let Some(allowed) = init_split_filter
                && !allowed.contains(&fact.var())
            {
                continue;
            }
            let Some(tree) = index.comparison_tree(fact.var()) else {
                continue;
            };
            for numeric_var_id in comparison_refinement_dimensions(task, tree) {
                if blacklisted_numeric_var_ids.contains(&numeric_var_id) {
                    continue;
                }
                let Some(_) = task.numeric_variables().get(numeric_var_id) else {
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
                let Some(&init_value) = initial_numeric.get(numeric_var_id) else {
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
) -> Result<()> {
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
                ensure!(
                    task.numeric_variables().get(numeric_var_id).is_some(),
                    "initial numeric seed references missing variable {numeric_var_id}"
                );
                ensure!(
                    is_refinable_numeric_dimension(task, numeric_var_id),
                    "initial numeric seed references unsupported abstraction dimension {numeric_var_id} ({})",
                    task.numeric_variables()[numeric_var_id].name()
                );
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
    Ok(())
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
