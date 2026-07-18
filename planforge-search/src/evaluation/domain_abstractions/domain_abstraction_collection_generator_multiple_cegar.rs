#[cfg(test)]
mod tests;

use std::cmp::{Ordering, Reverse};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::Hash;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use ordered_float::OrderedFloat;
use planforge_sas::axioms::ComparisonAxiom;
use planforge_sas::numeric_task::{AbstractNumericTask, ExplicitFact, NumericType, Operator};
use planforge_sas::utils::linear_effects::linearize_numeric_var;
use rand::seq::SliceRandom;
use rand::{RngCore, SeedableRng, rngs::SmallRng};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::evaluation::abstraction_collections::portfolio::mix_seed;
use crate::evaluation::abstraction_task::{AbstractionUse, SingleGoalTask};
use crate::evaluation::domain_abstractions::cegar::FlawKind;

use super::additive_numeric_views::{
    comparison_refinement_dimensions, initial_numeric_values_with_additive_views,
    is_operator_invariant_regular_dimension, is_refinable_numeric_dimension, numeric_effect_deltas,
};
use super::cegar::CegarConfig;
use super::cegar::InitialSeedSplit;
use super::cegar::SplitDirection;
pub use super::cegar::flaw_search::flaw_selection::{FlawTreatmentVariants, InitSplitMethod};
use super::cegar::flaw_search::numeric_requirement_for_comparison_fact;
use super::comparison_expression::{CompOp, Interval};
use super::domain_abstraction::ComparisonAxiomIndex;
use super::domain_abstraction_generator::{
    DomainAbstraction, DomainAbstractionGenerator, DomainAbstractionMetadata,
};
use super::utils::compute_abstraction_size_u128;
use crate::resource_limits;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VariableSubset {
    Goals,
    NonGoals,
    All,
}

impl fmt::Display for VariableSubset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Goals => write!(f, "goals"),
            Self::NonGoals => write!(f, "non_goals"),
            Self::All => write!(f, "all"),
        }
    }
}

impl crate::config::sealed::Sealed for VariableSubset {}

impl crate::config::FromOptionValue for VariableSubset {
    fn from_option_value(value: &crate::config::ConfigValue) -> Result<Self, String> {
        match crate::config::atom(value)? {
            "goals" => Ok(Self::Goals),
            "non_goals" => Ok(Self::NonGoals),
            "all" => Ok(Self::All),
            other => Err(format!("invalid VariableSubset `{other}`")),
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum InitSplitQuantity {
    None,
    Single,
    All,
}

impl fmt::Display for InitSplitQuantity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::Single => write!(f, "single"),
            Self::All => write!(f, "all"),
        }
    }
}

impl crate::config::sealed::Sealed for InitSplitQuantity {}

impl crate::config::FromOptionValue for InitSplitQuantity {
    fn from_option_value(value: &crate::config::ConfigValue) -> Result<Self, String> {
        match crate::config::atom(value)? {
            "none" => Ok(Self::None),
            "single" => Ok(Self::Single),
            "all" => Ok(Self::All),
            other => Err(format!("invalid InitSplitQuantity `{other}`")),
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NumericSplitStrategy {
    Standard,
    Exclusion,
}

impl fmt::Display for NumericSplitStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Standard => write!(f, "standard"),
            Self::Exclusion => write!(f, "exclusion"),
        }
    }
}

impl crate::config::sealed::Sealed for NumericSplitStrategy {}

impl crate::config::FromOptionValue for NumericSplitStrategy {
    fn from_option_value(value: &crate::config::ConfigValue) -> Result<Self, String> {
        match crate::config::atom(value)? {
            "standard" => Ok(Self::Standard),
            "exclusion" => Ok(Self::Exclusion),
            other => Err(format!("invalid NumericSplitStrategy `{other}`")),
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PortfolioStrategy {
    Standard,
    Complementary,
}

impl fmt::Display for PortfolioStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Standard => write!(f, "standard"),
            Self::Complementary => write!(f, "complementary"),
        }
    }
}

impl crate::config::sealed::Sealed for PortfolioStrategy {}

impl crate::config::FromOptionValue for PortfolioStrategy {
    fn from_option_value(value: &crate::config::ConfigValue) -> Result<Self, String> {
        match crate::config::atom(value)? {
            "standard" => Ok(Self::Standard),
            "complementary" => Ok(Self::Complementary),
            other => Err(format!("invalid PortfolioStrategy `{other}`")),
        }
    }
}

impl PortfolioStrategy {
    fn uses_ranked_goals(self) -> bool {
        matches!(self, Self::Complementary)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ComplementaryDirection {
    Regression,
    Progression,
}

#[derive(Debug, Clone)]
struct RootGroup {
    numeric_var_ids: HashSet<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct NumericRootGroupKey {
    coefficient_shape: Vec<OrderedFloat<f64>>,
    invariant_terms: Vec<(usize, OrderedFloat<f64>)>,
    constant: OrderedFloat<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum SeedIdentity {
    Propositional {
        var_id: usize,
        value: usize,
    },
    Numeric {
        numeric_var_id: usize,
        value_bits: u64,
        include_in_lower: bool,
    },
}

#[derive(
    Debug, Clone, Deserialize, Serialize, PartialEq, planforge_search::config::ApplyOptions,
)]
pub struct DomainAbstractionCollectionGeneratorMultipleCegarConfig {
    pub max_abstraction_size: usize,
    pub max_collection_size: usize,
    pub abstraction_generation_max_time: f64,
    pub total_max_time: f64,
    pub stagnation_limit: f64,
    pub blacklist_trigger_percentage: f64,
    pub enable_blacklist_on_stagnation: bool,
    pub blacklist_option: VariableSubset,
    pub init_split_candidates: VariableSubset,
    pub init_split_quantity: InitSplitQuantity,
    pub random_seed: Option<u64>,
    pub debug: bool,
    pub use_wildcard_plans: bool,
    pub combine_labels: bool,
    pub flaw_kind: FlawKind,
    pub flaw_treatment: FlawTreatmentVariants,
    pub init_split_method: InitSplitMethod,
    pub numeric_split_strategy: NumericSplitStrategy,
    pub portfolio_strategy: PortfolioStrategy,
    /// Overrides `FlawKind`'s default split direction when set; otherwise the
    /// flaw kind chooses its own default (`Forward` for everything except
    /// `TargetCentered`, which defaults to `Backward`).
    pub split_direction: Option<SplitDirection>,
    /// Pass-through for `CegarConfig::compute_operator_footprints`. Default
    /// `true`. SCP/fillSCP wrappers leave it on; canonical/max wrappers turn
    /// it off to skip ~12 GB of per-concrete-op `StateRegion` storage on
    /// large tasks like minecraft-sword-advanced/prob_30x30_5.
    /// Set internally by heuristic construction; not exposed as a CLI option.
    #[option(skip)]
    pub compute_operator_footprints: bool,
    /// Cap on the number of comparison-axiom propositional vars a single
    /// CEGAR run may refine into its pattern. `None` = unbounded (the
    /// historical behavior). When set, the refinement loop rejects any
    /// split that would introduce a new comparison-axiom prop var beyond
    /// this cap; `max_refined_single_atom` falls through to the next
    /// candidate (typically a numeric split). The cap exists to keep
    /// canonical-DA additive-subset diversity: when every abstraction
    /// refines every comparison var, cascade-relevance covers all
    /// numerics those comparisons depend on, every operator that
    /// modifies any of those numerics is marked relevant, and Bron-
    /// Kerbosch returns singleton subsets — canonical degenerates to
    /// `max h_i`. On counters/pfile4 with this cap unset, initial h=14
    /// and the search OOMs at 8 GB; numeric-fd with the same CEGAR
    /// config naturally yields abstractions with 3-5 refined comparison
    /// vars per pattern, gets initial h=21, and fits in 7.4 GB.
    /// Not currently exposed as a CLI option.
    #[option(skip)]
    pub max_refined_comparison_vars_per_abstraction: Option<usize>,
}

impl Default for DomainAbstractionCollectionGeneratorMultipleCegarConfig {
    fn default() -> Self {
        Self {
            max_abstraction_size: 10_000,
            max_collection_size: 10_000_000,
            abstraction_generation_max_time: f64::INFINITY,
            total_max_time: 10.0,
            stagnation_limit: 20.0,
            blacklist_trigger_percentage: 0.75,
            enable_blacklist_on_stagnation: true,
            blacklist_option: VariableSubset::All,
            init_split_candidates: VariableSubset::All,
            init_split_quantity: InitSplitQuantity::Single,
            random_seed: None,
            debug: false,
            use_wildcard_plans: true,
            combine_labels: true,
            flaw_kind: FlawKind::Progression,
            flaw_treatment: FlawTreatmentVariants::RandomSingleAtom,
            init_split_method: InitSplitMethod::InitValue,
            numeric_split_strategy: NumericSplitStrategy::Standard,
            portfolio_strategy: PortfolioStrategy::Standard,
            split_direction: None,
            compute_operator_footprints: true,
            max_refined_comparison_vars_per_abstraction: None,
        }
    }
}

fn fmt_f64(value: f64) -> String {
    if value.is_infinite() {
        "infinity".to_string()
    } else {
        value.to_string()
    }
}

fn fmt_optional_seed(seed: Option<u64>) -> String {
    seed.map_or_else(|| "none".to_string(), |seed| seed.to_string())
}

fn time_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(0x5EED_F00D_u64)
}

fn is_generation_deadline_error(error: &anyhow::Error) -> bool {
    crate::resource_limits::is_deadline_exceeded(error)
}

impl fmt::Display for DomainAbstractionCollectionGeneratorMultipleCegarConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            concat!(
                "max_abstraction_size={}, ",
                "max_collection_size={}, ",
                "abstraction_generation_max_time={}, ",
                "total_max_time={}, ",
                "stagnation_limit={}, ",
                "blacklist_trigger_percentage={}, ",
                "enable_blacklist_on_stagnation={}, ",
                "blacklist_option={}, ",
                "init_split_candidates={}, ",
                "init_split_quantity={}, ",
                "random_seed={}, ",
                "debug={}, ",
                "use_wildcard_plans={}, ",
                "combine_labels={}, ",
                "flaw_treatment={}, ",
                "init_split_method={}, ",
                "numeric_split_strategy={}, ",
                "portfolio_strategy={}, ",
            ),
            self.max_abstraction_size,
            self.max_collection_size,
            fmt_f64(self.abstraction_generation_max_time),
            fmt_f64(self.total_max_time),
            fmt_f64(self.stagnation_limit),
            fmt_f64(self.blacklist_trigger_percentage),
            self.enable_blacklist_on_stagnation,
            self.blacklist_option,
            self.init_split_candidates,
            self.init_split_quantity,
            fmt_optional_seed(self.random_seed),
            self.debug,
            self.use_wildcard_plans,
            self.combine_labels,
            self.flaw_treatment,
            self.init_split_method,
            self.numeric_split_strategy,
            self.portfolio_strategy,
        )
    }
}

#[derive(Debug, Clone)]
pub struct DomainAbstractionCollectionGeneratorMultipleCegar {
    config: DomainAbstractionCollectionGeneratorMultipleCegarConfig,
}

impl DomainAbstractionCollectionGeneratorMultipleCegar {
    pub fn new(config: DomainAbstractionCollectionGeneratorMultipleCegarConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &DomainAbstractionCollectionGeneratorMultipleCegarConfig {
        &self.config
    }

    fn validate_supported_options(&self) -> Result<()> {
        if self.config.numeric_split_strategy != NumericSplitStrategy::Standard {
            bail!("`numeric_split_strategy=exclusion` is not supported in the current Rust port");
        }
        Ok(())
    }

    fn create_rng(&self) -> SmallRng {
        SmallRng::seed_from_u64(self.config.random_seed.unwrap_or_else(time_seed))
    }

    fn build_cegar_config(
        &self,
        max_abstraction_size: usize,
        remaining_time: f64,
        init_split_var_ids: Option<HashSet<usize>>,
        initial_seed_splits: Vec<InitialSeedSplit>,
        blacklisted_prop_var_ids: HashSet<usize>,
        blacklisted_numeric_var_ids: HashSet<usize>,
        random_seed: Option<u64>,
        flaw_kind: FlawKind,
    ) -> CegarConfig {
        CegarConfig {
            max_abstraction_size,
            max_iterations: CegarConfig::default().max_iterations,
            max_time: if remaining_time.is_finite() {
                Some(Duration::from_secs_f64(remaining_time.max(0.0)))
            } else {
                None
            },
            use_wildcard_plans: self.config.use_wildcard_plans,
            combine_labels: self.config.combine_labels,
            debug: self.config.debug,
            random_seed,
            flaw_kind,
            flaw_treatment: self.config.flaw_treatment,
            init_split_method: match self.config.init_split_quantity {
                InitSplitQuantity::None => InitSplitMethod::Identity,
                InitSplitQuantity::Single | InitSplitQuantity::All => self.config.init_split_method,
            },
            init_split_var_ids,
            blacklisted_prop_var_ids,
            blacklisted_numeric_var_ids,
            initial_seed_splits,
            split_direction: self.config.split_direction,
            compute_operator_footprints: self.config.compute_operator_footprints,
            max_refined_comparison_vars_per_abstraction: self
                .config
                .max_refined_comparison_vars_per_abstraction,
        }
    }

    pub fn generate_collection(
        &self,
        task: &dyn AbstractNumericTask,
    ) -> Result<Vec<DomainAbstraction>> {
        self.validate_supported_options()?;

        let mut rng = self.create_rng();
        let mut goals: Vec<_> = (0..task.get_num_goals())
            .map(|goal_id| task.get_goal_fact(goal_id).clone())
            .collect();
        if self.config.portfolio_strategy.uses_ranked_goals() {
            goals.sort_by(|left, right| compare_goals_for_collection(task, left, right));
        } else {
            goals.shuffle(&mut rng);
        }
        let blacklist_candidates =
            collect_blacklist_candidate_var_ids(task, self.config.blacklist_option);

        let start = Instant::now();
        let mut remaining_collection_size = self.config.max_collection_size;
        let mut generated_keys: HashSet<AbstractionKey> = HashSet::new();
        let mut generated_abstractions: Vec<DomainAbstraction> = Vec::new();
        let mut time_point_of_last_new_abstraction = 0.0f64;
        let mut blacklisting = false;
        let blacklist_start_time =
            self.config.total_max_time * self.config.blacklist_trigger_percentage;
        let mut iteration = 1usize;
        let mut goal_index = 0usize;
        let mut group_index = 0usize;
        let mut complementary_direction = ComplementaryDirection::Regression;
        let mut complementary_round = 0usize;
        let stop_reason: &str;
        loop {
            let elapsed = start.elapsed().as_secs_f64();
            if !blacklisting && elapsed > blacklist_start_time {
                blacklisting = true;
                time_point_of_last_new_abstraction = elapsed;
            }

            let remaining_total_time = if self.config.total_max_time.is_finite() {
                (self.config.total_max_time - elapsed).max(0.0)
            } else {
                f64::INFINITY
            };
            let remaining_generation_time = self
                .config
                .abstraction_generation_max_time
                .min(remaining_total_time);
            let remaining_abstraction_size =
                remaining_collection_size.min(self.config.max_abstraction_size);

            if remaining_abstraction_size == 0 {
                stop_reason = "collection size limit";
                break;
            }
            // Numeric FD constructs one abstraction before checking
            // collection-level limits. A collection must never silently
            // become an empty successful heuristic.
            if remaining_generation_time <= 0.0 && !generated_abstractions.is_empty() {
                stop_reason = "collection time limit";
                break;
            }

            let full_goal_task = self.uses_full_goal_task(goals.len(), iteration);
            let single_goal_task = if full_goal_task {
                None
            } else {
                goals
                    .get(goal_index)
                    .map(|goal| SingleGoalTask::new(task, goal.clone()))
            };
            let abstraction_task: &dyn AbstractNumericTask = single_goal_task
                .as_ref()
                .map(|single_goal_task| single_goal_task as &dyn AbstractNumericTask)
                .unwrap_or(task);
            let root_groups = if full_goal_task {
                Vec::new()
            } else {
                let goal = goals
                    .get(goal_index)
                    .expect("complementary goal_index is in bounds");
                self.root_groups_for_goal(abstraction_task, abstraction_task, goal)
            };
            let active_root_group = if root_groups.is_empty() {
                None
            } else {
                Some(
                    root_groups
                        .get(group_index)
                        .expect("complementary group_index is in bounds"),
                )
            };
            let blacklisted_var_ids = if blacklisting {
                sample_blacklisted_variables(&blacklist_candidates, &mut rng)
            } else {
                HashSet::new()
            };
            let (blacklisted_prop_var_ids, blacklisted_numeric_var_ids) =
                split_blacklisted_variables(abstraction_task, blacklisted_var_ids);
            let (group_blacklisted_prop_var_ids, group_blacklisted_numeric_var_ids) =
                self.group_blacklisted_variable_ids(abstraction_task, active_root_group);
            let blacklisted_prop_var_ids = blacklisted_prop_var_ids
                .union(&group_blacklisted_prop_var_ids)
                .copied()
                .collect::<HashSet<_>>();
            let blacklisted_numeric_var_ids = blacklisted_numeric_var_ids
                .union(&group_blacklisted_numeric_var_ids)
                .copied()
                .collect::<HashSet<_>>();
            let initial_seed_splits = self.initial_seed_splits_for_goal_count(
                abstraction_task,
                iteration,
                goals.len(),
                complementary_direction,
                active_root_group,
            );
            let init_split_var_ids = if initial_seed_splits.is_empty() {
                self.initial_split_var_ids(abstraction_task, iteration)
            } else {
                None
            };
            let flaw_kind = self.flaw_kind_for_goal_count_and_direction(
                goals.len(),
                iteration,
                complementary_direction,
            );
            let seed_descriptions = initial_seed_splits
                .iter()
                .map(seed_split_description)
                .collect::<Vec<_>>();
            let cegar_random_seed = self.cegar_random_seed(
                &mut rng,
                iteration,
                goal_index,
                group_index,
                complementary_direction,
                full_goal_task,
            );
            let cegar_config = self.build_cegar_config(
                remaining_abstraction_size,
                remaining_generation_time,
                init_split_var_ids,
                initial_seed_splits,
                blacklisted_prop_var_ids,
                blacklisted_numeric_var_ids,
                Some(cegar_random_seed),
                flaw_kind,
            );
            let generator = DomainAbstractionGenerator::new(cegar_config)
                .context("failed to construct single-abstraction CEGAR generator")?;
            let generation_start = Instant::now();
            debug!(
                "domain abstraction collection: starting CEGAR generation iteration {}, remaining_generation_time={:.2}s, full_goal_task={}, flaw_kind={}, complementary_round={}, goal_index={}, group_index={}, direction={:?}",
                iteration,
                remaining_generation_time,
                full_goal_task,
                flaw_kind,
                complementary_round,
                goal_index,
                group_index,
                complementary_direction,
            );
            let mut abstraction = match generator.generate(abstraction_task) {
                Ok(abstraction) => abstraction,
                Err(error) if is_generation_deadline_error(&error) => {
                    info!(
                        "domain abstraction collection: generation deadline expired at iteration {}; keeping {} abstractions built so far",
                        iteration,
                        generated_abstractions.len()
                    );
                    stop_reason = "abstraction generation deadline";
                    break;
                }
                Err(error) => {
                    return Err(error).with_context(|| {
                        format!(
                            "failed to generate abstraction for collection iteration {iteration}"
                        )
                    });
                }
            };
            debug!(
                "domain abstraction collection: finished CEGAR generation iteration {} in {:.3}s",
                iteration,
                generation_start.elapsed().as_secs_f64()
            );
            let solved_by_self = abstraction.metadata.solved_by_self;
            let cegar_stop_reason = abstraction.metadata.stop_reason;
            abstraction.metadata = DomainAbstractionMetadata {
                collection_iteration: Some(iteration),
                portfolio_strategy: Some(self.config.portfolio_strategy.to_string()),
                flaw_kind: Some(flaw_kind.to_string()),
                full_goal_task: Some(full_goal_task),
                abstraction_use: AbstractionUse::CollectionMember,
                initial_seed_splits: seed_descriptions,
                max_abstraction_size: Some(remaining_abstraction_size),
                solved_by_self,
                stop_reason: cegar_stop_reason,
            };

            let abstraction_size = compute_abstraction_size_u128(
                abstraction.factory.domain_sizes(),
                abstraction.factory.numeric_domain_sizes(),
            )
            .unwrap_or(u128::MAX);

            let abstraction_key = AbstractionKey::from_abstraction(&abstraction);
            if generated_keys.insert(abstraction_key) {
                time_point_of_last_new_abstraction = elapsed;
                let consumed = abstraction_size.min(remaining_collection_size as u128) as usize;
                remaining_collection_size = remaining_collection_size.saturating_sub(consumed);
                generated_abstractions.push(abstraction);
                if self.config.debug {
                    if let Some(last) = generated_abstractions.last() {
                        log_collection_abstraction_debug(
                            generated_abstractions.len() - 1,
                            last,
                            abstraction_task,
                        );
                    }
                }
                debug!(
                    "domain abstraction collection: added abstraction at iteration {}, abstraction_size={}, elapsed={:.2}s, remaining_collection_size={}, next_max_abstraction_size={}, remaining_generation_time={:.2}s, blacklisting={}",
                    iteration,
                    abstraction_size,
                    start.elapsed().as_secs_f64(),
                    remaining_collection_size,
                    remaining_collection_size.min(self.config.max_abstraction_size),
                    remaining_generation_time,
                    blacklisting
                );
            }

            let stagnated =
                elapsed - time_point_of_last_new_abstraction > self.config.stagnation_limit;
            if remaining_collection_size == 0 {
                stop_reason = "collection size limit";
                break;
            }
            if self.config.total_max_time.is_finite() && elapsed >= self.config.total_max_time {
                stop_reason = "collection time limit";
                break;
            }
            if stagnated && (!self.config.enable_blacklist_on_stagnation || blacklisting) {
                stop_reason = "stagnation limit";
                break;
            }
            // Release the reserved padding before the process reaches its hard limit,
            // leaving enough memory for search to report a useful result.
            if !resource_limits::poll_and_release_if_exceeded() {
                info!(
                    "domain abstraction collection: stopping at iteration {iteration} \
                     because the memory padding was released (RSS limit reached)"
                );
                stop_reason = "memory limit";
                break;
            }
            if stagnated && self.config.enable_blacklist_on_stagnation {
                blacklisting = true;
                time_point_of_last_new_abstraction = elapsed;
            }

            if !full_goal_task {
                let group_count = root_groups.len();
                assert!(
                    group_count > 0,
                    "complementary iteration must have at least one root group"
                );
                if advance_complementary_schedule(
                    goals.len(),
                    group_count,
                    &mut goal_index,
                    &mut group_index,
                    &mut complementary_direction,
                ) {
                    complementary_round += 1;
                    debug!(
                        "domain abstraction collection: starting complementary round {complementary_round}"
                    );
                }
            }

            iteration += 1;
        }

        if generated_abstractions.is_empty() {
            bail!(
                "multi_domain_abstractions(...) failed to generate the mandatory first abstraction ({stop_reason})"
            )
        }
        let total_states = generated_abstractions.iter().try_fold(
            0u128,
            |total, abstraction| -> Result<u128> {
                let size = compute_abstraction_size_u128(
                    abstraction.factory.domain_sizes(),
                    abstraction.factory.numeric_domain_sizes(),
                )
                .context("domain abstraction size overflow in collection summary")?;
                total
                    .checked_add(size)
                    .context("domain abstraction collection state count overflow")
            },
        )?;
        info!(
            "domain abstraction collection finished: abstractions={}, states={}, elapsed={:.3}s, stop_reason={}",
            generated_abstractions.len(),
            total_states,
            start.elapsed().as_secs_f64(),
            stop_reason
        );
        if self.config.debug {
            log_collection_debug_summary(&generated_abstractions);
        }

        Ok(generated_abstractions)
    }

    fn initial_split_var_ids(
        &self,
        task: &dyn AbstractNumericTask,
        iteration: usize,
    ) -> Option<HashSet<usize>> {
        let candidate_var_ids =
            collect_init_split_candidate_var_ids(task, self.config.init_split_candidates);

        let selected_var_ids: HashSet<usize> = match self.config.init_split_quantity {
            InitSplitQuantity::None => HashSet::new(),
            InitSplitQuantity::All => candidate_var_ids.iter().copied().collect(),
            InitSplitQuantity::Single => {
                select_single_init_split_var(&candidate_var_ids, iteration)
                    .into_iter()
                    .collect()
            }
        };

        Some(selected_var_ids)
    }

    fn initial_seed_splits_for_goal_count(
        &self,
        task: &dyn AbstractNumericTask,
        iteration: usize,
        goal_count: usize,
        complementary_direction: ComplementaryDirection,
        active_root_group: Option<&RootGroup>,
    ) -> Vec<InitialSeedSplit> {
        match self.config.portfolio_strategy {
            PortfolioStrategy::Standard => return Vec::new(),
            PortfolioStrategy::Complementary => {
                if self.complementary_uses_target_centered_for_goal_count(
                    goal_count,
                    iteration,
                    complementary_direction,
                ) {
                    let mut seeds = self.backward_goal_seed_splits(task);
                    if let Some(root_group) = active_root_group {
                        seeds.retain(|seed| match seed {
                            InitialSeedSplit::Propositional { .. } => true,
                            InitialSeedSplit::Numeric { numeric_var_id, .. } => {
                                root_group.numeric_var_ids.contains(numeric_var_id)
                            }
                        });
                    }
                    return seeds;
                }
                return Vec::new();
            }
        }
    }

    fn uses_full_goal_task(&self, goal_count: usize, _iteration: usize) -> bool {
        match self.config.portfolio_strategy {
            PortfolioStrategy::Standard => true,
            PortfolioStrategy::Complementary => goal_count == 0,
        }
    }

    fn cegar_random_seed(
        &self,
        rng: &mut SmallRng,
        iteration: usize,
        goal_index: usize,
        group_index: usize,
        complementary_direction: ComplementaryDirection,
        full_goal_task: bool,
    ) -> u64 {
        match self.config.portfolio_strategy {
            PortfolioStrategy::Standard => rng.next_u64(),
            PortfolioStrategy::Complementary if full_goal_task => rng.next_u64(),
            PortfolioStrategy::Complementary => {
                let direction_id = match complementary_direction {
                    ComplementaryDirection::Regression => 0u64,
                    ComplementaryDirection::Progression => 1u64,
                };
                let key = (iteration as u64)
                    ^ ((goal_index as u64) << 17)
                    ^ ((group_index as u64) << 33)
                    ^ (direction_id << 49);
                mix_seed(key)
            }
        }
    }

    #[cfg(test)]
    fn flaw_kind_for_goal_count(&self, goal_count: usize, iteration: usize) -> FlawKind {
        self.flaw_kind_for_goal_count_and_direction(
            goal_count,
            iteration,
            ComplementaryDirection::Regression,
        )
    }

    fn flaw_kind_for_goal_count_and_direction(
        &self,
        goal_count: usize,
        iteration: usize,
        complementary_direction: ComplementaryDirection,
    ) -> FlawKind {
        match self.config.portfolio_strategy {
            PortfolioStrategy::Standard => return self.config.flaw_kind,
            PortfolioStrategy::Complementary => {
                if self.complementary_uses_target_centered_for_goal_count(
                    goal_count,
                    iteration,
                    complementary_direction,
                ) {
                    return FlawKind::TargetCentered;
                }
                return self.config.flaw_kind;
            }
        }
    }

    fn complementary_uses_target_centered_for_goal_count(
        &self,
        goal_count: usize,
        iteration: usize,
        complementary_direction: ComplementaryDirection,
    ) -> bool {
        // Target-centered (backward) splits place split points at the
        // boundaries of the regressed goal-required interval, which is the
        // natural granularity for the regression-side abstraction.
        !self.uses_full_goal_task(goal_count, iteration)
            && complementary_direction == ComplementaryDirection::Regression
    }

    fn root_groups_for_goal(
        &self,
        source_task: &dyn AbstractNumericTask,
        task: &dyn AbstractNumericTask,
        goal: &ExplicitFact,
    ) -> Vec<RootGroup> {
        let Some(groups) = complete_shape_root_groups(source_task, task, goal) else {
            let fallback = all_goal_relevant_numeric_vars(task, goal);
            if fallback.is_empty() {
                return vec![RootGroup {
                    numeric_var_ids: (0..task.numeric_variables().len()).collect(),
                }];
            }
            return vec![RootGroup {
                numeric_var_ids: fallback,
            }];
        };
        groups
    }

    fn group_blacklisted_variable_ids(
        &self,
        task: &dyn AbstractNumericTask,
        active_root_group: Option<&RootGroup>,
    ) -> (HashSet<usize>, HashSet<usize>) {
        let Some(active_root_group) = active_root_group else {
            return (HashSet::new(), HashSet::new());
        };
        let blacklisted_numeric_var_ids = (0..task.numeric_variables().len())
            .filter(|numeric_var_id| !active_root_group.numeric_var_ids.contains(numeric_var_id))
            .collect();
        let blacklisted_prop_var_ids = task
            .comparison_axioms()
            .iter()
            .filter_map(|comparison_axiom| {
                let roots = comparison_root_vars_for_comparison(task, comparison_axiom);
                (!roots.is_empty()
                    && !roots.iter().all(|numeric_var_id| {
                        active_root_group.numeric_var_ids.contains(numeric_var_id)
                    }))
                .then_some(comparison_axiom.get_affected_var_id())
            })
            .collect();
        (blacklisted_prop_var_ids, blacklisted_numeric_var_ids)
    }

    fn backward_goal_seed_splits(&self, task: &dyn AbstractNumericTask) -> Vec<InitialSeedSplit> {
        let Ok(comparison_index) = ComparisonAxiomIndex::from_task(task) else {
            return goal_seed_splits(task);
        };
        let initial_numeric = initial_numeric_values_with_additive_views(task);
        let deltas = numeric_effect_deltas(task);
        let mut seeds = Vec::new();
        let mut numeric_seed_groups = Vec::new();

        for goal_id in 0..task.get_num_goals() {
            let goal = task.get_goal_fact(goal_id);
            seeds.push(InitialSeedSplit::Propositional {
                var_id: goal.var(),
                value: goal.value(),
            });

            for op in task
                .get_operators()
                .iter()
                .filter(|op| operator_has_unconditional_effect(op, goal))
            {
                for precondition in op.preconditions() {
                    for requirement in target_centered_requirements_for_comparison_fact(
                        task,
                        &comparison_index,
                        precondition,
                        &initial_numeric,
                    ) {
                        let mut requirement_seeds = Vec::new();
                        add_requirement_bounds(&requirement, &mut requirement_seeds);
                        let Some(source_value) =
                            initial_numeric.get(requirement.numeric_var_id).copied()
                        else {
                            numeric_seed_groups.push(requirement_seeds);
                            continue;
                        };
                        add_shells_for_requirement(
                            task,
                            &deltas,
                            source_value,
                            &requirement,
                            &mut requirement_seeds,
                        );
                        numeric_seed_groups.push(requirement_seeds);
                    }
                }
            }
        }

        append_interleaved_numeric_seeds(&mut seeds, numeric_seed_groups);
        seeds
    }
}

fn advance_complementary_schedule(
    goal_count: usize,
    group_count: usize,
    goal_index: &mut usize,
    group_index: &mut usize,
    direction: &mut ComplementaryDirection,
) -> bool {
    assert!(goal_count > 0, "complementary schedule requires a goal");
    assert!(
        *goal_index < goal_count,
        "complementary goal index must be in bounds"
    );
    assert!(
        group_count > 0,
        "complementary schedule requires a root group"
    );
    assert!(
        *group_index < group_count,
        "complementary root-group index must be in bounds"
    );

    if *direction == ComplementaryDirection::Regression {
        *direction = ComplementaryDirection::Progression;
        return false;
    }

    *direction = ComplementaryDirection::Regression;
    *group_index += 1;
    if *group_index < group_count {
        return false;
    }

    *group_index = 0;
    *goal_index += 1;
    if *goal_index < goal_count {
        return false;
    }

    *goal_index = 0;
    true
}

fn operator_has_unconditional_effect(op: &Operator, fact: &ExplicitFact) -> bool {
    op.effects().iter().any(|effect| {
        effect.conditions().is_empty()
            && effect.var_id() == fact.var()
            && effect.value() == fact.value()
    })
}

fn complete_shape_root_groups(
    source_task: &dyn AbstractNumericTask,
    task: &dyn AbstractNumericTask,
    goal: &ExplicitFact,
) -> Option<Vec<RootGroup>> {
    let comparison_index = ComparisonAxiomIndex::from_task(task).ok()?;
    let achievers = task
        .get_operators()
        .iter()
        .filter(|op| operator_has_unconditional_effect(op, goal))
        .collect::<Vec<_>>();
    if achievers.is_empty() {
        return None;
    }

    let mut per_achiever = Vec::with_capacity(achievers.len());
    let mut per_achiever_coarse = Vec::with_capacity(achievers.len());
    for op in achievers {
        let mut by_shape: HashMap<NumericRootGroupKey, HashSet<usize>> = HashMap::new();
        let mut by_coarse_shape: HashMap<Vec<OrderedFloat<f64>>, HashSet<usize>> = HashMap::new();
        for precondition in op.preconditions() {
            for numeric_var_id in
                comparison_root_vars_for_fact(task, &comparison_index, precondition)
            {
                let shape = numeric_root_group_key(source_task, task, numeric_var_id)?;
                by_coarse_shape
                    .entry(shape.coefficient_shape.clone())
                    .or_default()
                    .insert(numeric_var_id);
                by_shape.entry(shape).or_default().insert(numeric_var_id);
            }
        }
        if by_shape.is_empty() {
            return None;
        }
        per_achiever.push(by_shape);
        per_achiever_coarse.push(by_coarse_shape);
    }

    let mut groups = complete_groups_for_keys(&per_achiever_coarse);
    groups.extend(complete_groups_for_keys(&per_achiever));
    let mut seen_groups = HashSet::new();
    groups.retain(|group| {
        let mut ids = group.numeric_var_ids.iter().copied().collect::<Vec<_>>();
        ids.sort_unstable();
        seen_groups.insert(ids)
    });

    if groups.is_empty() {
        return None;
    }
    groups.sort_by_key(|group| {
        let mut ids = group.numeric_var_ids.iter().copied().collect::<Vec<_>>();
        ids.sort_unstable();
        (
            ids.first().copied().unwrap_or(usize::MAX),
            Reverse(ids.len()),
            ids,
        )
    });
    Some(groups)
}

fn complete_groups_for_keys<K>(per_achiever: &[HashMap<K, HashSet<usize>>]) -> Vec<RootGroup>
where
    K: Eq + Hash,
{
    let Some(first) = per_achiever.first() else {
        return Vec::new();
    };
    first
        .keys()
        .filter(|key| {
            per_achiever
                .iter()
                .all(|achiever_shapes| achiever_shapes.contains_key(*key))
        })
        .map(|key| {
            let numeric_var_ids = per_achiever
                .iter()
                .flat_map(|achiever_shapes| {
                    achiever_shapes
                        .get(key)
                        .expect("shape key was checked above")
                        .iter()
                        .copied()
                })
                .collect();
            RootGroup { numeric_var_ids }
        })
        .collect()
}

fn all_goal_relevant_numeric_vars(
    task: &dyn AbstractNumericTask,
    goal: &ExplicitFact,
) -> HashSet<usize> {
    let Ok(comparison_index) = ComparisonAxiomIndex::from_task(task) else {
        return HashSet::new();
    };
    let mut relevant = HashSet::new();
    for op in task
        .get_operators()
        .iter()
        .filter(|op| operator_has_unconditional_effect(op, goal))
    {
        for precondition in op.preconditions() {
            relevant.extend(comparison_root_vars_for_fact(
                task,
                &comparison_index,
                precondition,
            ));
        }
    }
    relevant
}

fn comparison_root_vars_for_fact(
    task: &dyn AbstractNumericTask,
    comparison_index: &ComparisonAxiomIndex,
    fact: &ExplicitFact,
) -> Vec<usize> {
    let Some(tree) = comparison_index.comparison_tree(fact.var()) else {
        return Vec::new();
    };
    comparison_refinement_dimensions(task, tree)
}

fn comparison_root_vars_for_comparison(
    task: &dyn AbstractNumericTask,
    comparison_axiom: &ComparisonAxiom,
) -> Vec<usize> {
    comparison_root_vars_for_numeric_ids(
        task,
        [
            comparison_axiom.get_left_var_id(),
            comparison_axiom.get_right_var_id(),
        ],
    )
}

fn comparison_root_vars_for_numeric_ids(
    task: &dyn AbstractNumericTask,
    numeric_var_ids: [usize; 2],
) -> Vec<usize> {
    let mut roots = numeric_var_ids
        .into_iter()
        .filter(|&numeric_var_id| is_refinable_numeric_dimension(task, numeric_var_id))
        .collect::<Vec<_>>();
    roots.sort_unstable();
    roots.dedup();
    roots
}

fn numeric_root_group_key(
    source_task: &dyn AbstractNumericTask,
    task: &dyn AbstractNumericTask,
    numeric_var_id: usize,
) -> Option<NumericRootGroupKey> {
    let task_var = task.numeric_variables().get(numeric_var_id)?;
    if let Some(shape) = restricted_shape_key(task_var.name()) {
        return Some(NumericRootGroupKey {
            coefficient_shape: shape,
            invariant_terms: Vec::new(),
            constant: OrderedFloat(0.0),
        });
    }
    let expr = source_task
        .numeric_variables()
        .iter()
        .position(|var| var.name() == task_var.name())
        .and_then(|source_var_id| linearize_numeric_var(source_task, source_var_id).ok())
        .or_else(|| linearize_numeric_var(task, numeric_var_id).ok())?;
    let mut coefficients = expr
        .coefficients
        .iter()
        .copied()
        .filter(|coefficient| coefficient.abs() >= 1e-12)
        .map(OrderedFloat)
        .collect::<Vec<_>>();
    if coefficients.is_empty() {
        return None;
    }
    coefficients.sort_unstable();
    let invariant_terms = expr
        .coefficients
        .iter()
        .copied()
        .enumerate()
        .filter(|(dependency, coefficient)| {
            coefficient.abs() >= 1e-12
                && is_operator_invariant_regular_dimension(source_task, *dependency)
        })
        .map(|(dependency, coefficient)| (dependency, OrderedFloat(coefficient)))
        .collect();
    Some(NumericRootGroupKey {
        coefficient_shape: coefficients,
        invariant_terms,
        constant: OrderedFloat(expr.constant),
    })
}

fn restricted_shape_key(name: &str) -> Option<Vec<OrderedFloat<f64>>> {
    let shape = name.strip_prefix("rt-shape:")?.split('|').next()?;
    let coefficients = if shape.is_empty() {
        Vec::new()
    } else {
        shape
            .split(',')
            .map(|coefficient| coefficient.parse::<f64>().ok().map(OrderedFloat))
            .collect::<Option<Vec<_>>>()?
    };
    (!coefficients.is_empty()).then_some(coefficients)
}

fn seed_split_description(seed: &InitialSeedSplit) -> String {
    match seed {
        InitialSeedSplit::Propositional { var_id, value } => format!("p{var_id}={value}"),
        InitialSeedSplit::Numeric {
            numeric_var_id,
            value,
            include_in_lower,
        } => format!(
            "n{numeric_var_id}{}{}",
            if *include_in_lower { "<=" } else { "<" },
            value
        ),
    }
}

fn log_collection_abstraction_debug(
    abstraction_id: usize,
    abstraction: &DomainAbstraction,
    task: &dyn AbstractNumericTask,
) {
    let metadata = &abstraction.metadata;
    let split_numeric = abstraction
        .factory
        .numeric_domain_sizes()
        .iter()
        .enumerate()
        .filter(|(_, size)| **size > 1)
        .count();
    info!(
        "domain abstraction collection debug: id={abstraction_id}, iteration={:?}, strategy={:?}, flaw_kind={:?}, full_goal_task={:?}, max_abs_size={:?}, states={}, prop_split_vars={}, numeric_split_vars={}, abstract_ops={}, seed_splits={}",
        metadata.collection_iteration,
        metadata.portfolio_strategy,
        metadata.flaw_kind,
        metadata.full_goal_task,
        metadata.max_abstraction_size,
        compute_abstraction_size_u128(
            abstraction.factory.domain_sizes(),
            abstraction.factory.numeric_domain_sizes()
        )
        .unwrap_or(u128::MAX),
        abstraction
            .factory
            .domain_sizes()
            .iter()
            .filter(|&&size| size > 1)
            .count(),
        split_numeric,
        abstraction.abstract_operators.len(),
        metadata.initial_seed_splits.join(","),
    );
    log_split_propositional_vars(abstraction, task);
    log_split_numeric_partitions(abstraction, task);
}

fn log_collection_debug_summary(abstractions: &[DomainAbstraction]) {
    info!(
        "domain abstraction collection debug: generated {} abstractions",
        abstractions.len()
    );
    for (id, abstraction) in abstractions.iter().enumerate() {
        info!(
            "domain abstraction collection debug summary: id={id}, iteration={:?}, strategy={:?}, flaw_kind={:?}, states={}, abstract_ops={}, seed_splits={}",
            abstraction.metadata.collection_iteration,
            abstraction.metadata.portfolio_strategy,
            abstraction.metadata.flaw_kind,
            compute_abstraction_size_u128(
                abstraction.factory.domain_sizes(),
                abstraction.factory.numeric_domain_sizes()
            )
            .unwrap_or(u128::MAX),
            abstraction.abstract_operators.len(),
            abstraction.metadata.initial_seed_splits.join(","),
        );
    }
}

fn log_split_propositional_vars(abstraction: &DomainAbstraction, task: &dyn AbstractNumericTask) {
    let mut entries = Vec::new();
    for (var_id, &size) in abstraction
        .factory
        .domain_sizes()
        .iter()
        .enumerate()
        .filter(|(_, size)| **size > 1)
    {
        let name = task.get_variable_name(var_id).unwrap_or("<unknown>");
        entries.push(format!("p{var_id}={name}:size{size}"));
    }
    if !entries.is_empty() {
        info!(
            "domain abstraction collection debug prop vars: {}",
            entries.join(", ")
        );
    }
}

fn log_split_numeric_partitions(abstraction: &DomainAbstraction, task: &dyn AbstractNumericTask) {
    for (numeric_var_id, &size) in abstraction
        .factory
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
        let Some(parts) = abstraction.factory.partitions().partitions(numeric_var_id) else {
            continue;
        };
        let preview = partition_preview(parts);
        info!(
            "domain abstraction collection debug partitions: n{numeric_var_id}={name}, size={size}, {preview}"
        );
    }
}

fn partition_preview(parts: &[super::comparison_expression::Interval]) -> String {
    let mut entries = Vec::new();
    for interval in parts.iter().take(3) {
        entries.push(format!("{interval:?}"));
    }
    if parts.len() > 6 {
        entries.push("...".to_string());
    }
    let suffix_start = if parts.len() > 6 { parts.len() - 3 } else { 3 };
    for interval in parts.iter().skip(suffix_start) {
        entries.push(format!("{interval:?}"));
    }
    entries.join(" ")
}

#[derive(Debug, Clone, PartialEq)]
struct NumericRequirement {
    numeric_var_id: usize,
    lower: Option<f64>,
    upper: Option<f64>,
}

impl NumericRequirement {
    fn from_interval(
        numeric_var_id: usize,
        interval: super::comparison_expression::Interval,
    ) -> Self {
        Self {
            numeric_var_id,
            lower: interval.lower.is_finite().then_some(interval.lower),
            upper: interval.upper.is_finite().then_some(interval.upper),
        }
    }
}

fn target_centered_requirements_for_comparison_fact(
    task: &dyn AbstractNumericTask,
    comparison_index: &ComparisonAxiomIndex,
    fact: &ExplicitFact,
    numeric_state: &[f64],
) -> Vec<NumericRequirement> {
    if let Some((numeric_var_id, interval)) =
        numeric_requirement_for_comparison_fact(task, comparison_index, fact)
    {
        if is_refinable_numeric_dimension(task, numeric_var_id) {
            return vec![NumericRequirement::from_interval(numeric_var_id, interval)];
        }
    }

    let Some(tree) = comparison_index.comparison_tree(fact.var()) else {
        return Vec::new();
    };
    let Ok(left) = linearize_numeric_var(task, tree.left_numeric_var_id) else {
        return Vec::new();
    };
    let Ok(right) = linearize_numeric_var(task, tree.right_numeric_var_id) else {
        return Vec::new();
    };
    let Some(required_op) = required_comparison_op(tree.op, fact.value()) else {
        return Vec::new();
    };

    let expression = left.subtract(&right);
    let mut requirements = Vec::new();
    for (numeric_var_id, &coefficient) in expression.coefficients.iter().enumerate() {
        if coefficient.abs() < 1e-12 {
            continue;
        }
        if task
            .numeric_variables()
            .get(numeric_var_id)
            .is_none_or(|variable| variable.get_type() != &NumericType::Regular)
        {
            continue;
        }
        let mut fixed_constant = expression.constant;
        let mut has_all_values = true;
        for (var, other_coefficient) in expression.coefficients.iter().enumerate() {
            if var == numeric_var_id {
                continue;
            }
            let Some(value) = numeric_state.get(var).copied() else {
                has_all_values = false;
                break;
            };
            fixed_constant += other_coefficient * value;
        }
        if !has_all_values {
            continue;
        }
        let Some(interval) = single_var_interval(coefficient, fixed_constant, required_op) else {
            continue;
        };
        requirements.push(NumericRequirement::from_interval(numeric_var_id, interval));
    }

    merge_numeric_requirements(&mut requirements);
    requirements
}

fn required_comparison_op(op: CompOp, prop_value: usize) -> Option<CompOp> {
    match prop_value {
        0 => Some(op),
        1 => Some(match op {
            CompOp::Lt => CompOp::Ge,
            CompOp::Le => CompOp::Gt,
            CompOp::Gt => CompOp::Le,
            CompOp::Ge => CompOp::Lt,
            CompOp::Eq => CompOp::Ne,
            CompOp::Ne => CompOp::Eq,
        }),
        _ => None,
    }
}

fn single_var_interval(coefficient: f64, constant: f64, op: CompOp) -> Option<Interval> {
    if coefficient.abs() < 1e-12 || op == CompOp::Ne {
        return None;
    }
    let threshold = -constant / coefficient;
    if !threshold.is_finite() {
        return None;
    }
    Some(match (op, coefficient.is_sign_positive()) {
        (CompOp::Lt, true) | (CompOp::Gt, false) => {
            Interval::new(f64::NEG_INFINITY, threshold, false, false)
        }
        (CompOp::Le, true) | (CompOp::Ge, false) => {
            Interval::new(f64::NEG_INFINITY, threshold, false, true)
        }
        (CompOp::Gt, true) | (CompOp::Lt, false) => {
            Interval::new(threshold, f64::INFINITY, false, false)
        }
        (CompOp::Ge, true) | (CompOp::Le, false) => {
            Interval::new(threshold, f64::INFINITY, true, false)
        }
        (CompOp::Eq, _) => Interval::singleton(threshold),
        (CompOp::Ne, _) => return None,
    })
}

fn merge_numeric_requirements(requirements: &mut Vec<NumericRequirement>) {
    requirements.sort_by_key(|requirement| requirement.numeric_var_id);
    let mut merged: Vec<NumericRequirement> = Vec::new();
    for requirement in requirements.drain(..) {
        if let Some(last) = merged.last_mut()
            && last.numeric_var_id == requirement.numeric_var_id
        {
            last.lower = match (last.lower, requirement.lower) {
                (Some(left), Some(right)) => Some(left.max(right)),
                (Some(left), None) => Some(left),
                (None, Some(right)) => Some(right),
                (None, None) => None,
            };
            last.upper = match (last.upper, requirement.upper) {
                (Some(left), Some(right)) => Some(left.min(right)),
                (Some(left), None) => Some(left),
                (None, Some(right)) => Some(right),
                (None, None) => None,
            };
            continue;
        }
        merged.push(requirement);
    }
    *requirements = merged;
}

fn approximate_distance_from_initial(
    requirements: &[NumericRequirement],
    initial_numeric: &[f64],
) -> f64 {
    requirements
        .iter()
        .map(|requirement| {
            let Some(&initial_value) = initial_numeric.get(requirement.numeric_var_id) else {
                return f64::INFINITY;
            };
            if !initial_value.is_finite() {
                return f64::INFINITY;
            }
            match (requirement.lower, requirement.upper) {
                (Some(lower), _) if initial_value < lower => lower - initial_value,
                (_, Some(upper)) if initial_value > upper => initial_value - upper,
                _ => 0.0,
            }
        })
        .sum()
}

fn compare_goals_for_collection(
    task: &dyn AbstractNumericTask,
    left: &ExplicitFact,
    right: &ExplicitFact,
) -> Ordering {
    let left_distance = estimate_goal_distance_from_initial(task, left);
    let right_distance = estimate_goal_distance_from_initial(task, right);
    right_distance
        .total_cmp(&left_distance)
        .then_with(|| left.var().cmp(&right.var()))
        .then_with(|| left.value().cmp(&right.value()))
}

fn estimate_goal_distance_from_initial(task: &dyn AbstractNumericTask, goal: &ExplicitFact) -> f64 {
    let Ok(comparison_index) = ComparisonAxiomIndex::from_task(task) else {
        return 0.0;
    };
    let initial_numeric = initial_numeric_values_with_additive_views(task);
    let mut best_direct = 0.0f64;
    let direct_requirements = target_centered_requirements_for_comparison_fact(
        task,
        &comparison_index,
        goal,
        &initial_numeric,
    );
    if !direct_requirements.is_empty() {
        best_direct = approximate_distance_from_initial(&direct_requirements, &initial_numeric);
    }

    let mut best_achiever = 0.0f64;
    for op in task
        .get_operators()
        .iter()
        .filter(|op| operator_has_unconditional_effect(op, goal))
    {
        let mut requirements = Vec::new();
        for precondition in op.preconditions() {
            requirements.extend(target_centered_requirements_for_comparison_fact(
                task,
                &comparison_index,
                precondition,
                &initial_numeric,
            ));
        }
        merge_numeric_requirements(&mut requirements);
        best_achiever = best_achiever.max(approximate_distance_from_initial(
            &requirements,
            &initial_numeric,
        ));
    }

    best_direct.max(best_achiever)
}

fn add_requirement_bounds(requirement: &NumericRequirement, seeds: &mut Vec<InitialSeedSplit>) {
    if let Some(lower) = requirement.lower
        && lower.is_finite()
    {
        seeds.push(InitialSeedSplit::Numeric {
            numeric_var_id: requirement.numeric_var_id,
            value: lower,
            include_in_lower: false,
        });
    }
    if let Some(upper) = requirement.upper
        && upper.is_finite()
    {
        seeds.push(InitialSeedSplit::Numeric {
            numeric_var_id: requirement.numeric_var_id,
            value: upper,
            include_in_lower: true,
        });
    }
}

fn add_shells_for_requirement(
    task: &dyn AbstractNumericTask,
    deltas: &HashMap<usize, Vec<f64>>,
    source_value: f64,
    target: &NumericRequirement,
    seeds: &mut Vec<InitialSeedSplit>,
) {
    let Some(var_deltas) = deltas.get(&target.numeric_var_id) else {
        return;
    };
    let max_shells = max_shell_splits_for_task(task);
    if let Some(lower) = target.lower
        && source_value < lower
        && let Some(step) = smallest_positive_delta(var_deltas)
    {
        add_monotone_shells(
            target.numeric_var_id,
            source_value,
            lower,
            -step,
            false,
            max_shells,
            seeds,
        );
    }
    if let Some(upper) = target.upper
        && source_value > upper
        && let Some(step) = largest_negative_delta(var_deltas)
    {
        add_monotone_shells(
            target.numeric_var_id,
            source_value,
            upper,
            -step,
            true,
            max_shells,
            seeds,
        );
    }
}

fn max_shell_splits_for_task(task: &dyn AbstractNumericTask) -> usize {
    max_shell_splits_for_var_count(task.numeric_variables().len())
}

fn max_shell_splits_for_var_count(var_count: usize) -> usize {
    (256usize).max(var_count * 4)
}

fn smallest_positive_delta(deltas: &[f64]) -> Option<f64> {
    deltas
        .iter()
        .copied()
        .filter(|delta| *delta > 1e-12)
        .min_by_key(|delta| OrderedFloat(*delta))
}

fn largest_negative_delta(deltas: &[f64]) -> Option<f64> {
    deltas
        .iter()
        .copied()
        .filter(|delta| *delta < -1e-12)
        .max_by_key(|delta| OrderedFloat(*delta))
}

fn add_monotone_shells(
    numeric_var_id: usize,
    source_value: f64,
    target_value: f64,
    reverse_step: f64,
    include_in_lower: bool,
    max_shells: usize,
    seeds: &mut Vec<InitialSeedSplit>,
) {
    let mut value = target_value;
    for _ in 0..max_shells {
        if !value.is_finite() {
            break;
        }
        if reverse_step < 0.0 && value <= source_value {
            break;
        }
        if reverse_step > 0.0 && value >= source_value {
            break;
        }
        seeds.push(InitialSeedSplit::Numeric {
            numeric_var_id,
            value,
            include_in_lower,
        });
        value += reverse_step;
    }
}

fn append_interleaved_numeric_seeds(
    seeds: &mut Vec<InitialSeedSplit>,
    mut numeric_seed_groups: Vec<Vec<InitialSeedSplit>>,
) {
    numeric_seed_groups.sort_by(|left, right| seed_group_key(left).cmp(&seed_group_key(right)));
    let mut seen = seeds.iter().map(seed_identity).collect::<HashSet<_>>();
    let max_group_len = numeric_seed_groups.iter().map(Vec::len).max().unwrap_or(0);
    for layer in 0..max_group_len {
        for group in &numeric_seed_groups {
            if let Some(seed) = group.get(layer)
                && seen.insert(seed_identity(seed))
            {
                seeds.push(seed.clone());
            }
        }
    }
}

fn seed_identity(seed: &InitialSeedSplit) -> SeedIdentity {
    match seed {
        InitialSeedSplit::Propositional { var_id, value } => SeedIdentity::Propositional {
            var_id: *var_id,
            value: *value,
        },
        InitialSeedSplit::Numeric {
            numeric_var_id,
            value,
            include_in_lower,
        } => SeedIdentity::Numeric {
            numeric_var_id: *numeric_var_id,
            value_bits: value.to_bits(),
            include_in_lower: *include_in_lower,
        },
    }
}

fn seed_group_key(group: &[InitialSeedSplit]) -> (usize, OrderedFloat<f64>, bool) {
    group
        .iter()
        .find_map(|seed| match seed {
            InitialSeedSplit::Numeric {
                numeric_var_id,
                value,
                include_in_lower,
            } => Some((*numeric_var_id, OrderedFloat(*value), *include_in_lower)),
            InitialSeedSplit::Propositional { .. } => None,
        })
        .unwrap_or((usize::MAX, OrderedFloat(0.0), false))
}

fn goal_seed_splits(task: &dyn AbstractNumericTask) -> Vec<InitialSeedSplit> {
    let mut goal_axiom_map: HashMap<usize, Vec<ExplicitFact>> = HashMap::new();
    for axiom in task.axioms() {
        if !axiom.conditions().is_empty() {
            goal_axiom_map.insert(axiom.var_id(), axiom.conditions().to_vec());
        }
    }

    let mut seeds = Vec::new();
    for goal_id in 0..task.get_num_goals() {
        let goal = task.get_goal_fact(goal_id);
        if let Some(conditions) = goal_axiom_map.get(&goal.var()) {
            seeds.extend(
                conditions
                    .iter()
                    .map(|fact| InitialSeedSplit::Propositional {
                        var_id: fact.var(),
                        value: fact.value(),
                    }),
            );
        } else {
            seeds.push(InitialSeedSplit::Propositional {
                var_id: goal.var(),
                value: goal.value(),
            });
        }
    }
    seeds.sort_by_key(|seed| match seed {
        InitialSeedSplit::Propositional { var_id, value } => (0, *var_id, *value),
        InitialSeedSplit::Numeric {
            numeric_var_id,
            value,
            ..
        } => (1, *numeric_var_id, value.to_bits() as usize),
    });
    seeds.dedup();
    seeds
}

fn collect_logic_axiom_effect_vars(task: &dyn AbstractNumericTask) -> HashSet<usize> {
    task.axioms().iter().map(|axiom| axiom.var_id()).collect()
}

fn collect_comparison_axiom_var_ids(task: &dyn AbstractNumericTask) -> HashSet<usize> {
    task.comparison_axioms()
        .iter()
        .map(|axiom| axiom.get_affected_var_id())
        .collect()
}

fn collect_goal_related_propositional_vars(task: &dyn AbstractNumericTask) -> HashSet<usize> {
    let mut goal_axiom_map: HashMap<usize, Vec<usize>> = HashMap::new();
    for axiom in task.axioms() {
        if axiom.conditions().is_empty() {
            continue;
        }
        let affected_var_id = axiom.var_id();
        let condition_var_ids = axiom
            .conditions()
            .iter()
            .map(|condition| condition.var())
            .collect::<Vec<_>>();
        goal_axiom_map.insert(affected_var_id, condition_var_ids);
    }

    let logic_axiom_effect_vars = collect_logic_axiom_effect_vars(task);
    let mut goal_related: HashSet<usize> = HashSet::new();
    for goal_id in 0..task.get_num_goals() {
        let goal_var_id = task.get_goal_fact(goal_id).var();
        if let Some(preconditions) = goal_axiom_map.get(&goal_var_id) {
            goal_related.extend(preconditions.iter().copied());
        } else if !logic_axiom_effect_vars.contains(&goal_var_id) {
            goal_related.insert(goal_var_id);
        }
    }

    goal_related
}

fn collect_init_split_candidate_var_ids(
    task: &dyn AbstractNumericTask,
    subset: VariableSubset,
) -> Vec<usize> {
    let goal_related = collect_goal_related_propositional_vars(task);
    let logic_axiom_effect_vars = collect_logic_axiom_effect_vars(task);
    let comparison_axiom_vars = collect_comparison_axiom_var_ids(task);

    let mut candidates: Vec<usize> = match subset {
        VariableSubset::Goals => goal_related.iter().copied().collect(),
        VariableSubset::NonGoals => (0..task.variables().len())
            .filter(|var_id| {
                !goal_related.contains(var_id)
                    && !logic_axiom_effect_vars.contains(var_id)
                    && !comparison_axiom_vars.contains(var_id)
            })
            .collect(),
        VariableSubset::All => (0..task.variables().len())
            .filter(|var_id| {
                !logic_axiom_effect_vars.contains(var_id)
                    && (!comparison_axiom_vars.contains(var_id) || goal_related.contains(var_id))
            })
            .collect(),
    };
    if matches!(subset, VariableSubset::NonGoals | VariableSubset::All) {
        let encoded_numeric_offset = task.variables().len();
        candidates.extend(
            task.numeric_variables()
                .iter()
                .enumerate()
                .filter(|(_, variable)| variable.get_type() == &NumericType::Regular)
                .map(|(numeric_var_id, _)| encoded_numeric_offset + numeric_var_id),
        );
    }
    candidates.sort_unstable();
    candidates.dedup();
    candidates
}

fn collect_blacklist_candidate_var_ids(
    task: &dyn AbstractNumericTask,
    subset: VariableSubset,
) -> Vec<usize> {
    let mut candidates = collect_init_split_candidate_var_ids(task, subset);
    if matches!(subset, VariableSubset::NonGoals | VariableSubset::All) {
        let encoded_numeric_offset = task.variables().len();
        candidates.extend(
            task.numeric_variables()
                .iter()
                .enumerate()
                .filter(|(_, variable)| variable.get_type() == &NumericType::Regular)
                .map(|(numeric_var_id, _)| encoded_numeric_offset + numeric_var_id),
        );
    }
    candidates.sort_unstable();
    candidates.dedup();
    candidates
}

fn split_blacklisted_variables(
    task: &dyn AbstractNumericTask,
    encoded_var_ids: HashSet<usize>,
) -> (HashSet<usize>, HashSet<usize>) {
    let num_prop_vars = task.variables().len();
    let mut blacklisted_prop_var_ids = HashSet::new();
    let mut blacklisted_numeric_var_ids = HashSet::new();

    for encoded_var_id in encoded_var_ids {
        if encoded_var_id < num_prop_vars {
            blacklisted_prop_var_ids.insert(encoded_var_id);
        } else {
            let numeric_var_id = encoded_var_id - num_prop_vars;
            if numeric_var_id < task.numeric_variables().len() {
                blacklisted_numeric_var_ids.insert(numeric_var_id);
            }
        }
    }

    (blacklisted_prop_var_ids, blacklisted_numeric_var_ids)
}

fn sample_blacklisted_variables<R: rand::Rng + ?Sized>(
    candidates: &[usize],
    rng: &mut R,
) -> HashSet<usize> {
    if candidates.is_empty() {
        return HashSet::new();
    }

    let blacklist_size = rng.gen_range(1..=candidates.len());
    let mut shuffled = candidates.to_vec();
    shuffled.shuffle(rng);
    shuffled.into_iter().take(blacklist_size).collect()
}

fn select_single_init_split_var(candidate_var_ids: &[usize], iteration: usize) -> Option<usize> {
    if candidate_var_ids.is_empty() {
        return None;
    }
    let index = iteration % candidate_var_ids.len();
    candidate_var_ids.get(index).copied()
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct IntervalFingerprint {
    lower: OrderedFloat<f64>,
    upper: OrderedFloat<f64>,
    lower_closed: bool,
    upper_closed: bool,
}

impl IntervalFingerprint {
    fn from_interval(interval: super::comparison_expression::Interval) -> Self {
        Self {
            lower: OrderedFloat(interval.lower),
            upper: OrderedFloat(interval.upper),
            lower_closed: interval.lower_closed,
            upper_closed: interval.upper_closed,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct AbstractionKey {
    domain_mapping: Vec<Vec<usize>>,
    numeric_fingerprint: Vec<Vec<IntervalFingerprint>>,
}

impl AbstractionKey {
    fn from_abstraction(abstraction: &DomainAbstraction) -> Self {
        let factory = &abstraction.factory;
        let numeric_fingerprint = (0..factory.numeric_domain_sizes().len())
            .map(|numeric_var_id| {
                factory
                    .partitions()
                    .partitions(numeric_var_id)
                    .unwrap_or(&[])
                    .iter()
                    .copied()
                    .map(IntervalFingerprint::from_interval)
                    .collect::<Vec<_>>()
            })
            .collect();

        Self {
            domain_mapping: factory.domain_mapping().clone(),
            numeric_fingerprint,
        }
    }
}
