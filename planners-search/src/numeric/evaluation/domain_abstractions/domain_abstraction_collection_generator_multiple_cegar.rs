#[cfg(test)]
mod tests;

use std::cell::{Ref, RefMut};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use ordered_float::OrderedFloat;
use planners_sas::numeric::axioms::ComparisonOperator;
use planners_sas::numeric::axioms::{AssignmentAxiom, ComparisonAxiom, PropositionalAxiom};
use planners_sas::numeric::numeric_task::{
    AbstractNumericTask, AssignmentOperation, ExplicitFact, ExplicitVariable, Metric, NumericType,
    NumericVariable, Operator,
};
use planners_sas::numeric::utils::linear_effects::linearize_numeric_var;
use rand::seq::SliceRandom;
use rand::{RngCore, SeedableRng, rngs::SmallRng};
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::numeric::evaluation::domain_abstractions::cegar::FlawKind;

use super::cegar::CegarConfig;
use super::cegar::InitialSeedSplit;
pub use super::cegar::flaw_search::flaw_selection::{FlawTreatmentVariants, InitSplitMethod};
use super::cegar::flaw_search::numeric_requirement_for_comparison_fact;
use super::comparison_expression::{CompOp, Interval};
use super::domain_abstraction::ComparisonAxiomIndex;
use super::domain_abstraction_generator::{
    DomainAbstraction, DomainAbstractionGenerator, DomainAbstractionMetadata,
    prepare_domain_abstraction_task,
};
use super::utils::compute_abstraction_size_u128;

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

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PortfolioStrategy {
    Standard,
    ViewDiverse,
    Complementary,
    RegionLandmarks,
    BackwardGoals,
    ForwardBackwardGoals,
    RouteShells,
}

impl fmt::Display for PortfolioStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Standard => write!(f, "standard"),
            Self::ViewDiverse => write!(f, "view_diverse"),
            Self::Complementary => write!(f, "complementary"),
            Self::RegionLandmarks => write!(f, "region_landmarks"),
            Self::BackwardGoals => write!(f, "backward_goals"),
            Self::ForwardBackwardGoals => write!(f, "forward_backward_goals"),
            Self::RouteShells => write!(f, "route_shells"),
        }
    }
}

impl PortfolioStrategy {
    fn uses_ranked_goals(self) -> bool {
        matches!(
            self,
            Self::Complementary
                | Self::RegionLandmarks
                | Self::BackwardGoals
                | Self::ForwardBackwardGoals
                | Self::RouteShells
        )
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
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
    pub transform_linear_task: bool,
    pub portfolio_strategy: PortfolioStrategy,
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
            transform_linear_task: false,
            portfolio_strategy: PortfolioStrategy::Standard,
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
                "transform_linear_task={}, ",
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
            self.transform_linear_task,
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
            transform_linear_task: self.config.transform_linear_task,
            initial_seed_splits,
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
        let region_abstraction_budget =
            if self.config.portfolio_strategy == PortfolioStrategy::RegionLandmarks {
                Some((self.config.max_collection_size / 3).max(1))
            } else {
                None
            };

        loop {
            if self.config.portfolio_strategy == PortfolioStrategy::BackwardGoals
                && !goals.is_empty()
                && iteration > goals.len()
            {
                break;
            }

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
            let mut remaining_abstraction_size =
                remaining_collection_size.min(self.config.max_abstraction_size);
            if let Some(region_budget) = region_abstraction_budget {
                remaining_abstraction_size = remaining_abstraction_size.min(region_budget);
            }

            if remaining_abstraction_size == 0 || remaining_generation_time <= 0.0 {
                break;
            }

            if self.config.portfolio_strategy == PortfolioStrategy::RouteShells && !goals.is_empty()
            {
                goal_index = self.route_shell_goal_index(goals.len(), iteration);
            }
            let full_goal_task = self.full_goal_task_for_iteration(iteration);
            let goal_task = if full_goal_task {
                None
            } else {
                goals
                    .get(goal_index)
                    .map(|goal| SingleGoalTask::new(task, goal.clone()))
            };
            let abstraction_task: &dyn AbstractNumericTask = goal_task
                .as_ref()
                .map(|single_goal_task| single_goal_task as &dyn AbstractNumericTask)
                .unwrap_or(task);
            let prepared_task = prepare_domain_abstraction_task(
                abstraction_task,
                self.config.transform_linear_task,
            )
            .context("failed to prepare task for domain-abstraction collection")?;
            let generation_task = prepared_task.task_for(abstraction_task);
            let blacklisted_var_ids = if blacklisting {
                sample_blacklisted_variables(&blacklist_candidates, &mut rng)
            } else {
                HashSet::new()
            };
            let seed_iteration = if self.config.portfolio_strategy
                == PortfolioStrategy::Complementary
                && !full_goal_task
                && !goals.is_empty()
            {
                ((iteration - 2) / goals.len()) + 1
            } else if self.config.portfolio_strategy == PortfolioStrategy::RouteShells
                && !full_goal_task
                && !goals.is_empty()
            {
                self.route_shell_index(goals.len(), iteration) + 1
            } else {
                iteration
            };
            let initial_seed_splits = if full_goal_task {
                Vec::new()
            } else {
                self.initial_seed_splits_for_goal_count(
                    generation_task,
                    seed_iteration,
                    goals.len(),
                )
            };
            let (blacklisted_prop_var_ids, blacklisted_numeric_var_ids) =
                split_blacklisted_variables(generation_task, blacklisted_var_ids);
            let init_split_var_ids = if full_goal_task
                && self.config.portfolio_strategy == PortfolioStrategy::Complementary
            {
                None
            } else if initial_seed_splits.is_empty()
                || self.config.portfolio_strategy == PortfolioStrategy::Complementary
            {
                self.initial_split_var_ids(generation_task, iteration)
            } else {
                None
            };
            let flaw_kind = self.flaw_kind_for_goal_count(goals.len(), iteration);
            let seed_descriptions = initial_seed_splits
                .iter()
                .map(seed_split_description)
                .collect::<Vec<_>>();
            let cegar_config = self.build_cegar_config(
                remaining_abstraction_size,
                remaining_generation_time,
                init_split_var_ids,
                initial_seed_splits,
                blacklisted_prop_var_ids,
                blacklisted_numeric_var_ids,
                Some(rng.next_u64()),
                flaw_kind,
            );
            let generator = DomainAbstractionGenerator::new(cegar_config)
                .context("failed to construct single-abstraction CEGAR generator")?;
            let mut abstraction = generator
                .generate_prepared(abstraction_task, &prepared_task)
                .with_context(|| {
                    format!("failed to generate abstraction for collection iteration {iteration}")
                })?;
            abstraction.metadata = DomainAbstractionMetadata {
                collection_iteration: Some(iteration),
                portfolio_strategy: Some(self.config.portfolio_strategy.to_string()),
                flaw_kind: Some(flaw_kind.to_string()),
                full_goal_task: Some(full_goal_task),
                initial_seed_splits: seed_descriptions,
                max_abstraction_size: Some(remaining_abstraction_size),
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
                            generation_task,
                        );
                    }
                }
                info!(
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
            if remaining_collection_size == 0
                || (self.config.total_max_time.is_finite() && elapsed >= self.config.total_max_time)
                || (stagnated && (!self.config.enable_blacklist_on_stagnation || blacklisting))
            {
                break;
            }
            if stagnated && self.config.enable_blacklist_on_stagnation {
                blacklisting = true;
                time_point_of_last_new_abstraction = elapsed;
            }

            let completed_full_goal_task = full_goal_task;
            iteration += 1;
            if self.config.portfolio_strategy != PortfolioStrategy::RouteShells
                && !completed_full_goal_task
                && !goals.is_empty()
            {
                goal_index = (goal_index + 1) % goals.len();
                let _ = &goals[goal_index];
            }
        }

        if generated_abstractions.is_empty() {
            bail!("multi_domain_abstractions(...) failed to generate any abstractions")
        }
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

    #[cfg(test)]
    fn initial_seed_splits(
        &self,
        task: &dyn AbstractNumericTask,
        iteration: usize,
    ) -> Vec<InitialSeedSplit> {
        self.initial_seed_splits_for_goal_count(task, iteration, task.get_num_goals())
    }

    fn initial_seed_splits_for_goal_count(
        &self,
        task: &dyn AbstractNumericTask,
        iteration: usize,
        goal_count: usize,
    ) -> Vec<InitialSeedSplit> {
        match self.config.portfolio_strategy {
            PortfolioStrategy::Standard => return Vec::new(),
            PortfolioStrategy::ViewDiverse => {}
            PortfolioStrategy::Complementary => {
                return self.complementary_seed_splits(task, iteration);
            }
            PortfolioStrategy::RegionLandmarks => {
                return self.region_landmark_seed_splits(task, iteration);
            }
            PortfolioStrategy::BackwardGoals => {
                return self.backward_goal_seed_splits(task);
            }
            PortfolioStrategy::ForwardBackwardGoals => {
                if self.forward_backward_uses_target_centered_for_goal_count(goal_count, iteration)
                {
                    return self.backward_goal_seed_splits(task);
                }
                return Vec::new();
            }
            PortfolioStrategy::RouteShells => {
                return self.route_shell_seed_splits(task, iteration.saturating_sub(1));
            }
        }

        let mut candidates = collect_false_view_candidates(task);
        if candidates.is_empty() {
            return Vec::new();
        }
        candidates.sort_by(|left, right| {
            right
                .deficit
                .total_cmp(&left.deficit)
                .then_with(|| left.numeric_var_id.cmp(&right.numeric_var_id))
                .then_with(|| left.comparison_var_id.cmp(&right.comparison_var_id))
        });
        let candidate_iteration =
            if self.config.portfolio_strategy == PortfolioStrategy::ViewDiverse {
                (iteration - 1) / 2
            } else {
                iteration - 1
            };
        let selected = &candidates[candidate_iteration % candidates.len()];
        let mut seeds = Vec::new();
        seeds.push(InitialSeedSplit::Numeric {
            numeric_var_id: selected.numeric_var_id,
            value: selected.initial_value,
            include_in_lower: selected.include_in_lower,
        });
        seeds.push(InitialSeedSplit::Propositional {
            var_id: selected.comparison_var_id,
            value: COMPARISON_TRUE_VALUE,
        });
        seeds.extend(goal_seed_splits(task));
        seeds
    }

    #[cfg(test)]
    fn flaw_kind_for_iteration(&self, iteration: usize) -> FlawKind {
        self.flaw_kind_for_goal_count(0, iteration)
    }

    fn flaw_kind_for_goal_count(&self, goal_count: usize, iteration: usize) -> FlawKind {
        match self.config.portfolio_strategy {
            PortfolioStrategy::Standard => return self.config.flaw_kind,
            PortfolioStrategy::Complementary => {
                if self.full_goal_task_for_iteration(iteration) {
                    return self.config.flaw_kind;
                }
                return FlawKind::SequenceRegression;
            }
            PortfolioStrategy::RegionLandmarks => return FlawKind::SequenceBidirectional,
            PortfolioStrategy::BackwardGoals => return FlawKind::TargetCentered,
            PortfolioStrategy::ForwardBackwardGoals => {
                if self.forward_backward_uses_target_centered_for_goal_count(goal_count, iteration)
                {
                    return FlawKind::TargetCentered;
                }
                return self.config.flaw_kind;
            }
            PortfolioStrategy::RouteShells => return FlawKind::TargetCentered,
            PortfolioStrategy::ViewDiverse => {}
        }
        if iteration % 2 == 1 {
            FlawKind::SequenceProgression
        } else {
            FlawKind::SequenceRegression
        }
    }

    fn full_goal_task_for_iteration(&self, iteration: usize) -> bool {
        match self.config.portfolio_strategy {
            PortfolioStrategy::Complementary => iteration == 1,
            PortfolioStrategy::RegionLandmarks => true,
            PortfolioStrategy::Standard
            | PortfolioStrategy::ViewDiverse
            | PortfolioStrategy::BackwardGoals
            | PortfolioStrategy::ForwardBackwardGoals
            | PortfolioStrategy::RouteShells => false,
        }
    }

    fn route_shell_goal_index(&self, goal_count: usize, iteration: usize) -> usize {
        debug_assert!(goal_count > 0);
        let zero_based = iteration - 1;
        if goal_count >= ROUTE_SHELL_MANY_GOAL_THRESHOLD {
            (zero_based / ROUTE_SHELLS_PER_GOAL_PASS) % goal_count
        } else {
            zero_based % goal_count
        }
    }

    fn route_shell_index(&self, goal_count: usize, iteration: usize) -> usize {
        debug_assert!(goal_count > 0);
        let zero_based = iteration - 1;
        if goal_count >= ROUTE_SHELL_MANY_GOAL_THRESHOLD {
            zero_based % ROUTE_SHELLS_PER_GOAL_PASS
        } else {
            zero_based / goal_count
        }
    }

    fn forward_backward_uses_target_centered_for_goal_count(
        &self,
        goal_count: usize,
        iteration: usize,
    ) -> bool {
        let many_goal_task = goal_count >= 6;
        if many_goal_task {
            iteration % 3 != 1
        } else {
            iteration % 3 == 0
        }
    }

    fn complementary_seed_splits(
        &self,
        task: &dyn AbstractNumericTask,
        iteration: usize,
    ) -> Vec<InitialSeedSplit> {
        let mut seeds = complementary_propositional_achiever_seed_splits(task, iteration);
        let mut groups = collect_complementary_view_groups(task);
        if groups.is_empty() {
            dedup_seed_splits_preserve_order(&mut seeds);
            return seeds;
        }
        groups.sort_by(|left, right| {
            right
                .deficit
                .total_cmp(&left.deficit)
                .then_with(|| left.first_numeric_var_id.cmp(&right.first_numeric_var_id))
                .then_with(|| {
                    left.first_comparison_var_id
                        .cmp(&right.first_comparison_var_id)
                })
        });

        let selected = &groups[(iteration - 1) % groups.len()];
        for candidate in &selected.candidates {
            seeds.push(InitialSeedSplit::Numeric {
                numeric_var_id: candidate.numeric_var_id,
                value: candidate.initial_value,
                include_in_lower: candidate.include_in_lower,
            });
            seeds.push(InitialSeedSplit::Propositional {
                var_id: candidate.comparison_var_id,
                value: COMPARISON_TRUE_VALUE,
            });
            seeds.extend(complementary_route_seed_splits(task, candidate));
        }
        dedup_seed_splits_preserve_order(&mut seeds);
        seeds
    }

    fn region_landmark_seed_splits(
        &self,
        task: &dyn AbstractNumericTask,
        iteration: usize,
    ) -> Vec<InitialSeedSplit> {
        let anchors = collect_region_landmark_anchors(task);
        if anchors.is_empty() {
            return Vec::new();
        }

        let segment_id = (iteration - 1) % (anchors.len() + 1);
        let mut seeds = Vec::new();
        seeds.extend(goal_seed_splits(task));

        let initial_numeric = task.get_initial_numeric_state_values();
        let (source_requirements, target_requirements): (
            Vec<NumericRequirement>,
            Vec<NumericRequirement>,
        ) = if segment_id == 0 {
            let initial_requirements = anchors[0]
                .requirements
                .iter()
                .filter_map(|requirement| {
                    initial_numeric
                        .get(requirement.numeric_var_id)
                        .copied()
                        .map(|value| {
                            NumericRequirement::singleton(requirement.numeric_var_id, value)
                        })
                })
                .collect();
            (initial_requirements, anchors[0].requirements.clone())
        } else if segment_id < anchors.len() {
            (
                anchors[segment_id - 1].requirements.clone(),
                anchors[segment_id].requirements.clone(),
            )
        } else {
            let last = anchors
                .last()
                .expect("region landmark anchors must be non-empty");
            (last.requirements.clone(), last.requirements.clone())
        };

        if segment_id > 0 {
            seeds.extend(anchors[segment_id - 1].facts.iter().cloned().map(|fact| {
                InitialSeedSplit::Propositional {
                    var_id: fact.var,
                    value: fact.value,
                }
            }));
        }
        if let Some(anchor) = anchors.get(segment_id.min(anchors.len() - 1)) {
            seeds.extend(anchor.facts.iter().cloned().map(|fact| {
                InitialSeedSplit::Propositional {
                    var_id: fact.var,
                    value: fact.value,
                }
            }));
        }

        seeds.extend(shell_seed_splits(
            task,
            &source_requirements,
            &target_requirements,
        ));
        sort_and_dedup_seed_splits(&mut seeds);
        seeds
    }

    fn backward_goal_seed_splits(&self, task: &dyn AbstractNumericTask) -> Vec<InitialSeedSplit> {
        let Ok(comparison_index) = ComparisonAxiomIndex::from_task(task) else {
            return goal_seed_splits(task);
        };
        let initial_numeric = task.get_initial_numeric_state_values();
        let deltas = numeric_effect_deltas(task);
        let mut seeds = Vec::new();

        for goal_id in 0..task.get_num_goals() {
            let goal = task.get_goal_fact(goal_id);
            seeds.push(InitialSeedSplit::Propositional {
                var_id: goal.var,
                value: goal.value,
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
                        add_requirement_bounds(&requirement, &mut seeds);
                        let Some(source_value) =
                            initial_numeric.get(requirement.numeric_var_id).copied()
                        else {
                            continue;
                        };
                        add_shells_for_requirement(
                            task,
                            &deltas,
                            source_value,
                            &requirement,
                            &mut seeds,
                        );
                    }
                }
            }
        }

        sort_and_dedup_seed_splits(&mut seeds);
        seeds
    }

    fn route_shell_seed_splits(
        &self,
        task: &dyn AbstractNumericTask,
        shell_index: usize,
    ) -> Vec<InitialSeedSplit> {
        let Ok(comparison_index) = ComparisonAxiomIndex::from_task(task) else {
            return goal_seed_splits(task);
        };
        let initial_numeric = task.get_initial_numeric_state_values();
        let deltas = numeric_effect_deltas(task);
        let mut seeds = goal_seed_splits(task);

        for goal_id in 0..task.get_num_goals() {
            let goal = task.get_goal_fact(goal_id);
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
                        add_route_shell_for_requirement(
                            &requirement,
                            &deltas,
                            &initial_numeric,
                            shell_index,
                            &mut seeds,
                        );
                    }
                }
            }
        }

        sort_and_dedup_seed_splits(&mut seeds);
        seeds
    }
}

const COMPARISON_TRUE_VALUE: usize = 0;
const COMPLEMENTARY_PROPOSITIONAL_ACHIEVER_DEPTH: usize = 2;
const COMPLEMENTARY_MAX_PROPOSITIONAL_SEEDS: usize = 64;
const ROUTE_SHELL_STEPS: usize = 4;
const ROUTE_SHELLS_PER_GOAL_PASS: usize = 4;
const ROUTE_SHELL_MANY_GOAL_THRESHOLD: usize = 6;

#[derive(Debug, Clone)]
struct ViewCandidate {
    comparison_var_id: usize,
    numeric_var_id: usize,
    initial_value: f64,
    target_value: f64,
    deficit: f64,
    include_in_lower: bool,
    threshold_include_in_lower: bool,
    direction: ViewDirection,
}

#[derive(Debug, Clone)]
struct ViewCandidateGroup {
    candidates: Vec<ViewCandidate>,
    deficit: f64,
    first_numeric_var_id: usize,
    first_comparison_var_id: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ViewDirection {
    Increase,
    Decrease,
}

fn collect_false_view_candidates(task: &dyn AbstractNumericTask) -> Vec<ViewCandidate> {
    let initial_numeric = task.get_initial_numeric_state_values();
    let mut out = Vec::new();
    for comparison in task.comparison_axioms() {
        let Ok(is_true) = comparison.is_hold(&initial_numeric) else {
            continue;
        };
        if is_true {
            continue;
        }
        if let Some(candidate) = view_candidate_for_comparison(task, comparison, &initial_numeric) {
            out.push(candidate);
        }
    }
    out
}

fn collect_complementary_view_groups(task: &dyn AbstractNumericTask) -> Vec<ViewCandidateGroup> {
    let initial_numeric = task.get_initial_numeric_state_values();
    let mut groups_by_precondition_index: HashMap<usize, Vec<ViewCandidate>> = HashMap::new();

    for goal_id in 0..task.get_num_goals() {
        let goal = task.get_goal_fact(goal_id);
        for op in task.get_operators() {
            if !op.effects().iter().any(|effect| {
                effect.var_id() == goal.var
                    && effect.value() == goal.value
                    && effect.conditions().is_empty()
            }) {
                continue;
            }

            let false_view_preconditions = op
                .preconditions()
                .iter()
                .filter_map(|fact| view_candidate_for_comparison_fact(task, fact, &initial_numeric))
                .collect::<Vec<_>>();
            for (index, candidate) in false_view_preconditions.into_iter().enumerate() {
                groups_by_precondition_index
                    .entry(index)
                    .or_default()
                    .push(candidate);
            }
        }
    }

    let mut groups = groups_by_precondition_index
        .into_values()
        .filter_map(view_candidate_group)
        .collect::<Vec<_>>();
    if groups.is_empty() {
        groups = collect_false_view_candidates(task)
            .into_iter()
            .map(|candidate| view_candidate_group(vec![candidate]))
            .collect::<Option<Vec<_>>>()
            .unwrap_or_default();
    }
    groups
}

fn complementary_propositional_achiever_seed_splits(
    task: &dyn AbstractNumericTask,
    iteration: usize,
) -> Vec<InitialSeedSplit> {
    if task.get_num_goals() == 0 {
        return Vec::new();
    }
    let selected_goal_id = (iteration - 1) % task.get_num_goals();
    let selected_goal = task.get_goal_fact(selected_goal_id);
    let mut seeds = Vec::new();
    let mut seen_facts = HashSet::new();
    let mut frontier = vec![(selected_goal.clone(), 0usize)];
    let mut frontier_index = 0usize;

    while frontier_index < frontier.len() && seeds.len() < COMPLEMENTARY_MAX_PROPOSITIONAL_SEEDS {
        let (fact, depth) = frontier[frontier_index].clone();
        frontier_index += 1;
        if !seen_facts.insert((fact.var, fact.value)) {
            continue;
        }
        seeds.push(InitialSeedSplit::Propositional {
            var_id: fact.var,
            value: fact.value,
        });
        if depth >= COMPLEMENTARY_PROPOSITIONAL_ACHIEVER_DEPTH {
            continue;
        }
        for op in task.get_operators() {
            if !operator_has_unconditional_effect(op, &fact) {
                continue;
            }
            for precondition in op.preconditions() {
                if frontier.len() >= COMPLEMENTARY_MAX_PROPOSITIONAL_SEEDS {
                    break;
                }
                frontier.push((precondition.clone(), depth + 1));
            }
        }
    }
    dedup_seed_splits_preserve_order(&mut seeds);
    seeds
}

fn operator_has_unconditional_effect(op: &Operator, fact: &ExplicitFact) -> bool {
    op.effects().iter().any(|effect| {
        effect.conditions().is_empty()
            && effect.var_id() == fact.var
            && effect.value() == fact.value
    })
}

fn view_candidate_for_comparison_fact(
    task: &dyn AbstractNumericTask,
    fact: &ExplicitFact,
    initial_numeric: &[f64],
) -> Option<ViewCandidate> {
    if fact.value != COMPARISON_TRUE_VALUE {
        return None;
    }
    let comparison = task
        .comparison_axioms()
        .iter()
        .find(|comparison| comparison.get_affected_var_id() == fact.var)?;
    let Ok(is_true) = comparison.is_hold(initial_numeric) else {
        return None;
    };
    if is_true {
        return None;
    }
    view_candidate_for_comparison(task, comparison, initial_numeric)
}

fn view_candidate_group(mut candidates: Vec<ViewCandidate>) -> Option<ViewCandidateGroup> {
    candidates.sort_by(|left, right| {
        left.numeric_var_id
            .cmp(&right.numeric_var_id)
            .then_with(|| left.comparison_var_id.cmp(&right.comparison_var_id))
    });
    candidates.dedup_by(|left, right| {
        left.numeric_var_id == right.numeric_var_id
            && left.comparison_var_id == right.comparison_var_id
    });
    let first = candidates.first()?;
    let first_numeric_var_id = first.numeric_var_id;
    let first_comparison_var_id = first.comparison_var_id;
    let deficit = candidates
        .iter()
        .map(|candidate| candidate.deficit)
        .max_by_key(|deficit| OrderedFloat(*deficit))
        .expect("candidate group must be non-empty");
    Some(ViewCandidateGroup {
        candidates,
        deficit,
        first_numeric_var_id,
        first_comparison_var_id,
    })
}

fn view_candidate_for_comparison(
    task: &dyn AbstractNumericTask,
    comparison: &ComparisonAxiom,
    initial_numeric: &[f64],
) -> Option<ViewCandidate> {
    let left = comparison.get_left_var_id();
    let right = comparison.get_right_var_id();
    let left_type = task.numeric_variables().get(left)?.get_type();
    let right_type = task.numeric_variables().get(right)?.get_type();
    let left_value = *initial_numeric.get(left)?;
    let right_value = *initial_numeric.get(right)?;

    if left_type == &NumericType::Regular && right_type == &NumericType::Constant {
        view_candidate_from_values(
            comparison.get_affected_var_id(),
            left,
            left_value,
            right_value,
            comparison.get_operator(),
            true,
        )
    } else if left_type == &NumericType::Constant && right_type == &NumericType::Regular {
        view_candidate_from_values(
            comparison.get_affected_var_id(),
            right,
            right_value,
            left_value,
            comparison.get_operator(),
            false,
        )
    } else {
        None
    }
}

fn view_candidate_from_values(
    comparison_var_id: usize,
    numeric_var_id: usize,
    variable_value: f64,
    constant_value: f64,
    operator: &ComparisonOperator,
    variable_is_left: bool,
) -> Option<ViewCandidate> {
    let direction = match (operator, variable_is_left) {
        (ComparisonOperator::GreaterThan | ComparisonOperator::GreaterThanOrEqual, true)
        | (ComparisonOperator::LessThan | ComparisonOperator::LessThanOrEqual, false) => {
            ViewDirection::Increase
        }
        (ComparisonOperator::LessThan | ComparisonOperator::LessThanOrEqual, true)
        | (ComparisonOperator::GreaterThan | ComparisonOperator::GreaterThanOrEqual, false) => {
            ViewDirection::Decrease
        }
        (ComparisonOperator::Equal, _) if variable_value < constant_value => {
            ViewDirection::Increase
        }
        (ComparisonOperator::Equal, _) if variable_value > constant_value => {
            ViewDirection::Decrease
        }
        (ComparisonOperator::Equal, _) => {
            return None;
        }
        (ComparisonOperator::UnEqual, _) => return None,
    };
    let deficit = match direction {
        ViewDirection::Increase => constant_value - variable_value,
        ViewDirection::Decrease => variable_value - constant_value,
    };
    if !deficit.is_finite() || deficit <= 0.0 || !variable_value.is_finite() {
        return None;
    }
    Some(ViewCandidate {
        comparison_var_id,
        numeric_var_id,
        initial_value: variable_value,
        target_value: constant_value,
        deficit,
        include_in_lower: true,
        threshold_include_in_lower: direction == ViewDirection::Decrease,
        direction,
    })
}

fn complementary_route_seed_splits(
    task: &dyn AbstractNumericTask,
    candidate: &ViewCandidate,
) -> Vec<InitialSeedSplit> {
    let mut seeds = Vec::new();
    seeds.push(InitialSeedSplit::Numeric {
        numeric_var_id: candidate.numeric_var_id,
        value: candidate.target_value,
        include_in_lower: candidate.threshold_include_in_lower,
    });

    let target = match candidate.direction {
        ViewDirection::Increase => NumericRequirement {
            numeric_var_id: candidate.numeric_var_id,
            lower: Some(candidate.target_value),
            upper: None,
        },
        ViewDirection::Decrease => NumericRequirement {
            numeric_var_id: candidate.numeric_var_id,
            lower: None,
            upper: Some(candidate.target_value),
        },
    };
    add_shells_for_requirement(
        task,
        &numeric_effect_deltas(task),
        candidate.initial_value,
        &target,
        &mut seeds,
    );
    seeds
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
    fn singleton(numeric_var_id: usize, value: f64) -> Self {
        Self {
            numeric_var_id,
            lower: Some(value),
            upper: Some(value),
        }
    }

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

    fn representative_value(&self) -> Option<f64> {
        match (self.lower, self.upper) {
            (Some(lower), Some(upper)) => Some((lower + upper) / 2.0),
            (Some(lower), None) => Some(lower),
            (None, Some(upper)) => Some(upper),
            (None, None) => None,
        }
    }
}

#[derive(Debug, Clone)]
struct RegionLandmarkAnchor {
    facts: Vec<ExplicitFact>,
    requirements: Vec<NumericRequirement>,
    distance_from_initial: f64,
}

fn collect_region_landmark_anchors(task: &dyn AbstractNumericTask) -> Vec<RegionLandmarkAnchor> {
    let Ok(comparison_index) = ComparisonAxiomIndex::from_task(task) else {
        return Vec::new();
    };
    let goals = (0..task.get_num_goals())
        .map(|goal_id| task.get_goal_fact(goal_id).clone())
        .collect::<Vec<_>>();
    let initial_numeric = task.get_initial_numeric_state_values();
    let mut anchors = Vec::new();

    for goal in &goals {
        for op in task.get_operators() {
            if !op.effects().iter().any(|effect| {
                effect.var_id() == goal.var
                    && effect.value() == goal.value
                    && effect.conditions().is_empty()
            }) {
                continue;
            }

            let mut facts = vec![goal.clone()];
            facts.extend(op.preconditions().iter().cloned());
            facts.sort();
            facts.dedup();

            let mut requirements = Vec::new();
            for fact in op.preconditions() {
                let Some((numeric_var_id, interval)) =
                    numeric_requirement_for_comparison_fact(task, &comparison_index, fact)
                else {
                    continue;
                };
                if task
                    .numeric_variables()
                    .get(numeric_var_id)
                    .is_none_or(|variable| variable.get_type() != &NumericType::Regular)
                {
                    continue;
                }
                requirements.push(NumericRequirement::from_interval(numeric_var_id, interval));
            }
            merge_numeric_requirements(&mut requirements);
            if requirements.is_empty() {
                continue;
            }
            let distance_from_initial =
                approximate_distance_from_initial(&requirements, &initial_numeric);
            if !distance_from_initial.is_finite() {
                continue;
            }
            anchors.push(RegionLandmarkAnchor {
                facts,
                requirements,
                distance_from_initial,
            });
        }
    }

    anchors.sort_by(|left, right| {
        left.distance_from_initial
            .total_cmp(&right.distance_from_initial)
            .then_with(|| {
                left.facts
                    .first()
                    .map(|fact| (fact.var, fact.value))
                    .cmp(&right.facts.first().map(|fact| (fact.var, fact.value)))
            })
    });
    anchors.dedup_by(|left, right| {
        left.facts == right.facts && left.requirements == right.requirements
    });
    anchors
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
        if task
            .numeric_variables()
            .get(numeric_var_id)
            .is_some_and(|variable| variable.get_type() == &NumericType::Regular)
        {
            return vec![NumericRequirement::from_interval(numeric_var_id, interval)];
        }
    }

    let Some(tree) = comparison_index.comparison_tree(fact.var) else {
        return Vec::new();
    };
    let Ok(left) = linearize_numeric_var(task, tree.left_numeric_var_id) else {
        return Vec::new();
    };
    let Ok(right) = linearize_numeric_var(task, tree.right_numeric_var_id) else {
        return Vec::new();
    };
    let Some(required_op) = required_comparison_op(tree.op, fact.value) else {
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
        .then_with(|| left.var.cmp(&right.var))
        .then_with(|| left.value.cmp(&right.value))
}

fn estimate_goal_distance_from_initial(task: &dyn AbstractNumericTask, goal: &ExplicitFact) -> f64 {
    let Ok(comparison_index) = ComparisonAxiomIndex::from_task(task) else {
        return 0.0;
    };
    let initial_numeric = task.get_initial_numeric_state_values();
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

fn shell_seed_splits(
    task: &dyn AbstractNumericTask,
    source_requirements: &[NumericRequirement],
    target_requirements: &[NumericRequirement],
) -> Vec<InitialSeedSplit> {
    let deltas = numeric_effect_deltas(task);
    let mut seeds = Vec::new();
    for target in target_requirements {
        add_requirement_bounds(target, &mut seeds);
        let source_value = source_requirements
            .iter()
            .find(|source| source.numeric_var_id == target.numeric_var_id)
            .and_then(NumericRequirement::representative_value)
            .or_else(|| {
                task.get_initial_numeric_state_values()
                    .get(target.numeric_var_id)
                    .copied()
            });
        let Some(source_value) = source_value else {
            continue;
        };
        add_shells_for_requirement(task, &deltas, source_value, target, &mut seeds);
    }
    seeds
}

fn numeric_effect_deltas(task: &dyn AbstractNumericTask) -> HashMap<usize, Vec<f64>> {
    let initial_numeric = task.get_initial_numeric_state_values();
    let mut deltas: HashMap<usize, Vec<f64>> = HashMap::new();
    for op in task.get_operators() {
        for effect in op.assignment_effects() {
            let affected = effect.affected_var_id();
            if task
                .numeric_variables()
                .get(affected)
                .is_none_or(|variable| variable.get_type() != &NumericType::Regular)
            {
                continue;
            }
            let Some(&source_value) = initial_numeric.get(effect.var_id()) else {
                continue;
            };
            if !source_value.is_finite() {
                continue;
            }
            let delta = match effect.operation() {
                AssignmentOperation::Plus => source_value,
                AssignmentOperation::Minus => -source_value,
                AssignmentOperation::Assign
                | AssignmentOperation::Times
                | AssignmentOperation::Divide => continue,
            };
            if delta.abs() < 1e-12 {
                continue;
            }
            deltas.entry(affected).or_default().push(delta);
        }
    }
    for values in deltas.values_mut() {
        values.sort_by_key(|value| OrderedFloat(*value));
        values.dedup_by(|left, right| (*left - *right).abs() < 1e-12);
    }
    deltas
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

fn add_route_shell_for_requirement(
    requirement: &NumericRequirement,
    deltas: &HashMap<usize, Vec<f64>>,
    initial_numeric: &[f64],
    shell_index: usize,
    seeds: &mut Vec<InitialSeedSplit>,
) {
    let Some(&initial_value) = initial_numeric.get(requirement.numeric_var_id) else {
        return;
    };
    if !initial_value.is_finite() {
        return;
    }
    match (requirement.lower, requirement.upper) {
        (Some(lower), _) if initial_value < lower => {
            let Some(step) = deltas
                .get(&requirement.numeric_var_id)
                .and_then(|values| smallest_positive_delta(values))
            else {
                add_requirement_bounds(requirement, seeds);
                return;
            };
            add_increasing_route_shell(requirement.numeric_var_id, lower, step, shell_index, seeds);
        }
        (_, Some(upper)) if initial_value > upper => {
            let Some(step) = deltas
                .get(&requirement.numeric_var_id)
                .and_then(|values| largest_negative_delta(values))
                .map(f64::abs)
            else {
                add_requirement_bounds(requirement, seeds);
                return;
            };
            add_decreasing_route_shell(requirement.numeric_var_id, upper, step, shell_index, seeds);
        }
        _ => add_requirement_bounds(requirement, seeds),
    }
}

fn add_increasing_route_shell(
    numeric_var_id: usize,
    target_lower: f64,
    step: f64,
    shell_index: usize,
    seeds: &mut Vec<InitialSeedSplit>,
) {
    if step <= 1e-12 || !target_lower.is_finite() {
        return;
    }
    let shell_width = step * ROUTE_SHELL_STEPS as f64;
    let shell_near = target_lower - shell_index as f64 * shell_width;
    let shell_far = shell_near - shell_width;
    add_route_shell_points(numeric_var_id, shell_far, shell_near, step, false, seeds);
    seeds.push(InitialSeedSplit::Numeric {
        numeric_var_id,
        value: target_lower,
        include_in_lower: false,
    });
}

fn add_decreasing_route_shell(
    numeric_var_id: usize,
    target_upper: f64,
    step: f64,
    shell_index: usize,
    seeds: &mut Vec<InitialSeedSplit>,
) {
    if step <= 1e-12 || !target_upper.is_finite() {
        return;
    }
    let shell_width = step * ROUTE_SHELL_STEPS as f64;
    let shell_near = target_upper + shell_index as f64 * shell_width;
    let shell_far = shell_near + shell_width;
    add_route_shell_points(numeric_var_id, shell_near, shell_far, step, true, seeds);
    seeds.push(InitialSeedSplit::Numeric {
        numeric_var_id,
        value: target_upper,
        include_in_lower: true,
    });
}

fn add_route_shell_points(
    numeric_var_id: usize,
    start: f64,
    end: f64,
    step: f64,
    include_in_lower: bool,
    seeds: &mut Vec<InitialSeedSplit>,
) {
    if !start.is_finite() || !end.is_finite() || step <= 1e-12 {
        return;
    }
    let mut value = start;
    let limit = ROUTE_SHELL_STEPS + 1;
    for _ in 0..=limit {
        if value > end + 1e-9 {
            break;
        }
        seeds.push(InitialSeedSplit::Numeric {
            numeric_var_id,
            value,
            include_in_lower,
        });
        value += step;
    }
}

fn max_shell_splits_for_task(task: &dyn AbstractNumericTask) -> usize {
    (256usize).max(task.numeric_variables().len() * 4)
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

fn sort_and_dedup_seed_splits(seeds: &mut Vec<InitialSeedSplit>) {
    seeds.sort_by_key(|seed| match seed {
        InitialSeedSplit::Propositional { var_id, value } => (0, *var_id, *value, 0, 0),
        InitialSeedSplit::Numeric {
            numeric_var_id,
            value,
            include_in_lower,
        } => (
            1,
            *numeric_var_id,
            *include_in_lower as usize,
            value.to_bits() as usize,
            0,
        ),
    });
    seeds.dedup();
}

fn dedup_seed_splits_preserve_order(seeds: &mut Vec<InitialSeedSplit>) {
    let mut unique = Vec::with_capacity(seeds.len());
    for seed in seeds.drain(..) {
        if !unique.iter().any(|existing| existing == &seed) {
            unique.push(seed);
        }
    }
    *seeds = unique;
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
        if let Some(conditions) = goal_axiom_map.get(&goal.var) {
            seeds.extend(
                conditions
                    .iter()
                    .map(|fact| InitialSeedSplit::Propositional {
                        var_id: fact.var,
                        value: fact.value,
                    }),
            );
        } else {
            seeds.push(InitialSeedSplit::Propositional {
                var_id: goal.var,
                value: goal.value,
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
            .map(|condition| condition.var)
            .collect::<Vec<_>>();
        goal_axiom_map.insert(affected_var_id, condition_var_ids);
    }

    let logic_axiom_effect_vars = collect_logic_axiom_effect_vars(task);
    let mut goal_related: HashSet<usize> = HashSet::new();
    for goal_id in 0..task.get_num_goals() {
        let goal_var_id = task.get_goal_fact(goal_id).var;
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

struct SingleGoalTask<'task> {
    base: &'task dyn AbstractNumericTask,
    goal: ExplicitFact,
}

impl<'task> SingleGoalTask<'task> {
    fn new(base: &'task dyn AbstractNumericTask, goal: ExplicitFact) -> Self {
        Self { base, goal }
    }
}

impl AbstractNumericTask for SingleGoalTask<'_> {
    fn variables(&self) -> &Vec<ExplicitVariable> {
        self.base.variables()
    }

    fn numeric_variables(&self) -> &Vec<NumericVariable> {
        self.base.numeric_variables()
    }

    fn assignment_axioms(&self) -> &Vec<AssignmentAxiom> {
        self.base.assignment_axioms()
    }

    fn comparison_axioms(&self) -> &Vec<ComparisonAxiom> {
        self.base.comparison_axioms()
    }

    fn axioms(&self) -> &Vec<PropositionalAxiom> {
        self.base.axioms()
    }

    fn metric(&self) -> &Metric {
        self.base.metric()
    }

    fn get_num_variables(&self) -> usize {
        self.base.get_num_variables()
    }

    fn get_variable_name(&self, index: usize) -> Result<&str, &str> {
        self.base.get_variable_name(index)
    }

    fn get_variable_domain_size(&self, index: usize) -> Result<usize, &str> {
        self.base.get_variable_domain_size(index)
    }

    fn get_variable_axiom_layer(&self, index: usize) -> Result<Option<usize>, &str> {
        self.base.get_variable_axiom_layer(index)
    }

    fn get_variable_default_axiom_value(&self, index: usize) -> Result<usize, &str> {
        self.base.get_variable_default_axiom_value(index)
    }

    fn get_fact_name(&self, fact: &ExplicitFact) -> &str {
        self.base.get_fact_name(fact)
    }

    fn are_facts_mutex(&self, fact1: &ExplicitFact, fact2: &ExplicitFact) -> bool {
        self.base.are_facts_mutex(fact1, fact2)
    }

    fn get_operators(&self) -> &Vec<Operator> {
        self.base.get_operators()
    }

    fn get_operator_cost(&self, index: usize, is_axiom: bool) -> u64 {
        self.base.get_operator_cost(index, is_axiom)
    }

    fn get_operator_name(&self, index: usize, is_axiom: bool) -> &str {
        self.base.get_operator_name(index, is_axiom)
    }

    fn get_num_operators(&self) -> usize {
        self.base.get_num_operators()
    }

    fn get_num_operator_preconditions(&self, index: usize, is_axiom: bool) -> usize {
        self.base.get_num_operator_preconditions(index, is_axiom)
    }

    fn get_operator_precondition(
        &self,
        index: usize,
        precond_index: usize,
        is_axiom: bool,
    ) -> &ExplicitFact {
        self.base
            .get_operator_precondition(index, precond_index, is_axiom)
    }

    fn get_num_operator_effects(&self, index: usize, is_axiom: bool) -> usize {
        self.base.get_num_operator_effects(index, is_axiom)
    }

    fn get_num_operator_effect_conditions(
        &self,
        index: usize,
        eff_index: usize,
        is_axiom: bool,
    ) -> usize {
        self.base
            .get_num_operator_effect_conditions(index, eff_index, is_axiom)
    }

    fn get_operator_effect_condition(
        &self,
        index: usize,
        eff_index: usize,
        cond_index: usize,
        is_axiom: bool,
    ) -> &ExplicitFact {
        self.base
            .get_operator_effect_condition(index, eff_index, cond_index, is_axiom)
    }

    fn get_operator_effect(&self, index: usize, eff_index: usize, is_axiom: bool) -> &ExplicitFact {
        self.base.get_operator_effect(index, eff_index, is_axiom)
    }

    fn convert_operator_index(&self, index: usize, ancestor_task: &dyn AbstractNumericTask) {
        self.base.convert_operator_index(index, ancestor_task)
    }

    fn get_num_axioms(&self) -> usize {
        self.base.get_num_axioms()
    }

    fn get_num_goals(&self) -> usize {
        1
    }

    fn get_goal_fact(&self, index: usize) -> &ExplicitFact {
        assert_eq!(index, 0, "SingleGoalTask only exposes one goal fact");
        &self.goal
    }

    fn get_initial_propositional_state_values(&self) -> Ref<'_, Vec<usize>> {
        self.base.get_initial_propositional_state_values()
    }

    fn get_initial_numeric_state_values(&self) -> Ref<'_, Vec<f64>> {
        self.base.get_initial_numeric_state_values()
    }

    fn get_initial_propositional_state_values_mut(&self) -> RefMut<'_, Vec<usize>> {
        self.base.get_initial_propositional_state_values_mut()
    }

    fn get_initial_numeric_state_values_mut(&self) -> RefMut<'_, Vec<f64>> {
        self.base.get_initial_numeric_state_values_mut()
    }

    fn set_initial_numeric_state_values(&self, values: Vec<f64>) {
        self.base.set_initial_numeric_state_values(values)
    }

    fn set_initial_propositional_state_values(&self, values: Vec<usize>) {
        self.base.set_initial_propositional_state_values(values)
    }

    fn convert_ancestor_state_values(
        &self,
        ancestor_state_values: &[usize],
        ancestor_task: &dyn AbstractNumericTask,
    ) -> Vec<usize> {
        self.base
            .convert_ancestor_state_values(ancestor_state_values, ancestor_task)
    }

    fn get_num_cmp_axioms(&self) -> usize {
        self.base.get_num_cmp_axioms()
    }

    fn abstract_state_values(
        &self,
        propositional_values: &[usize],
        numeric_values: &[f64],
    ) -> Result<(Vec<usize>, Vec<f64>), String> {
        self.base
            .abstract_state_values(propositional_values, numeric_values)
    }

    fn evaluated_initial_abstract_state_values(&self) -> Result<(Vec<usize>, Vec<f64>), String> {
        self.base.evaluated_initial_abstract_state_values()
    }

    fn abstract_operator_cost(&self, operator_id: usize) -> f64 {
        self.base.abstract_operator_cost(operator_id)
    }
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
