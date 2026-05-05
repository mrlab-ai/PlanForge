#[cfg(test)]
mod tests;

use std::cell::{Ref, RefMut};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use ordered_float::OrderedFloat;
use planners_sas::numeric::axioms::ComparisonOperator;
use planners_sas::numeric::axioms::{AssignmentAxiom, ComparisonAxiom, PropositionalAxiom};
use planners_sas::numeric::numeric_task::{
    AbstractNumericTask, ExplicitFact, ExplicitVariable, Metric, NumericType, NumericVariable,
    Operator,
};
use rand::seq::SliceRandom;
use rand::{rngs::SmallRng, RngCore, SeedableRng};
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::numeric::evaluation::domain_abstractions::cegar::FlawKind;

pub use super::cegar::flaw_search::flaw_selection::{FlawTreatmentVariants, InitSplitMethod};
use super::cegar::CegarConfig;
use super::cegar::InitialSeedSplit;
use super::domain_abstraction_generator::{
    prepare_domain_abstraction_task, DomainAbstraction, DomainAbstractionGenerator,
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
}

impl fmt::Display for PortfolioStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Standard => write!(f, "standard"),
            Self::ViewDiverse => write!(f, "view_diverse"),
        }
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
        goals.shuffle(&mut rng);
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

            if remaining_abstraction_size == 0 || remaining_generation_time <= 0.0 {
                break;
            }

            let goal_task = goals
                .get(goal_index)
                .map(|goal| SingleGoalTask::new(task, goal.clone()));
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
            let initial_seed_splits = self.initial_seed_splits(generation_task, iteration);
            let (blacklisted_prop_var_ids, blacklisted_numeric_var_ids) =
                split_blacklisted_variables(generation_task, blacklisted_var_ids);
            let init_split_var_ids = if initial_seed_splits.is_empty() {
                self.initial_split_var_ids(generation_task, iteration)
            } else {
                None
            };
            let cegar_config = self.build_cegar_config(
                remaining_abstraction_size,
                remaining_generation_time,
                init_split_var_ids,
                initial_seed_splits,
                blacklisted_prop_var_ids,
                blacklisted_numeric_var_ids,
                Some(rng.next_u64()),
                self.flaw_kind_for_iteration(iteration),
            );
            let generator = DomainAbstractionGenerator::new(cegar_config)
                .context("failed to construct single-abstraction CEGAR generator")?;
            let abstraction = generator
                .generate_prepared(abstraction_task, &prepared_task)
                .with_context(|| {
                    format!("failed to generate abstraction for collection iteration {iteration}")
                })?;

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

            iteration += 1;
            if !goals.is_empty() {
                goal_index = (goal_index + 1) % goals.len();
                let _ = &goals[goal_index];
            }
        }

        if generated_abstractions.is_empty() {
            bail!("multi_domain_abstractions(...) failed to generate any abstractions")
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

    fn initial_seed_splits(
        &self,
        task: &dyn AbstractNumericTask,
        iteration: usize,
    ) -> Vec<InitialSeedSplit> {
        if self.config.portfolio_strategy != PortfolioStrategy::ViewDiverse {
            return Vec::new();
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

    fn flaw_kind_for_iteration(&self, iteration: usize) -> FlawKind {
        if self.config.portfolio_strategy != PortfolioStrategy::ViewDiverse {
            return self.config.flaw_kind;
        }
        if iteration % 2 == 1 {
            FlawKind::SequenceProgression
        } else {
            FlawKind::SequenceRegression
        }
    }
}

const COMPARISON_TRUE_VALUE: usize = 0;

#[derive(Debug, Clone)]
struct ViewCandidate {
    comparison_var_id: usize,
    numeric_var_id: usize,
    initial_value: f64,
    deficit: f64,
    include_in_lower: bool,
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
    let deficit = match (operator, variable_is_left) {
        (ComparisonOperator::GreaterThan | ComparisonOperator::GreaterThanOrEqual, true)
        | (ComparisonOperator::LessThan | ComparisonOperator::LessThanOrEqual, false) => {
            constant_value - variable_value
        }
        (ComparisonOperator::LessThan | ComparisonOperator::LessThanOrEqual, true)
        | (ComparisonOperator::GreaterThan | ComparisonOperator::GreaterThanOrEqual, false) => {
            variable_value - constant_value
        }
        (ComparisonOperator::Equal, _) => (variable_value - constant_value).abs(),
        (ComparisonOperator::UnEqual, _) => return None,
    };
    if !deficit.is_finite() || deficit <= 0.0 || !variable_value.is_finite() {
        return None;
    }
    Some(ViewCandidate {
        comparison_var_id,
        numeric_var_id,
        initial_value: variable_value,
        deficit,
        include_in_lower: true,
    })
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
