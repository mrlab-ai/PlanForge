use std::cell::{Ref, RefMut};
use std::collections::HashSet;
use std::fmt;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use ordered_float::OrderedFloat;
use planners_sas::numeric::axioms::{AssignmentAxiom, ComparisonAxiom, PropositionalAxiom};
use planners_sas::numeric::numeric_task::{
    AbstractNumericTask, ExplicitFact, ExplicitVariable, Metric, NumericVariable, Operator,
};
use rand::seq::SliceRandom;
use rand::{SeedableRng, rngs::SmallRng};
use serde::{Deserialize, Serialize};

use super::cegar::CegarConfig;
pub use super::cegar::{ExecEntirePlanMode, FlawTreatment, InitSplitMethod};
use super::domain_abstraction_generator::DomainAbstraction;
use super::domain_abstraction_generator::DomainAbstractionGenerator;
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
    pub random_seed: i32,
    pub use_wildcard_plans: bool,
    pub deviation_flaws: bool,
    pub flaw_treatment: FlawTreatment,
    pub init_split_method: InitSplitMethod,
    pub numeric_split_strategy: NumericSplitStrategy,
    pub exec_entire_plan: ExecEntirePlanMode,
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
            random_seed: -1,
            use_wildcard_plans: true,
            deviation_flaws: true,
            flaw_treatment: FlawTreatment::RandomSingleAtom,
            init_split_method: InitSplitMethod::InitValue,
            numeric_split_strategy: NumericSplitStrategy::Standard,
            exec_entire_plan: ExecEntirePlanMode::StopAtFirstFlaw,
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
                "use_wildcard_plans={}, ",
                "deviation_flaws={}, ",
                "flaw_treatment={}, ",
                "init_split_method={}, ",
                "numeric_split_strategy={}, ",
                "exec_entire_plan={}, ",
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
            self.random_seed,
            self.use_wildcard_plans,
            self.deviation_flaws,
            self.flaw_treatment,
            self.init_split_method,
            self.numeric_split_strategy,
            self.exec_entire_plan,
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
        if !self.config.deviation_flaws {
            bail!("`deviation_flaws=false` is not supported in the current Rust port");
        }
        if self.config.numeric_split_strategy != NumericSplitStrategy::Standard {
            bail!("`numeric_split_strategy=exclusion` is not supported in the current Rust port");
        }
        Ok(())
    }

    fn create_rng(&self) -> SmallRng {
        if self.config.random_seed >= 0 {
            SmallRng::seed_from_u64(self.config.random_seed as u64)
        } else {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0x5EED_F00D_u64);
            SmallRng::seed_from_u64(nanos)
        }
    }

    fn build_cegar_config(
        &self,
        max_abstraction_size: usize,
        remaining_time: f64,
        init_split_var_ids: Option<HashSet<usize>>,
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
            combine_labels: false,
            debug: false,
            flaw_treatment: self.config.flaw_treatment,
            init_split_method: match self.config.init_split_quantity {
                InitSplitQuantity::None => InitSplitMethod::Identity,
                InitSplitQuantity::Single | InitSplitQuantity::All => self.config.init_split_method,
            },
            exec_entire_plan: self.config.exec_entire_plan,
            init_split_var_ids,
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
            let init_split_var_ids = self.initial_split_var_ids(abstraction_task);
            let cegar_config = self.build_cegar_config(
                remaining_abstraction_size,
                remaining_generation_time,
                init_split_var_ids,
            );
            let generator = DomainAbstractionGenerator::new(cegar_config)
                .context("failed to construct single-abstraction CEGAR generator")?;
            let abstraction = generator.generate(abstraction_task).with_context(|| {
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

    fn initial_split_var_ids(&self, task: &dyn AbstractNumericTask) -> Option<HashSet<usize>> {
        let candidate_var_ids: HashSet<usize> = match self.config.init_split_candidates {
            VariableSubset::Goals => collect_goal_related_propositional_vars(task),
            VariableSubset::NonGoals => {
                let goal_related = collect_goal_related_propositional_vars(task);
                (0..task.variables().len())
                    .filter(|var_id| !goal_related.contains(var_id))
                    .collect()
            }
            VariableSubset::All => (0..task.variables().len()).collect(),
        };

        let selected_var_ids = match self.config.init_split_quantity {
            InitSplitQuantity::None => HashSet::new(),
            InitSplitQuantity::All => candidate_var_ids,
            InitSplitQuantity::Single => select_single_init_split_var(task, &candidate_var_ids)
                .into_iter()
                .collect(),
        };

        Some(selected_var_ids)
    }
}

fn collect_goal_related_propositional_vars(task: &dyn AbstractNumericTask) -> HashSet<usize> {
    let mut goal_related: HashSet<usize> = (0..task.get_num_goals())
        .filter_map(|goal_id| usize::try_from(task.get_goal_fact(goal_id).var).ok())
        .collect();

    loop {
        let mut changed = false;

        for axiom in task.axioms() {
            let affected_var_id = axiom.var_id() as usize;
            if goal_related.contains(&affected_var_id) {
                for condition in axiom.conditions() {
                    changed |= goal_related.insert(condition.var);
                }
            }
        }

        if !changed {
            break;
        }
    }

    goal_related
}

fn select_single_init_split_var(
    task: &dyn AbstractNumericTask,
    candidate_var_ids: &HashSet<usize>,
) -> Option<usize> {
    (0..task.get_num_goals())
        .filter_map(|goal_id| usize::try_from(task.get_goal_fact(goal_id).var).ok())
        .find(|var_id| candidate_var_ids.contains(var_id))
        .or_else(|| candidate_var_ids.iter().min().copied())
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

    fn abstract_propositional_var_ids(&self) -> &[usize] {
        self.base.abstract_propositional_var_ids()
    }

    fn abstract_numeric_var_ids(&self) -> &[usize] {
        self.base.abstract_numeric_var_ids()
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
