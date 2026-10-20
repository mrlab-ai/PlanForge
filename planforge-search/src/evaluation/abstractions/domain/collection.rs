#[cfg(test)]
mod tests;

use std::collections::HashSet;
use std::fmt;
use std::time::Duration;

use anyhow::{Context, Result};

use planforge_sas::numeric_task::{AbstractNumericTask, VariableIndex};
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::evaluation::abstractions::collections::goal_focused::{
    BuiltReport, CollectionMembership, GoalFocusedBackend, GoalFocusedSchedule, GoalVariant,
    MemberBudget, build_goal_focused_collection,
};
use crate::evaluation::abstractions::collections::standard::{
    StandardBackend, StandardMember, StandardSchedule, build_standard_collection,
};
use crate::evaluation::abstractions::collections::strategy::{
    CollectionOptions, CollectionStrategy, validate_collection_options,
};
use crate::evaluation::abstractions::common::cegar::{
    CegarStopReason, FlawKind, FlawScope, SplitKind, SplitSelection,
};
use crate::evaluation::abstractions::common::task::AbstractionUse;
use crate::evaluation::abstractions::domain::abstract_operators::DomainMapping;
use planforge_sas::explicit_state::ExplicitState;

use super::abstraction::compute_abstraction_size_u128;
use super::abstraction::{DomainAbstraction, DomainAbstractionMetadata};
use super::config::DomainAbstractionConfig;
use super::generator::DomainAbstractionGenerator;
use planforge_sas::utils::float_tolerance;
use planforge_sas::utils::interval::Interval;

#[derive(
    Debug, Clone, Deserialize, Serialize, PartialEq, planforge_search::config::ApplyOptions,
)]
pub struct DomainAbstractionCollectionConfig {
    pub max_states: usize,
    pub max_collection_size: usize,
    pub max_time: f64,
    pub total_max_time: f64,
    pub stagnation_limit: f64,
    pub blacklist_trigger_percentage: f64,
    pub enable_blacklist_on_stagnation: bool,
    pub random_seed: Option<u64>,
    pub debug: bool,
    pub use_wildcard_plans: bool,
    pub combine_labels: bool,
    pub flaw_kind: FlawKind,
    pub flaw_scope: FlawScope,
    pub split_selection: SplitSelection,
    /// `standard` refines full-task abstractions with random restarts,
    /// blacklisting, stagnation detection, and deduplication.
    /// `goal_focused` builds the goal-focused collection shared with
    /// Cartesian abstractions (see
    /// [`goal_focused`](crate::evaluation::abstractions::collections::goal_focused)):
    /// `variants_per_goal` members per goal, each refined from the trivial
    /// abstraction of its single-goal task with the configured `flaw_kind`,
    /// concrete splits for even and wanted splits for odd variants (unless
    /// `alternate_split_kinds` is off), flaw ties ranked by `variant / 2`. The
    /// init-split, blacklist, and stagnation options only apply to `standard`.
    pub collection_strategy: CollectionStrategy,
    /// Members per goal of a `goal_focused` collection. Must be 1 for
    /// `standard`.
    pub variants_per_goal: usize,
    /// Let every variant of a `goal_focused` collection refine from the
    /// concrete states its earlier members' plans reached. Must be false for
    /// `standard`.
    pub progress_states: bool,
    /// Alternate concrete and wanted splits between members: by iteration in
    /// a `standard` collection, by variant parity in a `goal_focused` one.
    /// Unset, exactly goal-focused collections alternate; otherwise every
    /// member uses the configured `split_kind` (concrete splits when unset).
    pub alternate_split_kinds: Option<bool>,
    /// Build every member of a `goal_focused` collection on its goal's
    /// single-goal task. When false, every member abstracts the full task and
    /// goals only label the `G * V` members, which are scheduled round-robin
    /// and carry no collection membership. Must be true for `standard` and
    /// with `progress_states`.
    pub single_goal_tasks: bool,
    /// The fixed split kind; concrete splits when unset.
    pub split_kind: Option<SplitKind>,
    /// Pass-through for `DomainAbstractionConfig::compute_source_sets`. Default
    /// `true`. Hierarchical combinators disable it during CEGAR and construct
    /// operator regions after collection generation when region SCP needs them.
    /// Set internally by heuristic construction; not exposed as a CLI option.
    #[option(skip)]
    pub compute_source_sets: bool,
}

impl Default for DomainAbstractionCollectionConfig {
    fn default() -> Self {
        Self {
            max_states: 1_000_000,
            max_collection_size: 10_000_000,
            max_time: f64::INFINITY,
            total_max_time: 10.0,
            stagnation_limit: 20.0,
            blacklist_trigger_percentage: 0.75,
            enable_blacklist_on_stagnation: true,
            random_seed: Some(2011),
            debug: false,
            use_wildcard_plans: false,
            combine_labels: true,
            flaw_kind: FlawKind::Progression,
            flaw_scope: FlawScope::First,
            split_selection: SplitSelection::Random,
            collection_strategy: CollectionStrategy::Standard,
            variants_per_goal: 1,
            progress_states: false,
            alternate_split_kinds: None,
            single_goal_tasks: true,
            split_kind: None,
            compute_source_sets: true,
        }
    }
}

impl DomainAbstractionCollectionConfig {
    /// Bound both collection-level and per-abstraction construction by the
    /// time remaining to the caller.
    pub(crate) fn cap_construction_time(&mut self, max_seconds: f64) {
        self.total_max_time = self.total_max_time.min(max_seconds);
        self.max_time = self.max_time.min(max_seconds);
    }

    /// Select whether generated abstractions retain operator regions.
    ///
    /// This is an internal construction policy rather than a user option.
    pub(crate) fn set_compute_source_sets(&mut self, enabled: bool) {
        self.compute_source_sets = enabled;
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

/// A time limit in seconds, `None` when unbounded.
fn optional_duration(option: &str, seconds: f64) -> Result<Option<Duration>> {
    if seconds.is_infinite() {
        return Ok(None);
    }
    Duration::try_from_secs_f64(seconds)
        .map(Some)
        .with_context(|| format!("invalid domain abstraction collection {option}={seconds}"))
}

impl fmt::Display for DomainAbstractionCollectionConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            concat!(
                "max_states={}, ",
                "max_collection_size={}, ",
                "max_time={}, ",
                "total_max_time={}, ",
                "stagnation_limit={}, ",
                "blacklist_trigger_percentage={}, ",
                "enable_blacklist_on_stagnation={}, ",
                "random_seed={}, ",
                "debug={}, ",
                "use_wildcard_plans={}, ",
                "combine_labels={}, ",
                "flaw_kind={}, ",
                "flaw_scope={}, ",
                "split_selection={}, ",
                "collection_strategy={}, ",
                "variants_per_goal={}, ",
                "progress_states={}, ",
                "alternate_split_kinds={:?}, ",
                "single_goal_tasks={}, ",
            ),
            self.max_states,
            self.max_collection_size,
            fmt_f64(self.max_time),
            fmt_f64(self.total_max_time),
            fmt_f64(self.stagnation_limit),
            fmt_f64(self.blacklist_trigger_percentage),
            self.enable_blacklist_on_stagnation,
            fmt_optional_seed(self.random_seed),
            self.debug,
            self.use_wildcard_plans,
            self.combine_labels,
            self.flaw_kind,
            self.flaw_scope,
            self.split_selection,
            self.collection_strategy,
            self.variants_per_goal,
            self.progress_states,
            self.alternate_split_kinds,
            self.single_goal_tasks,
        )
    }
}

#[derive(Debug, Clone)]
pub struct DomainAbstractionCollectionGenerator {
    config: DomainAbstractionCollectionConfig,
}

impl DomainAbstractionCollectionGenerator {
    pub fn new(config: DomainAbstractionCollectionConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &DomainAbstractionCollectionConfig {
        &self.config
    }

    fn validate_supported_options(&self) -> Result<()> {
        validate_collection_options(CollectionOptions {
            strategy: self.config.collection_strategy,
            variants_per_goal: self.config.variants_per_goal,
            max_collection_size: self.config.max_collection_size,
            progress_states: self.config.progress_states,
            alternate_split_kinds: self.config.alternate_split_kinds,
            single_goal_tasks: self.config.single_goal_tasks,
            fixed_split_kind: self.config.split_kind.is_some(),
            flaw_kind: self.config.flaw_kind,
        })
    }

    /// Configuration for one restart of a standard collection.
    fn standard_member_config(&self, member: &StandardMember) -> DomainAbstractionConfig {
        DomainAbstractionConfig {
            max_states: member.max_states,
            max_iterations: DomainAbstractionConfig::default().max_iterations,
            max_time: member
                .max_time
                .is_finite()
                .then(|| Duration::from_secs_f64(member.max_time.max(0.0))),
            use_wildcard_plans: self.config.use_wildcard_plans,
            combine_labels: self.config.combine_labels,
            debug: self.config.debug,
            random_seed: Some(member.seed),
            flaw_kind: self.config.flaw_kind,
            flaw_scope: self.config.flaw_scope,
            split_kind: member.split_kind.unwrap_or(SplitKind::Concrete),
            split_selection: self.config.split_selection,
            init_split_var_ids: Some(member.initial_split_var.into_iter().collect()),
            blacklisted_prop_var_ids: member.blacklisted_prop_var_ids.clone(),
            blacklisted_numeric_var_ids: member.blacklisted_numeric_var_ids.clone(),
            compute_source_sets: self.config.compute_source_sets,
            flaw_selection_rank: None,
        }
    }

    pub fn generate(&self, task: &dyn AbstractNumericTask) -> Result<Vec<DomainAbstraction>> {
        self.validate_supported_options()?;
        let abstractions = match self.config.collection_strategy {
            CollectionStrategy::Standard => self.generate_standard_collection(task)?,
            CollectionStrategy::PerGoal | CollectionStrategy::GoalFocused => {
                self.generate_goal_focused_collection(task)?
            }
        };
        if self.config.debug {
            log_collection_debug_summary(&abstractions);
        }
        Ok(abstractions)
    }

    /// The goal-focused collection shared with Cartesian abstractions.
    fn generate_goal_focused_collection(
        &self,
        task: &dyn AbstractNumericTask,
    ) -> Result<Vec<DomainAbstraction>> {
        let backend = GoalFocusedDomainBackend {
            config: &self.config,
            member_max_time: optional_duration("max_time", self.config.max_time)?,
        };
        let mut abstractions = build_goal_focused_collection(
            task,
            &GoalFocusedSchedule {
                strategy: self.config.collection_strategy,
                variants_per_goal: self.config.variants_per_goal,
                random_seed: self.config.random_seed,
                max_collection_states: self.config.max_collection_size,
                total_max_time: optional_duration("total_max_time", self.config.total_max_time)?,
                progress_states: self.config.progress_states,
                alternate_split_kinds: self
                    .config
                    .collection_strategy
                    .alternates_split_kinds(self.config.alternate_split_kinds),
                single_goal_tasks: self.config.single_goal_tasks,
            },
            &backend,
        )?;
        for (index, abstraction) in abstractions.iter_mut().enumerate() {
            abstraction.metadata.collection_iteration = Some(index + 1);
            if self.config.debug {
                log_collection_abstraction_debug(index, abstraction, task);
            }
        }
        Ok(abstractions)
    }

    /// The standard collection of `task`; see
    /// [`standard`](crate::evaluation::abstractions::collections::standard).
    fn generate_standard_collection(
        &self,
        task: &dyn AbstractNumericTask,
    ) -> Result<Vec<DomainAbstraction>> {
        let abstractions = build_standard_collection(
            task,
            &self.standard_schedule(),
            &StandardDomainBackend { generator: self },
        )?;
        if self.config.debug {
            for (index, abstraction) in abstractions.iter().enumerate() {
                log_collection_abstraction_debug(index, abstraction, task);
            }
        }
        Ok(abstractions)
    }

    fn standard_schedule(&self) -> StandardSchedule {
        StandardSchedule {
            max_states: self.config.max_states,
            max_collection_size: self.config.max_collection_size,
            max_time: self.config.max_time,
            total_max_time: self.config.total_max_time,
            stagnation_limit: self.config.stagnation_limit,
            blacklist_trigger_percentage: self.config.blacklist_trigger_percentage,
            enable_blacklist_on_stagnation: self.config.enable_blacklist_on_stagnation,
            random_seed: self.config.random_seed,
            alternate_split_kinds: self
                .config
                .collection_strategy
                .alternates_split_kinds(self.config.alternate_split_kinds),
            split_kind: self.config.split_kind,
        }
    }
}

/// Builds the restarts of a standard domain collection.
struct StandardDomainBackend<'a> {
    generator: &'a DomainAbstractionCollectionGenerator,
}

impl StandardBackend for StandardDomainBackend<'_> {
    type Member = DomainAbstraction;
    type Key = AbstractionKey;

    const NAME: &'static str = "domain";

    fn build(
        &self,
        task: &dyn AbstractNumericTask,
        member: &StandardMember,
    ) -> Result<DomainAbstraction> {
        let config = &self.generator.config;
        let mut abstraction =
            DomainAbstractionGenerator::new(self.generator.standard_member_config(member))?
                .generate(task)?;
        abstraction.metadata = DomainAbstractionMetadata {
            collection_iteration: Some(member.iteration),
            collection_strategy: Some(config.collection_strategy.to_string()),
            flaw_kind: Some(config.flaw_kind),
            flaw_scope: Some(config.flaw_scope),
            full_goal_task: Some(true),
            abstraction_use: AbstractionUse::CollectionMember,
            max_states: Some(member.max_states),
            ..std::mem::take(&mut abstraction.metadata)
        };
        Ok(abstraction)
    }

    fn key(&self, abstraction: &DomainAbstraction) -> AbstractionKey {
        AbstractionKey::from_abstraction(abstraction)
    }

    fn num_states(&self, abstraction: &DomainAbstraction) -> usize {
        abstraction.distance_table.distances.len()
    }

    fn settles(&self, abstraction: &DomainAbstraction) -> Option<&'static str> {
        if abstraction.metadata.solved_by_self {
            Some("concrete plan")
        } else if abstraction.metadata.stop_reason == Some(CegarStopReason::DeadEnd) {
            Some("initial state is a dead end")
        } else {
            None
        }
    }
}

/// Builds the members of a goal-focused domain collection.
struct GoalFocusedDomainBackend<'config> {
    config: &'config DomainAbstractionCollectionConfig,
    member_max_time: Option<Duration>,
}

impl GoalFocusedDomainBackend<'_> {
    /// CEGAR configuration for one member: the collection's options, capped by
    /// what is left of the collection's budgets and specialised for the
    /// variant. Members start from the trivial abstraction: no initial splits
    /// and no blacklisted variables.
    fn member_cegar_config(
        &self,
        variant: &GoalVariant,
        budget: MemberBudget,
    ) -> DomainAbstractionConfig {
        DomainAbstractionConfig {
            max_states: self.config.max_states.min(budget.remaining_states),
            max_iterations: DomainAbstractionConfig::default().max_iterations,
            // Whichever of the two budgets exist bind; the tighter one wins.
            max_time: self
                .member_max_time
                .into_iter()
                .chain(budget.remaining_time)
                .min(),
            use_wildcard_plans: self.config.use_wildcard_plans,
            combine_labels: self.config.combine_labels,
            debug: self.config.debug,
            random_seed: variant.seed,
            flaw_kind: self.config.flaw_kind,
            flaw_scope: self.config.flaw_scope,
            split_kind: variant
                .split_kind
                .or(self.config.split_kind)
                .unwrap_or(SplitKind::Concrete),
            split_selection: self.config.split_selection,
            init_split_var_ids: Some(HashSet::new()),
            blacklisted_prop_var_ids: HashSet::new(),
            blacklisted_numeric_var_ids: HashSet::new(),
            compute_source_sets: self.config.compute_source_sets,
            flaw_selection_rank: variant.tie_rank,
        }
    }
}

impl GoalFocusedBackend for GoalFocusedDomainBackend<'_> {
    type Member = DomainAbstraction;

    const NAME: &'static str = "domain";

    fn build(
        &self,
        task: &dyn AbstractNumericTask,
        root: Option<&ExplicitState>,
        variant: &GoalVariant,
        budget: MemberBudget,
    ) -> Result<(DomainAbstraction, BuiltReport)> {
        let cegar_config = self.member_cegar_config(variant, budget);
        let max_states = cegar_config.max_states;
        let mut abstraction = DomainAbstractionGenerator::new(cegar_config)
            .context("failed to construct single-abstraction CEGAR generator")?
            .generate_from_root(task, root)?;
        let metadata = &mut abstraction.metadata;
        metadata.collection_strategy = Some(self.config.collection_strategy.to_string());
        metadata.flaw_kind = Some(self.config.flaw_kind);
        metadata.flaw_scope = Some(self.config.flaw_scope);
        metadata.full_goal_task = Some(variant.full_task);
        metadata.max_states = Some(max_states);
        let report = BuiltReport {
            initial_h: abstraction.distance_table.distances
                [abstraction.distance_table.initial_state_hash],
            num_states: abstraction.distance_table.distances.len(),
            concrete_plan: abstraction.metadata.concrete_plan_operator_ids.clone(),
            root_is_dead_end: abstraction.metadata.stop_reason == Some(CegarStopReason::DeadEnd),
        };
        Ok((abstraction, report))
    }

    fn record_membership(
        &self,
        abstraction: &mut DomainAbstraction,
        membership: Option<CollectionMembership>,
    ) {
        abstraction.metadata.collection_member = membership;
        abstraction.metadata.abstraction_use = AbstractionUse::CollectionMember;
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
        "domain abstraction collection debug: id={abstraction_id}, iteration={:?}, strategy={:?}, flaw_kind={:?}, full_goal_task={:?}, member={:?}, max_abs_size={:?}, states={}, prop_split_vars={}, numeric_split_vars={}, abstract_ops={}",
        metadata.collection_iteration,
        metadata.collection_strategy,
        metadata.flaw_kind,
        metadata.full_goal_task,
        metadata.collection_member,
        metadata.max_states,
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
            "domain abstraction collection debug summary: id={id}, iteration={:?}, strategy={:?}, flaw_kind={:?}, member={:?}, states={}, abstract_ops={}",
            abstraction.metadata.collection_iteration,
            abstraction.metadata.collection_strategy,
            abstraction.metadata.flaw_kind,
            abstraction.metadata.collection_member,
            compute_abstraction_size_u128(
                abstraction.factory.domain_sizes(),
                abstraction.factory.numeric_domain_sizes()
            )
            .unwrap_or(u128::MAX),
            abstraction.abstract_operators.len(),
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
        let name = task
            .get_variable_name(VariableIndex::from_usize(var_id))
            .unwrap_or("<unknown>");
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
        let Some(parts) = abstraction
            .factory
            .partitions()
            .partitions(VariableIndex::from_usize(numeric_var_id))
        else {
            continue;
        };
        let preview = partition_preview(parts);
        info!(
            "domain abstraction collection debug partitions: n{numeric_var_id}={name}, size={size}, {preview}"
        );
    }
}

fn partition_preview(parts: &[Interval]) -> String {
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct IntervalFingerprint {
    lower: u64,
    upper: u64,
    lower_closed: bool,
    upper_closed: bool,
}

impl IntervalFingerprint {
    fn from_interval(interval: Interval) -> Self {
        Self {
            lower: float_tolerance::canonical_bits(interval.lower),
            upper: float_tolerance::canonical_bits(interval.upper),
            lower_closed: interval.lower_closed,
            upper_closed: interval.upper_closed,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct AbstractionKey {
    domain_mapping: DomainMapping,
    numeric_fingerprint: Vec<Vec<IntervalFingerprint>>,
}

impl AbstractionKey {
    fn from_abstraction(abstraction: &DomainAbstraction) -> Self {
        let factory = &abstraction.factory;
        let numeric_fingerprint = (0..factory.numeric_domain_sizes().len())
            .map(|numeric_var_id| {
                factory
                    .partitions()
                    .partitions(VariableIndex::from_usize(numeric_var_id))
                    .unwrap_or_else(|| {
                        panic!(
                            "numeric variable {numeric_var_id} of the abstraction has no partitions"
                        )
                    })
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
