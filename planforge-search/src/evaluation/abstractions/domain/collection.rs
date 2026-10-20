#[cfg(test)]
mod tests;

use std::collections::HashSet;
use std::fmt;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};

use planforge_sas::numeric_task::{AbstractNumericTask, NumericType, VariableIndex};
use rand::seq::SliceRandom;
use rand::{RngCore, SeedableRng, rngs::SmallRng};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::evaluation::abstractions::collections::goal_focused::{
    BuiltReport, CollectionMembership, FULL_TASK_PROGRESS_STATES_ERROR, GoalFocusedBackend,
    GoalFocusedSchedule, GoalVariant, MemberBudget, build_goal_focused_collection,
};
use crate::evaluation::abstractions::collections::portfolio::CollectionStrategy;
use crate::evaluation::abstractions::common::cegar::{
    CegarStopReason, FlawKind, FlawScope, SplitKind, SplitSelection,
};
use crate::evaluation::abstractions::common::concrete_state::ConcreteState;
use crate::evaluation::abstractions::common::task::AbstractionUse;
use crate::evaluation::abstractions::domain::abstract_operators::DomainMapping;

use super::abstraction::compute_abstraction_size_u128;
use super::abstraction::{DomainAbstraction, DomainAbstractionMetadata};
use super::config::CegarConfig;
use super::generator::DomainAbstractionGenerator;
use crate::resource_limits;
use planforge_sas::utils::float_tolerance;
use planforge_sas::utils::interval::Interval;

#[derive(
    Debug, Clone, Deserialize, Serialize, PartialEq, planforge_search::config::ApplyOptions,
)]
pub struct DomainAbstractionCollectionGeneratorMultipleCegarConfig {
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
    /// Alternate concrete and wanted splits by variant parity in a
    /// `goal_focused` collection. When false, every variant uses the
    /// configured `split_kind` (concrete splits when unset).
    /// Must be true for `standard`.
    pub alternate_split_kinds: bool,
    /// Build every member of a `goal_focused` collection on its goal's
    /// single-goal task. When false, every member abstracts the full task and
    /// goals only label the `G * V` members, which are scheduled round-robin
    /// and carry no collection membership. Must be true for `standard` and
    /// with `progress_states`.
    pub single_goal_tasks: bool,
    /// Alternate concrete and wanted splits between the members of a
    /// `standard` collection, independently of the flaw kind and scope.
    pub interleave_split_directions: bool,
    /// The fixed split kind; concrete splits when unset.
    pub split_kind: Option<SplitKind>,
    /// Pass-through for `CegarConfig::compute_source_sets`. Default
    /// `true`. Hierarchical combinators disable it during CEGAR and construct
    /// operator regions after collection generation when region SCP needs them.
    /// Set internally by heuristic construction; not exposed as a CLI option.
    #[option(skip)]
    pub compute_source_sets: bool,
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
            alternate_split_kinds: true,
            single_goal_tasks: true,
            interleave_split_directions: false,
            split_kind: None,
            compute_source_sets: true,
            max_refined_comparison_vars_per_abstraction: None,
        }
    }
}

impl DomainAbstractionCollectionGeneratorMultipleCegarConfig {
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

fn is_generation_deadline_error(error: &anyhow::Error) -> bool {
    crate::resource_limits::is_deadline_exceeded(error)
}

impl fmt::Display for DomainAbstractionCollectionGeneratorMultipleCegarConfig {
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
                "alternate_split_kinds={}, ",
                "single_goal_tasks={}, ",
                "interleave_split_directions={}, ",
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
            self.interleave_split_directions,
        )
    }
}

/// Which variables one standard collection member may split.
struct MemberVariableChoice {
    init_split_var_ids: Option<HashSet<VariableIndex>>,
    blacklisted_prop_var_ids: HashSet<VariableIndex>,
    blacklisted_numeric_var_ids: HashSet<VariableIndex>,
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
        ensure!(
            self.config.variants_per_goal > 0,
            "`variants_per_goal` must be > 0"
        );
        match self.config.collection_strategy {
            CollectionStrategy::Standard => {
                ensure!(
                    self.config.variants_per_goal == 1,
                    "`variants_per_goal` requires `collection_strategy=goal_focused`"
                );
                ensure!(
                    !self.config.progress_states,
                    "`progress_states=true` requires `collection_strategy=goal_focused`"
                );
                ensure!(
                    self.config.alternate_split_kinds,
                    "`alternate_split_kinds=false` requires `collection_strategy=goal_focused`"
                );
                ensure!(
                    self.config.single_goal_tasks,
                    "`single_goal_tasks=false` requires `collection_strategy=goal_focused`"
                );
                if self.config.interleave_split_directions && self.config.split_kind.is_some() {
                    bail!(
                        "`interleave_split_directions=true` cannot be combined with a fixed `split_kind`"
                    );
                }
            }
            CollectionStrategy::GoalFocused => {
                ensure!(
                    !self.config.interleave_split_directions,
                    "`interleave_split_directions=true` requires `collection_strategy=standard`"
                );
                ensure!(
                    !self.config.alternate_split_kinds || self.config.split_kind.is_none(),
                    "`alternate_split_kinds=true` chooses the split kind per variant; it cannot be combined with a fixed `split_kind`"
                );
                ensure!(
                    self.config.flaw_kind == FlawKind::Progression,
                    "`collection_strategy=goal_focused` refines along executed concrete plans and supports flaw_kind=progression, not flaw_kind={}",
                    self.config.flaw_kind
                );
                ensure!(
                    self.config.max_collection_size > 0,
                    "`max_collection_size` must be > 0"
                );
                ensure!(
                    self.config.single_goal_tasks || !self.config.progress_states,
                    "{FULL_TASK_PROGRESS_STATES_ERROR}"
                );
            }
        }
        Ok(())
    }

    fn create_rng(&self) -> SmallRng {
        SmallRng::seed_from_u64(
            self.config
                .random_seed
                .unwrap_or(crate::evaluation::DEFAULT_RANDOM_SEED),
        )
    }

    fn build_cegar_config(
        &self,
        max_states: usize,
        remaining_time: f64,
        variable_choice: MemberVariableChoice,
        random_seed: u64,
        split_kind: Option<SplitKind>,
    ) -> CegarConfig {
        let MemberVariableChoice {
            init_split_var_ids,
            blacklisted_prop_var_ids,
            blacklisted_numeric_var_ids,
        } = variable_choice;
        CegarConfig {
            max_states,
            max_iterations: CegarConfig::default().max_iterations,
            max_time: if remaining_time.is_finite() {
                Some(Duration::from_secs_f64(remaining_time.max(0.0)))
            } else {
                None
            },
            use_wildcard_plans: self.config.use_wildcard_plans,
            combine_labels: self.config.combine_labels,
            debug: self.config.debug,
            random_seed: Some(random_seed),
            flaw_kind: self.config.flaw_kind,
            flaw_scope: self.config.flaw_scope,
            split_kind: split_kind.unwrap_or(SplitKind::Concrete),
            split_selection: self.config.split_selection,
            init_split_var_ids,
            blacklisted_prop_var_ids,
            blacklisted_numeric_var_ids,
            compute_source_sets: self.config.compute_source_sets,
            max_refined_comparison_vars_per_abstraction: self
                .config
                .max_refined_comparison_vars_per_abstraction,
            flaw_selection_rank: None,
        }
    }

    pub fn generate_collection(
        &self,
        task: &dyn AbstractNumericTask,
    ) -> Result<Vec<DomainAbstraction>> {
        self.validate_supported_options()?;
        let abstractions = match self.config.collection_strategy {
            CollectionStrategy::Standard => self.generate_standard_collection(task)?,
            CollectionStrategy::GoalFocused => self.generate_goal_focused_collection(task)?,
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
                alternate_split_kinds: self.config.alternate_split_kinds,
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

    /// Full-task members with random restarts until the size, time,
    /// stagnation, or memory limit, or until a member settles the task: its
    /// abstract plan is a concrete plan, so it is perfect along an optimal
    /// plan, or the initial state is an abstract dead end, so the task is
    /// unsolvable. No further member can help the search then.
    fn generate_standard_collection(
        &self,
        task: &dyn AbstractNumericTask,
    ) -> Result<Vec<DomainAbstraction>> {
        let mut rng = self.create_rng();
        let blacklist_candidates = collect_initial_split_candidate_var_ids(task);

        let start = Instant::now();
        let mut remaining_collection_size = self.config.max_collection_size;
        let mut generated_keys: HashSet<AbstractionKey> = HashSet::new();
        let mut generated_abstractions: Vec<DomainAbstraction> = Vec::new();
        let mut time_point_of_last_new_abstraction = 0.0f64;
        let mut blacklisting = false;
        let blacklist_start_time =
            self.config.total_max_time * self.config.blacklist_trigger_percentage;
        let mut iteration = 1usize;
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
            let remaining_generation_time = self.config.max_time.min(remaining_total_time);
            let remaining_abstraction_size = remaining_collection_size.min(self.config.max_states);

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

            let blacklisted_var_ids = if blacklisting {
                sample_blacklisted_variables(&blacklist_candidates, &mut rng)
            } else {
                HashSet::new()
            };
            let (blacklisted_prop_var_ids, blacklisted_numeric_var_ids) =
                split_blacklisted_variables(task, blacklisted_var_ids);
            let split_kind = self.split_kind_for_iteration(iteration);
            let cegar_config = self.build_cegar_config(
                remaining_abstraction_size,
                remaining_generation_time,
                MemberVariableChoice {
                    init_split_var_ids: self.initial_split_var_ids(task, iteration),
                    blacklisted_prop_var_ids,
                    blacklisted_numeric_var_ids,
                },
                rng.next_u64(),
                split_kind,
            );
            let generator = DomainAbstractionGenerator::new(cegar_config)
                .context("failed to construct single-abstraction CEGAR generator")?;
            let generation_start = Instant::now();
            debug!(
                "domain abstraction collection: starting CEGAR generation iteration {}, remaining_generation_time={:.2}s, flaw_kind={}, flaw_scope={}, split_kind={:?}",
                iteration,
                remaining_generation_time,
                self.config.flaw_kind,
                self.config.flaw_scope,
                split_kind,
            );
            let mut abstraction = match generator.generate(task) {
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
            abstraction.metadata = DomainAbstractionMetadata {
                collection_iteration: Some(iteration),
                collection_strategy: Some(self.config.collection_strategy.to_string()),
                flaw_kind: Some(self.config.flaw_kind),
                flaw_scope: Some(self.config.flaw_scope),
                full_goal_task: Some(true),
                abstraction_use: AbstractionUse::CollectionMember,
                max_states: Some(remaining_abstraction_size),
                ..std::mem::take(&mut abstraction.metadata)
            };

            let abstraction_size = compute_abstraction_size_u128(
                abstraction.factory.domain_sizes(),
                abstraction.factory.numeric_domain_sizes(),
            )
            .unwrap_or(u128::MAX);

            let settled_by = if abstraction.metadata.solved_by_self {
                Some("concrete plan")
            } else if abstraction.metadata.stop_reason == Some(CegarStopReason::DeadEnd) {
                Some("initial state is a dead end")
            } else {
                None
            };
            let abstraction_key = AbstractionKey::from_abstraction(&abstraction);
            let elapsed_after_generation = start.elapsed().as_secs_f64();
            if generated_keys.insert(abstraction_key) {
                time_point_of_last_new_abstraction = elapsed_after_generation;
                let consumed = abstraction_size.min(remaining_collection_size as u128) as usize;
                remaining_collection_size = remaining_collection_size.saturating_sub(consumed);
                if self.config.debug {
                    log_collection_abstraction_debug(
                        generated_abstractions.len(),
                        &abstraction,
                        task,
                    );
                }
                generated_abstractions.push(abstraction);
                debug!(
                    "domain abstraction collection: added abstraction at iteration {}, abstraction_size={}, elapsed={:.2}s, remaining_collection_size={}, next_max_abstraction_size={}, remaining_generation_time={:.2}s, blacklisting={}",
                    iteration,
                    abstraction_size,
                    start.elapsed().as_secs_f64(),
                    remaining_collection_size,
                    remaining_collection_size.min(self.config.max_states),
                    remaining_generation_time,
                    blacklisting
                );
            }

            if let Some(reason) = settled_by {
                stop_reason = reason;
                break;
            }
            let stagnated = elapsed_after_generation - time_point_of_last_new_abstraction
                > self.config.stagnation_limit;
            if remaining_collection_size == 0 {
                stop_reason = "collection size limit";
                break;
            }
            if self.config.total_max_time.is_finite()
                && elapsed_after_generation >= self.config.total_max_time
            {
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

            iteration += 1;
        }

        if generated_abstractions.is_empty() {
            bail!(
                "the domain collection failed to generate its mandatory first abstraction ({stop_reason})"
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
        Ok(generated_abstractions)
    }

    /// The variable member `iteration` splits initially: the candidates take
    /// turns.
    fn initial_split_var_ids(
        &self,
        task: &dyn AbstractNumericTask,
        iteration: usize,
    ) -> Option<HashSet<VariableIndex>> {
        let candidate_var_ids = collect_initial_split_candidate_var_ids(task);
        Some(
            candidate_var_ids
                .get(iteration % candidate_var_ids.len().max(1))
                .copied()
                .into_iter()
                .collect(),
        )
    }

    fn split_kind_for_iteration(&self, iteration: usize) -> Option<SplitKind> {
        if !self.config.interleave_split_directions {
            return self.config.split_kind;
        }
        assert!(iteration > 0, "collection iterations are one-based");
        Some(if iteration % 2 == 1 {
            SplitKind::Concrete
        } else {
            SplitKind::Wanted
        })
    }
}

/// Builds the members of a goal-focused domain collection.
struct GoalFocusedDomainBackend<'config> {
    config: &'config DomainAbstractionCollectionGeneratorMultipleCegarConfig,
    member_max_time: Option<Duration>,
}

impl GoalFocusedDomainBackend<'_> {
    /// CEGAR configuration for one member: the collection's options, capped by
    /// what is left of the collection's budgets and specialised for the
    /// variant. Members start from the trivial abstraction: no initial splits
    /// and no blacklisted variables.
    fn member_cegar_config(&self, variant: &GoalVariant, budget: MemberBudget) -> CegarConfig {
        CegarConfig {
            max_states: self.config.max_states.min(budget.remaining_states),
            max_iterations: CegarConfig::default().max_iterations,
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
            max_refined_comparison_vars_per_abstraction: self
                .config
                .max_refined_comparison_vars_per_abstraction,
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
        root: Option<&ConcreteState>,
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

fn collect_logic_axiom_effect_vars(task: &dyn AbstractNumericTask) -> HashSet<VariableIndex> {
    task.axioms().iter().map(|axiom| axiom.var_id()).collect()
}

/// The propositional variables the goal names.
///
/// This used to be the bodies of the rules deriving a derived goal, keyed by the
/// goal's variable, and dropped the goal entirely when no rule derived that value.
/// `validate_abstractable_goal` refuses such a goal, so what is left is the goal
/// variables themselves -- and for a numeric goal that is the comparison variable
/// the rule body used to hold.
fn goal_var_ids(task: &dyn AbstractNumericTask) -> HashSet<VariableIndex> {
    (0..task.get_num_goals())
        .map(|goal_id| task.get_goal_fact(goal_id).var_index())
        .collect()
}

/// The variables initial splits and blacklisting choose from: every
/// propositional variable no axiom writes, except condition variables that no
/// goal mentions, and every regular numeric variable (encoded after the
/// propositional ones).
fn collect_initial_split_candidate_var_ids(task: &dyn AbstractNumericTask) -> Vec<VariableIndex> {
    let goal_related = goal_var_ids(task);
    let logic_axiom_effect_vars = collect_logic_axiom_effect_vars(task);
    let mut candidates: Vec<VariableIndex> = (0..task.variables().len())
        .map(VariableIndex::from_usize)
        .filter(|var_id| {
            !logic_axiom_effect_vars.contains(var_id)
                && (!task.numeric_conditions().is_condition_var(*var_id)
                    || goal_related.contains(var_id))
        })
        .collect();
    let encoded_numeric_offset = task.variables().len();
    candidates.extend(
        task.numeric_variables()
            .iter()
            .enumerate()
            .filter(|(_, variable)| variable.get_type() == &NumericType::Regular)
            .map(|(numeric_var_id, _)| {
                VariableIndex::from_usize(encoded_numeric_offset + numeric_var_id)
            }),
    );
    candidates
}

fn split_blacklisted_variables(
    task: &dyn AbstractNumericTask,
    encoded_var_ids: HashSet<VariableIndex>,
) -> (HashSet<VariableIndex>, HashSet<VariableIndex>) {
    let num_prop_vars = task.variables().len();
    let mut blacklisted_prop_var_ids = HashSet::new();
    let mut blacklisted_numeric_var_ids = HashSet::new();

    for encoded_var_id in encoded_var_ids {
        if encoded_var_id.index() < num_prop_vars {
            blacklisted_prop_var_ids.insert(encoded_var_id);
        } else {
            let numeric_var_id = encoded_var_id.index() - num_prop_vars;
            if numeric_var_id < task.numeric_variables().len() {
                blacklisted_numeric_var_ids.insert(VariableIndex::from_usize(numeric_var_id));
            }
        }
    }

    (blacklisted_prop_var_ids, blacklisted_numeric_var_ids)
}

fn sample_blacklisted_variables<R: rand::Rng + ?Sized>(
    candidates: &[VariableIndex],
    rng: &mut R,
) -> HashSet<VariableIndex> {
    if candidates.is_empty() {
        return HashSet::new();
    }

    let blacklist_size = rng.gen_range(1..=candidates.len());
    let mut shuffled = candidates.to_vec();
    shuffled.shuffle(rng);
    shuffled.into_iter().take(blacklist_size).collect()
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
