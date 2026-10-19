use std::cell::{Ref, RefCell};
use std::time::Instant;

use anyhow::{Context, Result, ensure};

use planforge_sas::numeric_task::{AbstractNumericTask, OperatorIndex};

use super::abstract_operator_generator::AbstractOperator;
use super::cegar::{Cegar, CegarConfig, CegarStopReason, SplitDirection};
use super::factory::{AbstractDistanceTable, DistanceTableOptions, DomainAbstractionFactory};
use crate::evaluation::abstractions::collections::goal_focused::CollectionMembership;
use crate::evaluation::abstractions::common::concrete_state::ConcreteState;
use crate::evaluation::abstractions::common::task::{AbstractionUse, RootedTask};
use crate::evaluation::abstractions::cost_partitioning::costs::{
    AbstractOperatorSourceSets, AbstractTransitionSystem,
};

/// Fully built abstraction artifact that can be used to evaluate concrete states.
#[derive(Debug, Clone)]
pub struct DomainAbstraction {
    pub factory: DomainAbstractionFactory,
    pub distance_table: AbstractDistanceTable,
    pub hash_multipliers: Vec<usize>,
    pub combine_labels: bool,
    pub relevant_operator_ids: Vec<OperatorIndex>,
    pub abstract_operators: Vec<AbstractOperator>,
    pub abstract_source_sets: Vec<AbstractOperatorSourceSets>,
    pub(crate) interval_transition_system: RefCell<Option<AbstractTransitionSystem>>,
    pub metadata: DomainAbstractionMetadata,
}

impl DomainAbstraction {
    pub fn lookup_only(
        factory: DomainAbstractionFactory,
        distance_table: AbstractDistanceTable,
        hash_multipliers: Vec<usize>,
        combine_labels: bool,
        metadata: DomainAbstractionMetadata,
    ) -> Self {
        Self {
            factory,
            distance_table,
            hash_multipliers,
            combine_labels,
            relevant_operator_ids: Vec::new(),
            abstract_operators: Vec::new(),
            abstract_source_sets: Vec::new(),
            interval_transition_system: RefCell::new(None),
            metadata,
        }
    }

    pub fn task_for_factory<'task>(
        &'task self,
        fallback: &'task dyn AbstractNumericTask,
    ) -> &'task dyn AbstractNumericTask {
        fallback
    }

    pub fn discard_transition_data(&mut self) {
        self.abstract_operators.clear();
        self.abstract_source_sets.clear();
        self.interval_transition_system.get_mut().take();
    }

    pub fn ensure_abstract_source_sets(&mut self, task: &dyn AbstractNumericTask) -> Result<()> {
        if !self.abstract_source_sets.is_empty() {
            ensure!(
                self.abstract_source_sets.len() == self.abstract_operators.len(),
                "domain abstraction has {} operator regions for {} abstract operators",
                self.abstract_source_sets.len(),
                self.abstract_operators.len()
            );
            return Ok(());
        }
        ensure!(
            !self.abstract_operators.is_empty(),
            "cannot construct regional operator regions after abstract operators were discarded"
        );
        self.abstract_source_sets = self
            .factory
            .build_abstract_source_sets(task, &self.abstract_operators)
            .context("failed to build abstract-operator regions")?;
        ensure!(
            self.abstract_source_sets.len() == self.abstract_operators.len(),
            "domain operator-region construction produced {} entries for {} abstract operators",
            self.abstract_source_sets.len(),
            self.abstract_operators.len()
        );
        Ok(())
    }

    pub fn interval_transition_system<'a>(
        &'a self,
        task: &dyn AbstractNumericTask,
        deadline: Option<Instant>,
    ) -> Result<Ref<'a, AbstractTransitionSystem>> {
        if self.interval_transition_system.borrow().is_none() {
            let mut transition_system = self
                .factory
                .build_abstract_transition_system_from_operators(
                    task,
                    self.combine_labels,
                    &self.abstract_operators,
                    DistanceTableOptions::default()
                        .without_cartesian_sets()
                        .with_deadline(deadline),
                )?;
            transition_system.forward.clear();
            *self.interval_transition_system.borrow_mut() = Some(transition_system);
        }
        Ok(Ref::map(
            self.interval_transition_system.borrow(),
            |transition_system| {
                transition_system
                    .as_ref()
                    .expect("regional transition system was initialized")
            },
        ))
    }

    pub fn lookup_clone(&self) -> Self {
        let mut abstraction = self.clone();
        abstraction.discard_transition_data();
        abstraction
    }
}

#[derive(Debug, Clone, Default)]
pub struct DomainAbstractionMetadata {
    pub collection_iteration: Option<usize>,
    pub collection_strategy: Option<String>,
    pub flaw_kind: Option<String>,
    pub full_goal_task: Option<bool>,
    pub abstraction_use: AbstractionUse,
    pub stop_reason: Option<CegarStopReason>,
    pub max_abstraction_size: Option<usize>,
    /// CEGAR exited because the wildcard plan has no flaws. This proves
    /// `h(init)` optimal only when `abstraction_use` is `Standalone`;
    /// collection combinators deliberately do not expose that search shortcut.
    pub solved_by_self: bool,
    /// The split direction CEGAR refined with.
    pub split_direction: SplitDirection,
    /// The rank CEGAR used among tied best flaws, if any.
    pub flaw_selection_rank: Option<usize>,
    /// Operator ids of the concrete plan CEGAR executed when it solved the task
    /// from its refinement root.
    pub concrete_plan_operator_ids: Option<Vec<OperatorIndex>>,
    /// Set when the abstraction is a member of a goal-focused collection.
    pub collection_member: Option<CollectionMembership>,
}

/// Numeric-fd style generator that constructs a domain abstraction via CEGAR.
#[derive(Debug, Clone)]
pub struct DomainAbstractionGenerator {
    cegar: Cegar,
    config: CegarConfig,
}

impl DomainAbstractionGenerator {
    pub fn new(config: CegarConfig) -> Result<Self> {
        let cegar = Cegar::new(config.clone()).context("failed to construct CEGAR")?;
        Ok(Self { cegar, config })
    }

    pub fn config(&self) -> &CegarConfig {
        &self.config
    }

    /// Builds a domain abstraction and its abstract distance table.
    pub fn generate(&self, task: &dyn AbstractNumericTask) -> Result<DomainAbstraction> {
        self.generate_from_root(task, None)
    }

    /// Builds a domain abstraction whose CEGAR loop searches for flaws from
    /// `refinement_root` instead of the task initial state.
    ///
    /// The root only chooses counterexamples. The abstraction still partitions
    /// the full state space of `task`, operator costs stay those of `task`, and
    /// the distance table's initial state is the initial state of `task`.
    pub(crate) fn generate_from_root(
        &self,
        task: &dyn AbstractNumericTask,
        refinement_root: Option<&ConcreteState>,
    ) -> Result<DomainAbstraction> {
        let outcome = match refinement_root {
            Some(root) => {
                let rooted_task = RootedTask::new(task, root)?;
                self.cegar.build_abstraction(&rooted_task)
            }
            None => self.cegar.build_abstraction(task),
        }
        .context("CEGAR failed to build abstraction")?;
        let solved_by_self = outcome.solved_by_self;
        let stop_reason = outcome.stop_reason;
        let concrete_plan_operator_ids = outcome.concrete_plan_operator_ids;
        let factory = outcome.final_state.factory;
        let mut operator_generator =
            factory.make_operator_generator(task, self.config.combine_labels)?;
        let abstract_operators = operator_generator
            .build_abstract_operators_with_deadline(task, None)
            .context("failed to build abstract operators")?;
        let abstract_source_sets = if self.config.compute_source_sets {
            factory
                .build_abstract_source_sets(task, &abstract_operators)
                .context("failed to build abstract-operator regions")?
        } else {
            // Heuristics that read only the distance table (canonical, max,
            // single domain abstraction) do not consume operator regions; skipping
            // saves ~12 GB on minecraft-sword-advanced/prob_30x30_5. SCP /
            // Callers that need regional SCP can construct operator regions from
            // the finalized abstraction after collection generation.
            Vec::new()
        };
        let distance_table = factory
            .build_distance_table_with_operators(
                task,
                &operator_generator,
                &abstract_operators,
                false,
            )
            .context("failed to build abstract distance table")?;
        let initial_h = distance_table
            .distances
            .get(distance_table.initial_state_hash)
            .copied()
            .with_context(|| {
                format!(
                    "abstract initial state hash {} out of bounds for distance table of length {}",
                    distance_table.initial_state_hash,
                    distance_table.distances.len()
                )
            })?;
        ensure!(
            initial_h.is_finite(),
            "domain abstraction initial state is abstract-dead after CEGAR; initial_hash={}, states={}, abstract_ops={}, prop_domains={:?}, numeric_domains={:?}",
            distance_table.initial_state_hash,
            distance_table.distances.len(),
            abstract_operators.len(),
            factory.domain_sizes(),
            factory.numeric_domain_sizes()
        );
        let hash_multipliers =
            compute_hash_multipliers(factory.domain_sizes(), factory.numeric_domain_sizes())
                .context("failed to compute hash multipliers")?;
        let relevant_operator_ids = factory
            .relevant_operator_ids_from_operators(
                task,
                self.config.combine_labels,
                &abstract_operators,
                DistanceTableOptions::default(),
            )
            .context("failed to compute relevant operator ids")?;

        Ok(DomainAbstraction {
            factory,
            distance_table,
            hash_multipliers,
            combine_labels: self.config.combine_labels,
            relevant_operator_ids,
            abstract_operators,
            abstract_source_sets,
            interval_transition_system: RefCell::new(None),
            metadata: DomainAbstractionMetadata {
                solved_by_self,
                abstraction_use: AbstractionUse::Standalone,
                stop_reason: Some(stop_reason),
                split_direction: self.config.effective_split_direction(),
                flaw_selection_rank: self.config.flaw_selection_rank,
                concrete_plan_operator_ids,
                ..DomainAbstractionMetadata::default()
            },
        })
    }
}

/// Computes mixed-radix hash multipliers for propositional and numeric variables.
///
/// This mirrors the hashing scheme used by `AbstractOperatorGenerator`.
pub fn compute_hash_multipliers(
    domain_sizes: &[usize],
    numeric_domain_sizes: &[usize],
) -> Result<Vec<usize>> {
    let total = domain_sizes
        .len()
        .checked_add(numeric_domain_sizes.len())
        .context("variable count overflow")?;
    ensure!(total > 0, "cannot compute hash multipliers for 0 variables");

    let mut multipliers: Vec<usize> = vec![0; total];
    let mut mult: usize = 1;
    for idx in 0..total {
        multipliers[idx] = mult;

        let radix: usize = if idx < domain_sizes.len() {
            domain_sizes[idx]
        } else {
            let n = idx - domain_sizes.len();
            *numeric_domain_sizes
                .get(n)
                .context("numeric domain size out of range")?
        };

        mult = mult
            .checked_mul(radix)
            .context("hash multiplier overflow")?;
    }

    Ok(multipliers)
}
