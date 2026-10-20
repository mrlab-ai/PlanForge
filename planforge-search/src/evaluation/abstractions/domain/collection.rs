#[cfg(test)]
mod tests;

use anyhow::{Context, Result};

use planforge_sas::numeric_task::{AbstractNumericTask, VariableIndex};
use tracing::info;

use crate::evaluation::abstractions::collections::goal_focused::{
    BuiltReport, CollectionMembership, GoalFocusedBackend, GoalVariant, MemberBudget,
    build_goal_focused_collection,
};
use crate::evaluation::abstractions::collections::standard::{
    StandardBackend, StandardMember, build_standard_collection,
};
use crate::evaluation::abstractions::collections::strategy::CollectionStrategy;
use crate::evaluation::abstractions::common::cegar::CegarStopReason;
use crate::evaluation::abstractions::common::task::AbstractionUse;
use crate::evaluation::abstractions::domain::abstract_operators::DomainMapping;
use planforge_sas::explicit_state::ExplicitState;

use super::abstraction::compute_abstraction_size_u128;
use super::abstraction::{DomainAbstraction, DomainAbstractionMetadata};
use super::config::DomainAbstractionCollectionConfig;
use super::generator::DomainAbstractionGenerator;
use crate::evaluation::abstractions::collections::config::MemberSpecialization;
use planforge_sas::utils::float_tolerance;
use planforge_sas::utils::interval::Interval;

#[derive(Debug, Clone)]
pub struct DomainAbstractionCollectionGenerator {
    config: DomainAbstractionCollectionConfig,
}

impl DomainAbstractionCollectionGenerator {
    pub fn new(config: DomainAbstractionCollectionConfig) -> Result<Self> {
        config.validate()?;
        DomainAbstractionGenerator::new(config.member.clone())?;
        Ok(Self { config })
    }

    pub fn config(&self) -> &DomainAbstractionCollectionConfig {
        &self.config
    }

    pub fn generate(&self, task: &dyn AbstractNumericTask) -> Result<Vec<DomainAbstraction>> {
        let abstractions = match self.config.collection_strategy {
            CollectionStrategy::Standard => self.generate_standard_collection(task)?,
            CollectionStrategy::PerGoal | CollectionStrategy::GoalFocused => {
                self.generate_goal_focused_collection(task)?
            }
        };
        if self.config.member.cegar.debug {
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
        };
        let mut abstractions =
            build_goal_focused_collection(task, &self.config.goal_focused_schedule(), &backend)?;
        for (index, abstraction) in abstractions.iter_mut().enumerate() {
            abstraction.metadata.collection_iteration = Some(index + 1);
            if self.config.member.cegar.debug {
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
            &self.config.standard_schedule(),
            &StandardDomainBackend { generator: self },
        )?;
        if self.config.member.cegar.debug {
            for (index, abstraction) in abstractions.iter().enumerate() {
                log_collection_abstraction_debug(index, abstraction, task);
            }
        }
        Ok(abstractions)
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
            DomainAbstractionGenerator::new(config.member.for_restart(member))?.generate(task)?;
        abstraction.metadata = DomainAbstractionMetadata {
            collection_iteration: Some(member.iteration),
            collection_strategy: Some(config.collection_strategy.to_string()),
            flaw_kind: Some(config.member.cegar.flaw_kind),
            flaw_scope: Some(config.member.cegar.flaw_scope),
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
        let cegar_config = self
            .config
            .member
            .for_variant(variant, self.config.split_kind, budget);
        let max_states = cegar_config.cegar.max_states;
        let mut abstraction = DomainAbstractionGenerator::new(cegar_config)
            .context("failed to construct single-abstraction CEGAR generator")?
            .generate_from_root(task, root)?;
        let metadata = &mut abstraction.metadata;
        metadata.collection_strategy = Some(self.config.collection_strategy.to_string());
        metadata.flaw_kind = Some(self.config.member.cegar.flaw_kind);
        metadata.flaw_scope = Some(self.config.member.cegar.flaw_scope);
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
