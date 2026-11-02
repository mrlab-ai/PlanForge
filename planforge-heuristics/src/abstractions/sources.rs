//! The abstraction sources a heuristic is built from -- single abstractions,
//! collections and pattern databases -- and their construction under a shared
//! time budget.

use planforge_sas::numeric_task::Task;
use std::time::{Duration, Instant};

use planforge_search::config::{ApplyOptions, ConfigError, TimeLimit};
use planforge_search::heuristic_factory::{EmptyConfig, HeuristicBuildError, TaskRequirements};

use crate::requirements::{ABSTRACTABLE_GOAL, RESTRICTED};
use planforge_search::resource_limits::{Deadline, min_time_limit};
use planforge_search::task_transformations::TransformationFeature;

use crate::abstractions::cartesian::CartesianAbstractionGenerator;
use crate::abstractions::collections::config::{CollectionConfig, MemberConfig};
use crate::abstractions::collections::generator::{AbstractionGenerator, CollectionGenerator};
use crate::abstractions::common::abstraction::CegarAbstraction;
use crate::abstractions::component::AbstractionComponent;
use crate::abstractions::domain::generator::DomainAbstractionGenerator;
use crate::abstractions::icaps26::{Icaps26CartesianConfig, Icaps26CartesianGenerator};
use crate::abstractions::pdb::pattern_collection::PatternCollection;
use crate::abstractions::pdb::pattern_database::{
    PatternDatabase, PdbBuildConfig, PdbHeuristicConfig,
};
use crate::abstractions::pdb::pattern_generator_greedy::{
    GreedyPatternGeneratorConfig, generate_greedy_pattern,
};
use crate::abstractions::pdb::pattern_generator_hillclimbing::{
    HillClimbingConfig, generate_hill_climbing_pdbs,
};
use crate::abstractions::pdb::pattern_generator_systematic::{
    SystematicPatternGeneratorConfig, generate_systematic_patterns,
};
use crate::abstractions::pdb::pdb_collection::build_pattern_databases;
use tracing::info;

use crate::abstractions::cartesian::{
    CartesianAbstractionCollectionConfig, CartesianAbstractionConfig,
};
use crate::abstractions::domain::config::{
    DomainAbstractionCollectionConfig, DomainAbstractionConfig,
};
use crate::abstractions::pdb::pattern_database::PdbInternalHeuristic;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbstractionPurpose {
    Standalone,
    LabelCostPartitioning,
    RegionalCostPartitioning,
}

impl AbstractionPurpose {
    pub(crate) fn needs_source_sets(self) -> bool {
        matches!(self, Self::RegionalCostPartitioning)
    }
    pub(crate) fn needs_transition_system(self) -> bool {
        !matches!(self, Self::Standalone)
    }
}

#[derive(Debug, Clone, planforge_search::config::FromOptionValue)]
pub enum AbstractionSource {
    #[option(rename = "domain")]
    Domain(DomainAbstractionConfig),
    #[option(rename = "domain_collection")]
    DomainCollection(DomainAbstractionCollectionConfig),
    #[option(rename = "cartesian")]
    Cartesian(CartesianAbstractionConfig),
    #[option(rename = "cartesian_collection")]
    CartesianCollection(CartesianAbstractionCollectionConfig),
    #[option(rename = "pdb")]
    Pdb(PdbSourceConfig),
    #[option(rename = "icaps26_cartesian")]
    Icaps26Cartesian(Icaps26CartesianConfig),
}

impl AbstractionSource {
    pub fn validate(&self) -> Result<(), ConfigError> {
        let result = match self {
            Self::Domain(config) => config.cegar.validate(),
            Self::Cartesian(config) => config.cegar.validate(),
            Self::DomainCollection(config) => config
                .validate()
                .and_then(|_| config.member.cegar.validate()),
            Self::CartesianCollection(config) => config
                .validate()
                .and_then(|_| config.member.cegar.validate()),
            Self::Icaps26Cartesian(config) => {
                Icaps26CartesianGenerator::new(config.clone()).map(|_| ())
            }
            Self::Pdb(config) => match &config.patterns {
                PdbPatternsConfig::Systematic(patterns) => {
                    patterns.validate().map_err(anyhow::Error::msg)
                }
                PdbPatternsConfig::HillClimbing(patterns) => {
                    patterns.validate().map_err(anyhow::Error::msg)
                }
                PdbPatternsConfig::Greedy(_) => Ok(()),
            },
        };
        result.map_err(|error| ConfigError::from(error.to_string()).at(self.name()))
    }

    pub fn requirements(&self) -> TaskRequirements {
        let restricted = matches!(self, Self::Pdb(_) | Self::Icaps26Cartesian(_));
        let icaps26 = matches!(self, Self::Icaps26Cartesian(_));
        TaskRequirements {
            transformations: icaps26
                .then_some(
                    <crate::abstractions::icaps26::Icaps26RestrictFeature as TransformationFeature>::NAME,
                )
                .into_iter()
                .collect(),
            checks: restricted
                .then_some(RESTRICTED)
                .into_iter()
                .chain([ABSTRACTABLE_GOAL])
                .collect(),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Domain(_) => "domain",
            Self::DomainCollection(_) => "domain_collection",
            Self::Cartesian(_) => "cartesian",
            Self::CartesianCollection(_) => "cartesian_collection",
            Self::Pdb(_) => "pdb",
            Self::Icaps26Cartesian(_) => "icaps26_cartesian",
        }
    }

    pub fn build(
        &self,
        task: &Task,
        purpose: AbstractionPurpose,
        remaining: Option<Duration>,
    ) -> Result<Vec<AbstractionComponent>, HeuristicBuildError> {
        self.requirements().validate(task)?;
        self.build_validated(task, purpose, remaining)
    }

    fn build_validated(
        &self,
        task: &Task,
        purpose: AbstractionPurpose,
        remaining: Option<Duration>,
    ) -> Result<Vec<AbstractionComponent>, HeuristicBuildError> {
        match self {
            Self::Domain(config) => build_cegar_source::<DomainAbstractionGenerator>(
                task,
                config.clone(),
                purpose,
                remaining,
            ),
            Self::DomainCollection(config) => build_cegar_collection_source::<
                DomainAbstractionGenerator,
            >(
                task, config.clone(), purpose, remaining
            ),
            Self::Cartesian(config) => build_cegar_source::<CartesianAbstractionGenerator>(
                task,
                config.clone(),
                purpose,
                remaining,
            ),
            Self::CartesianCollection(config) => build_cegar_collection_source::<
                CartesianAbstractionGenerator,
            >(
                task, config.clone(), purpose, remaining
            ),
            Self::Icaps26Cartesian(config) => {
                let mut config = config.clone();
                config.compute_source_sets = purpose.needs_source_sets();
                config.retain_transition_system = purpose.needs_transition_system();
                config.max_time = min_time_limit(config.max_time, remaining);
                let abstraction = Icaps26CartesianGenerator::new(config)
                    .and_then(|generator| generator.generate(task))
                    .map_err(|error| {
                        format!("failed to build ICAPS 2026 Cartesian source: {error:#}")
                    })?;
                Ok(vec![AbstractionComponent::from(abstraction)])
            }
            Self::Pdb(config) => {
                let pdbs = config
                    .build(task, purpose.needs_transition_system(), remaining)
                    .map_err(|error| format!("failed to build numeric PDB source: {error}"))?;
                Ok(pdbs.into_iter().map(AbstractionComponent::from).collect())
            }
        }
    }
}

#[derive(Debug, Clone, planforge_search::config::FromOptionValue)]
enum PdbPatternsConfig {
    #[option(rename = "systematic")]
    Systematic(SystematicPatternGeneratorConfig),
    #[option(rename = "greedy")]
    Greedy(GreedyPatternGeneratorConfig),
    #[option(rename = "hillclimbing")]
    HillClimbing(HillClimbingConfig),
}
impl Default for PdbPatternsConfig {
    fn default() -> Self {
        Self::Systematic(SystematicPatternGeneratorConfig::default())
    }
}

/// A pattern collection and the heuristics its pattern databases use inside
/// their projections.
#[derive(Debug, Clone, Default, planforge_search::config::ApplyOptions)]
pub struct PdbSourceConfig {
    patterns: PdbPatternsConfig,
    exploration_heuristic: PdbInternalHeuristic,
    frontier_heuristic: PdbInternalHeuristic,
    failed_lookup_heuristic: PdbInternalHeuristic,
}

impl PdbSourceConfig {
    fn build(
        &self,
        task: &Task,
        retain_transitions: bool,
        remaining: Option<Duration>,
    ) -> Result<Vec<PatternDatabase>, String> {
        let heuristics = PdbHeuristicConfig {
            exploration_heuristic: self.exploration_heuristic,
            frontier_heuristic: self.frontier_heuristic,
            failed_lookup_heuristic: self.failed_lookup_heuristic,
        };
        let pdb_config = |max_states| PdbBuildConfig {
            max_states,
            heuristics,
            retain_transitions,
        };
        match self.patterns {
            PdbPatternsConfig::Systematic(config) => build_pattern_databases(
                task,
                &generate_systematic_patterns(task, config),
                pdb_config(config.max_pdb_states),
            ),
            PdbPatternsConfig::Greedy(config) => build_pattern_databases(
                task,
                &PatternCollection::new(vec![generate_greedy_pattern(task, config)]),
                pdb_config(config.max_pdb_states),
            ),
            PdbPatternsConfig::HillClimbing(config) => {
                generate_hill_climbing_pdbs(task, config, heuristics, retain_transitions, remaining)
            }
        }
    }
}

/// The configuration of a combination of abstraction heuristics: its
/// sources, given first, the construction budget they share, and the
/// combination's own options.
#[derive(Debug, Clone, Default, planforge_search::config::ApplyOptions)]
pub struct CombinationConfig<O: ApplyOptions = EmptyConfig> {
    #[option(positional)]
    pub sources: Vec<AbstractionSource>,
    pub construction_max_time: TimeLimit,
    #[option(flatten)]
    pub options: O,
}

impl<O: ApplyOptions> CombinationConfig<O> {
    /// Rejects an empty source list, an invalid source and a construction
    /// budget that is zero or beyond the clock.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.sources.is_empty() {
            return Err(
                ConfigError::from("at least one abstraction source is required")
                    .at_option("sources"),
            );
        }
        for (index, source) in self.sources.iter().enumerate() {
            source
                .validate()
                .map_err(|error| error.at(format!("[{index}]")).at_option("sources"))?;
        }
        self.deadline().map_err(|error| {
            ConfigError::from(error.to_string()).at_option("construction_max_time")
        })?;
        Ok(())
    }

    pub fn requirements(&self) -> TaskRequirements {
        self.sources
            .iter()
            .fold(TaskRequirements::default(), |requirements, source| {
                requirements.union(source.requirements())
            })
    }

    /// The deadline of the construction, starting now.
    pub(crate) fn deadline(&self) -> Result<Deadline, HeuristicBuildError> {
        let limit = self.construction_max_time.duration()?;
        if limit == Some(Duration::ZERO) {
            return Err(format!(
                "construction_max_time must be at least one nanosecond, got {}",
                self.construction_max_time.seconds()
            )
            .into());
        }
        Deadline::after(Instant::now(), limit)
            .map_err(|error| format!("invalid construction_max_time: {error}").into())
    }
}

pub(crate) fn build_components(
    task: &Task,
    sources: &[AbstractionSource],
    purpose: AbstractionPurpose,
    deadline: Deadline,
) -> Result<Vec<AbstractionComponent>, HeuristicBuildError> {
    assert!(
        !sources.is_empty(),
        "a configured abstraction collection has sources"
    );
    let construction_start = Instant::now();
    let mut components = Vec::new();
    for source in sources {
        let remaining = remaining_construction_time(deadline)?;
        let built = source.build_validated(task, purpose, remaining)?;
        if built.is_empty() {
            return Err(format!(
                "abstraction source `{}` produced no components",
                source.name()
            )
            .into());
        }
        components.extend(built);
    }
    let states = components
        .iter()
        .map(AbstractionComponent::num_states)
        .collect::<Vec<_>>();
    let total_states = states.iter().try_fold(0usize, |total, states| {
        total.checked_add(*states).ok_or_else(|| {
            "abstraction collection state count exceeds addressable memory".to_string()
        })
    })?;
    let abstract_operators: usize = components
        .iter()
        .filter_map(AbstractionComponent::cegar)
        .map(CegarAbstraction::num_abstract_operators)
        .sum();
    info!(
        "Abstraction collection: abstractions={}, total_states={}, states={:?}, abstract_operators={}, construction_time={:.3}s",
        components.len(),
        total_states,
        states,
        abstract_operators,
        construction_start.elapsed().as_secs_f64(),
    );

    Ok(components)
}

fn build_cegar_source<G>(
    task: &Task,
    mut config: G::Config,
    purpose: AbstractionPurpose,
    remaining: Option<Duration>,
) -> Result<Vec<AbstractionComponent>, HeuristicBuildError>
where
    G: AbstractionGenerator,
    G::Abstraction: Into<AbstractionComponent>,
{
    let cegar = config.cegar_mut();
    cegar.keep_transitions = purpose.needs_transition_system();
    cegar.max_time = min_time_limit(cegar.max_time.duration()?, remaining).into();
    info!("Building {} source...", G::KIND);
    let abstraction = G::new(config)
        .and_then(|generator| generator.generate(task))
        .map_err(|error| {
            HeuristicBuildError::from_source(error, format!("failed to build {}", G::KIND))
        })?;
    Ok(vec![abstraction.into()])
}

fn build_cegar_collection_source<G>(
    task: &Task,
    mut config: CollectionConfig<G::Config>,
    purpose: AbstractionPurpose,
    remaining: Option<Duration>,
) -> Result<Vec<AbstractionComponent>, HeuristicBuildError>
where
    G: AbstractionGenerator,
    G::Abstraction: 'static,
{
    config.member.cegar_mut().keep_transitions = purpose.needs_transition_system();
    if let Some(remaining) = remaining {
        config
            .cap_construction_time(remaining)
            .map_err(|error| format!("invalid construction time: {error}"))?;
    }
    info!("Building {} collection source...", G::KIND);
    let abstractions = CollectionGenerator::<G>::new(config)
        .and_then(|generator| generator.generate(task))
        .map_err(|error| {
            HeuristicBuildError::from_source(
                error,
                format!("failed to build {} collection", G::KIND),
            )
        })?;
    Ok(abstractions.into_iter().map(Into::into).collect())
}
#[cfg(test)]
mod tests;

pub(crate) fn remaining_construction_time(
    deadline: Deadline,
) -> Result<Option<Duration>, HeuristicBuildError> {
    match deadline.remaining() {
        Some(remaining) if remaining.is_zero() => Err(HeuristicBuildError::ConstructionTimeout),
        remaining => Ok(remaining),
    }
}
