use std::time::{Duration, Instant};

use crate::resource_limits::{Deadline, min_time_limit};

use crate::config::{
    ApplyOptions, ConfigArg, ConfigError, ConfigValue, FromOptionValue, TimeLimit, for_each_option,
};
use crate::evaluation::abstractions::cartesian::CartesianAbstractionGenerator;
use crate::evaluation::abstractions::collections::config::{CollectionConfig, MemberConfig};
use crate::evaluation::abstractions::collections::generator::{
    AbstractionGenerator, CollectionGenerator,
};
use crate::evaluation::abstractions::common::abstraction::CegarAbstraction;
use crate::evaluation::abstractions::component::AbstractionComponent;
use crate::evaluation::abstractions::cost_partitioning::saturated::ScpConfig;
use crate::evaluation::abstractions::domain::generator::DomainAbstractionGenerator;
use crate::evaluation::abstractions::icaps26::{Icaps26CartesianConfig, Icaps26CartesianGenerator};
use crate::evaluation::abstractions::pdb::canonical_pdb_heuristic::CanonicalNumericPdbConfig;
use crate::evaluation::abstractions::pdb::pattern_collection::PatternCollection;
use crate::evaluation::abstractions::pdb::pattern_database::PdbHeuristicConfig;
use crate::evaluation::abstractions::pdb::pattern_generator_greedy::{
    GreedyPatternGeneratorConfig, generate_greedy_pattern,
};
use crate::evaluation::abstractions::pdb::pattern_generator_systematic::{
    SystematicPatternGeneratorConfig, generate_systematic_patterns,
};
use crate::evaluation::abstractions::pdb::pdb_collection::PdbCollection;
use planforge_sas::numeric_task::AbstractNumericTask;
use tracing::info;

use super::HeuristicBuildError;

use super::TaskRequirements;
use crate::evaluation::abstractions::cartesian::{
    CartesianAbstractionCollectionConfig, CartesianAbstractionConfig,
};
use crate::evaluation::abstractions::domain::config::{
    DomainAbstractionCollectionConfig, DomainAbstractionConfig,
};
use crate::evaluation::abstractions::pdb::pattern_database::PdbInternalHeuristic;

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

#[derive(Debug, Clone, crate::config::FromOptionValue)]
pub enum AbstractionSource {
    #[option(rename = "domain")]
    Domain(DomainAbstractionConfig),
    #[option(rename = "domain_collection")]
    DomainCollection(DomainAbstractionCollectionConfig),
    #[option(rename = "cartesian")]
    Cartesian(CartesianAbstractionConfig),
    #[option(rename = "cartesian_collection")]
    CartesianCollection(CartesianAbstractionCollectionConfig),
    #[option(rename = "pdb", alias = "numeric_pdb")]
    Pdb(PdbSourceConfig),
    #[option(rename = "icaps26_cartesian")]
    Icaps26Cartesian(Icaps26CartesianConfig),
}

impl AbstractionSource {
    pub fn requirements(&self) -> TaskRequirements {
        TaskRequirements {
            restricted: matches!(self, Self::Pdb(_) | Self::Icaps26Cartesian(_)),
            abstractable_goal: true,
            icaps26_restriction: matches!(self, Self::Icaps26Cartesian(_)),
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

    pub fn build<'task>(
        &self,
        task: &'task dyn AbstractNumericTask,
        purpose: AbstractionPurpose,
        remaining: Option<Duration>,
    ) -> Result<Vec<AbstractionComponent<'task>>, HeuristicBuildError> {
        self.requirements().validate(task)?;
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
                let (patterns, max_pdb_states, heuristic_config) =
                    config.clone().build_patterns(task);
                let pdbs = PdbCollection::with_heuristic_config(
                    task,
                    patterns,
                    max_pdb_states,
                    heuristic_config,
                )
                .map_err(|error| format!("failed to build numeric PDB source: {error}"))?
                .into_pdbs();
                Ok(pdbs.into_iter().map(AbstractionComponent::from).collect())
            }
        }
    }
}

#[derive(Debug, Clone, crate::config::FromOptionValue)]
enum PdbPatternsConfig {
    #[option(rename = "systematic")]
    Systematic(SystematicPatternGeneratorConfig),
    #[option(rename = "greedy")]
    Greedy(GreedyPatternGeneratorConfig),
}
impl Default for PdbPatternsConfig {
    fn default() -> Self {
        Self::Systematic(SystematicPatternGeneratorConfig::default())
    }
}

#[derive(Debug, Clone, Default)]
pub struct PdbSourceConfig {
    patterns: PdbPatternsConfig,
    systematic_heuristic: PdbHeuristicConfig,
}

#[derive(Default, crate::config::ApplyOptions)]
struct ExplicitPdbSourceConfig {
    patterns: PdbPatternsConfig,
    exploration_heuristic: PdbInternalHeuristic,
    frontier_heuristic: PdbInternalHeuristic,
    failed_lookup_heuristic: PdbInternalHeuristic,
}

impl ApplyOptions for PdbSourceConfig {
    fn apply_options(&mut self, args: &[ConfigArg]) -> Result<(), crate::config::ConfigError> {
        // The old flat PDB form is a typed alias for systematic patterns.
        if !args.iter().any(|arg| arg.key() == Some("patterns")) {
            let mut config = CanonicalNumericPdbConfig::default();
            config.apply_options(args)?;
            *self = Self::systematic(config);
        } else {
            let mut config = ExplicitPdbSourceConfig::default();
            config.apply_options(args)?;
            self.patterns = config.patterns;
            self.systematic_heuristic = PdbHeuristicConfig {
                exploration_heuristic: config.exploration_heuristic,
                frontier_heuristic: config.frontier_heuristic,
                failed_lookup_heuristic: config.failed_lookup_heuristic,
            };
        }
        Ok(())
    }
}

impl PdbSourceConfig {
    pub(crate) fn systematic(config: CanonicalNumericPdbConfig) -> Self {
        Self {
            patterns: PdbPatternsConfig::Systematic(SystematicPatternGeneratorConfig {
                max_pdb_states: config.max_pdb_states,
                max_pattern_size: config.max_pattern_size,
                only_interesting_patterns: config.only_interesting_patterns,
            }),
            systematic_heuristic: config.pdb_heuristic_config(),
        }
    }
    pub(crate) fn greedy(config: GreedyPatternGeneratorConfig) -> Self {
        Self {
            patterns: PdbPatternsConfig::Greedy(config),
            systematic_heuristic: config.pdb_heuristic_config(),
        }
    }
    fn build_patterns(
        self,
        task: &dyn AbstractNumericTask,
    ) -> (PatternCollection, usize, PdbHeuristicConfig) {
        match self.patterns {
            PdbPatternsConfig::Systematic(config) => (
                generate_systematic_patterns(task, config),
                config.max_pdb_states,
                self.systematic_heuristic,
            ),
            PdbPatternsConfig::Greedy(config) => {
                let max_pdb_states = config.max_pdb_states;
                let heuristic = config.pdb_heuristic_config();
                let pattern = generate_greedy_pattern(task, config);
                (
                    PatternCollection::new(vec![pattern]),
                    max_pdb_states,
                    heuristic,
                )
            }
        }
    }
}

/// Typed abstraction sources and their shared construction budget.
#[derive(Debug, Clone, Default)]
pub struct SourcesConfig {
    pub sources: Vec<AbstractionSource>,
    pub construction_max_time: TimeLimit,
}

struct NoOptions {}
impl ApplyOptions for NoOptions {
    fn apply_options(&mut self, args: &[ConfigArg]) -> Result<(), crate::config::ConfigError> {
        for_each_option(args, &[], |key, _| {
            Err(format!("unknown option `{key}`").into())
        })
    }
}

impl SourcesConfig {
    fn apply_with(
        &mut self,
        args: &[ConfigArg],
        options: &mut impl ApplyOptions,
    ) -> Result<(), ConfigError> {
        let mut named = Vec::new();
        let mut listed = false;
        for (index, arg) in args.iter().enumerate() {
            if arg.key().is_some() {
                named.push(arg.clone());
                continue;
            }
            match arg.value() {
                ConfigValue::List(_) => {
                    if listed {
                        return Err("duplicate abstraction source list".into());
                    }
                    if index != 0 {
                        return Err("the abstraction source list must be the first argument".into());
                    }
                    listed = true;
                    self.sources = Vec::from_option_value(arg.value())
                        .map_err(|error| error.at_option("sources"))?;
                }
                value => {
                    if listed {
                        return Err("all abstraction sources must appear inside the list".into());
                    }
                    self.sources
                        .push(AbstractionSource::from_option_value(value)?);
                }
            }
        }
        let mut other = Vec::new();
        for_each_option(&named, &[], |key, value| {
            if key == "construction_max_time" {
                self.construction_max_time = TimeLimit::positive_bounded(value)
                    .map_err(|error| format!("construction_max_time: {error}"))?;
            } else {
                other.push(ConfigArg::new(Some(key.into()), value.clone()));
            }
            Ok(())
        })?;
        options.apply_options(&other)?;
        if self.sources.is_empty() {
            return Err("at least one abstraction source is required".into());
        }
        Ok(())
    }

    pub(crate) fn deadline(&self) -> Result<Deadline, HeuristicBuildError> {
        Deadline::after(Instant::now(), self.construction_max_time.duration()?)
            .map_err(|error| format!("invalid construction_max_time: {error}").into())
    }
}

impl ApplyOptions for SourcesConfig {
    fn apply_options(&mut self, args: &[ConfigArg]) -> Result<(), crate::config::ConfigError> {
        self.apply_with(args, &mut NoOptions {})
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ScpHeuristicConfig {
    pub sources: SourcesConfig,
    pub scp: ScpConfig,
}
impl ApplyOptions for ScpHeuristicConfig {
    fn apply_options(&mut self, args: &[ConfigArg]) -> Result<(), crate::config::ConfigError> {
        self.sources.apply_with(args, &mut self.scp)
    }
}

pub(crate) fn build_components<'task>(
    task: &'task dyn AbstractNumericTask,
    sources: &[AbstractionSource],
    purpose: AbstractionPurpose,
    deadline: Deadline,
) -> Result<Vec<AbstractionComponent<'task>>, HeuristicBuildError> {
    assert!(
        !sources.is_empty(),
        "a configured abstraction collection has sources"
    );
    let construction_start = Instant::now();
    let mut components = Vec::new();
    for source in sources {
        let remaining = remaining_construction_time(deadline)?;
        let built = source.build(task, purpose, remaining)?;
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

fn build_cegar_source<'task, G>(
    task: &dyn AbstractNumericTask,
    mut config: G::Config,
    purpose: AbstractionPurpose,
    remaining: Option<Duration>,
) -> Result<Vec<AbstractionComponent<'task>>, HeuristicBuildError>
where
    G: AbstractionGenerator,
    G::Abstraction: Into<AbstractionComponent<'task>>,
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

fn build_cegar_collection_source<'task, G>(
    task: &dyn AbstractNumericTask,
    mut config: CollectionConfig<G::Config>,
    purpose: AbstractionPurpose,
    remaining: Option<Duration>,
) -> Result<Vec<AbstractionComponent<'task>>, HeuristicBuildError>
where
    G: AbstractionGenerator,
    G::Abstraction: Into<AbstractionComponent<'task>>,
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
pub(crate) fn remaining_construction_time(
    deadline: Deadline,
) -> Result<Option<Duration>, HeuristicBuildError> {
    match deadline.remaining() {
        Some(remaining) if remaining.is_zero() => Err(HeuristicBuildError::ConstructionTimeout),
        remaining => Ok(remaining),
    }
}
