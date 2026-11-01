#[cfg(test)]
mod tests;

use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use planforge_sas::explicit_state::{ExplicitState, StateValues};
use planforge_sas::numeric_task::{AbstractNumericTask, OperatorIndex};
use planforge_sas::state_registry::{ConcreteState, StateRegistry};
use planforge_sas::utils::float_tolerance;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use tracing::{debug, info};

use super::causal_graph::{CausalGraph, CausalGraphVariable};
use super::numeric_size_estimator::NumericSizeEstimator;
use super::pattern_database::{PatternDatabase, PdbBuildConfig, PdbHeuristicConfig};
use super::pattern_generator_systematic::goal_variables;
use crate::config::TimeLimit;
use crate::evaluation::abstractions::cost_partitioning::canonical::{
    are_operator_sets_additive, compute_max_additive_subsets_from_relevant_operators,
};
use crate::evaluation::evaluator::EvaluationError;
use crate::evaluation::sampling::RandomWalkSampler;
use crate::evaluation::seed::DEFAULT_RANDOM_SEED;
use crate::resource_limits::{Deadline, min_time_limit};
use crate::task_transformations::projection::{Pattern, ProjectedTask};

/// The options of iPDB, with the defaults of numeric Fast Downward.
#[derive(Debug, Clone, Copy, PartialEq, planforge_search::config::ApplyOptions)]
pub struct HillClimbingConfig {
    /// The newly reached states after which the exploration of a pattern
    /// with numeric variables stops.
    pub max_pdb_states: usize,
    /// The largest product of a pattern database's size and the domain size
    /// of the variable a candidate adds to its pattern.
    pub max_pdb_size: usize,
    /// The most abstract states the collection may have.
    pub collection_max_size: usize,
    pub num_samples: usize,
    /// The samples a candidate must raise the heuristic on to be added.
    pub min_improvement: usize,
    pub max_time: TimeLimit,
    pub random_seed: u64,
}

impl Default for HillClimbingConfig {
    fn default() -> Self {
        Self {
            max_pdb_states: 10_000,
            max_pdb_size: 1_000_000,
            collection_max_size: 10_000_000,
            num_samples: 1000,
            min_improvement: 10,
            max_time: TimeLimit::UNLIMITED,
            random_seed: DEFAULT_RANDOM_SEED,
        }
    }
}

impl HillClimbingConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.max_pdb_states == 0 || self.max_pdb_size == 0 || self.collection_max_size == 0 {
            return Err(
                "max_pdb_states, max_pdb_size and collection_max_size must be positive".into(),
            );
        }
        if self.min_improvement == 0 || self.min_improvement > self.num_samples {
            return Err(format!(
                "min_improvement must lie between 1 and num_samples={}, got {}",
                self.num_samples, self.min_improvement
            ));
        }
        self.max_time.duration().map(|_| ())
    }
}

impl fmt::Display for HillClimbingConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "max_pdb_states={}, max_pdb_size={}, collection_max_size={}, num_samples={}, \
             min_improvement={}, max_time={}, random_seed={}",
            self.max_pdb_states,
            self.max_pdb_size,
            self.collection_max_size,
            self.num_samples,
            self.min_improvement,
            self.max_time.seconds(),
            self.random_seed,
        )
    }
}

/// iPDB (Haslum et al. 2007) as numeric Fast Downward adapts it to numeric
/// planning: hill climbing in the space of pattern collections, starting
/// from one pattern per goal variable.
///
/// Each step builds the pattern database of every new candidate, a pattern
/// of the collection extended by one variable its operators have a
/// precondition on, and adds the candidate that raises the canonical
/// heuristic on the most states sampled by random walks. The climb stops
/// when no candidate raises it on `min_improvement` samples, when the
/// initial state is a dead end, or when `max_time` or `remaining` passes.
/// Every pattern database explores up to `config.max_pdb_states` states with
/// `heuristics`.
pub fn generate_hill_climbing_pdbs<'task>(
    task: &'task dyn AbstractNumericTask,
    config: HillClimbingConfig,
    heuristics: PdbHeuristicConfig,
    retain_transitions: bool,
    remaining: Option<Duration>,
) -> Result<Vec<PatternDatabase<'task>>, String> {
    let pdb_config = PdbBuildConfig {
        max_states: config.max_pdb_states,
        heuristics,
        retain_transitions,
    };
    let mut climbing = HillClimbing::new(task, config, pdb_config)?;
    let mut collection = Collection::default();
    for variable in goal_variables(task, &climbing.causal_graph) {
        let mut pattern = Pattern::new(Vec::new(), Vec::new());
        add_variable(&mut pattern, variable);
        let pdb = climbing.build(&pattern)?;
        collection.add(pattern, pdb);
    }
    info!(
        "iPDB: initial collection of {} patterns with {} states: {:?}",
        collection.pdbs.len(),
        collection.size,
        collection.patterns
    );
    let deadline = Deadline::after(
        Instant::now(),
        min_time_limit(config.max_time.duration()?, remaining),
    )
    .map_err(|error| error.to_string())?;
    climbing.climb(&mut collection, deadline)?;
    Ok(collection.pdbs)
}

struct HillClimbing<'task> {
    task: &'task dyn AbstractNumericTask,
    config: HillClimbingConfig,
    pdb_config: PdbBuildConfig,
    causal_graph: CausalGraph,
    numeric_sizes: NumericSizeEstimator,
    rng: SmallRng,
    initial_state: ExplicitState,
    /// Extensions rejected for exceeding `max_pdb_size`.
    rejected: usize,
}

/// The outcome of evaluating the candidates on the samples.
enum Selection {
    Best { index: usize, improvement: usize },
    NoneImproves,
    DeadlinePassed,
}

/// Why hill climbing stopped.
#[derive(Debug)]
enum Stop {
    DeadEnd,
    NoImprovement,
    Timeout,
}

impl<'task> HillClimbing<'task> {
    fn new(
        task: &'task dyn AbstractNumericTask,
        config: HillClimbingConfig,
        pdb_config: PdbBuildConfig,
    ) -> Result<Self, String> {
        let mut registry = StateRegistry::for_task(Arc::new(task));
        let mut initial_state = ExplicitState::default();
        let initial = registry.get_initial_state();
        registry
            .view(&initial)
            .fill_explicit(&mut initial_state)
            .map_err(|error| format!("failed to evaluate the initial state: {error:?}"))?;
        Ok(Self {
            task,
            config,
            pdb_config,
            causal_graph: CausalGraph::new(task)?,
            numeric_sizes: NumericSizeEstimator::new(task),
            rng: SmallRng::seed_from_u64(config.random_seed),
            initial_state,
            rejected: 0,
        })
    }

    fn build(&self, pattern: &Pattern) -> Result<PatternDatabase<'task>, String> {
        let projected_task = ProjectedTask::new(self.task, pattern).map_err(|error| {
            format!("failed to project the task to pattern {pattern:?}: {error}")
        })?;
        PatternDatabase::build(projected_task, self.pdb_config)
    }

    fn climb(
        &mut self,
        collection: &mut Collection<'task>,
        deadline: Deadline,
    ) -> Result<(), String> {
        let start = Instant::now();
        let mut new_patterns = Vec::new();
        for (pattern, pdb) in collection.patterns.iter().zip(&collection.pdbs) {
            self.extend(pattern, pdb.num_states(), &mut new_patterns);
        }
        let mut generated: BTreeSet<Pattern> = BTreeSet::new();
        let mut candidates: Vec<Option<Candidate<'task>>> = Vec::new();
        let mut iterations = 0usize;

        let stop = 'climb: loop {
            iterations += 1;
            let initial_h = collection.value(&self.initial_state);
            info!(
                "iPDB iteration {iterations}: {} patterns, {} states, initial h {initial_h}",
                collection.pdbs.len(),
                collection.size
            );
            if initial_h.is_infinite() {
                break Stop::DeadEnd;
            }
            for pattern in new_patterns.drain(..) {
                if !generated.insert(pattern.clone()) {
                    continue;
                }
                if deadline.passed() {
                    break 'climb Stop::Timeout;
                }
                let pdb = self.build(&pattern)?;
                candidates.push(Some(Candidate::new(pattern, pdb)));
            }
            let Some(samples) = self.sample_states(collection, initial_h, deadline)? else {
                break Stop::Timeout;
            };
            let (index, improvement) =
                match self.select_candidate(collection, &mut candidates, &samples, deadline) {
                    Selection::Best { index, improvement }
                        if improvement >= self.config.min_improvement =>
                    {
                        (index, improvement)
                    }
                    Selection::Best { .. } | Selection::NoneImproves => break Stop::NoImprovement,
                    Selection::DeadlinePassed => break Stop::Timeout,
                };
            let candidate = candidates[index]
                .take()
                .expect("a selected candidate is still a candidate");
            info!(
                "iPDB: adding pattern {:?} with improvement {improvement}",
                candidate.pattern
            );
            self.extend(
                &candidate.pattern,
                candidate.pdb.num_states(),
                &mut new_patterns,
            );
            collection.add(candidate.pattern, candidate.pdb);
        };

        info!(
            "iPDB: stopped ({stop:?}) after {iterations} iterations and {:.2}s: {} patterns, \
             {} states, {} generated, {} rejected",
            start.elapsed().as_secs_f64(),
            collection.pdbs.len(),
            collection.size,
            generated.len(),
            self.rejected
        );
        Ok(())
    }

    /// Pushes onto `extensions` every extension of `pattern` by one variable
    /// that the operators changing a variable of `pattern` have a
    /// precondition on, and whose domain size times `pdb_size` stays within
    /// `max_pdb_size`.
    fn extend(&mut self, pattern: &Pattern, pdb_size: usize, extensions: &mut Vec<Pattern>) {
        for variable in pattern_variables(pattern) {
            for neighbor in self.causal_graph.eff_pre_neighbors_of(variable) {
                let mut extended = pattern.clone();
                if !add_variable(&mut extended, neighbor) {
                    continue;
                }
                let domain_size = match neighbor {
                    CausalGraphVariable::Propositional(var_id) => {
                        self.task.variables()[var_id.index()].domain_size()
                    }
                    CausalGraphVariable::Numeric(var_id) => {
                        self.numeric_sizes.estimate_domain_size(var_id)
                    }
                };
                if pdb_size
                    .checked_mul(domain_size)
                    .is_some_and(|size| size <= self.config.max_pdb_size)
                {
                    extensions.push(extended);
                } else {
                    self.rejected += 1;
                }
            }
        }
    }

    /// The end states of `num_samples` random walks from the initial state
    /// (Haslum et al. 2007), whose lengths follow the goal distance
    /// `initial_h` estimates. A walk that reaches a dead end of `collection`
    /// restarts from the initial state. `None` when `deadline` passes.
    fn sample_states(
        &mut self,
        collection: &Collection<'task>,
        initial_h: f64,
        deadline: Deadline,
    ) -> Result<Option<Vec<ExplicitState>>, String> {
        let mut sampler = RandomWalkSampler::new(Arc::new(self.task));
        let mut values = ExplicitState::default();
        let mut samples = Vec::with_capacity(self.config.num_samples);
        while samples.len() < self.config.num_samples {
            let walk_end = sampler
                .sample(initial_h, &mut self.rng, deadline, |registry, state| {
                    fill_values(registry, state, &mut values)?;
                    Ok(collection.is_dead_end(&values))
                })
                .map_err(|error| error.to_string())?;
            let Some(walk_end) = walk_end else {
                return Ok(None);
            };
            let mut sample = ExplicitState::default();
            fill_values(sampler.registry(), &walk_end, &mut sample)
                .map_err(|error| error.to_string())?;
            samples.push(sample);
        }
        Ok(Some(samples))
    }

    /// The candidate that raises the canonical heuristic on the most samples,
    /// the first one among equals. Drops the candidates the collection has no
    /// room for.
    fn select_candidate(
        &self,
        collection: &Collection<'task>,
        candidates: &mut [Option<Candidate<'task>>],
        samples: &[ExplicitState],
        deadline: Deadline,
    ) -> Selection {
        let sample_values: Vec<Vec<f64>> = samples
            .iter()
            .map(|sample| collection.values(sample))
            .collect();
        let sample_h: Vec<f64> = sample_values
            .iter()
            .map(|values| collection.canonical_value(values))
            .collect();
        let mut selection = Selection::NoneImproves;
        for (index, slot) in candidates.iter_mut().enumerate() {
            if deadline.passed() {
                return Selection::DeadlinePassed;
            }
            let Some(candidate) = slot else {
                continue;
            };
            if collection.size + candidate.pdb.num_states() > self.config.collection_max_size {
                *slot = None;
                continue;
            }
            let subsets = collection.subsets_additive_with(&candidate.relevant_operators);
            let improvement = samples
                .iter()
                .zip(&sample_values)
                .zip(&sample_h)
                .filter(|&((sample, values), &h_collection)| {
                    raises_canonical_value(
                        candidate.pdb.value(StateValues::from(sample)),
                        values,
                        h_collection,
                        &subsets,
                    )
                })
                .count();
            debug!(
                "iPDB: candidate {:?} with {} states improves {improvement} samples",
                candidate.pattern,
                candidate.pdb.num_states()
            );
            let best_improvement = match selection {
                Selection::Best { improvement, .. } => improvement,
                _ => 0,
            };
            if improvement > best_improvement {
                selection = Selection::Best { index, improvement };
            }
        }
        selection
    }
}

/// Whether a pattern database with value `h_pattern` raises the canonical
/// heuristic value `h_collection` of a state whose values in the collection
/// are `values`, given the maximal additive subsets it would join.
///
/// A raise must exceed float noise: summing the same values in another order
/// must not count as an improvement.
fn raises_canonical_value(
    h_pattern: f64,
    values: &[f64],
    h_collection: f64,
    subsets: &[Vec<usize>],
) -> bool {
    if h_pattern.is_infinite() {
        return true;
    }
    if h_collection.is_infinite() {
        return false;
    }
    subsets.iter().any(|subset| {
        h_pattern + subset.iter().map(|&pdb| values[pdb]).sum::<f64>()
            > h_collection + float_tolerance::SEARCH_EPSILON
    })
}

struct Candidate<'task> {
    pattern: Pattern,
    pdb: PatternDatabase<'task>,
    relevant_operators: BTreeSet<OperatorIndex>,
}

impl<'task> Candidate<'task> {
    fn new(pattern: Pattern, pdb: PatternDatabase<'task>) -> Self {
        let relevant_operators = pdb.relevant_operator_ids().iter().copied().collect();
        Self {
            pattern,
            pdb,
            relevant_operators,
        }
    }
}

/// The pattern databases hill climbing selected, and their maximal additive
/// subsets.
#[derive(Default)]
struct Collection<'task> {
    patterns: Vec<Pattern>,
    pdbs: Vec<PatternDatabase<'task>>,
    relevant_operators: Vec<BTreeSet<OperatorIndex>>,
    max_additive_subsets: Vec<Vec<usize>>,
    /// The abstract states of all pattern databases.
    size: usize,
}

impl<'task> Collection<'task> {
    fn add(&mut self, pattern: Pattern, pdb: PatternDatabase<'task>) {
        self.size += pdb.num_states();
        self.relevant_operators
            .push(pdb.relevant_operator_ids().iter().copied().collect());
        self.patterns.push(pattern);
        self.pdbs.push(pdb);
        self.max_additive_subsets =
            compute_max_additive_subsets_from_relevant_operators(&self.relevant_operators);
    }

    /// The value of every pattern database for `state`.
    fn values(&self, state: &ExplicitState) -> Vec<f64> {
        self.pdbs
            .iter()
            .map(|pdb| pdb.value(StateValues::from(state)))
            .collect()
    }

    /// The canonical heuristic value of a state with pattern database
    /// `values`: infinite if one is, since every pattern database lies in a
    /// maximal additive subset.
    fn canonical_value(&self, values: &[f64]) -> f64 {
        if values.iter().any(|value| value.is_infinite()) {
            return f64::INFINITY;
        }
        self.max_additive_subsets
            .iter()
            .map(|subset| subset.iter().map(|&pdb| values[pdb]).sum::<f64>())
            .fold(0.0, f64::max)
    }

    fn value(&self, state: &ExplicitState) -> f64 {
        self.canonical_value(&self.values(state))
    }

    fn is_dead_end(&self, state: &ExplicitState) -> bool {
        self.pdbs
            .iter()
            .any(|pdb| pdb.value(StateValues::from(state)).is_infinite())
    }

    /// Every maximal additive subset, restricted to the pattern databases
    /// additive with one whose relevant operators are `relevant_operators`.
    fn subsets_additive_with(
        &self,
        relevant_operators: &BTreeSet<OperatorIndex>,
    ) -> Vec<Vec<usize>> {
        let mut subsets: Vec<Vec<usize>> = self
            .max_additive_subsets
            .iter()
            .map(|subset| {
                subset
                    .iter()
                    .copied()
                    .filter(|&pdb| {
                        are_operator_sets_additive(
                            &self.relevant_operators[pdb],
                            relevant_operators,
                        )
                    })
                    .collect()
            })
            .collect();
        subsets.sort();
        subsets.dedup();
        subsets
    }
}

fn pattern_variables(pattern: &Pattern) -> impl Iterator<Item = CausalGraphVariable> + '_ {
    pattern
        .regular
        .iter()
        .map(|&var_id| CausalGraphVariable::Propositional(var_id))
        .chain(
            pattern
                .numeric
                .iter()
                .map(|&var_id| CausalGraphVariable::Numeric(var_id)),
        )
}

/// Adds `variable` to `pattern`; false if it was there already.
fn add_variable(pattern: &mut Pattern, variable: CausalGraphVariable) -> bool {
    match variable {
        CausalGraphVariable::Propositional(var_id) => pattern.add_regular_var(var_id),
        CausalGraphVariable::Numeric(var_id) => pattern.add_numeric_var(var_id),
    }
}

fn fill_values(
    registry: &StateRegistry<'_>,
    state: &ConcreteState,
    values: &mut ExplicitState,
) -> Result<(), EvaluationError> {
    registry.view(state).fill_explicit(values).map_err(|error| {
        EvaluationError::ComputationFailed(format!("failed to evaluate a sampled state: {error:?}"))
    })
}
