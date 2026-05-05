use std::cell::RefCell;
use std::collections::BTreeSet;
use std::fmt;
use std::time::{Duration, Instant};

use planners_sas::numeric::numeric_task::{
    AbstractNumericTask, metric_operator_cost_from_initial_values,
};
use rand::seq::SliceRandom;
use rand::{SeedableRng, rngs::SmallRng};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::numeric::evaluation::evaluator::{EvaluationError, EvaluationState};
use crate::numeric::evaluation::heuristic::Heuristic;
use crate::numeric::evaluation::pattern_databases::pattern_database::{
    PatternDatabase, PdbHeuristicConfig, PdbInternalHeuristic,
};

use super::abstract_operator_generator::AbstractOperator;
use super::domain_abstraction_collection_generator_multiple_cegar::DomainAbstractionCollectionGeneratorMultipleCegarConfig;
use super::domain_abstraction_factory::AbstractDistanceTable;
use super::domain_abstraction_generator::DomainAbstraction;
use super::domain_abstraction_heuristic::{
    DomainAbstractionHeuristic, DomainAbstractionLookupScratch,
    compute_collection_abstract_state_ids,
};
use super::transition_cost_partitioning::{
    AbstractOperatorFootprint, AbstractTransitionSystem, TransitionResidualCosts,
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ScoringFunction {
    MaxHeuristic,
    MinStolenCosts,
    MaxHeuristicPerStolenCosts,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Saturator {
    All,
    Perim,
    Perimstar,
}

impl fmt::Display for Saturator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Saturator::All => write!(f, "all"),
            Saturator::Perim => write!(f, "perim"),
            Saturator::Perimstar => write!(f, "perimstar"),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct ScpOnlineConfig {
    pub max_time: f64,
    pub max_size: usize,
    pub interval: usize,
    pub combine_labels: bool,
    pub collection_config: DomainAbstractionCollectionGeneratorMultipleCegarConfig,
    pub use_numeric_pdbs: bool,
    pub max_pdb_states: usize,
    pub max_pattern_size: usize,
    pub only_interesting_patterns: bool,
    pub pdb_exploration_heuristic: PdbInternalHeuristic,
    pub pdb_frontier_heuristic: PdbInternalHeuristic,
    pub pdb_failed_lookup_heuristic: PdbInternalHeuristic,
    pub scoring_function: ScoringFunction,
    pub saturator: Saturator,
    pub random_seed: Option<u64>,
    pub use_abstract_operator_cost_partitioning: bool,
}

impl Default for ScpOnlineConfig {
    fn default() -> Self {
        let collection_config = DomainAbstractionCollectionGeneratorMultipleCegarConfig {
            combine_labels: false,
            ..Default::default()
        };
        let random_seed = collection_config.random_seed;
        Self {
            max_time: 200.0,
            max_size: usize::MAX,
            interval: 10_000,
            combine_labels: false,
            collection_config,
            use_numeric_pdbs: false,
            max_pdb_states: 50_000,
            max_pattern_size: 2,
            only_interesting_patterns: true,
            pdb_exploration_heuristic: PdbInternalHeuristic::Blind,
            pdb_frontier_heuristic: PdbInternalHeuristic::Zero,
            pdb_failed_lookup_heuristic: PdbInternalHeuristic::Zero,
            scoring_function: ScoringFunction::MaxHeuristicPerStolenCosts,
            saturator: Saturator::All,
            random_seed,
            use_abstract_operator_cost_partitioning: false,
        }
    }
}

impl ScpOnlineConfig {
    pub fn pdb_heuristic_config(&self) -> PdbHeuristicConfig {
        PdbHeuristicConfig {
            exploration_heuristic: self.pdb_exploration_heuristic,
            frontier_heuristic: self.pdb_frontier_heuristic,
            failed_lookup_heuristic: self.pdb_failed_lookup_heuristic,
        }
    }
}

// ---------------------------------------------------------------------------
// Lookup tables and CP heuristic
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct LookupTable {
    abstraction_id: usize,
    distances: Vec<f64>,
    unknown_value: f64,
}

#[derive(Debug, Clone, Default)]
struct CostPartitioningHeuristic {
    lookup_tables: Vec<LookupTable>,
}

impl CostPartitioningHeuristic {
    fn is_empty(&self) -> bool {
        self.lookup_tables.is_empty()
    }

    fn add_h_values(&mut self, abstraction_id: usize, distances: Vec<f64>) {
        if distances.iter().any(|value| *value > 0.0) {
            self.lookup_tables.push(LookupTable {
                abstraction_id,
                distances,
                unknown_value: f64::INFINITY,
            });
        }
    }

    fn add_pdb_h_values(&mut self, abstraction_id: usize, distances: Vec<f64>) {
        if distances.iter().any(|value| *value > 0.0) {
            self.lookup_tables.push(LookupTable {
                abstraction_id,
                distances,
                unknown_value: 0.0,
            });
        }
    }

    fn compute_heuristic(&self, abstract_state_ids: &[Option<usize>]) -> f64 {
        let mut sum = 0.0;
        for table in &self.lookup_tables {
            let Some(state_id) = abstract_state_ids
                .get(table.abstraction_id)
                .copied()
                .flatten()
            else {
                sum += table.unknown_value;
                continue;
            };
            let Some(&value) = table.distances.get(state_id) else {
                sum += table.unknown_value;
                continue;
            };
            if value.is_infinite() && value.is_sign_positive() {
                return f64::INFINITY;
            }
            sum += value;
        }
        sum
    }

    fn estimate_size_in_kb(&self) -> usize {
        let values = self
            .lookup_tables
            .iter()
            .map(|table| table.distances.len())
            .sum::<usize>();
        values.saturating_mul(std::mem::size_of::<f64>()) / 1024
    }
}

// ---------------------------------------------------------------------------
// Main heuristic
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct ScpOnlineState {
    start_time: Instant,
    evaluated_states: usize,
    improve_heuristic: bool,
    size_kb: usize,
    cp_heuristics: Vec<CostPartitioningHeuristic>,
    h_values_by_abstraction: Vec<Vec<f64>>,
    stolen_costs_by_abstraction: Vec<f64>,
    rng: SmallRng,
    improvement_ended: bool,
    required_lookup_ids: Vec<usize>,
}

impl ScpOnlineState {
    fn new(seed: Option<u64>) -> Self {
        let seed = seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|duration| duration.as_nanos() as u64)
                .unwrap_or(0x5C9_0A11_u64)
        });
        Self {
            start_time: Instant::now(),
            evaluated_states: 0,
            improve_heuristic: true,
            size_kb: 0,
            cp_heuristics: Vec::new(),
            h_values_by_abstraction: Vec::new(),
            stolen_costs_by_abstraction: Vec::new(),
            rng: SmallRng::seed_from_u64(seed),
            improvement_ended: false,
            required_lookup_ids: Vec::new(),
        }
    }
}

pub struct SaturatedCostPartitioningOnlineHeuristic<'task> {
    name: String,
    abstractions: RefCell<Option<Vec<DomainAbstraction>>>,
    abstraction_heuristics: Vec<DomainAbstractionHeuristic>,
    pdbs: Vec<PatternDatabase<'task>>,
    config: ScpOnlineConfig,
    state: RefCell<ScpOnlineState>,
    transition_system_cache: RefCell<Vec<Option<AbstractTransitionSystem>>>,
    lookup_scratch: RefCell<DomainAbstractionLookupScratch>,
    component_ids_scratch: RefCell<Vec<Option<usize>>>,
    prop_scratch: RefCell<Vec<usize>>,
    numeric_scratch: RefCell<Vec<f64>>,
    expanded_numeric_scratch: RefCell<Vec<f64>>,
}

impl<'task> SaturatedCostPartitioningOnlineHeuristic<'task> {
    pub fn new(
        name: Option<String>,
        abstractions: Vec<DomainAbstraction>,
        pdbs: Vec<PatternDatabase<'task>>,
        config: ScpOnlineConfig,
        task: &dyn AbstractNumericTask,
    ) -> Result<Self, EvaluationError> {
        let abstraction_heuristics = abstractions
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, abstraction)| {
                DomainAbstractionHeuristic::new(Some(format!("scp_online_{index}")), abstraction)
            })
            .collect();

        let original_costs: Vec<f64> = task
            .get_operators()
            .iter()
            .map(|op| metric_operator_cost_from_initial_values(task, op))
            .collect();

        let num_abstractions = abstractions.len();
        let pdbs_count = pdbs.len();
        let total_components = num_abstractions + pdbs_count;

        let mut h_values: Vec<Vec<f64>> = Vec::with_capacity(total_components);
        let mut saturated_costs_by_abstraction: Vec<Vec<f64>> =
            Vec::with_capacity(total_components);
        let mut debug_initial_h_values = Vec::new();

        for (abstraction_id, abstraction) in abstractions.iter().enumerate() {
            let abstraction_task = abstraction.task_for_factory(task);
            let goal_facts = &abstraction.distance_table.goal_facts;
            let table = abstraction
                .factory
                .build_goal_distances_for_goals(
                    abstraction_task,
                    config.combine_labels,
                    &original_costs,
                    goal_facts,
                )
                .map_err(|error| {
                    EvaluationError::ComputationFailed(format!(
                        "failed to compute goal distances for order generator: {error:#}"
                    ))
                })?;
            let (_, saturated) = abstraction
                .factory
                .build_cost_partitioned_distance_table_for_goals(
                    abstraction_task,
                    config.combine_labels,
                    &original_costs,
                    false,
                    goal_facts,
                )
                .map_err(|error| {
                    EvaluationError::ComputationFailed(format!(
                        "failed to compute saturated costs for order generator: {error:#}"
                    ))
                })?;
            if config.collection_config.debug {
                let initial_h = table
                    .distances
                    .get(table.initial_state_hash)
                    .copied()
                    .unwrap_or(f64::INFINITY);
                debug_initial_h_values.push(initial_h);
                info!(
                    "scp_online debug: collection abstraction {abstraction_id}: original_initial_h={initial_h}, states={}, goal_facts={}",
                    abstraction_state_count(abstraction),
                    goal_facts.len()
                );
            }
            h_values.push(table.distances);
            saturated_costs_by_abstraction.push(saturated);
        }
        if config.collection_config.debug && !debug_initial_h_values.is_empty() {
            let max_initial_h = debug_initial_h_values
                .iter()
                .copied()
                .fold(0.0_f64, f64::max);
            info!(
                "scp_online debug: collection max original-cost initial h before cost partitioning = {max_initial_h}"
            );
        }

        for pdb in &pdbs {
            let distances = pdb.build_goal_distances(&original_costs).map_err(|error| {
                EvaluationError::ComputationFailed(format!(
                    "failed to compute PDB goal distances for order generator: {error}"
                ))
            })?;
            let (_, saturated) = pdb
                .build_cost_partitioned_distance_table(&original_costs)
                .map_err(|error| {
                    EvaluationError::ComputationFailed(format!(
                        "failed to compute PDB saturated costs for order generator: {error}"
                    ))
                })?;
            h_values.push(distances);
            saturated_costs_by_abstraction.push(saturated);
        }

        let surplus_costs =
            compute_all_surplus_costs(&original_costs, &saturated_costs_by_abstraction);
        let stolen_costs: Vec<f64> = saturated_costs_by_abstraction
            .iter()
            .map(|saturated| compute_costs_stolen_by_heuristic(saturated, &surplus_costs))
            .collect();

        let mut st = ScpOnlineState::new(config.random_seed);
        st.h_values_by_abstraction = h_values;
        st.stolen_costs_by_abstraction = stolen_costs;

        Ok(Self {
            name: name.unwrap_or_else(|| "scp_online".to_string()),
            abstractions: RefCell::new(Some(abstractions)),
            abstraction_heuristics,
            pdbs,
            config,
            state: RefCell::new(st),
            transition_system_cache: RefCell::new(vec![None; num_abstractions]),
            lookup_scratch: RefCell::new(DomainAbstractionLookupScratch::new()),
            component_ids_scratch: RefCell::new(Vec::new()),
            prop_scratch: RefCell::new(Vec::new()),
            numeric_scratch: RefCell::new(Vec::new()),
            expanded_numeric_scratch: RefCell::new(Vec::new()),
        })
    }

    fn compute_order_for_state(
        state: &mut ScpOnlineState,
        abstract_state_ids: &[Option<usize>],
        scoring_function: ScoringFunction,
    ) -> Vec<usize> {
        let total = state.h_values_by_abstraction.len();
        let mut order: Vec<usize> = (0..total).collect();
        order.shuffle(&mut state.rng);

        let scores: Vec<f64> = (0..total)
            .map(|abs_id| {
                let h = abstract_state_ids
                    .get(abs_id)
                    .copied()
                    .flatten()
                    .and_then(|sid| {
                        state
                            .h_values_by_abstraction
                            .get(abs_id)
                            .and_then(|values| values.get(sid))
                            .copied()
                    })
                    .unwrap_or(0.0);
                let stolen = state
                    .stolen_costs_by_abstraction
                    .get(abs_id)
                    .copied()
                    .unwrap_or(0.0);
                compute_score(h, stolen, scoring_function)
            })
            .collect();

        order.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        order
    }

    fn compute_abstract_state_ids_into(
        &self,
        eval_state: &EvaluationState<'_, '_>,
        required_ids: Option<&[usize]>,
        ids: &mut Vec<Option<usize>>,
    ) -> Result<usize, EvaluationError> {
        let num_domain = self.abstraction_heuristics.len();
        let total_components = num_domain + self.pdbs.len();
        ids.clear();
        ids.resize(total_components, None);
        let needs_all = required_ids.is_none();
        {
            let mut scratch = self.lookup_scratch.borrow_mut();
            compute_collection_abstract_state_ids(
                &self.abstraction_heuristics,
                eval_state,
                required_ids,
                &mut scratch,
            )?;
            for (id, abstract_id) in scratch.abstract_state_ids.iter().copied().enumerate() {
                ids[id] = abstract_id;
            }
        }

        if !self.pdbs.is_empty() {
            let pdb_required =
                needs_all || required_ids.is_some_and(|ids| ids.iter().any(|&id| id >= num_domain));
            if !pdb_required {
                return Ok(num_domain);
            }
            let registry = eval_state.state_registry().ok_or_else(|| {
                EvaluationError::InvalidState(
                    "SCP online PDB lookup requires state registry".to_string(),
                )
            })?;
            let mut numeric = self.numeric_scratch.borrow_mut();
            registry
                .fill_numeric_vars(eval_state.state(), &mut numeric)
                .map_err(|err| {
                    EvaluationError::ComputationFailed(format!(
                        "failed to read numeric state: {err:?}"
                    ))
                })?;
            let mut prop = self.prop_scratch.borrow_mut();
            eval_state.state().fill_state(registry, &mut prop);
            let mut expanded = self.expanded_numeric_scratch.borrow_mut();
            if let Some(first) = self.pdbs.first() {
                first
                    .expand_numeric_state_values_into(&numeric, &mut expanded)
                    .map_err(EvaluationError::ComputationFailed)?;
            }
            for (pdb_id, pdb) in self.pdbs.iter().enumerate() {
                let sid = pdb
                    .abstract_state_id_from_expanded_state_values(&prop, &expanded)
                    .map_err(EvaluationError::ComputationFailed)?;
                ids[num_domain + pdb_id] = sid;
            }
        }

        Ok(num_domain)
    }

    fn compute_max_h(state: &ScpOnlineState, ids: &[Option<usize>]) -> f64 {
        state
            .cp_heuristics
            .iter()
            .map(|cp| cp.compute_heuristic(ids))
            .fold(0.0, f64::max)
    }

    fn required_lookup_ids(state: &ScpOnlineState) -> Vec<usize> {
        let mut ids: Vec<usize> = state
            .cp_heuristics
            .iter()
            .flat_map(|cp| cp.lookup_tables.iter().map(|table| table.abstraction_id))
            .collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    fn operator_costs(task: &dyn AbstractNumericTask) -> Vec<f64> {
        task.get_operators()
            .iter()
            .map(|op| metric_operator_cost_from_initial_values(task, op))
            .collect()
    }

    fn is_online_deadline_error(error: &anyhow::Error) -> bool {
        error.to_string().contains("online SCP deadline exceeded")
    }

    fn update_improvement_status(&self, state: &mut ScpOnlineState) {
        let time_limit_reached = self.config.max_time.is_finite()
            && state.start_time.elapsed() >= Duration::from_secs_f64(self.config.max_time);

        if state.improve_heuristic && (time_limit_reached || state.size_kb >= self.config.max_size)
        {
            state.improve_heuristic = false;
        }
    }

    fn release_abstractions_if_finished(&self, state: &mut ScpOnlineState) {
        if !state.improve_heuristic && !state.improvement_ended {
            let mut abs_guard = self.abstractions.borrow_mut();
            if abs_guard.is_some() {
                abs_guard.take();
                self.transition_system_cache.borrow_mut().clear();
                state.improvement_ended = true;
            }
        }
    }

    fn should_build_cp(&self, state: &ScpOnlineState) -> bool {
        state.improve_heuristic && state.evaluated_states.is_multiple_of(self.config.interval)
    }

    fn maybe_build_cp(
        &self,
        task: &dyn AbstractNumericTask,
        state: &mut ScpOnlineState,
        abstract_state_ids: &[Option<usize>],
        num_domain_abstractions: usize,
    ) -> Result<Option<CostPartitioningHeuristic>, EvaluationError> {
        if !self.should_build_cp(state) {
            return Ok(None);
        }

        let original_costs = Self::operator_costs(task);
        let order =
            Self::compute_order_for_state(state, abstract_state_ids, self.config.scoring_function);

        let abstractions_guard = self.abstractions.borrow();
        let abstractions: &[DomainAbstraction] = match &*abstractions_guard {
            Some(abs) => abs.as_slice(),
            None => &[],
        };

        if abstractions.is_empty() && self.pdbs.is_empty() {
            return Ok(None);
        }

        let deadline = (self.config.max_time.is_finite())
            .then(|| state.start_time + Duration::from_secs_f64(self.config.max_time));
        let mode = if self.config.use_abstract_operator_cost_partitioning {
            "abstract-operator"
        } else {
            "label"
        };
        info!(
            "scp_online: building {mode} CP at evaluation {}, stored_cps={}, current_h={}, size={} KiB, order_len={}, saturator={}, elapsed={:.3}s",
            state.evaluated_states,
            state.cp_heuristics.len(),
            Self::compute_max_h(state, abstract_state_ids),
            state.size_kb,
            order.len(),
            self.config.saturator,
            state.start_time.elapsed().as_secs_f64(),
        );

        let cp = if self.config.use_abstract_operator_cost_partitioning {
            self.build_abstract_operator_cp(
                task,
                abstractions,
                &order,
                abstract_state_ids,
                num_domain_abstractions,
                &original_costs,
                deadline,
            )?
        } else {
            self.build_label_cp(
                task,
                abstractions,
                &order,
                abstract_state_ids,
                num_domain_abstractions,
                &original_costs,
                deadline,
            )?
        };

        if cp.is_empty() {
            info!("scp_online: {mode} CP attempt produced no lookup tables");
        }
        Ok((!cp.is_empty()).then_some(cp))
    }

    fn with_transition_system<R>(
        &self,
        task: &dyn AbstractNumericTask,
        abstraction: &DomainAbstraction,
        abstraction_id: usize,
        deadline: Option<Instant>,
        f: impl FnOnce(&AbstractTransitionSystem) -> Result<R, EvaluationError>,
    ) -> Result<Option<R>, EvaluationError> {
        if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            return Ok(None);
        }
        {
            let mut cache = self.transition_system_cache.borrow_mut();
            if abstraction_id >= cache.len() {
                return Err(EvaluationError::ComputationFailed(format!(
                    "transition-system cache missing abstraction {abstraction_id}"
                )));
            }
            if cache[abstraction_id].is_none() {
                let start = Instant::now();
                info!("scp_online: building transition system for abstraction {abstraction_id}");
                let abstraction_task = abstraction.task_for_factory(task);
                let build_without_regions = abstraction_state_count(abstraction) > 10_000;
                let transition_system_result = if build_without_regions {
                    abstraction.factory.build_abstract_transition_system_from_operators_without_regions_with_deadline(
                        abstraction_task,
                        self.config.combine_labels,
                        &abstraction.abstract_operators,
                        deadline,
                    )
                } else {
                    abstraction
                        .factory
                        .build_abstract_transition_system_from_operators_with_deadline(
                            abstraction_task,
                            self.config.combine_labels,
                            &abstraction.abstract_operators,
                            deadline,
                        )
                };
                let transition_system = match transition_system_result {
                    Ok(transition_system) => transition_system,
                    Err(error) => {
                        if error.to_string().contains("online SCP deadline exceeded") {
                            return Ok(None);
                        }
                        return Err(EvaluationError::ComputationFailed(format!(
                            "failed to build transition system for abstraction {abstraction_id}: {error:#}"
                        )));
                    }
                };
                log_transition_system_summary(abstraction_id, &transition_system, start.elapsed());
                cache[abstraction_id] = Some(transition_system);
            } else {
                info!(
                    "scp_online: reusing cached transition system for abstraction {abstraction_id}"
                );
            }
        }

        let cache = self.transition_system_cache.borrow();
        let transition_system = cache
            .get(abstraction_id)
            .and_then(|entry| entry.as_ref())
            .ok_or_else(|| {
                EvaluationError::ComputationFailed(format!(
                    "transition-system cache missing abstraction {abstraction_id}"
                ))
            })?;
        f(transition_system).map(Some)
    }

    #[allow(clippy::too_many_arguments)]
    fn build_abstract_operator_cp(
        &self,
        task: &dyn AbstractNumericTask,
        abstractions: &[DomainAbstraction],
        order: &[usize],
        abstract_state_ids: &[Option<usize>],
        num_domain_abstractions: usize,
        original_costs: &[f64],
        deadline: Option<Instant>,
    ) -> Result<CostPartitioningHeuristic, EvaluationError> {
        let mut cp = CostPartitioningHeuristic::default();
        let mut remaining_costs = TransitionResidualCosts::from_operator_costs(original_costs);

        for &pos in order {
            if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                break;
            }
            if pos < num_domain_abstractions {
                let abstraction = &abstractions[pos];
                info!(
                    "scp_online: abstract-operator CP step abstraction {pos}, abstract_states={}",
                    abstraction_state_count(abstraction)
                );
                log_abstract_operator_footprint_summary(
                    pos,
                    &abstraction.abstract_operator_footprints,
                );
                match self.config.saturator {
                    Saturator::All => {
                        let completed = self.with_transition_system(
                            task,
                            abstraction,
                            pos,
                            deadline,
                            |transition_system| {
                                let (table, tcf) = match abstraction
                                .factory
                                .build_abstract_operator_cost_partitioned_distance_table_from_system_and_footprints_with_deadline(
                                    transition_system,
                                    Some(&abstraction.abstract_operator_footprints),
                                    &remaining_costs,
                                    pos,
                                    None,
                                    deadline,
                                ) {
                                    Ok(result) => result,
                                    Err(error) if Self::is_online_deadline_error(&error) => {
                                        info!(
                                            "scp_online: abstract-operator all abstraction {pos} stopped while computing table (deadline)"
                                        );
                                        return Ok(true);
                                    }
                                    Err(error) => {
                                        return Err(EvaluationError::ComputationFailed(format!(
                                        "failed to compute abstract-operator SCP table: {error:#}"
                                    )));
                                    }
                                };
                                log_transition_table_summary(
                                    "all",
                                    pos,
                                    &table.distances,
                                    &tcf.operator_costs,
                                    abstract_state_ids,
                                );
                                if should_skip_zero_current_table(
                                    "all",
                                    pos,
                                    &table.distances,
                                    abstract_state_ids,
                                ) {
                                    return Ok(true);
                                }
                                cp.add_h_values(pos, table.distances);
                                remaining_costs
                                    .reduce_by_abstract_operator_footprints(
                                        pos,
                                        &abstraction.abstract_operator_footprints,
                                        &tcf,
                                    )
                                    .map_err(|error| {
                                        EvaluationError::ComputationFailed(format!(
                                            "failed to reduce abstract-operator residual costs: {error:#}"
                                        ))
                                    })?;
                                log_transition_residual_summary(&remaining_costs);
                                Ok(true)
                            },
                        )?;
                        if completed.is_none() {
                            info!(
                                "scp_online: abstract-operator CP stopped while building abstraction {pos} (deadline)"
                            );
                            break;
                        }
                        if completed == Some(false) {
                            break;
                        }
                    }
                    Saturator::Perim => {
                        let cap_state_id = abstract_state_ids.get(pos).copied().flatten();
                        let completed = self.with_transition_system(
                            task,
                            abstraction,
                            pos,
                            deadline,
                            |transition_system| {
                                let (table, tcf) = match abstraction
                                .factory
                                .build_abstract_operator_cost_partitioned_distance_table_from_system_and_footprints_with_deadline(
                                    transition_system,
                                    Some(&abstraction.abstract_operator_footprints),
                                    &remaining_costs,
                                    pos,
                                    cap_state_id,
                                    deadline,
                                ) {
                                    Ok(result) => result,
                                    Err(error) if Self::is_online_deadline_error(&error) => {
                                        info!(
                                            "scp_online: abstract-operator perim abstraction {pos} stopped while computing table (deadline)"
                                        );
                                        return Ok(true);
                                    }
                                    Err(error) => {
                                        return Err(EvaluationError::ComputationFailed(format!(
                                        "failed to compute abstract-operator PERIM table: {error:#}"
                                    )));
                                    }
                                };
                                log_transition_table_summary(
                                    "perim",
                                    pos,
                                    &table.distances,
                                    &tcf.operator_costs,
                                    abstract_state_ids,
                                );
                                if should_skip_zero_current_table(
                                    "perim",
                                    pos,
                                    &table.distances,
                                    abstract_state_ids,
                                ) {
                                    return Ok(true);
                                }
                                cp.add_h_values(pos, table.distances);
                                remaining_costs
                                    .reduce_by_abstract_operator_footprints(
                                        pos,
                                        &abstraction.abstract_operator_footprints,
                                        &tcf,
                                    )
                                    .map_err(|error| {
                                        EvaluationError::ComputationFailed(format!(
                                            "failed to reduce abstract-operator PERIM residual costs: {error:#}"
                                        ))
                                    })?;
                                log_transition_residual_summary(&remaining_costs);
                                Ok(true)
                            },
                        )?;
                        if completed.is_none() {
                            info!(
                                "scp_online: abstract-operator PERIM CP stopped while building abstraction {pos} (deadline)"
                            );
                            break;
                        }
                        if completed == Some(false) {
                            break;
                        }
                    }
                    Saturator::Perimstar => {
                        let cap_state_id = abstract_state_ids.get(pos).copied().flatten();
                        let completed = self.with_transition_system(
                            task,
                            abstraction,
                            pos,
                            deadline,
                            |transition_system| {
                                let (perim_table, perim_tcf) = match abstraction
                                .factory
                                .build_abstract_operator_cost_partitioned_distance_table_from_system_and_footprints_with_deadline(
                                    transition_system,
                                    Some(&abstraction.abstract_operator_footprints),
                                    &remaining_costs,
                                    pos,
                                    cap_state_id,
                                    deadline,
                                ) {
                                    Ok(result) => result,
                                    Err(error) if Self::is_online_deadline_error(&error) => {
                                        info!(
                                            "scp_online: abstract-operator perimstar/perim abstraction {pos} stopped while computing table (deadline)"
                                        );
                                        return Ok(true);
                                    }
                                    Err(error) => {
                                        return Err(EvaluationError::ComputationFailed(format!(
                                        "failed to compute abstract-operator Perim step for Perimstar: {error:#}"
                                    )));
                                    }
                                };
                                log_transition_table_summary(
                                    "perimstar/perim",
                                    pos,
                                    &perim_table.distances,
                                    &perim_tcf.operator_costs,
                                    abstract_state_ids,
                                );
                                if should_skip_zero_current_table(
                                    "perimstar/perim",
                                    pos,
                                    &perim_table.distances,
                                    abstract_state_ids,
                                ) {
                                    return Ok(true);
                                }
                                cp.add_h_values(pos, perim_table.distances);
                                remaining_costs
                                    .reduce_by_abstract_operator_footprints(
                                        pos,
                                        &abstraction.abstract_operator_footprints,
                                        &perim_tcf,
                                    )
                                    .map_err(|error| {
                                        EvaluationError::ComputationFailed(format!(
                                            "failed to reduce abstract-operator Perim residual costs: {error:#}"
                                        ))
                                    })?;
                                log_transition_residual_summary(&remaining_costs);

                                let (all_table, all_tcf) = match abstraction
                                .factory
                                .build_abstract_operator_cost_partitioned_distance_table_from_system_and_footprints_with_deadline(
                                    transition_system,
                                    Some(&abstraction.abstract_operator_footprints),
                                    &remaining_costs,
                                    pos,
                                    None,
                                    deadline,
                                ) {
                                    Ok(result) => result,
                                    Err(error) if Self::is_online_deadline_error(&error) => {
                                        info!(
                                            "scp_online: abstract-operator perimstar/all abstraction {pos} stopped while computing table (deadline)"
                                        );
                                        return Ok(true);
                                    }
                                    Err(error) => {
                                        return Err(EvaluationError::ComputationFailed(format!(
                                        "failed to compute abstract-operator All step for Perimstar: {error:#}"
                                    )));
                                    }
                                };
                                log_transition_table_summary(
                                    "perimstar/all",
                                    pos,
                                    &all_table.distances,
                                    &all_tcf.operator_costs,
                                    abstract_state_ids,
                                );
                                if should_skip_zero_current_table(
                                    "perimstar/all",
                                    pos,
                                    &all_table.distances,
                                    abstract_state_ids,
                                ) {
                                    return Ok(true);
                                }
                                cp.add_h_values(pos, all_table.distances);
                                remaining_costs
                                    .reduce_by_abstract_operator_footprints(
                                        pos,
                                        &abstraction.abstract_operator_footprints,
                                        &all_tcf,
                                    )
                                    .map_err(|error| {
                                        EvaluationError::ComputationFailed(format!(
                                            "failed to reduce abstract-operator All residual costs: {error:#}"
                                        ))
                                    })?;
                                log_transition_residual_summary(&remaining_costs);
                                Ok(true)
                            },
                        )?;
                        if completed.is_none() {
                            info!(
                                "scp_online: abstract-operator PERIMSTAR CP stopped while building abstraction {pos} (deadline)"
                            );
                            break;
                        }
                        if completed == Some(false) {
                            break;
                        }
                    }
                }
            } else {
                self.add_transition_pdb_step(
                    &mut cp,
                    &mut remaining_costs,
                    pos,
                    order,
                    abstract_state_ids,
                    num_domain_abstractions,
                )?;
            }
        }

        Ok(cp)
    }

    fn add_transition_pdb_step(
        &self,
        cp: &mut CostPartitioningHeuristic,
        remaining_costs: &mut TransitionResidualCosts,
        pos: usize,
        _order: &[usize],
        abstract_state_ids: &[Option<usize>],
        num_domain_abstractions: usize,
    ) -> Result<(), EvaluationError> {
        let pdb_id = pos - num_domain_abstractions;
        let Some(pdb) = self.pdbs.get(pdb_id) else {
            return Ok(());
        };

        let mut remaining_operator_costs = remaining_costs.operator_costs_for_label_cp();
        match self.config.saturator {
            Saturator::All => {
                let (distances, saturated) = pdb
                    .build_cost_partitioned_distance_table(&remaining_operator_costs)
                    .map_err(|error| {
                        EvaluationError::ComputationFailed(format!(
                            "failed to compute PDB SCP table {pdb_id}: {error}"
                        ))
                    })?;
                cp.add_pdb_h_values(pos, distances);
                remaining_costs
                    .reduce_operator_costs_uniform(&saturated)
                    .map_err(|error| {
                        EvaluationError::ComputationFailed(format!(
                            "failed to reduce PDB residual costs: {error:#}"
                        ))
                    })?;
            }
            Saturator::Perim => {
                let h_cap = abstract_state_ids
                    .get(pos)
                    .copied()
                    .flatten()
                    .and_then(|sid| {
                        pdb.build_goal_distances(&remaining_operator_costs)
                            .ok()
                            .and_then(|dists| dists.get(sid).copied())
                    })
                    .unwrap_or(f64::INFINITY);
                let (distances, saturated) = pdb
                    .build_cost_partitioned_distance_table_capped(&remaining_operator_costs, h_cap)
                    .map_err(|error| {
                        EvaluationError::ComputationFailed(format!(
                            "failed to compute PDB PERIM table {pdb_id}: {error}"
                        ))
                    })?;
                cp.add_pdb_h_values(pos, distances);
                remaining_costs
                    .reduce_operator_costs_uniform(&saturated)
                    .map_err(|error| {
                        EvaluationError::ComputationFailed(format!(
                            "failed to reduce PDB PERIM residual costs: {error:#}"
                        ))
                    })?;
            }
            Saturator::Perimstar => {
                let h_cap = abstract_state_ids
                    .get(pos)
                    .copied()
                    .flatten()
                    .and_then(|sid| {
                        pdb.build_goal_distances(&remaining_operator_costs)
                            .ok()
                            .and_then(|dists| dists.get(sid).copied())
                    })
                    .unwrap_or(f64::INFINITY);
                let (perim_dists, perim_sat) = pdb
                    .build_cost_partitioned_distance_table_capped(&remaining_operator_costs, h_cap)
                    .map_err(|error| {
                        EvaluationError::ComputationFailed(format!(
                            "failed to compute PDB Perim step for Perimstar {pdb_id}: {error}"
                        ))
                    })?;
                cp.add_pdb_h_values(pos, perim_dists);
                remaining_costs
                    .reduce_operator_costs_uniform(&perim_sat)
                    .map_err(|error| {
                        EvaluationError::ComputationFailed(format!(
                            "failed to reduce PDB Perim residual costs: {error:#}"
                        ))
                    })?;

                remaining_operator_costs = remaining_costs.operator_costs_for_label_cp();
                let (all_dists, all_sat) = pdb
                    .build_cost_partitioned_distance_table(&remaining_operator_costs)
                    .map_err(|error| {
                        EvaluationError::ComputationFailed(format!(
                            "failed to compute PDB All step for Perimstar {pdb_id}: {error}"
                        ))
                    })?;
                cp.add_pdb_h_values(pos, all_dists);
                remaining_costs
                    .reduce_operator_costs_uniform(&all_sat)
                    .map_err(|error| {
                        EvaluationError::ComputationFailed(format!(
                            "failed to reduce PDB All residual costs: {error:#}"
                        ))
                    })?;
            }
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn build_label_cp(
        &self,
        task: &dyn AbstractNumericTask,
        abstractions: &[DomainAbstraction],
        order: &[usize],
        abstract_state_ids: &[Option<usize>],
        num_domain_abstractions: usize,
        original_costs: &[f64],
        deadline: Option<Instant>,
    ) -> Result<CostPartitioningHeuristic, EvaluationError> {
        let mut cp = CostPartitioningHeuristic::default();
        let mut remaining_costs: Vec<f64> = original_costs.to_vec();

        for &pos in order {
            if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                break;
            }
            if pos < num_domain_abstractions {
                let abstraction = &abstractions[pos];
                info!(
                    "scp_online: label CP step abstraction {pos}, abstract_states={}",
                    abstraction_state_count(abstraction)
                );
                match self.config.saturator {
                    Saturator::All => {
                        let abstraction_task = abstraction.task_for_factory(task);
                        let (distances, saturated) = Self::compute_domain_cp_entry(
                            abstraction,
                            abstraction_task,
                            self.config.combine_labels,
                            &remaining_costs,
                        )?;
                        log_label_table_summary(
                            "all",
                            pos,
                            &distances,
                            &saturated,
                            abstract_state_ids,
                        );
                        if should_skip_zero_current_table(
                            "label all",
                            pos,
                            &distances,
                            abstract_state_ids,
                        ) {
                            continue;
                        }
                        cp.add_h_values(pos, distances);
                        reduce_costs(&mut remaining_costs, &saturated)?;
                    }
                    Saturator::Perim => {
                        let h_cap = abstract_state_ids
                            .get(pos)
                            .copied()
                            .flatten()
                            .and_then(|sid| {
                                let abstraction_task = abstraction.task_for_factory(task);
                                abstraction
                                    .factory
                                    .build_goal_distances_for_goals(
                                        abstraction_task,
                                        self.config.combine_labels,
                                        &remaining_costs,
                                        &abstraction.distance_table.goal_facts,
                                    )
                                    .ok()
                                    .and_then(|t| t.distances.get(sid).copied())
                            })
                            .unwrap_or(f64::INFINITY);
                        let abstraction_task = abstraction.task_for_factory(task);
                        let (distances, saturated) = Self::compute_domain_perim_entry(
                            abstraction,
                            abstraction_task,
                            self.config.combine_labels,
                            &remaining_costs,
                            h_cap,
                        )?;
                        log_label_table_summary(
                            "perim",
                            pos,
                            &distances,
                            &saturated,
                            abstract_state_ids,
                        );
                        if should_skip_zero_current_table(
                            "label perim",
                            pos,
                            &distances,
                            abstract_state_ids,
                        ) {
                            continue;
                        }
                        cp.add_h_values(pos, distances);
                        reduce_costs(&mut remaining_costs, &saturated)?;
                    }
                    Saturator::Perimstar => {
                        let h_cap = abstract_state_ids
                            .get(pos)
                            .copied()
                            .flatten()
                            .and_then(|sid| {
                                let abstraction_task = abstraction.task_for_factory(task);
                                abstraction
                                    .factory
                                    .build_goal_distances_for_goals(
                                        abstraction_task,
                                        self.config.combine_labels,
                                        &remaining_costs,
                                        &abstraction.distance_table.goal_facts,
                                    )
                                    .ok()
                                    .and_then(|t| t.distances.get(sid).copied())
                            })
                            .unwrap_or(f64::INFINITY);
                        let abstraction_task = abstraction.task_for_factory(task);
                        let (perim_distances, perim_saturated) = Self::compute_domain_perim_entry(
                            abstraction,
                            abstraction_task,
                            self.config.combine_labels,
                            &remaining_costs,
                            h_cap,
                        )?;
                        log_label_table_summary(
                            "perimstar/perim",
                            pos,
                            &perim_distances,
                            &perim_saturated,
                            abstract_state_ids,
                        );
                        if !should_skip_zero_current_table(
                            "label perimstar/perim",
                            pos,
                            &perim_distances,
                            abstract_state_ids,
                        ) {
                            cp.add_h_values(pos, perim_distances);
                            reduce_costs(&mut remaining_costs, &perim_saturated)?;
                        }

                        let (all_distances, all_saturated) = Self::compute_domain_cp_entry(
                            abstraction,
                            abstraction_task,
                            self.config.combine_labels,
                            &remaining_costs,
                        )?;
                        log_label_table_summary(
                            "perimstar/all",
                            pos,
                            &all_distances,
                            &all_saturated,
                            abstract_state_ids,
                        );
                        if should_skip_zero_current_table(
                            "label perimstar/all",
                            pos,
                            &all_distances,
                            abstract_state_ids,
                        ) {
                            continue;
                        }
                        cp.add_h_values(pos, all_distances);
                        reduce_costs(&mut remaining_costs, &all_saturated)?;
                    }
                }
            } else {
                self.add_label_pdb_step(
                    &mut cp,
                    &mut remaining_costs,
                    pos,
                    abstract_state_ids,
                    num_domain_abstractions,
                )?;
            }
        }

        Ok(cp)
    }

    fn add_label_pdb_step(
        &self,
        cp: &mut CostPartitioningHeuristic,
        remaining_costs: &mut [f64],
        pos: usize,
        abstract_state_ids: &[Option<usize>],
        num_domain_abstractions: usize,
    ) -> Result<(), EvaluationError> {
        let pdb_id = pos - num_domain_abstractions;
        let Some(pdb) = self.pdbs.get(pdb_id) else {
            return Ok(());
        };

        match self.config.saturator {
            Saturator::All => {
                let (distances, saturated) = pdb
                    .build_cost_partitioned_distance_table(remaining_costs)
                    .map_err(|error| {
                        EvaluationError::ComputationFailed(format!(
                            "failed to compute PDB SCP table {pdb_id}: {error}"
                        ))
                    })?;
                if should_skip_zero_current_table("pdb all", pos, &distances, abstract_state_ids) {
                    return Ok(());
                }
                cp.add_pdb_h_values(pos, distances);
                reduce_costs(remaining_costs, &saturated)?;
            }
            Saturator::Perim => {
                let h_cap = abstract_state_ids
                    .get(pos)
                    .copied()
                    .flatten()
                    .and_then(|sid| {
                        pdb.build_goal_distances(remaining_costs)
                            .ok()
                            .and_then(|dists| dists.get(sid).copied())
                    })
                    .unwrap_or(f64::INFINITY);
                let (distances, saturated) = pdb
                    .build_cost_partitioned_distance_table_capped(remaining_costs, h_cap)
                    .map_err(|error| {
                        EvaluationError::ComputationFailed(format!(
                            "failed to compute PDB PERIM table {pdb_id}: {error}"
                        ))
                    })?;
                if should_skip_zero_current_table("pdb perim", pos, &distances, abstract_state_ids)
                {
                    return Ok(());
                }
                cp.add_pdb_h_values(pos, distances);
                reduce_costs(remaining_costs, &saturated)?;
            }
            Saturator::Perimstar => {
                let h_cap = abstract_state_ids
                    .get(pos)
                    .copied()
                    .flatten()
                    .and_then(|sid| {
                        pdb.build_goal_distances(remaining_costs)
                            .ok()
                            .and_then(|dists| dists.get(sid).copied())
                    })
                    .unwrap_or(f64::INFINITY);
                let (perim_dists, perim_sat) = pdb
                    .build_cost_partitioned_distance_table_capped(remaining_costs, h_cap)
                    .map_err(|error| {
                        EvaluationError::ComputationFailed(format!(
                            "failed to compute PDB Perim step for Perimstar {pdb_id}: {error}"
                        ))
                    })?;
                if !should_skip_zero_current_table(
                    "pdb perimstar/perim",
                    pos,
                    &perim_dists,
                    abstract_state_ids,
                ) {
                    cp.add_pdb_h_values(pos, perim_dists);
                    reduce_costs(remaining_costs, &perim_sat)?;
                }

                let (all_dists, all_sat) = pdb
                    .build_cost_partitioned_distance_table(remaining_costs)
                    .map_err(|error| {
                        EvaluationError::ComputationFailed(format!(
                            "failed to compute PDB All step for Perimstar {pdb_id}: {error}"
                        ))
                    })?;
                if should_skip_zero_current_table(
                    "pdb perimstar/all",
                    pos,
                    &all_dists,
                    abstract_state_ids,
                ) {
                    return Ok(());
                }
                cp.add_pdb_h_values(pos, all_dists);
                reduce_costs(remaining_costs, &all_sat)?;
            }
        }

        Ok(())
    }

    fn accept_improved_cp(
        state: &mut ScpOnlineState,
        cp: CostPartitioningHeuristic,
        abstract_state_ids: &[Option<usize>],
        max_h: &mut f64,
    ) {
        let new_h = cp.compute_heuristic(abstract_state_ids);
        if new_h > *max_h {
            let size_kb = cp.estimate_size_in_kb();
            info!(
                "scp_online: accepted CP, h {} -> {}, lookup_tables={}, size={} KiB",
                *max_h,
                new_h,
                cp.lookup_tables.len(),
                size_kb,
            );
            state.size_kb = state.size_kb.saturating_add(cp.estimate_size_in_kb());
            state.cp_heuristics.push(cp);
            state.required_lookup_ids = Self::required_lookup_ids(state);
            *max_h = new_h;
        } else {
            info!(
                "scp_online: rejected CP, candidate_h={} did not improve current_h={}, lookup_tables={}",
                new_h,
                *max_h,
                cp.lookup_tables.len(),
            );
        }
    }

    /// Build a single SCP for one domain abstraction using its own factory.
    fn compute_domain_cp_entry(
        abstraction: &DomainAbstraction,
        task: &dyn AbstractNumericTask,
        combine_labels: bool,
        remaining_costs: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>), EvaluationError> {
        let (table, saturated) = abstraction
            .factory
            .build_cost_partitioned_distance_table_for_goals(
                task,
                combine_labels,
                remaining_costs,
                false,
                &abstraction.distance_table.goal_facts,
            )
            .map_err(|error| {
                EvaluationError::ComputationFailed(format!(
                    "failed to compute SCP table: {error:#}"
                ))
            })?;
        Ok((table.distances, saturated))
    }

    /// Build a PERIM domain CP entry.
    ///
    /// Saturated costs are computed from a table where states outside the
    /// perimeter are inactive. The returned lookup table is then recomputed
    /// globally from those saturated costs, so stored online CPs remain valid
    /// for later states outside the original perimeter.
    fn compute_domain_perim_entry(
        abstraction: &DomainAbstraction,
        task: &dyn AbstractNumericTask,
        combine_labels: bool,
        remaining_costs: &[f64],
        h_cap: f64,
    ) -> Result<(Vec<f64>, Vec<f64>), EvaluationError> {
        let (table, _) = abstraction
            .factory
            .build_cost_partitioned_distance_table_for_goals(
                task,
                combine_labels,
                remaining_costs,
                false,
                &abstraction.distance_table.goal_facts,
            )
            .map_err(|error| {
                EvaluationError::ComputationFailed(format!(
                    "failed to compute SCP table for PERIM: {error:#}"
                ))
            })?;
        let mut perim_distances = table.distances;
        if h_cap.is_finite() {
            for h in &mut perim_distances {
                if !h.is_finite() || *h > h_cap {
                    *h = f64::NEG_INFINITY;
                }
            }
        }
        let mut generator = abstraction
            .factory
            .make_operator_generator(task, combine_labels)
            .map_err(|error| {
                EvaluationError::ComputationFailed(format!(
                    "failed to create operator generator for PERIM: {error:#}"
                ))
            })?;
        let mut operators = generator.build_abstract_operators(task).map_err(|error| {
            EvaluationError::ComputationFailed(format!(
                "failed to build abstract operators for PERIM: {error:#}"
            ))
        })?;
        apply_operator_costs_from_slice(&mut operators, remaining_costs)?;
        let perim_table = AbstractDistanceTable {
            distances: perim_distances,
            generating_op_ids: table.generating_op_ids,
            initial_state_hash: table.initial_state_hash,
            goal_facts: table.goal_facts,
            hash_multipliers: table.hash_multipliers,
            numeric_domain_sizes: table.numeric_domain_sizes,
        };
        let saturated = abstraction
            .factory
            .saturated_costs_for_table(task, combine_labels, &operators, &perim_table)
            .map_err(|error| {
                EvaluationError::ComputationFailed(format!(
                    "failed to compute PERIM saturated costs: {error:#}"
                ))
            })?;
        let global_table = abstraction
            .factory
            .build_goal_distances_for_goals(
                task,
                combine_labels,
                &saturated,
                &abstraction.distance_table.goal_facts,
            )
            .map_err(|error| {
                EvaluationError::ComputationFailed(format!(
                    "failed to compute global PERIM lookup table: {error:#}"
                ))
            })?;
        Ok((global_table.distances, saturated))
    }
}

fn abstraction_state_count(abstraction: &DomainAbstraction) -> u128 {
    abstraction
        .factory
        .domain_sizes()
        .iter()
        .chain(abstraction.factory.numeric_domain_sizes().iter())
        .fold(1_u128, |acc, &size| acc.saturating_mul(size as u128))
}

fn current_h_for_distances(
    abstraction_id: usize,
    distances: &[f64],
    abstract_state_ids: &[Option<usize>],
) -> f64 {
    abstract_state_ids
        .get(abstraction_id)
        .copied()
        .flatten()
        .and_then(|state_id| distances.get(state_id).copied())
        .unwrap_or(0.0)
}

fn should_skip_zero_current_table(
    step: &str,
    abstraction_id: usize,
    distances: &[f64],
    abstract_state_ids: &[Option<usize>],
) -> bool {
    let current_h = current_h_for_distances(abstraction_id, distances, abstract_state_ids);
    if current_h > 1e-9 {
        return false;
    }
    debug!("scp_online: skipping {step} abstraction {abstraction_id}: current_h=0");
    true
}

fn positive_cost_stats(costs: &[f64]) -> (usize, f64) {
    costs
        .iter()
        .copied()
        .filter(|cost| cost.is_finite() && *cost > 0.0)
        .fold((0, 0.0), |(count, total), cost| (count + 1, total + cost))
}

fn log_label_table_summary(
    step: &str,
    abstraction_id: usize,
    distances: &[f64],
    saturated_costs: &[f64],
    abstract_state_ids: &[Option<usize>],
) {
    let (positive_count, total_positive) = positive_cost_stats(saturated_costs);
    let current_h = current_h_for_distances(abstraction_id, distances, abstract_state_ids);
    info!(
        "scp_online: label {step} abstraction {abstraction_id}: current_h={current_h}, positive_saturated_labels={positive_count}, total_positive_saturated={total_positive:.6}"
    );
}

fn log_transition_table_summary(
    step: &str,
    abstraction_id: usize,
    distances: &[f64],
    operator_costs: &[f64],
    abstract_state_ids: &[Option<usize>],
) {
    let (positive_count, total_positive) = positive_cost_stats(operator_costs);
    let current_h = current_h_for_distances(abstraction_id, distances, abstract_state_ids);
    info!(
        "scp_online: abstract-operator {step} abstraction {abstraction_id}: current_h={current_h}, positive_saturated_abstract_ops={positive_count}, total_positive_saturated={total_positive:.6}"
    );
}

fn log_abstract_operator_footprint_summary(
    abstraction_id: usize,
    footprints: &[AbstractOperatorFootprint],
) {
    let total_labels: usize = footprints.iter().map(|fp| fp.labels.len()).sum();
    let non_allocable_labels: usize = footprints
        .iter()
        .flat_map(|fp| fp.labels.iter())
        .filter(|fp| !fp.allocable)
        .count();
    info!(
        "scp_online: abstract-operator footprints abstraction {abstraction_id}: labels={total_labels}, non_allocable_labels={non_allocable_labels}"
    );
}

fn log_transition_residual_summary(remaining_costs: &TransitionResidualCosts) {
    info!(
        "scp_online: abstract-operator residuals now store {} region reductions",
        remaining_costs.num_reductions()
    );
}

fn log_transition_system_summary(
    abstraction_id: usize,
    transition_system: &AbstractTransitionSystem,
    elapsed: Duration,
) {
    let mut abstract_ops = BTreeSet::new();
    let mut concrete_ops = BTreeSet::new();
    let mut label_refs = 0usize;
    for transition in &transition_system.transitions {
        abstract_ops.insert(transition.abstract_op_id);
        label_refs = label_refs.saturating_add(transition.concrete_op_ids.len());
        concrete_ops.extend(transition.concrete_op_ids.iter().copied());
    }
    let transitions = transition_system.transitions.len();
    let avg_labels = if transitions == 0 {
        0.0
    } else {
        label_refs as f64 / transitions as f64
    };
    info!(
        "scp_online: transition system abstraction {abstraction_id}: states={}, transitions={}, duplicate_attempts_skipped={}, abstract_ops={}, concrete_labels={}, avg_labels_per_transition={avg_labels:.3}, build_time={:.3}s",
        transition_system.backward.len(),
        transitions,
        transition_system.duplicate_transition_attempts,
        abstract_ops.len(),
        concrete_ops.len(),
        elapsed.as_secs_f64(),
    );
}

impl Heuristic for SaturatedCostPartitioningOnlineHeuristic<'_> {
    fn compute_heuristic(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError> {
        let task = eval_state.task().ok_or_else(|| {
            EvaluationError::InvalidState(
                "SaturatedCostPartitioningOnlineHeuristic requires task".to_string(),
            )
        })?;
        let build_cp = {
            let state = self.state.borrow();
            self.should_build_cp(&state)
        };

        let mut component_ids = self.component_ids_scratch.borrow_mut();
        let num_domain_abstractions = if build_cp {
            self.compute_abstract_state_ids_into(eval_state, None, &mut component_ids)?
        } else {
            let state = self.state.borrow();
            self.compute_abstract_state_ids_into(
                eval_state,
                Some(&state.required_lookup_ids),
                &mut component_ids,
            )?
        };
        let abstract_state_ids = component_ids.as_slice();

        let mut state = self.state.borrow_mut();
        let mut max_h = Self::compute_max_h(&state, abstract_state_ids);
        if max_h.is_infinite() {
            return Ok(max_h);
        }

        if build_cp {
            self.update_improvement_status(&mut state);
            self.release_abstractions_if_finished(&mut state);
        } else if !state.improve_heuristic && !state.improvement_ended {
            self.release_abstractions_if_finished(&mut state);
        }

        if let Some(cp) = self.maybe_build_cp(
            task,
            &mut state,
            abstract_state_ids,
            num_domain_abstractions,
        )? {
            Self::accept_improved_cp(&mut state, cp, abstract_state_ids, &mut max_h);
        }

        state.evaluated_states = state.evaluated_states.saturating_add(1);
        Ok(max_h)
    }

    fn heuristic_name(&self) -> String {
        self.name.clone()
    }
}

// ---------------------------------------------------------------------------
// Greedy order utilities
// ---------------------------------------------------------------------------

fn compute_score(h: f64, stolen_costs: f64, scoring_function: ScoringFunction) -> f64 {
    match scoring_function {
        ScoringFunction::MaxHeuristic => h,
        ScoringFunction::MinStolenCosts => -stolen_costs,
        ScoringFunction::MaxHeuristicPerStolenCosts => h / stolen_costs.max(1.0),
    }
}

fn compute_stolen_costs(wanted: f64, surplus: f64) -> f64 {
    if !wanted.is_finite() || !surplus.is_finite() {
        return 0.0;
    }
    let rest = surplus + wanted;
    if rest >= 0.0 {
        (0.0_f64).max(wanted - rest)
    } else {
        wanted.max(rest)
    }
}

fn compute_costs_stolen_by_heuristic(saturated: &[f64], surplus: &[f64]) -> f64 {
    saturated
        .iter()
        .zip(surplus.iter())
        .map(|(&s, &su)| compute_stolen_costs(s, su))
        .sum()
}

fn compute_surplus_cost(saturated_by_abs: &[Vec<f64>], op_id: usize, remaining_cost: f64) -> f64 {
    let sum: f64 = saturated_by_abs
        .iter()
        .map(|costs| costs.get(op_id).copied().unwrap_or(f64::NEG_INFINITY))
        .filter(|&w| w > f64::NEG_INFINITY)
        .sum();
    if !remaining_cost.is_finite() || !sum.is_finite() {
        return f64::INFINITY;
    }
    remaining_cost - sum
}

fn compute_all_surplus_costs(costs: &[f64], saturated_by_abs: &[Vec<f64>]) -> Vec<f64> {
    (0..costs.len())
        .map(|op_id| compute_surplus_cost(saturated_by_abs, op_id, costs[op_id]))
        .collect()
}

fn apply_operator_costs_from_slice(
    operators: &mut [AbstractOperator],
    operator_costs: &[f64],
) -> Result<(), EvaluationError> {
    for op in operators {
        if op.concrete_op_ids.is_empty() {
            return Err(EvaluationError::ComputationFailed(
                "abstract operator without concrete labels".to_string(),
            ));
        }
        let mut cost = f64::INFINITY;
        for &concrete_op_id in &op.concrete_op_ids {
            let concrete_cost = operator_costs.get(concrete_op_id).copied().ok_or_else(|| {
                EvaluationError::ComputationFailed(format!(
                    "missing residual cost for concrete operator {concrete_op_id}"
                ))
            })?;
            if !concrete_cost.is_finite() {
                return Err(EvaluationError::ComputationFailed(format!(
                    "residual cost for concrete operator {concrete_op_id} must be finite"
                )));
            }
            cost = cost.min(concrete_cost);
        }
        op.cost = cost;
    }
    Ok(())
}

fn reduce_costs(
    remaining_costs: &mut [f64],
    saturated_costs: &[f64],
) -> Result<(), EvaluationError> {
    if remaining_costs.len() != saturated_costs.len() {
        return Err(EvaluationError::ComputationFailed(format!(
            "cost vector length mismatch: remaining={}, saturated={}",
            remaining_costs.len(),
            saturated_costs.len()
        )));
    }
    for (op_id, (r, s)) in remaining_costs
        .iter_mut()
        .zip(saturated_costs.iter())
        .enumerate()
    {
        if !s.is_finite() {
            continue;
        }
        if *s <= 1e-9 {
            continue;
        }
        let new_remaining = *r - *s;
        if new_remaining < -1e-9 {
            return Err(EvaluationError::ComputationFailed(format!(
                "label residual cost underflow for operator {op_id}: remaining={r}, saturated={s}, result={new_remaining}"
            )));
        }
        *r = new_remaining.max(0.0);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduce_costs_rejects_significant_underflow() {
        let mut remaining = vec![1.0];
        let saturated = vec![1.5];

        let err = reduce_costs(&mut remaining, &saturated).unwrap_err();
        assert!(format!("{err}").contains("underflow"));
    }

    #[test]
    fn reduce_costs_clamps_tiny_negative_roundoff() {
        let mut remaining = vec![1.0];
        let saturated = vec![1.0 + 1e-12];

        reduce_costs(&mut remaining, &saturated).unwrap();
        assert_eq!(remaining, vec![0.0]);
    }
}
