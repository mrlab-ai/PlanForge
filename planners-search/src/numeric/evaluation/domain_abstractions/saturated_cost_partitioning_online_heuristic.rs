use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::time::{Duration, Instant};

use planners_sas::numeric::numeric_task::{
    AbstractNumericTask, metric_operator_cost_from_initial_values,
};
use rand::seq::SliceRandom;
use rand::{SeedableRng, rngs::SmallRng};
use serde::{Deserialize, Serialize};
use tracing::{Level, debug, enabled, info};

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
    AbstractOperatorCostBudget, AbstractOperatorFootprint, NonAllocableFootprintReason,
    TransitionResidualCosts,
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

impl fmt::Display for ScoringFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScoringFunction::MaxHeuristic => write!(f, "max_heuristic"),
            ScoringFunction::MinStolenCosts => write!(f, "min_stolen_costs"),
            ScoringFunction::MaxHeuristicPerStolenCosts => {
                write!(f, "max_heuristic_per_stolen_costs")
            }
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OrderGenerator {
    Greedy,
    DynamicGreedy,
    Random,
}

impl fmt::Display for OrderGenerator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrderGenerator::Greedy => write!(f, "greedy_orders"),
            OrderGenerator::DynamicGreedy => write!(f, "dynamic_greedy_orders"),
            OrderGenerator::Random => write!(f, "random_orders"),
        }
    }
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
    pub table_construction_max_time: f64,
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
    pub order_generator: OrderGenerator,
    pub order_optimization_max_time: f64,
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
            table_construction_max_time: 30.0,
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
            order_generator: OrderGenerator::Greedy,
            order_optimization_max_time: 0.0,
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
    original_operator_costs: Vec<f64>,
    state: RefCell<ScpOnlineState>,
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
            original_operator_costs: original_costs,
            state: RefCell::new(st),
            lookup_scratch: RefCell::new(DomainAbstractionLookupScratch::new()),
            component_ids_scratch: RefCell::new(Vec::new()),
            prop_scratch: RefCell::new(Vec::new()),
            numeric_scratch: RefCell::new(Vec::new()),
            expanded_numeric_scratch: RefCell::new(Vec::new()),
        })
    }

    fn compute_order_for_state(
        &self,
        task: &dyn AbstractNumericTask,
        state: &mut ScpOnlineState,
        abstract_state_ids: &[Option<usize>],
        abstractions: &[DomainAbstraction],
        num_domain_abstractions: usize,
        deadline: Option<Instant>,
    ) -> Result<Vec<usize>, EvaluationError> {
        match self.config.order_generator {
            OrderGenerator::Greedy => Ok(Self::compute_greedy_order_for_state(
                state,
                abstract_state_ids,
                self.config.scoring_function,
                abstractions,
                self.config.use_abstract_operator_cost_partitioning,
            )),
            OrderGenerator::Random => {
                let total = state.h_values_by_abstraction.len();
                let mut order: Vec<usize> = (0..total).collect();
                order.shuffle(&mut state.rng);
                Ok(order)
            }
            OrderGenerator::DynamicGreedy => self.compute_dynamic_greedy_order_for_state(
                task,
                state,
                abstract_state_ids,
                abstractions,
                num_domain_abstractions,
                deadline,
            ),
        }
    }

    fn compute_greedy_order_for_state(
        state: &mut ScpOnlineState,
        abstract_state_ids: &[Option<usize>],
        scoring_function: ScoringFunction,
        abstractions: &[DomainAbstraction],
        use_abstract_operator_cost_partitioning: bool,
    ) -> Vec<usize> {
        let total = state.h_values_by_abstraction.len();
        let mut order: Vec<usize> = (0..total).collect();
        order.shuffle(&mut state.rng);
        let current_h: Vec<f64> = (0..total)
            .map(|abs_id| {
                abstract_state_ids
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
                    .unwrap_or(0.0)
            })
            .collect();

        if use_abstract_operator_cost_partitioning
            && abstractions.iter().any(|abstraction| {
                abstraction.metadata.portfolio_strategy.as_deref() == Some("complementary")
            })
        {
            order.sort_by(|&a, &b| {
                let a_inactive = current_h.get(a).copied().unwrap_or(0.0) <= 1e-9;
                let b_inactive = current_h.get(b).copied().unwrap_or(0.0) <= 1e-9;
                a_inactive
                    .cmp(&b_inactive)
                    .then_with(|| {
                        abstraction_collection_iteration(abstractions, a)
                            .cmp(&abstraction_collection_iteration(abstractions, b))
                    })
                    .then_with(|| a.cmp(&b))
            });
            return order;
        }

        let scores: Vec<f64> = (0..total)
            .map(|abs_id| {
                let h = current_h[abs_id];
                let stolen = state
                    .stolen_costs_by_abstraction
                    .get(abs_id)
                    .copied()
                    .unwrap_or(0.0);
                if use_abstract_operator_cost_partitioning
                    && let Some(abstraction) = abstractions.get(abs_id)
                {
                    compute_abstract_operator_order_score(h, stolen, scoring_function, abstraction)
                } else {
                    compute_score(h, stolen, scoring_function)
                }
            })
            .collect();

        order.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        order
    }

    fn compute_dynamic_greedy_order_for_state(
        &self,
        task: &dyn AbstractNumericTask,
        state: &mut ScpOnlineState,
        abstract_state_ids: &[Option<usize>],
        abstractions: &[DomainAbstraction],
        num_domain_abstractions: usize,
        deadline: Option<Instant>,
    ) -> Result<Vec<usize>, EvaluationError> {
        if self.config.use_abstract_operator_cost_partitioning {
            return Err(EvaluationError::ComputationFailed(
                "dynamic_greedy_orders is only implemented for label SCP; abstract-operator SCP needs residual abstract-operator scoring, not label-order scoring".to_string(),
            ));
        }

        let total = state.h_values_by_abstraction.len();
        let mut remaining_components: Vec<usize> = (0..total).collect();
        let mut remaining_costs = self.original_operator_costs.clone();
        let mut order = Vec::with_capacity(total);

        while !remaining_components.is_empty() {
            if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                break;
            }

            remaining_components.shuffle(&mut state.rng);
            let mut candidate_saturated_costs = Vec::with_capacity(remaining_components.len());
            let mut candidate_h_values = Vec::with_capacity(remaining_components.len());

            for &pos in &remaining_components {
                if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                    break;
                }
                let (distances, saturated) = if pos < num_domain_abstractions {
                    let abstraction = &abstractions[pos];
                    let abstraction_task = abstraction.task_for_factory(task);
                    Self::compute_domain_cp_entry(
                        abstraction,
                        abstraction_task,
                        self.config.combine_labels,
                        &remaining_costs,
                    )?
                } else {
                    let pdb_id = pos - num_domain_abstractions;
                    let pdb = self.pdbs.get(pdb_id).ok_or_else(|| {
                        EvaluationError::ComputationFailed(format!(
                            "dynamic order requested missing PDB component {pdb_id}"
                        ))
                    })?;
                    pdb.build_cost_partitioned_distance_table(&remaining_costs)
                        .map_err(|error| {
                            EvaluationError::ComputationFailed(format!(
                                "failed to compute PDB dynamic-order table {pdb_id}: {error}"
                            ))
                        })?
                };
                let h = current_h_for_distances(pos, &distances, abstract_state_ids);
                candidate_h_values.push(h);
                candidate_saturated_costs.push(saturated);
            }

            if candidate_saturated_costs.len() != remaining_components.len() {
                break;
            }

            let surplus_costs =
                compute_all_surplus_costs(&remaining_costs, &candidate_saturated_costs);
            let mut best_index = None;
            let mut best_score = f64::NEG_INFINITY;
            for rem_index in 0..remaining_components.len() {
                let stolen = compute_costs_stolen_by_heuristic(
                    &candidate_saturated_costs[rem_index],
                    &surplus_costs,
                );
                let score = compute_score(
                    candidate_h_values[rem_index],
                    stolen,
                    self.config.scoring_function,
                );
                if best_index.is_none() || score > best_score {
                    best_index = Some(rem_index);
                    best_score = score;
                }
            }
            let best_index = best_index.ok_or_else(|| {
                EvaluationError::ComputationFailed(
                    "dynamic order generator had no remaining candidate".to_string(),
                )
            })?;
            order.push(remaining_components[best_index]);
            reduce_costs(&mut remaining_costs, &candidate_saturated_costs[best_index])?;
            remaining_components.swap_remove(best_index);
        }

        order.extend(remaining_components);
        Ok(order)
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
            for (pdb_id, pdb) in self.pdbs.iter().enumerate() {
                let mut expanded = self.expanded_numeric_scratch.borrow_mut();
                expanded.clear();
                pdb.expand_numeric_state_values_into(&numeric, &mut expanded)
                    .map_err(EvaluationError::ComputationFailed)?;
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

        let abstractions_guard = self.abstractions.borrow();
        let abstractions: &[DomainAbstraction] = match &*abstractions_guard {
            Some(abs) => abs.as_slice(),
            None => &[],
        };

        if abstractions.is_empty() && self.pdbs.is_empty() {
            return Ok(None);
        }
        let original_costs = self.original_operator_costs.as_slice();
        let deadline = self
            .config
            .table_construction_max_time
            .is_finite()
            .then(|| {
                Instant::now() + Duration::from_secs_f64(self.config.table_construction_max_time)
            });
        let standalone_current_h =
            standalone_current_h_values(state, abstract_state_ids, num_domain_abstractions);
        let mut order = self.compute_order_for_state(
            task,
            state,
            abstract_state_ids,
            abstractions,
            num_domain_abstractions,
            deadline,
        )?;
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
        if self.config.collection_config.debug {
            log_abstraction_candidate_report(
                mode,
                state,
                abstractions,
                &order,
                abstract_state_ids,
                self.config.scoring_function,
                self.config.use_abstract_operator_cost_partitioning,
            );
        }

        let mut cp = if self.config.use_abstract_operator_cost_partitioning {
            self.build_abstract_operator_cp(
                task,
                abstractions,
                &order,
                abstract_state_ids,
                &standalone_current_h,
                num_domain_abstractions,
                original_costs,
                deadline,
                self.config.saturator,
                None,
            )?
        } else {
            self.build_label_cp(
                task,
                abstractions,
                &order,
                abstract_state_ids,
                num_domain_abstractions,
                original_costs,
                deadline,
            )?
        };

        if self.config.order_optimization_max_time > 0.0 {
            let optimization_deadline =
                self.config
                    .order_optimization_max_time
                    .is_finite()
                    .then(|| {
                        Instant::now()
                            + Duration::from_secs_f64(self.config.order_optimization_max_time)
                    });
            self.optimize_order_with_hill_climbing(
                task,
                abstractions,
                &standalone_current_h,
                num_domain_abstractions,
                original_costs,
                abstract_state_ids,
                &mut order,
                &mut cp,
                optimization_deadline,
            )?;
        }

        if cp.is_empty() {
            info!("scp_online: {mode} CP attempt produced no lookup tables");
        }
        Ok((!cp.is_empty()).then_some(cp))
    }

    #[allow(clippy::too_many_arguments)]
    fn optimize_order_with_hill_climbing(
        &self,
        task: &dyn AbstractNumericTask,
        abstractions: &[DomainAbstraction],
        standalone_current_h: &[f64],
        num_domain_abstractions: usize,
        original_costs: &[f64],
        abstract_state_ids: &[Option<usize>],
        incumbent_order: &mut [usize],
        incumbent_cp: &mut CostPartitioningHeuristic,
        optimization_deadline: Option<Instant>,
    ) -> Result<(), EvaluationError> {
        let mut incumbent_h = incumbent_cp.compute_heuristic(abstract_state_ids);
        if self.config.collection_config.debug {
            info!("scp_online: order optimization incumbent_h={incumbent_h}");
        }

        loop {
            if optimization_deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                break;
            }
            let mut improved = false;
            for i in 0..incumbent_order.len() {
                for j in (i + 1)..incumbent_order.len() {
                    if optimization_deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                        return Ok(());
                    }

                    incumbent_order.swap(i, j);
                    let neighbor_cp = if self.config.use_abstract_operator_cost_partitioning {
                        self.build_abstract_operator_cp(
                            task,
                            abstractions,
                            incumbent_order,
                            abstract_state_ids,
                            standalone_current_h,
                            num_domain_abstractions,
                            original_costs,
                            optimization_deadline,
                            self.config.saturator,
                            None,
                        )?
                    } else {
                        self.build_label_cp(
                            task,
                            abstractions,
                            incumbent_order,
                            abstract_state_ids,
                            num_domain_abstractions,
                            original_costs,
                            optimization_deadline,
                        )?
                    };
                    let neighbor_h = neighbor_cp.compute_heuristic(abstract_state_ids);
                    if neighbor_h > incumbent_h {
                        if self.config.collection_config.debug {
                            info!(
                                "scp_online: order optimization swapped positions {i}/{j}, h {incumbent_h} -> {neighbor_h}"
                            );
                        }
                        *incumbent_cp = neighbor_cp;
                        incumbent_h = neighbor_h;
                        improved = true;
                        break;
                    }

                    incumbent_order.swap(i, j);
                }
                if improved {
                    break;
                }
            }
            if !improved {
                break;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn build_abstract_operator_cp(
        &self,
        task: &dyn AbstractNumericTask,
        abstractions: &[DomainAbstraction],
        order: &[usize],
        abstract_state_ids: &[Option<usize>],
        standalone_current_h: &[f64],
        num_domain_abstractions: usize,
        original_costs: &[f64],
        deadline: Option<Instant>,
        saturator: Saturator,
        budgets: Option<&[Vec<AbstractOperatorCostBudget>]>,
    ) -> Result<CostPartitioningHeuristic, EvaluationError> {
        let mut cp = CostPartitioningHeuristic::default();
        let mut remaining_costs = TransitionResidualCosts::from_operator_costs(original_costs);
        let label_rescue_operator_ids_by_abstraction =
            compute_suffix_label_rescue_operator_ids(abstractions, order, num_domain_abstractions);

        for &pos in order {
            if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                break;
            }
            if pos < num_domain_abstractions {
                let abstraction = &abstractions[pos];
                info!(
                    "scp_online: abstract-operator CP step abstraction {pos}, abstract_states={}, standalone_h={}, metadata={}",
                    abstraction_state_count(abstraction),
                    standalone_current_h.get(pos).copied().unwrap_or(0.0),
                    abstraction_metadata_summary(abstraction),
                );
                log_abstract_operator_footprint_summary(
                    pos,
                    &abstraction.abstract_operator_footprints,
                );
                let abstraction_task = abstraction.task_for_factory(task);
                match saturator {
                    Saturator::All => {
                        self.log_abstract_operator_label_diagnostic(
                            abstraction,
                            abstraction_task,
                            pos,
                            abstract_state_ids,
                            &remaining_costs,
                        )?;
                        let (table, tcf) = match abstraction
                                .factory
                                .build_abstract_operator_cost_partitioned_distance_table_with_operators_and_footprints_with_deadline(
                                    abstraction_task,
                                    abstraction.combine_labels,
                                    &abstraction.abstract_operators,
                                    &abstraction.abstract_operator_footprints,
                                    budgets.and_then(|budgets| budgets.get(pos).map(Vec::as_slice)),
                                    label_rescue_operator_ids_by_abstraction
                                        .get(pos),
                                    &remaining_costs,
                                    pos,
                                    abstract_state_ids.get(pos).copied().flatten(),
                                    None,
                                    deadline,
                                ) {
                                    Ok(result) => result,
                                    Err(error) if Self::is_online_deadline_error(&error) => {
                                        info!(
                                            "scp_online: abstract-operator all abstraction {pos} stopped while computing table (deadline)"
                                        );
                                        break;
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
                            continue;
                        }
                        cp.add_h_values(pos, table.distances);
                        remaining_costs
                            .reduce_by_abstract_operator_footprints(
                                pos,
                                &abstraction.abstract_operator_footprints,
                                label_rescue_operator_ids_by_abstraction.get(pos),
                                &tcf,
                            )
                            .map_err(|error| {
                                EvaluationError::ComputationFailed(format!(
                                    "failed to reduce abstract-operator residual costs: {error:#}"
                                ))
                            })?;
                        log_transition_residual_summary(&remaining_costs);
                    }
                    Saturator::Perim => {
                        let cap_state_id = abstract_state_ids.get(pos).copied().flatten();
                        self.log_abstract_operator_label_diagnostic(
                            abstraction,
                            abstraction_task,
                            pos,
                            abstract_state_ids,
                            &remaining_costs,
                        )?;
                        let (table, tcf) = match abstraction
                                .factory
                                .build_abstract_operator_cost_partitioned_distance_table_with_operators_and_footprints_with_deadline(
                                    abstraction_task,
                                    abstraction.combine_labels,
                                    &abstraction.abstract_operators,
                                    &abstraction.abstract_operator_footprints,
                                    budgets.and_then(|budgets| budgets.get(pos).map(Vec::as_slice)),
                                    label_rescue_operator_ids_by_abstraction
                                        .get(pos),
                                    &remaining_costs,
                                    pos,
                                    cap_state_id,
                                    cap_state_id,
                                    deadline,
                                ) {
                                    Ok(result) => result,
                                    Err(error) if Self::is_online_deadline_error(&error) => {
                                        info!(
                                            "scp_online: abstract-operator perim abstraction {pos} stopped while computing table (deadline)"
                                        );
                                        break;
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
                            continue;
                        }
                        cp.add_h_values(pos, table.distances);
                        remaining_costs
                            .reduce_by_abstract_operator_footprints(
                                pos,
                                &abstraction.abstract_operator_footprints,
                                label_rescue_operator_ids_by_abstraction
                                        .get(pos),
                                &tcf,
                            )
                            .map_err(|error| {
                                EvaluationError::ComputationFailed(format!(
                                    "failed to reduce abstract-operator PERIM residual costs: {error:#}"
                                ))
                            })?;
                        log_transition_residual_summary(&remaining_costs);
                    }
                    Saturator::Perimstar => {
                        let cap_state_id = abstract_state_ids.get(pos).copied().flatten();
                        self.log_abstract_operator_label_diagnostic(
                            abstraction,
                            abstraction_task,
                            pos,
                            abstract_state_ids,
                            &remaining_costs,
                        )?;
                        let (perim_table, perim_tcf) = match abstraction
                                .factory
                                .build_abstract_operator_cost_partitioned_distance_table_with_operators_and_footprints_with_deadline(
                                    abstraction_task,
                                    abstraction.combine_labels,
                                    &abstraction.abstract_operators,
                                    &abstraction.abstract_operator_footprints,
                                    budgets.and_then(|budgets| budgets.get(pos).map(Vec::as_slice)),
                                    label_rescue_operator_ids_by_abstraction
                                        .get(pos),
                                    &remaining_costs,
                                    pos,
                                    cap_state_id,
                                    cap_state_id,
                                    deadline,
                                ) {
                                    Ok(result) => result,
                                    Err(error) if Self::is_online_deadline_error(&error) => {
                                        info!(
                                            "scp_online: abstract-operator perimstar/perim abstraction {pos} stopped while computing table (deadline)"
                                        );
                                        break;
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
                            continue;
                        }
                        cp.add_h_values(pos, perim_table.distances);
                        remaining_costs
                            .reduce_by_abstract_operator_footprints(
                                pos,
                                &abstraction.abstract_operator_footprints,
                                label_rescue_operator_ids_by_abstraction
                                        .get(pos),
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
                                .build_abstract_operator_cost_partitioned_distance_table_with_operators_and_footprints_with_deadline(
                                    abstraction_task,
                                    abstraction.combine_labels,
                                            &abstraction.abstract_operators,
                                            &abstraction.abstract_operator_footprints,
                                            budgets.and_then(|budgets| budgets.get(pos).map(Vec::as_slice)),
                                            label_rescue_operator_ids_by_abstraction
                                        .get(pos),
                                            &remaining_costs,
                                            pos,
                                            cap_state_id,
                                    None,
                                    deadline,
                                ) {
                                    Ok(result) => result,
                                    Err(error) if Self::is_online_deadline_error(&error) => {
                                        info!(
                                            "scp_online: abstract-operator perimstar/all abstraction {pos} stopped while computing table (deadline)"
                                        );
                                        break;
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
                            continue;
                        }
                        cp.add_h_values(pos, all_table.distances);
                        remaining_costs
                            .reduce_by_abstract_operator_footprints(
                                pos,
                                &abstraction.abstract_operator_footprints,
                                label_rescue_operator_ids_by_abstraction
                                        .get(pos),
                                &all_tcf,
                            )
                            .map_err(|error| {
                                EvaluationError::ComputationFailed(format!(
                                    "failed to reduce abstract-operator All residual costs: {error:#}"
                                ))
                            })?;
                        log_transition_residual_summary(&remaining_costs);
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

    fn log_abstract_operator_label_diagnostic(
        &self,
        abstraction: &DomainAbstraction,
        abstraction_task: &dyn AbstractNumericTask,
        abstraction_id: usize,
        abstract_state_ids: &[Option<usize>],
        remaining_costs: &TransitionResidualCosts,
    ) -> Result<(), EvaluationError> {
        if !enabled!(Level::INFO) {
            return Ok(());
        }
        let label_remaining_costs = remaining_costs.operator_costs_for_label_cp();
        let (label_distances, label_saturated) = Self::compute_domain_cp_entry(
            abstraction,
            abstraction_task,
            self.config.combine_labels,
            &label_remaining_costs,
        )?;
        let label_h = current_h_for_distances(abstraction_id, &label_distances, abstract_state_ids);
        let (positive_labels, total_label_saturated) = positive_cost_stats(&label_saturated);
        let stats = abstract_operator_footprint_stats(&abstraction.abstract_operator_footprints);
        let potentially_lost_labels = count_positive_saturated_non_allocable_labels(
            &abstraction.abstract_operator_footprints,
            &label_saturated,
        );
        info!(
            "scp_online: abstract-operator label diagnostic abstraction {abstraction_id}: label_equivalent_h={label_h}, positive_saturated_labels={positive_labels}, total_label_saturated={total_label_saturated:.6}, footprint_labels={}, non_allocable_labels={}, positive_label_non_allocable_footprint_labels={potentially_lost_labels}",
            stats.total_labels, stats.non_allocable_labels,
        );
        log_positive_label_footprint_diagnostics(
            abstraction_id,
            abstraction_task,
            &abstraction.abstract_operator_footprints,
            &label_saturated,
        );
        Ok(())
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

fn abstraction_collection_iteration(
    abstractions: &[DomainAbstraction],
    abstraction_id: usize,
) -> usize {
    abstractions
        .get(abstraction_id)
        .and_then(|abstraction| abstraction.metadata.collection_iteration)
        .unwrap_or(usize::MAX)
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
    if !enabled!(Level::INFO) {
        return;
    }
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
    if !enabled!(Level::INFO) {
        return;
    }
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
    if !enabled!(Level::INFO) {
        return;
    }
    let stats = abstract_operator_footprint_stats(footprints);
    let non_allocable_ratio = if stats.total_labels == 0 {
        0.0
    } else {
        stats.non_allocable_labels as f64 / stats.total_labels as f64
    };
    info!(
        "scp_online: abstract-operator footprints abstraction {abstraction_id}: labels={}, non_allocable_labels={}, non_allocable_ratio={non_allocable_ratio:.3}, infinite_source={}, unsupported_effect={}",
        stats.total_labels,
        stats.non_allocable_labels,
        stats.infinite_active_source,
        stats.unsupported_effect_image,
    );
}

fn log_abstraction_candidate_report(
    mode: &str,
    state: &ScpOnlineState,
    abstractions: &[DomainAbstraction],
    order: &[usize],
    abstract_state_ids: &[Option<usize>],
    scoring_function: ScoringFunction,
    use_abstract_operator_cost_partitioning: bool,
) {
    let inactive = order
        .iter()
        .filter(|&&abstraction_id| {
            state
                .h_values_by_abstraction
                .get(abstraction_id)
                .map(|distances| {
                    current_h_for_distances(abstraction_id, distances, abstract_state_ids)
                })
                .unwrap_or(0.0)
                <= 1e-9
        })
        .count();
    info!(
        "scp_online: {mode} abstraction candidate report, candidates={}, inactive_current_state={inactive}, showing_top={}",
        order.len(),
        order.len().min(25),
    );

    for (rank, &abstraction_id) in order.iter().take(25).enumerate() {
        let h = state
            .h_values_by_abstraction
            .get(abstraction_id)
            .map(|distances| current_h_for_distances(abstraction_id, distances, abstract_state_ids))
            .unwrap_or(0.0);
        let stolen = state
            .stolen_costs_by_abstraction
            .get(abstraction_id)
            .copied()
            .unwrap_or(0.0);
        let Some(abstraction) = abstractions.get(abstraction_id) else {
            info!(
                "scp_online: candidate rank={rank}, id={abstraction_id}, h={h}, stolen={stolen}, missing_domain_abstraction=true"
            );
            continue;
        };
        let score = if use_abstract_operator_cost_partitioning {
            compute_abstract_operator_order_score(h, stolen, scoring_function, abstraction)
        } else {
            compute_score(h, stolen, scoring_function)
        };
        let stats = abstract_operator_footprint_stats(&abstraction.abstract_operator_footprints);
        let allocable_ratio = if stats.total_labels == 0 {
            0.0
        } else {
            (stats
                .total_labels
                .saturating_sub(stats.non_allocable_labels)) as f64
                / stats.total_labels as f64
        };
        let metadata = &abstraction.metadata;
        let seeds = truncate_for_log(&metadata.initial_seed_splits.join("|"), 220);
        info!(
            "scp_online: candidate rank={rank}, id={abstraction_id}, score={score:.6}, h={h}, stolen={stolen:.6}, states={}, abstract_ops={}, footprint_labels={}, allocable_ratio={allocable_ratio:.3}, infinite_source={}, unsupported_effect={}, iteration={:?}, flaw_kind={:?}, full_goal_task={:?}, seeds={seeds}",
            abstraction_state_count(abstraction),
            abstraction.abstract_operators.len(),
            stats.total_labels,
            stats.infinite_active_source,
            stats.unsupported_effect_image,
            metadata.collection_iteration,
            metadata.flaw_kind,
            metadata.full_goal_task,
        );
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct AbstractOperatorFootprintStats {
    total_labels: usize,
    non_allocable_labels: usize,
    infinite_active_source: usize,
    unsupported_effect_image: usize,
}

fn abstract_operator_footprint_stats(
    footprints: &[AbstractOperatorFootprint],
) -> AbstractOperatorFootprintStats {
    let mut stats = AbstractOperatorFootprintStats::default();
    for label in footprints.iter().flat_map(|fp| fp.labels.iter()) {
        stats.total_labels = stats.total_labels.saturating_add(1);
        if !label.allocable {
            stats.non_allocable_labels = stats.non_allocable_labels.saturating_add(1);
        }
        match label.non_allocable_reason {
            Some(NonAllocableFootprintReason::InfiniteActiveSource) => {
                stats.infinite_active_source = stats.infinite_active_source.saturating_add(1);
            }
            Some(NonAllocableFootprintReason::UnsupportedEffectImage) => {
                stats.unsupported_effect_image = stats.unsupported_effect_image.saturating_add(1);
            }
            Some(NonAllocableFootprintReason::UninformativeSource) => {}
            None => {}
        }
    }
    stats
}

fn compute_suffix_label_rescue_operator_ids(
    abstractions: &[DomainAbstraction],
    order: &[usize],
    num_domain_abstractions: usize,
) -> Vec<HashSet<usize>> {
    let mut rescue_by_abstraction = vec![HashSet::new(); abstractions.len()];
    let mut suffix_allocable = HashSet::new();

    for &abstraction_id in order.iter().rev() {
        if abstraction_id >= num_domain_abstractions {
            continue;
        }
        let abstraction = &abstractions[abstraction_id];
        let mut current_allocable = HashSet::new();
        let mut current_rescue_candidates = HashSet::new();
        for label in abstraction
            .abstract_operator_footprints
            .iter()
            .flat_map(|footprint| footprint.labels.iter())
        {
            if label.allocable {
                current_allocable.insert(label.concrete_op_id);
            } else if matches!(
                label.non_allocable_reason,
                Some(
                    NonAllocableFootprintReason::InfiniteActiveSource
                        | NonAllocableFootprintReason::UninformativeSource
                )
            ) {
                current_rescue_candidates.insert(label.concrete_op_id);
            }
        }

        let mut current_and_later_allocable = suffix_allocable.clone();
        current_and_later_allocable.extend(current_allocable.iter().copied());
        let blocking_allocable = if abstraction.metadata.full_goal_task == Some(true) {
            &current_allocable
        } else {
            &current_and_later_allocable
        };
        rescue_by_abstraction[abstraction_id] = current_rescue_candidates
            .into_iter()
            .filter(|op_id| !blocking_allocable.contains(op_id))
            .collect();
        suffix_allocable = current_and_later_allocable;
    }

    rescue_by_abstraction
}

fn count_positive_saturated_non_allocable_labels(
    footprints: &[AbstractOperatorFootprint],
    label_saturated_costs: &[f64],
) -> usize {
    footprints
        .iter()
        .flat_map(|footprint| footprint.labels.iter())
        .filter(|label| {
            !label.allocable
                && label_saturated_costs
                    .get(label.concrete_op_id)
                    .is_some_and(|cost| cost.is_finite() && *cost > 1e-9)
        })
        .count()
}

#[derive(Debug, Default)]
struct LabelFootprintReasonCounts {
    allocable: usize,
    infinite_source: usize,
    uninformative_source: usize,
    unsupported_effect: usize,
}

fn log_positive_label_footprint_diagnostics(
    abstraction_id: usize,
    task: &dyn AbstractNumericTask,
    footprints: &[AbstractOperatorFootprint],
    label_saturated_costs: &[f64],
) {
    if !enabled!(Level::INFO) {
        return;
    }
    let mut counts_by_label: HashMap<usize, LabelFootprintReasonCounts> = HashMap::new();
    for label in footprints
        .iter()
        .flat_map(|footprint| footprint.labels.iter())
    {
        let counts = counts_by_label.entry(label.concrete_op_id).or_default();
        if label.allocable {
            counts.allocable += 1;
        } else {
            match label.non_allocable_reason {
                Some(NonAllocableFootprintReason::InfiniteActiveSource) => {
                    counts.infinite_source += 1;
                }
                Some(NonAllocableFootprintReason::UninformativeSource) => {
                    counts.uninformative_source += 1;
                }
                Some(NonAllocableFootprintReason::UnsupportedEffectImage) => {
                    counts.unsupported_effect += 1;
                }
                None => {
                    counts.unsupported_effect += 1;
                }
            }
        }
    }

    let mut positive_labels = label_saturated_costs
        .iter()
        .enumerate()
        .filter_map(|(concrete_op_id, &cost)| {
            if cost.is_finite() && cost > 1e-9 {
                Some((concrete_op_id, cost))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    positive_labels.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });

    for (rank, (concrete_op_id, saturated_cost)) in positive_labels.into_iter().take(12).enumerate()
    {
        let counts = counts_by_label.get(&concrete_op_id);
        let (allocable, infinite, uninformative, unsupported) = counts
            .map(|counts| {
                (
                    counts.allocable,
                    counts.infinite_source,
                    counts.uninformative_source,
                    counts.unsupported_effect,
                )
            })
            .unwrap_or((0, 0, 0, 0));
        let op = task.get_operators().get(concrete_op_id);
        let op_name = op.map(|op| op.name()).unwrap_or("<missing operator>");
        let numeric_effects = op.map(|op| op.assignment_effects().len()).unwrap_or(0);
        info!(
            "scp_online: abstract-operator label diagnostic detail abstraction {abstraction_id}: rank={rank}, label={concrete_op_id}, saturated={saturated_cost:.6}, numeric_effects={numeric_effects}, footprint_allocable={allocable}, footprint_infinite={infinite}, footprint_uninformative={uninformative}, footprint_unsupported={unsupported}, op={op_name}"
        );
    }
}

fn abstraction_metadata_summary(abstraction: &DomainAbstraction) -> String {
    let metadata = &abstraction.metadata;
    format!(
        "iteration={:?},strategy={:?},flaw_kind={:?},full_goal_task={:?},seeds={}",
        metadata.collection_iteration,
        metadata.portfolio_strategy,
        metadata.flaw_kind,
        metadata.full_goal_task,
        metadata.initial_seed_splits.join("|"),
    )
}

fn truncate_for_log(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    let mut truncated = value.chars().take(max_chars).collect::<String>();
    truncated.push_str("...");
    truncated
}

fn log_transition_residual_summary(remaining_costs: &TransitionResidualCosts) {
    if !enabled!(Level::INFO) {
        return;
    }
    info!(
        "scp_online: abstract-operator residuals now store {} region reductions",
        remaining_costs.num_reductions()
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

#[cfg(test)]
mod handcrafted_sailing_tests {
    use std::cell::{Ref, RefMut};
    use std::collections::HashSet;
    use std::path::{Path, PathBuf};

    use planners_preprocess::run_preprocess_to_output;
    use planners_sas::numeric::axioms::{AssignmentAxiom, ComparisonAxiom, PropositionalAxiom};
    use planners_sas::numeric::numeric_task::{
        AbstractNumericTask, ExplicitFact, ExplicitVariable, Metric, NumericRootTask, NumericType,
        NumericVariable, Operator,
    };
    use planners_translator::translate_to_sas_to_path_fast;

    use super::*;
    use crate::numeric::evaluation::domain_abstractions::cegar::InitialSeedSplit;
    use crate::numeric::evaluation::domain_abstractions::domain_abstraction::NumericPartitions;
    use crate::numeric::evaluation::domain_abstractions::domain_abstraction_factory::DomainAbstractionFactory;
    use crate::numeric::evaluation::domain_abstractions::domain_abstraction_generator::{
        DomainAbstractionMetadata, compute_hash_multipliers, prepare_domain_abstraction_task,
    };

    #[test]
    #[ignore = "diagnostic full-task handcrafted sailing abstract-operator SCP report"]
    fn sailing_handcrafted_four_abstractions_full_task_abstract_operator_scp_initial_h_report() {
        let task = translated_sailing_2_2_task();
        let prepared = prepare_domain_abstraction_task(&task, true)
            .expect("sailing task should support transformed linear views");
        let transformed_task = prepared.task_for(&task);
        let specs = handcrafted_full_task_specs(transformed_task);
        assert_eq!(specs.len(), 4);

        let mut abstractions = Vec::new();
        for (index, spec) in specs.iter().enumerate() {
            let single_goal_task = SingleGoalTask::new(transformed_task, spec.goal.clone());
            let mut abstraction = build_handcrafted_abstraction(&single_goal_task, &prepared, spec)
                .unwrap_or_else(|error| panic!("failed to build {}: {error:#}", spec.name));
            abstraction.metadata = DomainAbstractionMetadata {
                collection_iteration: Some(index + 1),
                portfolio_strategy: Some("handcrafted_full_task_sailing".to_string()),
                flaw_kind: None,
                full_goal_task: Some(false),
                initial_seed_splits: spec.seed_splits.iter().map(seed_description).collect(),
                max_abstraction_size: Some(10_000),
            };
            let states = abstraction_state_count(&abstraction);
            assert!(
                states <= 10_000,
                "{} has {states} states, expected at most 10000",
                spec.name
            );
            abstractions.push(abstraction);
        }

        let config = ScpOnlineConfig {
            max_time: 300.0,
            table_construction_max_time: 30.0,
            max_size: 10_000_000,
            interval: usize::MAX,
            combine_labels: false,
            collection_config: DomainAbstractionCollectionGeneratorMultipleCegarConfig {
                debug: true,
                transform_linear_task: true,
                ..Default::default()
            },
            use_numeric_pdbs: false,
            max_pdb_states: 0,
            max_pattern_size: 0,
            only_interesting_patterns: true,
            pdb_exploration_heuristic: PdbInternalHeuristic::Blind,
            pdb_frontier_heuristic: PdbInternalHeuristic::Zero,
            pdb_failed_lookup_heuristic: PdbInternalHeuristic::Zero,
            scoring_function: ScoringFunction::MaxHeuristic,
            order_generator: OrderGenerator::Greedy,
            order_optimization_max_time: 0.0,
            saturator: Saturator::All,
            random_seed: Some(1),
            use_abstract_operator_cost_partitioning: true,
        };

        let heuristic = SaturatedCostPartitioningOnlineHeuristic::new(
            None,
            abstractions,
            vec![],
            config,
            &task,
        )
        .expect("failed to construct SCP heuristic");
        let abstract_state_ids = initial_abstract_state_ids(&heuristic, &task);
        {
            let mut state = heuristic.state.borrow_mut();
            let mut max_h = SaturatedCostPartitioningOnlineHeuristic::compute_max_h(
                &state,
                &abstract_state_ids,
            );
            let cp = heuristic
                .maybe_build_cp(&task, &mut state, &abstract_state_ids, specs.len())
                .expect("initial SCP construction failed")
                .expect("initial SCP should produce a cost partitioning");
            SaturatedCostPartitioningOnlineHeuristic::accept_improved_cp(
                &mut state,
                cp,
                &abstract_state_ids,
                &mut max_h,
            );
        }

        let state = heuristic.state.borrow();
        let initial_h =
            SaturatedCostPartitioningOnlineHeuristic::compute_max_h(&state, &abstract_state_ids);
        let abstractions_ref = heuristic.abstractions.borrow();
        let abstractions = abstractions_ref
            .as_ref()
            .expect("diagnostic keeps abstractions alive");

        println!("HANDCRAFTED_FULL_TASK_INITIAL_H {initial_h}");
        let mut contributions = vec![0.0; specs.len()];
        for cp in &state.cp_heuristics {
            for table in &cp.lookup_tables {
                let contribution = abstract_state_ids
                    .get(table.abstraction_id)
                    .copied()
                    .flatten()
                    .and_then(|state_id| table.distances.get(state_id).copied())
                    .unwrap_or(table.unknown_value);
                if let Some(total) = contributions.get_mut(table.abstraction_id) {
                    *total += contribution;
                }
            }
        }

        for (index, (spec, abstraction)) in specs.iter().zip(abstractions).enumerate() {
            let standalone_h = current_h_for_distances(
                index,
                &abstraction.distance_table.distances,
                &abstract_state_ids,
            );
            println!(
                "HANDCRAFTED_FULL_TASK_ABS index={index} name={} standalone_h={standalone_h} scp_contribution={} states={} abstract_ops={} views={}",
                spec.name,
                contributions[index],
                abstraction_state_count(abstraction),
                abstraction.abstract_operators.len(),
                partition_report(transformed_task, abstraction, &spec.view_ids),
            );
        }
        println!("HANDCRAFTED_FULL_TASK_CONTRIBUTIONS {contributions:?}");

        assert!(
            initial_h.is_finite(),
            "initial h must be finite, got {initial_h}"
        );
        assert!(initial_h > 0.0, "initial h should be positive");
        assert!(
            initial_h <= 76.0,
            "full-task diagnostic should not exceed the known optimal cost 76, got {initial_h}"
        );
    }

    #[derive(Debug, Clone)]
    struct HandcraftedSpec {
        name: String,
        goal: ExplicitFact,
        view_ids: Vec<usize>,
        seed_splits: Vec<InitialSeedSplit>,
    }

    fn handcrafted_full_task_specs(task: &dyn AbstractNumericTask) -> Vec<HandcraftedSpec> {
        [
            ("p1-u", "p1", ViewKind::Sum),
            ("p1-v", "p1", ViewKind::Difference),
            ("p0-u", "p0", ViewKind::Sum),
            ("p0-v", "p0", ViewKind::Difference),
        ]
        .into_iter()
        .map(|(name, person, view_kind)| {
            let view_ids = ["b0", "b1"]
                .into_iter()
                .map(|boat| find_sailing_view(task, boat, person, view_kind))
                .collect::<Vec<_>>();
            let seed_splits = route_seed_splits(task, &view_ids);
            HandcraftedSpec {
                name: name.to_string(),
                goal: find_saved_goal_fact(task, person),
                view_ids,
                seed_splits,
            }
        })
        .collect()
    }

    fn build_handcrafted_abstraction(
        transformed_task: &dyn AbstractNumericTask,
        prepared: &crate::numeric::evaluation::domain_abstractions::domain_abstraction_generator::PreparedDomainAbstractionTask,
        spec: &HandcraftedSpec,
    ) -> anyhow::Result<DomainAbstraction> {
        let mut partitions = NumericPartitions::trivial(transformed_task);
        for seed in &spec.seed_splits {
            let InitialSeedSplit::Numeric {
                numeric_var_id,
                value,
                include_in_lower,
            } = seed
            else {
                continue;
            };
            partitions.split_at(*numeric_var_id, *value, *include_in_lower);
        }

        let goal_vars = (0..transformed_task.get_num_goals())
            .map(|goal_id| transformed_task.get_goal_fact(goal_id).var)
            .collect::<HashSet<_>>();
        let domain_mapping = (0..transformed_task.get_num_variables())
            .map(|var_id| {
                let domain_size = transformed_task
                    .get_variable_domain_size(var_id)
                    .expect("valid transformed prop var id");
                if goal_vars.contains(&var_id) {
                    (0..domain_size).collect::<Vec<_>>()
                } else {
                    vec![0; domain_size]
                }
            })
            .collect::<Vec<_>>();
        let domain_sizes = domain_mapping
            .iter()
            .map(|mapping| mapping.iter().copied().max().map_or(0, |value| value + 1))
            .collect::<Vec<_>>();
        let numeric_domain_sizes = (0..transformed_task.numeric_variables().len())
            .map(|numeric_var_id| {
                partitions
                    .partitions(numeric_var_id)
                    .expect("trivial partitions contain every numeric variable")
                    .len()
            })
            .collect::<Vec<_>>();
        let factory = DomainAbstractionFactory::new(
            transformed_task,
            domain_mapping,
            domain_sizes,
            partitions,
            numeric_domain_sizes,
        )?;
        let mut operator_generator = factory.make_operator_generator(transformed_task, false)?;
        let abstract_operators = operator_generator.build_abstract_operators(transformed_task)?;
        let abstract_operator_footprints =
            factory.build_abstract_operator_footprints(transformed_task, &abstract_operators)?;
        let distance_table = factory.build_distance_table_with_operators(
            transformed_task,
            &operator_generator,
            &abstract_operators,
            false,
        )?;
        let transition_system = factory
            .build_abstract_transition_system_from_operators_without_regions_with_deadline(
                transformed_task,
                false,
                &abstract_operators,
                None,
            )?;
        let hash_multipliers =
            compute_hash_multipliers(factory.domain_sizes(), factory.numeric_domain_sizes())?;
        let mut relevant_operator_ids = transition_system
            .transitions
            .iter()
            .flat_map(|transition| transition.concrete_op_ids.iter().copied())
            .collect::<Vec<_>>();
        relevant_operator_ids.sort_unstable();
        relevant_operator_ids.dedup();
        Ok(DomainAbstraction {
            factory,
            distance_table,
            hash_multipliers,
            combine_labels: false,
            task_projection: prepared.task_projection.clone(),
            transformed_task: prepared.transformed_task.clone(),
            relevant_operator_ids,
            abstract_operators,
            abstract_operator_footprints,
            metadata: DomainAbstractionMetadata::default(),
        })
    }

    fn initial_abstract_state_ids(
        heuristic: &SaturatedCostPartitioningOnlineHeuristic<'_>,
        task: &dyn AbstractNumericTask,
    ) -> Vec<Option<usize>> {
        let prop = task.get_initial_propositional_state_values();
        let numeric = task.get_initial_numeric_state_values();
        heuristic
            .abstraction_heuristics
            .iter()
            .map(|component| {
                Some(
                    component
                        .abstract_state_hash_from_state_values(&prop, &numeric)
                        .expect("failed to hash initial state for handcrafted abstraction"),
                )
            })
            .collect()
    }

    #[derive(Debug, Clone, Copy)]
    enum ViewKind {
        Sum,
        Difference,
    }

    fn find_sailing_view(
        task: &dyn AbstractNumericTask,
        boat: &str,
        person: &str,
        view_kind: ViewKind,
    ) -> usize {
        let tuple = format!("({boat}, {boat}, {person})");
        let candidates = task
            .numeric_variables()
            .iter()
            .enumerate()
            .filter(|(_, variable)| variable.get_type() == &NumericType::Regular)
            .filter(|(_, variable)| {
                let name = variable.name();
                name.contains(&tuple)
                    && !name.contains("25.0")
                    && match view_kind {
                        ViewKind::Sum => name.contains("derived!sum_PNE x"),
                        ViewKind::Difference => name.contains("derived!difference_PNE y"),
                    }
            })
            .map(|(id, _)| id)
            .collect::<Vec<_>>();
        assert_eq!(
            candidates.len(),
            1,
            "expected one {view_kind:?} view for {boat}/{person}, got {candidates:?}"
        );
        candidates[0]
    }

    fn find_saved_goal_fact(task: &dyn AbstractNumericTask, person: &str) -> ExplicitFact {
        let suffix = format!(" {person}");
        let mut candidates = task
            .get_operators()
            .iter()
            .filter(|operator| operator.name().starts_with("save_person "))
            .filter(|operator| operator.name().ends_with(&suffix))
            .flat_map(|operator| operator.effects().iter())
            .filter(|effect| effect.conditions().is_empty())
            .map(|effect| ExplicitFact::new(effect.var_id(), effect.value()))
            .collect::<Vec<_>>();
        candidates.sort();
        candidates.dedup();
        assert_eq!(
            candidates.len(),
            1,
            "expected one saved fact from save_person operators for {person}, got {candidates:?}"
        );
        candidates[0].clone()
    }

    fn route_seed_splits(
        task: &dyn AbstractNumericTask,
        view_ids: &[usize],
    ) -> Vec<InitialSeedSplit> {
        let initial = task.get_initial_numeric_state_values();
        let mut seeds = Vec::new();
        for &view_id in view_ids {
            add_split(&mut seeds, view_id, initial[view_id], false);
            add_split(&mut seeds, view_id, 0.0, true);
            add_split(&mut seeds, view_id, 25.0, true);
            add_route_grid_values(&mut seeds, view_id, initial[view_id], 25.0, 3.0);
        }
        seeds.sort_by(|left, right| seed_description(left).cmp(&seed_description(right)));
        seeds.dedup();
        seeds
    }

    fn add_route_grid_values(
        seeds: &mut Vec<InitialSeedSplit>,
        numeric_var_id: usize,
        start: f64,
        end: f64,
        step: f64,
    ) {
        assert!(start.is_finite() && end.is_finite() && step.is_finite() && step > 0.0);
        let direction = if start <= end { 1.0 } else { -1.0 };
        let mut value = start;
        while (end - value) * direction > step {
            value += direction * step;
            add_split(seeds, numeric_var_id, value, true);
        }
    }

    fn add_split(
        seeds: &mut Vec<InitialSeedSplit>,
        numeric_var_id: usize,
        value: f64,
        include_in_lower: bool,
    ) {
        seeds.push(InitialSeedSplit::Numeric {
            numeric_var_id,
            value,
            include_in_lower,
        });
    }

    fn partition_report(
        task: &dyn AbstractNumericTask,
        abstraction: &DomainAbstraction,
        view_ids: &[usize],
    ) -> String {
        view_ids
            .iter()
            .map(|&view_id| {
                let name = task.numeric_variables()[view_id].name();
                let num_parts = abstraction
                    .factory
                    .partitions()
                    .partitions(view_id)
                    .expect("missing partition for handcrafted view")
                    .len();
                format!("n{view_id}:{name} parts={num_parts}")
            })
            .collect::<Vec<_>>()
            .join(" || ")
    }

    fn seed_description(seed: &InitialSeedSplit) -> String {
        match seed {
            InitialSeedSplit::Propositional { var_id, value } => format!("p{var_id}={value}"),
            InitialSeedSplit::Numeric {
                numeric_var_id,
                value,
                include_in_lower,
            } => format!(
                "n{numeric_var_id}{}{}",
                if *include_in_lower { "<=" } else { "<" },
                value
            ),
        }
    }

    fn translated_sailing_2_2_task() -> NumericRootTask {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let domain = root.join("others/sailing/domain.pddl");
        let problem = root.join("others/sailing/prob_2_2_1229.pddl");
        assert!(domain.is_file(), "missing {}", domain.display());
        assert!(problem.is_file(), "missing {}", problem.display());
        let temp_dir = unique_temp_dir("sailing_handcrafted_full_task_scp")
            .expect("failed to create sailing diagnostic temp dir");
        let output_sas = temp_dir.join("output.sas");
        let preprocessed = temp_dir.join("output");
        translate_to_sas_to_path_fast(
            domain.to_str().expect("non-utf8 sailing domain path"),
            problem.to_str().expect("non-utf8 sailing problem path"),
            &output_sas,
        )
        .expect("sailing translation failed");
        run_preprocess_to_output(
            &[
                "preprocess".to_string(),
                output_sas.to_string_lossy().to_string(),
            ],
            &preprocessed,
        );
        NumericRootTask::from_file(&preprocessed)
    }

    fn unique_temp_dir(prefix: &str) -> std::io::Result<PathBuf> {
        let base = std::env::temp_dir().join("numeric_planneRS");
        std::fs::create_dir_all(&base)?;
        let dir = base.join(format!(
            "{prefix}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        std::fs::create_dir(&dir)?;
        Ok(dir)
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
        fn get_operator_effect(
            &self,
            index: usize,
            eff_index: usize,
            is_axiom: bool,
        ) -> &ExplicitFact {
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
            assert_eq!(index, 0, "SingleGoalTask only exposes one goal");
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
        fn evaluated_initial_abstract_state_values(
            &self,
        ) -> Result<(Vec<usize>, Vec<f64>), String> {
            self.base.evaluated_initial_abstract_state_values()
        }
        fn abstract_operator_cost(&self, operator_id: usize) -> f64 {
            self.base.abstract_operator_cost(operator_id)
        }
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

fn compute_abstract_operator_order_score(
    h: f64,
    stolen_costs: f64,
    scoring_function: ScoringFunction,
    abstraction: &DomainAbstraction,
) -> f64 {
    if h <= 1e-9 {
        return -1.0e100;
    }
    let base = compute_score(h, stolen_costs, scoring_function);
    let size_penalty = ((abstraction_state_count(abstraction) as f64) / 10_000.0)
        .max(1.0)
        .sqrt();
    base / size_penalty
}

fn standalone_current_h_values(
    state: &ScpOnlineState,
    abstract_state_ids: &[Option<usize>],
    num_domain_abstractions: usize,
) -> Vec<f64> {
    (0..num_domain_abstractions)
        .map(|abstraction_id| {
            state
                .h_values_by_abstraction
                .get(abstraction_id)
                .map(|distances| {
                    current_h_for_distances(abstraction_id, distances, abstract_state_ids)
                })
                .unwrap_or(0.0)
        })
        .collect()
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
