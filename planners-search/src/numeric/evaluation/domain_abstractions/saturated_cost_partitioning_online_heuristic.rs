use std::cell::RefCell;
use std::time::{Duration, Instant};

use planners_sas::numeric::numeric_task::{
    AbstractNumericTask, metric_operator_cost_from_initial_values,
};
use serde::{Deserialize, Serialize};

use crate::numeric::evaluation::evaluator::{EvaluationError, EvaluationState};
use crate::numeric::evaluation::heuristic::Heuristic;
use crate::numeric::evaluation::pattern_databases::pattern_database::{
    PatternDatabase, PdbHeuristicConfig, PdbInternalHeuristic,
};

use super::domain_abstraction_collection_generator_multiple_cegar::DomainAbstractionCollectionGeneratorMultipleCegarConfig;
use super::domain_abstraction_factory::AbstractDistanceTable;
use super::domain_abstraction_generator::DomainAbstraction;
use super::domain_abstraction_heuristic::DomainAbstractionHeuristic;

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
}

impl Default for ScpOnlineConfig {
    fn default() -> Self {
        let mut collection_config =
            DomainAbstractionCollectionGeneratorMultipleCegarConfig::default();
        collection_config.combine_labels = false;
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
    fn add_h_values(&mut self, abstraction_id: usize, table: AbstractDistanceTable) {
        if table.distances.iter().any(|value| *value > 0.0) {
            self.lookup_tables.push(LookupTable {
                abstraction_id,
                distances: table.distances,
                unknown_value: f64::INFINITY,
            });
        }
    }

    fn add_pdb_h_values(&mut self, abstraction_id: usize, distances: Vec<f64>) {
        if distances.iter().any(|value| *value > 0.0) {
            self.lookup_tables.push(LookupTable {
                abstraction_id,
                distances,
                // Numeric PDBs can be truncated and therefore need not contain every
                // projected reachable state. Treat missing states as 0 instead of a
                // dead end; explicit +inf entries still signal abstract dead ends.
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

#[derive(Debug, Clone)]
struct ScpOnlineState {
    start_time: Instant,
    evaluated_states: usize,
    improve_heuristic: bool,
    size_kb: usize,
    cp_heuristics: Vec<CostPartitioningHeuristic>,
}

impl Default for ScpOnlineState {
    fn default() -> Self {
        Self {
            start_time: Instant::now(),
            evaluated_states: 0,
            improve_heuristic: true,
            size_kb: 0,
            cp_heuristics: Vec::new(),
        }
    }
}

pub struct SaturatedCostPartitioningOnlineHeuristic<'task> {
    name: String,
    abstractions: Vec<DomainAbstraction>,
    abstraction_heuristics: Vec<DomainAbstractionHeuristic>,
    pdbs: Vec<PatternDatabase<'task>>,
    config: ScpOnlineConfig,
    state: RefCell<ScpOnlineState>,
}

impl<'task> SaturatedCostPartitioningOnlineHeuristic<'task> {
    pub fn new(
        name: Option<String>,
        abstractions: Vec<DomainAbstraction>,
        pdbs: Vec<PatternDatabase<'task>>,
        config: ScpOnlineConfig,
    ) -> Self {
        let abstraction_heuristics = abstractions
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, abstraction)| {
                DomainAbstractionHeuristic::new(Some(format!("scp_online_{index}")), abstraction)
            })
            .collect();
        Self {
            name: name.unwrap_or_else(|| "scp_online".to_string()),
            abstractions,
            abstraction_heuristics,
            pdbs,
            config,
            state: RefCell::new(ScpOnlineState::default()),
        }
    }

    fn compute_abstract_state_ids(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<Vec<Option<usize>>, EvaluationError> {
        let mut ids = self
            .abstraction_heuristics
            .iter()
            .map(|heuristic| heuristic.abstract_state_hash(eval_state))
            .map(|result| result.map(Some))
            .collect::<Result<Vec<_>, _>>()?;
        if !self.pdbs.is_empty() {
            let state = eval_state.state();
            let registry = eval_state.state_registry().ok_or_else(|| {
                EvaluationError::InvalidState(
                    "SCP online PDB lookup requires state registry in EvaluationState".to_string(),
                )
            })?;
            let mut prop = Vec::new();
            let mut numeric = Vec::new();
            state.fill_state(registry, &mut prop);
            registry
                .fill_numeric_vars(state, &mut numeric)
                .map_err(|err| {
                    EvaluationError::ComputationFailed(format!(
                        "failed to read numeric state for SCP online PDB lookup: {err:?}"
                    ))
                })?;
            let mut expanded = Vec::new();
            if let Some(first_pdb) = self.pdbs.first() {
                first_pdb
                    .expand_numeric_state_values_into(&numeric, &mut expanded)
                    .map_err(EvaluationError::ComputationFailed)?;
            } else {
                expanded = numeric;
            }
            for pdb in &self.pdbs {
                let state_id = pdb
                    .abstract_state_id_from_expanded_state_values(&prop, &expanded)
                    .map_err(EvaluationError::ComputationFailed)?;
                ids.push(state_id);
            }
        }
        Ok(ids)
    }

    fn compute_max_h(state: &ScpOnlineState, abstract_state_ids: &[Option<usize>]) -> f64 {
        state
            .cp_heuristics
            .iter()
            .map(|cp| cp.compute_heuristic(abstract_state_ids))
            .fold(0.0, f64::max)
    }

    fn compute_scp(
        &self,
        task: &dyn AbstractNumericTask,
    ) -> Result<CostPartitioningHeuristic, EvaluationError> {
        let mut remaining_costs: Vec<f64> = task
            .get_operators()
            .iter()
            .map(|op| metric_operator_cost_from_initial_values(task, op))
            .collect();
        let mut cp = CostPartitioningHeuristic::default();

        for (abstraction_id, abstraction) in self.abstractions.iter().enumerate() {
            let (table, saturated_costs) = abstraction
                .factory
                .build_cost_partitioned_distance_table(
                    task,
                    self.config.combine_labels,
                    &remaining_costs,
                    false,
                )
                .map_err(|error| {
                    EvaluationError::ComputationFailed(format!(
                        "failed to compute SCP table for abstraction {abstraction_id}: {error:#}"
                    ))
                })?;
            cp.add_h_values(abstraction_id, table);
            reduce_costs(&mut remaining_costs, &saturated_costs)?;
        }

        let pdb_offset = self.abstractions.len();
        for (pdb_id, pdb) in self.pdbs.iter().enumerate() {
            let (distances, saturated_costs) = pdb
                .build_cost_partitioned_distance_table(&remaining_costs)
                .map_err(|error| {
                    EvaluationError::ComputationFailed(format!(
                        "failed to compute SCP table for numeric PDB {pdb_id}: {error}"
                    ))
                })?;
            cp.add_pdb_h_values(pdb_offset + pdb_id, distances);
            reduce_costs(&mut remaining_costs, &saturated_costs)?;
        }

        Ok(cp)
    }
}

impl Heuristic for SaturatedCostPartitioningOnlineHeuristic<'_> {
    fn compute_heuristic(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError> {
        let task = eval_state.task().ok_or_else(|| {
            EvaluationError::InvalidState(
                "SaturatedCostPartitioningOnlineHeuristic requires task in EvaluationState"
                    .to_string(),
            )
        })?;
        let abstract_state_ids = self.compute_abstract_state_ids(eval_state)?;

        let mut state = self.state.borrow_mut();
        let mut max_h = Self::compute_max_h(&state, &abstract_state_ids);
        if max_h.is_infinite() {
            return Ok(max_h);
        }

        let time_limit_reached = self.config.max_time.is_finite()
            && state.start_time.elapsed() >= Duration::from_secs_f64(self.config.max_time);
        if state.improve_heuristic && (time_limit_reached || state.size_kb >= self.config.max_size)
        {
            state.improve_heuristic = false;
        }

        if state.improve_heuristic && state.evaluated_states % self.config.interval == 0 {
            let cp = self.compute_scp(task)?;
            let new_h = cp.compute_heuristic(&abstract_state_ids);
            if new_h > max_h {
                state.size_kb = state.size_kb.saturating_add(cp.estimate_size_in_kb());
                state.cp_heuristics.push(cp);
                max_h = new_h;
            }
        }

        state.evaluated_states = state.evaluated_states.saturating_add(1);
        Ok(max_h)
    }

    fn heuristic_name(&self) -> String {
        self.name.clone()
    }
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

    for (remaining, saturated) in remaining_costs.iter_mut().zip(saturated_costs.iter()) {
        if !saturated.is_finite() {
            continue;
        }
        *remaining -= saturated;
        if *remaining < 0.0 && *remaining > -1e-9 {
            *remaining = 0.0;
        }
    }

    Ok(())
}
