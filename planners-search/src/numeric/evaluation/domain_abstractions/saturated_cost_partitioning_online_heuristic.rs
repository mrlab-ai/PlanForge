use std::cell::RefCell;
use std::time::{Duration, Instant};

use planners_sas::numeric::numeric_task::{
    AbstractNumericTask, metric_operator_cost_from_initial_values,
};
use rand::seq::SliceRandom;
use rand::{SeedableRng, rngs::SmallRng};
use serde::{Deserialize, Serialize};

use crate::numeric::evaluation::evaluator::{EvaluationError, EvaluationState};
use crate::numeric::evaluation::heuristic::Heuristic;
use crate::numeric::evaluation::pattern_databases::pattern_database::{
    PatternDatabase, PdbHeuristicConfig, PdbInternalHeuristic,
};

use super::domain_abstraction_collection_generator_multiple_cegar::DomainAbstractionCollectionGeneratorMultipleCegarConfig;
use super::domain_abstraction_factory::{AbstractDistanceTable, DomainAbstractionFactory};
use super::domain_abstraction_generator::DomainAbstraction;
use super::domain_abstraction_heuristic::DomainAbstractionHeuristic;

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
    pub random_seed: i32,
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
            scoring_function: ScoringFunction::MaxHeuristicPerStolenCosts,
            saturator: Saturator::All,
            random_seed: 0,
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
    /// Pre-computed h-values (original costs) per abstraction.
    h_values_by_abstraction: Vec<Vec<f64>>,
    /// Pre-computed stolen costs per abstraction (original costs).
    stolen_costs_by_abstraction: Vec<f64>,
    /// RNG for tie-breaking in order generation.
    rng: SmallRng,
    /// True after the improvement phase ends (memory released).
    improvement_ended: bool,
}

impl ScpOnlineState {
    fn new(seed: i32) -> Self {
        Self {
            start_time: Instant::now(),
            evaluated_states: 0,
            improve_heuristic: true,
            size_kb: 0,
            cp_heuristics: Vec::new(),
            h_values_by_abstraction: Vec::new(),
            stolen_costs_by_abstraction: Vec::new(),
            rng: SmallRng::seed_from_u64(seed as u64),
            improvement_ended: false,
        }
    }
}

pub struct SaturatedCostPartitioningOnlineHeuristic<'task> {
    name: String,
    /// Full abstractions – may be released once the improvement phase ends.
    abstractions: RefCell<Option<Vec<DomainAbstraction>>>,
    /// Light-weight abstraction heuristics for state ID computation.
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

        // Pre-compute order-generator state with original costs.
        let original_costs: Vec<f64> = task
            .get_operators()
            .iter()
            .map(|op| metric_operator_cost_from_initial_values(task, op))
            .collect();

        let num_abstractions = abstractions.len();
        let pdbs_count = pdbs.len();
        let total_components = num_abstractions + pdbs_count;

        let mut h_values_by_abstraction: Vec<Vec<f64>> =
            Vec::with_capacity(total_components);
        let mut saturated_costs_by_abstraction: Vec<Vec<f64>> =
            Vec::with_capacity(total_components);

        for abstraction in &abstractions {
            let table = abstraction
                .factory
                .build_goal_distances(task, config.combine_labels, &original_costs)
                .map_err(|error| {
                    EvaluationError::ComputationFailed(format!(
                        "failed to compute goal distances for order generator: {error:#}"
                    ))
                })?;
            // Compute saturated costs for the original-cost distances.
            let (_, saturated) = abstraction
                .factory
                .build_cost_partitioned_distance_table(
                    task,
                    config.combine_labels,
                    &original_costs,
                    false,
                )
                .map_err(|error| {
                    EvaluationError::ComputationFailed(format!(
                        "failed to compute saturated costs for order generator: {error:#}"
                    ))
                })?;
            h_values_by_abstraction.push(table.distances);
            saturated_costs_by_abstraction.push(saturated);
        }

        for pdb in &pdbs {
            let distances = pdb
                .build_goal_distances(&original_costs)
                .map_err(|error| {
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
            h_values_by_abstraction.push(distances);
            saturated_costs_by_abstraction.push(saturated);
        }

        // Compute surplus costs and stolen costs.
        let surplus_costs =
            compute_all_surplus_costs(&original_costs, &saturated_costs_by_abstraction);
        let stolen_costs_by_abstraction: Vec<f64> = saturated_costs_by_abstraction
            .iter()
            .map(|saturated| {
                compute_costs_stolen_by_heuristic(saturated, &surplus_costs)
            })
            .collect();

        let mut state = ScpOnlineState::new(config.random_seed);
        state.h_values_by_abstraction = h_values_by_abstraction;
        state.stolen_costs_by_abstraction = stolen_costs_by_abstraction;

        Ok(Self {
            name: name.unwrap_or_else(|| "scp_online".to_string()),
            abstractions: RefCell::new(Some(abstractions)),
            abstraction_heuristics,
            pdbs,
            config,
            state: RefCell::new(state),
        })
    }

    fn compute_order_for_state(
        state: &mut ScpOnlineState,
        abstract_state_ids: &[Option<usize>],
        scoring_function: ScoringFunction,
    ) -> Vec<usize> {
        let total_components = state.h_values_by_abstraction.len();
        // Default order: 0, 1, 2, ...
        let mut order: Vec<usize> = (0..total_components).collect();
        // Shuffle to break ties randomly (scorpion does this).
        order.shuffle(&mut state.rng);

        // Compute scores for each component.
        let scores: Vec<f64> = (0..total_components)
            .map(|abs_id| {
                let h = abstract_state_ids
                    .get(abs_id)
                    .copied()
                    .flatten()
                    .and_then(|sid| {
                        state.h_values_by_abstraction
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

        // Sort by score descending.
        order.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        order
    }

    fn compute_abstract_state_ids(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<(Vec<Option<usize>>, usize), EvaluationError> {
        let num_domain_abstractions = self.abstraction_heuristics.len();
        let mut ids: Vec<Option<usize>> = self
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

        Ok((ids, num_domain_abstractions))
    }

    fn compute_max_h(state: &ScpOnlineState, abstract_state_ids: &[Option<usize>]) -> f64 {
        state
            .cp_heuristics
            .iter()
            .map(|cp| cp.compute_heuristic(abstract_state_ids))
            .fold(0.0, f64::max)
    }

    /// Compute an SCP using the given order.
    fn compute_scp_ordered(
        factory: &DomainAbstractionFactory,
        pdbs: &[PatternDatabase<'_>],
        combine_labels: bool,
        order: &[usize],
        task: &dyn AbstractNumericTask,
        original_costs: &[f64],
        abstract_state_ids: &[Option<usize>],
        saturator: Saturator,
        num_domain_abstractions: usize,
    ) -> Result<CostPartitioningHeuristic, EvaluationError> {
        let mut remaining_costs: Vec<f64> = original_costs.to_vec();
        let pdb_offset = num_domain_abstractions;

        match saturator {
            Saturator::All => {
                let mut cp = CostPartitioningHeuristic::default();
                for &pos in order {
                    if pos < num_domain_abstractions {
                        let (table, saturated_costs) = factory
                            .build_cost_partitioned_distance_table(
                                task,
                                combine_labels,
                                &remaining_costs,
                                false,
                            )
                            .map_err(|error| {
                                EvaluationError::ComputationFailed(format!(
                                    "failed to compute SCP table for abstraction {pos}: {error:#}"
                                ))
                            })?;
                        cp.add_h_values(pos, table.distances);
                        reduce_costs(&mut remaining_costs, &saturated_costs)?;
                    } else {
                        let pdb_id = pos - pdb_offset;
                        if let Some(pdb) = pdbs.get(pdb_id) {
                            let (distances, saturated_costs) = pdb
                                .build_cost_partitioned_distance_table(&remaining_costs)
                                .map_err(|error| {
                                    EvaluationError::ComputationFailed(format!(
                                        "failed to compute SCP table for PDB {pdb_id}: {error}"
                                    ))
                                })?;
                            cp.add_pdb_h_values(pos, distances);
                            reduce_costs(&mut remaining_costs, &saturated_costs)?;
                        }
                    }
                }
                Ok(cp)
            }
            Saturator::Perim => {
                let mut cp = CostPartitioningHeuristic::default();
                for &pos in order {
                    if pos < num_domain_abstractions {
                        let (table, _) = factory
                            .build_cost_partitioned_distance_table(
                                task,
                                combine_labels,
                                &remaining_costs,
                                false,
                            )
                            .map_err(|error| {
                                EvaluationError::ComputationFailed(format!(
                                    "failed to compute SCP table for abstraction {pos} during PERIM: {error:#}"
                                ))
                            })?;

                        // Cap h-values at h(current_state) for this abstraction.
                        let h_cap = abstract_state_ids
                            .get(pos)
                            .copied()
                            .flatten()
                            .and_then(|sid| table.distances.get(sid).copied())
                            .unwrap_or(f64::INFINITY);
                        let mut capped_distances = table.distances;
                        if h_cap.is_finite() {
                            for h in &mut capped_distances {
                                if h.is_finite() && *h > h_cap {
                                    *h = h_cap;
                                }
                            }
                        }
                        // Recompute saturated costs from capped distances using
                        // the factory.  Build operators, apply costs, and compute
                        // saturated costs.
                        let mut generator = factory
                            .make_operator_generator(task, combine_labels)
                            .map_err(|error| {
                                EvaluationError::ComputationFailed(format!(
                                    "failed to create operator generator for PERIM: {error:#}"
                                ))
                            })?;
                        let mut operators = generator
                            .build_abstract_operators(task)
                            .map_err(|error| {
                                EvaluationError::ComputationFailed(format!(
                                    "failed to build abstract operators for PERIM: {error:#}"
                                ))
                            })?;
                        apply_operator_costs_from_slice(
                            &mut operators,
                            &remaining_costs,
                        )?;
                        let capped_table = AbstractDistanceTable {
                            distances: capped_distances.clone(),
                            generating_op_ids: table.generating_op_ids,
                            initial_state_hash: table.initial_state_hash,
                            goal_facts: table.goal_facts,
                            hash_multipliers: table.hash_multipliers,
                            numeric_domain_sizes: table.numeric_domain_sizes,
                        };
                        let saturated = factory
                            .saturated_costs_for_table(
                                task,
                                combine_labels,
                                &operators,
                                &capped_table,
                            )
                            .map_err(|error| {
                                EvaluationError::ComputationFailed(format!(
                                    "failed to compute PERIM saturated costs: {error:#}"
                                ))
                            })?;

                        cp.add_h_values(pos, capped_distances);
                        reduce_costs(&mut remaining_costs, &saturated)?;
                    } else {
                        let pdb_id = pos - pdb_offset;
                        if let Some(pdb) = pdbs.get(pdb_id) {
                            let h_cap = abstract_state_ids
                                .get(pos)
                                .copied()
                                .flatten()
                                .and_then(|sid| {
                                    pdb.build_goal_distances(&remaining_costs)
                                        .ok()
                                        .and_then(|dists| dists.get(sid).copied())
                                })
                                .unwrap_or(f64::INFINITY);
                            let (distances, saturated_costs) = pdb
                                .build_cost_partitioned_distance_table_capped(
                                    &remaining_costs,
                                    h_cap,
                                )
                                .map_err(|error| {
                                    EvaluationError::ComputationFailed(format!(
                                        "failed to compute PERIM capped SCP table for PDB {pdb_id}: {error}"
                                    ))
                                })?;
                            cp.add_pdb_h_values(pos, distances);
                            reduce_costs(&mut remaining_costs, &saturated_costs)?;
                        }
                    }
                }
                Ok(cp)
            }
            Saturator::Perimstar => {
                let mut cp = Self::compute_scp_ordered(
                    factory,
                    pdbs,
                    combine_labels,
                    order,
                    task,
                    original_costs,
                    abstract_state_ids,
                    Saturator::Perim,
                    num_domain_abstractions,
                )?;
                let extra_cp = Self::compute_scp_ordered(
                    factory,
                    pdbs,
                    combine_labels,
                    order,
                    task,
                    original_costs,
                    abstract_state_ids,
                    Saturator::All,
                    num_domain_abstractions,
                )?;
                cp.lookup_tables.extend(extra_cp.lookup_tables);
                Ok(cp)
            }
        }
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
        let (abstract_state_ids, num_domain_abstractions) =
            self.compute_abstract_state_ids(eval_state)?;

        let mut state = self.state.borrow_mut();
        let mut max_h = Self::compute_max_h(&state, &abstract_state_ids);
        if max_h.is_infinite() {
            return Ok(max_h);
        }

        let time_limit_reached = self.config.max_time.is_finite()
            && state.start_time.elapsed() >= Duration::from_secs_f64(self.config.max_time);

        if state.improve_heuristic && (time_limit_reached || state.size_kb >= self.config.max_size) {
            state.improve_heuristic = false;
        }

        // Release full abstractions and keep only abstraction heuristics once
        // the improvement phase ends (scorpion: extract_abstraction_functions).
        if !state.improve_heuristic && !state.improvement_ended {
            let mut abs_guard = self.abstractions.borrow_mut();
            if abs_guard.is_some() {
                abs_guard.take();
                state.improvement_ended = true;
            }
        }

        if state.improve_heuristic && state.evaluated_states % self.config.interval == 0 {
            let original_costs: Vec<f64> = task
                .get_operators()
                .iter()
                .map(|op| metric_operator_cost_from_initial_values(task, op))
                .collect();

            let order = Self::compute_order_for_state(
                &mut state,
                &abstract_state_ids,
                self.config.scoring_function,
            );

            // Access abstractions (if still active) or skip.
            let abstractions_guard = self.abstractions.borrow();
            let abstractions: &[DomainAbstraction] = match &*abstractions_guard {
                Some(abs) => abs.as_slice(),
                None => &[],
            };

            // For domain abstractions we need a factory.  All abstractions in the
            // same scp_online share the same combine_labels.  Use the factory from
            // the first abstraction (all CEGAR results share task semantics).
            let factory_opt = abstractions.first().map(|abs| &abs.factory);

            if let Some(factory) = factory_opt {
                let cp = Self::compute_scp_ordered(
                    factory,
                    &self.pdbs,
                    self.config.combine_labels,
                    &order,
                    task,
                    &original_costs,
                    &abstract_state_ids,
                    self.config.saturator,
                    num_domain_abstractions,
                )?;

                let new_h = cp.compute_heuristic(&abstract_state_ids);
                if new_h > max_h {
                    state.size_kb = state.size_kb.saturating_add(cp.estimate_size_in_kb());
                    state.cp_heuristics.push(cp);
                    max_h = new_h;
                }
            }
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
        ScoringFunction::MaxHeuristicPerStolenCosts => {
            h / stolen_costs.max(1.0)
        }
    }
}

fn compute_stolen_costs(wanted_by_abs: f64, surplus_cost: f64) -> f64 {
    if !wanted_by_abs.is_finite() || !surplus_cost.is_finite() {
        return 0.0;
    }

    let surplus_for_rest = surplus_cost + wanted_by_abs;
    if surplus_for_rest >= 0.0 {
        (0.0_f64).max(wanted_by_abs - surplus_for_rest)
    } else {
        wanted_by_abs.max(surplus_for_rest)
    }
}

fn compute_costs_stolen_by_heuristic(
    saturated_costs: &[f64],
    surplus_costs: &[f64],
) -> f64 {
    saturated_costs
        .iter()
        .zip(surplus_costs.iter())
        .map(|(&saturated, &surplus)| compute_stolen_costs(saturated, surplus))
        .sum()
}

fn compute_surplus_cost(
    saturated_costs_by_abstraction: &[Vec<f64>],
    op_id: usize,
    remaining_cost: f64,
) -> f64 {
    let sum_wanted: f64 = saturated_costs_by_abstraction
        .iter()
        .map(|costs| {
            costs
                .get(op_id)
                .copied()
                .unwrap_or(f64::NEG_INFINITY)
        })
        .filter(|&w| w > f64::NEG_INFINITY)
        .sum();

    if !remaining_cost.is_finite() || !sum_wanted.is_finite() {
        return f64::INFINITY;
    }
    remaining_cost - sum_wanted
}

fn compute_all_surplus_costs(
    costs: &[f64],
    saturated_costs_by_abstraction: &[Vec<f64>],
) -> Vec<f64> {
    costs
        .iter()
        .enumerate()
        .map(|(op_id, &cost)| {
            compute_surplus_cost(saturated_costs_by_abstraction, op_id, cost)
        })
        .collect()
}

fn apply_operator_costs_from_slice(
    operators: &mut [crate::numeric::evaluation::domain_abstractions::abstract_operator_generator::AbstractOperator],
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