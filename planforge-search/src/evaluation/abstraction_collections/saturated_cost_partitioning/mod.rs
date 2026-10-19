use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use planforge_sas::numeric_task::{
    AbstractNumericTask, NumericValue, TaskRef, metric_operator_cost_from_initial_values,
};
use planforge_sas::state_registry::{ConcreteState, ExpansionContext, StateRegistry};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng, rngs::SmallRng};
use tracing::{Level, debug, enabled, info};

use crate::evaluation::cartesian_abstractions::{
    CartesianAbstraction, CartesianAbstractionHeuristic,
};
use crate::evaluation::evaluator::{EvaluationError, EvaluationState};
use crate::evaluation::heuristic::Heuristic;
use crate::evaluation::numeric_landmarks::lm_cut_numeric_heuristic::LandmarkCutNumericHeuristic;
use crate::evaluation::pattern_databases::pattern_database::PatternDatabase;
use crate::successor_generator::SuccessorTree;

use super::component::AbstractionComponent;
use super::cost_partitioning::{
    AbstractOperatorCostFunction, AbstractOperatorRegions, LmCutResidualOperatorCostPartition,
    RegionalCostAllocation, StateRegion, TransitionResidualCosts,
    build_explicit_label_cost_partitioning_table, build_explicit_regional_cost_partitioning_table,
};
use super::goal_focused::MemberDirection;
use crate::evaluation::domain_abstractions::abstract_operator_generator::AbstractOperator;
use crate::evaluation::domain_abstractions::domain_abstraction_factory::{
    AbstractDistanceTable, DistanceTableOptions, SaturationStep,
};
use crate::evaluation::domain_abstractions::domain_abstraction_generator::DomainAbstraction;
use crate::evaluation::domain_abstractions::domain_abstraction_heuristic::{
    DomainAbstractionHeuristic, DomainAbstractionLookupScratch,
    compute_collection_abstract_state_ids,
};

mod config;
mod deadline;
mod diagnostics;
mod diverse_orders;
mod fill_scp;
mod lookup;
mod orders;
mod sampling;
mod saturation;
mod scoring;
pub(crate) use config::LegacyScpConfig;
pub use config::{
    CostPartitioningMethod, FillScpConfig, OrderGenerator, Saturator, ScoringFunction, ScpConfig,
};
use deadline::*;
use diagnostics::*;
pub use fill_scp::FillScpHeuristic;
use lookup::*;
use saturation::*;
use scoring::*;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Main heuristic
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct ScpState {
    start_time: Instant,
    evaluated_states: usize,
    improve_heuristic: bool,
    size_kb: usize,
    cp_heuristics: Vec<CostPartitioningHeuristic>,
    goal_distances_by_abstraction: Vec<Vec<f64>>,
    abstraction_lookup_tables: Vec<LookupTable>,
    stolen_costs_by_abstraction: Vec<f64>,
    rng: SmallRng,
    improvement_ended: bool,
    lookup_mask: Vec<bool>,
    sample_abstract_state_ids: Vec<Vec<Option<usize>>>,
}

impl ScpState {
    fn new(seed: Option<u64>) -> Self {
        let seed = seed.unwrap_or(crate::evaluation::DEFAULT_RANDOM_SEED);
        Self {
            start_time: Instant::now(),
            evaluated_states: 0,
            improve_heuristic: true,
            size_kb: 0,
            cp_heuristics: Vec::new(),
            goal_distances_by_abstraction: Vec::new(),
            abstraction_lookup_tables: Vec::new(),
            stolen_costs_by_abstraction: Vec::new(),
            rng: SmallRng::seed_from_u64(seed),
            improvement_ended: false,
            lookup_mask: Vec::new(),
            sample_abstract_state_ids: Vec::new(),
        }
    }
}

pub struct SaturatedCostPartitioningHeuristic<'task> {
    name: String,
    task: &'task dyn AbstractNumericTask,
    components: RefCell<Vec<AbstractionComponent<'task>>>,
    config: ScpConfig,
    debug_diagnostics: bool,
    original_operator_costs: Vec<f64>,
    state: RefCell<ScpState>,
    lookup_scratch: RefCell<DomainAbstractionLookupScratch>,
    component_ids_scratch: RefCell<Vec<Option<usize>>>,
    sampling_task: RefCell<Option<TaskRef<'task>>>,
}

/// What a [`SaturatedCostPartitioningHeuristic`] needs besides its abstractions
/// and configuration.
#[derive(Default)]
pub struct ScpBuildOptions<'task> {
    /// An owned task to sample states from by random walks; required for
    /// diverse orders.
    pub sampling_task: Option<TaskRef<'task>>,
    /// Log per-abstraction diagnostics while building cost partitionings.
    pub debug_diagnostics: bool,
}

/// The components of a collection of domain abstractions, Cartesian
/// abstractions and pattern databases, in that order, named `{prefix}_{i}` and
/// `{prefix}_cartesian_{i}`.
pub fn collection_components<'task>(
    prefix: &str,
    domain_abstractions: Vec<DomainAbstraction>,
    cartesian_abstractions: Vec<CartesianAbstraction>,
    pdbs: Vec<PatternDatabase<'task>>,
) -> Vec<AbstractionComponent<'task>> {
    let mut components =
        Vec::with_capacity(domain_abstractions.len() + cartesian_abstractions.len() + pdbs.len());
    components.extend(
        domain_abstractions
            .into_iter()
            .enumerate()
            .map(|(index, abstraction)| {
                AbstractionComponent::domain(Some(format!("{prefix}_{index}")), abstraction)
            }),
    );
    components.extend(cartesian_abstractions.into_iter().enumerate().map(
        |(index, abstraction)| {
            AbstractionComponent::cartesian(
                Some(format!("{prefix}_cartesian_{index}")),
                abstraction,
            )
        },
    ));
    components.extend(pdbs.into_iter().map(AbstractionComponent::pattern_database));
    components
}

impl<'task> SaturatedCostPartitioningHeuristic<'task> {
    pub fn new(
        name: Option<String>,
        components: Vec<AbstractionComponent<'task>>,
        config: ScpConfig,
        task: &'task dyn AbstractNumericTask,
        options: ScpBuildOptions<'task>,
    ) -> Result<Self, EvaluationError> {
        let ScpBuildOptions {
            sampling_task,
            debug_diagnostics,
        } = options;
        if components.is_empty() {
            return Err(EvaluationError::ComputationFailed(
                "SCP requires at least one abstraction component".to_string(),
            ));
        }
        config
            .validate(sampling_task.is_some())
            .map_err(EvaluationError::ComputationFailed)?;
        if config.partitioning.uses_regions() {
            for (component_id, component) in components.iter().enumerate() {
                match component {
                    AbstractionComponent::Cartesian(_) => info!(
                        "Cartesian component {component_id}: partitioning=region requested, using abstract-operator cost partitioning (Cartesian abstractions do not yet support regional granularity)"
                    ),
                    AbstractionComponent::PatternDatabase(_) => info!(
                        "PDB component {component_id}: partitioning=region requested, using label cost partitioning (PDBs do not yet support regional granularity)"
                    ),
                    AbstractionComponent::Domain(_) => {}
                }
            }
        }
        let original_costs: Vec<f64> = task
            .get_operators()
            .iter()
            .map(|op| metric_operator_cost_from_initial_values(task, op).value())
            .collect();

        let mut h_values: Vec<Vec<f64>> = Vec::with_capacity(components.len());
        let mut saturated_costs_by_abstraction: Vec<Vec<f64>> =
            Vec::with_capacity(components.len());
        let mut debug_initial_h_values = Vec::new();

        for (component_id, component) in components.iter().enumerate() {
            let (distances, saturated, initial_state_id) = match component {
                AbstractionComponent::Domain(heuristic) => {
                    let abstraction = heuristic.abstraction();
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
                        .build_cost_partitioned_distance_table(
                            abstraction_task,
                            config.combine_labels,
                            &original_costs,
                            DistanceTableOptions::default().with_goal_facts(goal_facts),
                        )
                        .map_err(|error| {
                            EvaluationError::ComputationFailed(format!(
                                "failed to compute saturated costs for order generator: {error:#}"
                            ))
                        })?;
                    if debug_diagnostics {
                        info!(
                            "scp debug: collection abstraction {component_id}: states={}, goal_facts={}",
                            abstraction_state_count(abstraction),
                            goal_facts.len()
                        );
                    }
                    (table.distances, saturated, table.initial_state_hash)
                }
                AbstractionComponent::Cartesian(heuristic) => {
                    let abstraction = heuristic.abstraction();
                    let (distances, saturated) = build_explicit_label_cost_partitioning_table(
                        &abstraction.transition_system,
                        &original_costs,
                        None,
                        None,
                    )
                    .map_err(|error| {
                        EvaluationError::ComputationFailed(format!(
                            "failed to compute Cartesian order-generator table {component_id}: {error:#}"
                        ))
                    })?;
                    (
                        distances,
                        saturated,
                        abstraction.transition_system.initial_state_hash,
                    )
                }
                AbstractionComponent::PatternDatabase(pdb) => {
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
                    (distances, saturated, 0)
                }
            };
            if debug_diagnostics {
                let initial_h = distances.get(initial_state_id).copied().ok_or_else(|| {
                    EvaluationError::InvalidState(format!(
                        "{} component {component_id} initial state {initial_state_id} out of bounds for {} states",
                        component.kind(),
                        distances.len()
                    ))
                })?;
                debug_initial_h_values.push(initial_h);
                info!(
                    "scp debug: {} component {component_id}: original_initial_h={initial_h}, states={}",
                    component.kind(),
                    component.num_states()
                );
            }
            h_values.push(distances);
            saturated_costs_by_abstraction.push(saturated);
        }

        if debug_diagnostics {
            let max_initial_h = debug_initial_h_values
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            info!(
                "scp debug: collection max original-cost initial h before cost partitioning = {max_initial_h}"
            );
        }

        let surplus_costs =
            compute_all_surplus_costs(&original_costs, &saturated_costs_by_abstraction);
        let mut stolen_costs: Vec<f64> = saturated_costs_by_abstraction
            .iter()
            .map(|saturated| compute_costs_stolen_by_heuristic(saturated, &surplus_costs))
            .collect();
        if config.partitioning.uses_regions() {
            let regional_scores = interval_stolen_costs(
                &components,
                &saturated_costs_by_abstraction,
                &original_costs,
            )?;
            for (component_id, score) in regional_scores.into_iter().enumerate() {
                if let Some(score) = score {
                    stolen_costs[component_id] = score;
                }
            }
        }

        let mut st = ScpState::new(config.random_seed);
        st.goal_distances_by_abstraction = h_values;
        st.stolen_costs_by_abstraction = stolen_costs;
        st.lookup_mask = vec![false; components.len()];

        Ok(Self {
            name: name.unwrap_or_else(|| "scp_online".to_string()),
            task,
            components: RefCell::new(components),
            config,
            debug_diagnostics,
            original_operator_costs: original_costs,
            state: RefCell::new(st),
            lookup_scratch: RefCell::new(DomainAbstractionLookupScratch::new()),
            component_ids_scratch: RefCell::new(Vec::new()),
            sampling_task: RefCell::new(sampling_task),
        })
    }

    fn update_improvement_status(&self, state: &mut ScpState) {
        if !self.config.online {
            return;
        }
        let time_limit_reached = self.config.max_time.is_finite()
            && state.start_time.elapsed() >= Duration::from_secs_f64(self.config.max_time);

        if state.improve_heuristic && (time_limit_reached || state.size_kb >= self.config.max_size)
        {
            state.improve_heuristic = false;
        }
    }

    fn release_abstractions_if_finished(&self, state: &mut ScpState) {
        if !state.improve_heuristic && !state.improvement_ended {
            for component in self.components.borrow_mut().iter_mut() {
                component.discard_transition_data();
            }
            state.improvement_ended = true;
        }
    }

    fn should_build_cp(&self, state: &ScpState) -> bool {
        state.improve_heuristic
            && (state.evaluated_states == 0
                || (self.config.online
                    && state.evaluated_states.is_multiple_of(self.config.interval)))
    }

    fn maybe_build_cp(
        &self,
        task: &dyn AbstractNumericTask,
        state: &mut ScpState,
        abstract_state_ids: &[Option<usize>],
    ) -> Result<Vec<CostPartitioningHeuristic>, EvaluationError> {
        if !self.should_build_cp(state) {
            return Ok(Vec::new());
        }

        let components = self.components.borrow();
        if components.is_empty() {
            return Ok(Vec::new());
        }
        let original_costs = self.original_operator_costs.as_slice();
        let deadline = self
            .config
            .table_construction_max_time
            .is_finite()
            .then(|| {
                Instant::now() + Duration::from_secs_f64(self.config.table_construction_max_time)
            });
        let abstraction_h = abstraction_heuristic_values(state, abstract_state_ids);
        let collection = PartitionedCollection {
            task,
            components: &components,
            abstract_state_ids,
            abstraction_h: &abstraction_h,
            original_costs,
        };
        let mut order =
            self.compute_order_for_state(task, state, abstract_state_ids, &components, deadline)?;
        let mode = if self.config.partitioning.uses_regions() {
            "regional"
        } else {
            "label"
        };
        info!(
            "scp: building {mode} CP at evaluation {}, stored_cps={}, current_h={}, size={} KiB, order_len={}, saturator={}, elapsed={:.3}s",
            state.evaluated_states,
            state.cp_heuristics.len(),
            Self::compute_max_h(state, abstract_state_ids),
            state.size_kb,
            order.len(),
            self.config.saturator,
            state.start_time.elapsed().as_secs_f64(),
        );
        if self.debug_diagnostics {
            log_abstraction_candidate_report(
                mode,
                state,
                &components,
                &order,
                abstract_state_ids,
                self.config.scoring_function,
            );
        }

        let initial_order_generation_max_time = if self.config.diversify {
            self.config.initial_order_generation_max_time
        } else {
            self.config.order_optimization_max_time
        };
        let mut candidates = self.build_best_cp_from_candidate_orders(
            collection,
            &mut order,
            deadline,
            initial_order_generation_max_time,
        )?;

        if self.config.order_optimization_max_time > 0.0 {
            let local_deadline = optimization_deadline(self.config.order_optimization_max_time);
            self.optimize_order_with_hill_climbing(
                collection,
                &mut order,
                &mut candidates.partitions[candidates.best_index],
                earliest_deadline(deadline, local_deadline),
            )?;
        }

        if self.config.online {
            let best = candidates.partitions.swap_remove(candidates.best_index);
            if best.is_empty() {
                info!("scp: {mode} CP attempt produced no lookup tables");
                return Ok(Vec::new());
            }
            return Ok(vec![best]);
        }
        if self.config.diversify {
            let best = candidates.partitions.swap_remove(candidates.best_index);
            candidates.partitions.retain(|cp| !cp.is_empty());
            if best.is_empty() && candidates.partitions.is_empty() {
                info!("scp: {mode} CP attempt produced no positive lookup tables");
                return Ok(Vec::new());
            }
            let mut initial_candidates = Vec::with_capacity(candidates.partitions.len() + 1);
            if !best.is_empty() {
                initial_candidates.push(best);
            }
            initial_candidates.extend(candidates.partitions);
            return self.select_diverse_orders(collection, state, initial_candidates, deadline);
        }
        candidates.partitions.retain(|cp| !cp.is_empty());
        if candidates.partitions.is_empty() {
            info!("scp: {mode} CP attempt produced no lookup tables");
            return Ok(Vec::new());
        }
        info!(
            "scp: retaining {} offline SCP order partitions",
            candidates.partitions.len()
        );
        Ok(candidates.partitions)
    }

    fn retain_cp(
        state: &mut ScpState,
        cp: CostPartitioningHeuristic,
        abstract_state_ids: &[Option<usize>],
        max_h: &mut f64,
        retain_alternative: bool,
        max_size_kb: usize,
    ) {
        if state.cp_heuristics.contains(&cp) {
            info!(
                "scp: discarded duplicate CP with {} lookup tables",
                cp.lookup_tables.len()
            );
            return;
        }
        let new_h = cp.compute_heuristic(abstract_state_ids);
        let improves_current_state = new_h > *max_h;
        let size_kb = cp.estimate_size_in_kb();
        let fits_size_limit =
            state.cp_heuristics.is_empty() || state.size_kb.saturating_add(size_kb) <= max_size_kb;
        if (improves_current_state || retain_alternative) && fits_size_limit {
            let component_values = cp
                .lookup_tables
                .iter()
                .map(|table| {
                    let value = abstract_state_ids
                        .get(table.abstraction_id)
                        .copied()
                        .flatten()
                        .and_then(|state_id| table.distances.get(state_id))
                        .copied()
                        .unwrap_or(table.unknown_value);
                    (table.abstraction_id, value)
                })
                .collect::<Vec<_>>();
            info!(
                "scp: retained CP, current-state h {} -> {}, lookup_tables={}, components={:?}, size={} KiB, alternative={}",
                *max_h,
                new_h,
                cp.lookup_tables.len(),
                component_values,
                size_kb,
                !improves_current_state,
            );
            state.size_kb = state.size_kb.saturating_add(size_kb);
            state.cp_heuristics.push(cp);
            state.lookup_mask = Self::compute_lookup_mask(state, abstract_state_ids.len());
            *max_h = (*max_h).max(new_h);
        } else if !fits_size_limit {
            info!(
                "scp: discarded CP because storing {} KiB would exceed max_size={} KiB (stored={} KiB)",
                size_kb, max_size_kb, state.size_kb
            );
        } else {
            info!(
                "scp: rejected CP, candidate_h={} did not improve current_h={}, lookup_tables={}",
                new_h,
                *max_h,
                cp.lookup_tables.len(),
            );
        }
    }
}

impl Heuristic for SaturatedCostPartitioningHeuristic<'_> {
    fn compute_heuristic(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError> {
        let task = eval_state.task();
        let build_cp = {
            let state = self.state.borrow();
            self.should_build_cp(&state)
        };

        let mut component_ids = self.component_ids_scratch.borrow_mut();
        if build_cp {
            self.compute_abstract_state_ids_into(eval_state, None, &mut component_ids)?;
        } else {
            let state = self.state.borrow();
            self.compute_abstract_state_ids_into(
                eval_state,
                Some(&state.lookup_mask),
                &mut component_ids,
            )?;
        }
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

        let candidate_partitions = self.maybe_build_cp(task, &mut state, abstract_state_ids)?;
        if build_cp && !self.config.online {
            Self::retain_abstraction_lookup_tables(&mut state, abstract_state_ids.len());
            max_h = max_h.max(Self::compute_max_h(&state, abstract_state_ids));
        }
        for cp in candidate_partitions {
            Self::retain_cp(
                &mut state,
                cp,
                abstract_state_ids,
                &mut max_h,
                !self.config.online,
                self.config.max_size,
            );
        }

        if build_cp && (!self.config.online || self.config.interval == usize::MAX) {
            state.improve_heuristic = false;
        }
        if build_cp && !self.config.online {
            assert!(
                state.goal_distances_by_abstraction.is_empty(),
                "offline standalone tables must move into the retained envelope"
            );
            state.stolen_costs_by_abstraction.clear();
            state.stolen_costs_by_abstraction.shrink_to_fit();
        }
        self.update_improvement_status(&mut state);
        self.release_abstractions_if_finished(&mut state);

        state.evaluated_states = state.evaluated_states.saturating_add(1);
        Ok(max_h)
    }

    fn heuristic_name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests;
