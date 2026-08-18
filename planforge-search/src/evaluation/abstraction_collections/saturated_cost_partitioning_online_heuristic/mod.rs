use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use planforge_sas::numeric_task::{
    AbstractNumericTask, TaskRef, metric_operator_cost_from_initial_values,
};
use planforge_sas::state_registry::{ConcreteState, ExpansionContext, StateRegistry};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng, rngs::SmallRng};
use tracing::{Level, debug, enabled, info};

use crate::evaluation::cartesian_abstractions::{
    CartesianAbstraction, CartesianAbstractionHeuristic, CartesianRefinementDirection,
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
mod diagnostics;
mod fill_scp;
pub(crate) use config::LegacyScpOnlineConfig;
pub use config::{
    CostPartitioningMethod, FillScpConfig, OrderGenerator, Saturator, ScoringFunction,
    ScpCollectionConfig,
};
/// Backward-compatible library name for the runtime collection configuration.
pub type ScpOnlineConfig = ScpCollectionConfig;
use diagnostics::*;
pub use fill_scp::FillScpHeuristic;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
struct LookupTable {
    abstraction_id: usize,
    distances: Vec<f64>,
    unknown_value: f64,
}

#[derive(Debug, Clone, Default, PartialEq)]
struct CostPartitioningHeuristic {
    lookup_tables: Vec<LookupTable>,
    specialist_goal_id: Option<usize>,
}

struct CandidateCostPartitions {
    partitions: Vec<CostPartitioningHeuristic>,
    best_index: usize,
}

/// Everything a cost partitioning over this collection needs except the order
/// it is built in: the task, its abstraction components, where one state lands
/// in every component and what each scores there standalone, and the original
/// operator costs the partition divides up.
///
/// Diversification evaluates the same collection at a sample rather than at the
/// state being expanded, which is `PartitionedCollection { abstract_state_ids,
/// standalone_current_h, ..collection }`.
#[derive(Clone, Copy)]
struct PartitionedCollection<'a, 'task> {
    task: &'a dyn AbstractNumericTask,
    components: &'a [AbstractionComponent<'task>],
    abstract_state_ids: &'a [Option<usize>],
    standalone_current_h: &'a [f64],
    original_costs: &'a [f64],
}

enum SaturatedCosts {
    Uniform(Vec<f64>),
    AbstractOperator(AbstractOperatorCostFunction),
    Regional(RegionalCostAllocation),
}

#[derive(Clone, Copy)]
enum SaturationCostSpace {
    Label,
    AbstractOperator,
    Regional,
}

struct ComponentSaturation<'a, 'task> {
    component: &'a AbstractionComponent<'task>,
    task: &'a dyn AbstractNumericTask,
    combine_labels: bool,
    cost_space: SaturationCostSpace,
    current_state_id: Option<usize>,
}

struct SaturationStepContext<'a> {
    component_id: usize,
    current_state_id: Option<usize>,
    abstract_state_ids: &'a [Option<usize>],
    saturator: Saturator,
    step_prefix: &'a str,
}

struct ComponentStepContext<'a, 'task> {
    component_id: usize,
    component: &'a AbstractionComponent<'task>,
    task: &'a dyn AbstractNumericTask,
    abstract_state_ids: &'a [Option<usize>],
    deadline: Option<Instant>,
    cost_space: SaturationCostSpace,
    saturator: Saturator,
    step_prefix: &'a str,
}

impl ComponentSaturation<'_, '_> {
    fn saturate(
        &self,
        residual: &TransitionResidualCosts,
        component_id: usize,
        cap_state_id: Option<usize>,
        deadline: Option<Instant>,
    ) -> Result<(Vec<f64>, SaturatedCosts), EvaluationError> {
        match (self.component, self.cost_space) {
            (AbstractionComponent::Domain(heuristic), SaturationCostSpace::Label) => {
                let abstraction = heuristic.abstraction();
                let abstraction_task = abstraction.task_for_factory(self.task);
                let operator_costs = residual.operator_costs_for_label_cp();
                let (distances, saturated) = if let Some(state_id) = cap_state_id {
                    let table = abstraction
                        .factory
                        .build_goal_distances_for_goals(
                            abstraction_task,
                            self.combine_labels,
                            &operator_costs,
                            &abstraction.distance_table.goal_facts,
                        )
                        .map_err(|error| {
                            EvaluationError::ComputationFailed(format!(
                                "failed to compute domain PERIM cap table: {error:#}"
                            ))
                        })?;
                    let h_cap = table.distances.get(state_id).copied().ok_or_else(|| {
                        EvaluationError::InvalidState(format!(
                            "domain component {component_id} state id {state_id} out of bounds for {} states",
                            table.distances.len()
                        ))
                    })?;
                    SaturatedCostPartitioningOnlineHeuristic::compute_domain_perim_entry(
                        abstraction,
                        abstraction_task,
                        self.combine_labels,
                        &operator_costs,
                        h_cap,
                    )?
                } else {
                    SaturatedCostPartitioningOnlineHeuristic::compute_domain_cp_entry(
                        abstraction,
                        abstraction_task,
                        self.combine_labels,
                        &operator_costs,
                        deadline,
                    )?
                };
                Ok((distances, SaturatedCosts::Uniform(saturated)))
            }
            (AbstractionComponent::Domain(heuristic), SaturationCostSpace::AbstractOperator) => {
                let abstraction = heuristic.abstraction();
                let abstraction_task = abstraction.task_for_factory(self.task);
                let (table, saturated) = abstraction
                    .factory
                    .build_abstract_operator_cost_partitioned_distance_table_with_operators_and_operator_regions(
                        abstraction_task,
                        abstraction.combine_labels,
                        &abstraction.abstract_operators,
                        &abstraction.abstract_operator_regions,
                        SaturationStep {
                            residual_costs: residual,
                            abstraction_id: component_id,
                            current_state_id: self.current_state_id,
                            cap_state_id,
                        },
                        DistanceTableOptions::default().with_deadline(deadline),
                    )
                    .map_err(|error| {
                        SaturatedCostPartitioningOnlineHeuristic::construction_error(
                            "failed to compute domain abstract-operator saturation",
                            error,
                        )
                    })?;
                Ok((table.distances, SaturatedCosts::AbstractOperator(saturated)))
            }
            (AbstractionComponent::Domain(heuristic), SaturationCostSpace::Regional) => {
                let abstraction = heuristic.abstraction();
                let abstraction_task = abstraction.task_for_factory(self.task);
                let transition_system = abstraction
                    .regional_transition_system(abstraction_task, deadline)
                    .map_err(|error| {
                        SaturatedCostPartitioningOnlineHeuristic::construction_error(
                            "failed to build domain regional transition system",
                            error,
                        )
                    })?;
                let (table, saturated) = abstraction
                    .factory
                    .build_precise_regional_cost_partitioned_distance_table(
                        &transition_system,
                        &abstraction.abstract_operator_regions,
                        residual,
                        component_id,
                        DistanceTableOptions::default()
                            .with_cap_state(cap_state_id)
                            .with_deadline(deadline),
                    )
                    .map_err(|error| {
                        SaturatedCostPartitioningOnlineHeuristic::construction_error(
                            "failed to compute domain regional saturation",
                            error,
                        )
                    })?;
                Ok((table.distances, SaturatedCosts::Regional(saturated)))
            }
            (AbstractionComponent::Cartesian(heuristic), SaturationCostSpace::Label) => {
                let (distances, saturated) = build_explicit_label_cost_partitioning_table(
                    &heuristic.abstraction().transition_system,
                    &residual.operator_costs_for_label_cp(),
                    cap_state_id,
                    deadline,
                )
                .map_err(|error| {
                    SaturatedCostPartitioningOnlineHeuristic::construction_error(
                        &format!(
                            "failed to compute Cartesian label saturation for component {component_id}"
                        ),
                        error,
                    )
                })?;
                Ok((distances, SaturatedCosts::Uniform(saturated)))
            }
            (
                AbstractionComponent::Cartesian(heuristic),
                SaturationCostSpace::AbstractOperator | SaturationCostSpace::Regional,
            ) => {
                // Phase 5 should give Cartesian components native regional allocations;
                // the Regional arm currently degrades to abstract-operator granularity.
                let abstraction = heuristic.abstraction();
                let (distances, saturated) = build_explicit_regional_cost_partitioning_table(
                    &abstraction.transition_system,
                    &abstraction.abstract_operator_regions,
                    residual,
                    component_id,
                    cap_state_id,
                    deadline,
                )
                .map_err(|error| {
                    SaturatedCostPartitioningOnlineHeuristic::construction_error(
                        &format!(
                            "failed to compute Cartesian regional saturation for component {component_id}"
                        ),
                        error,
                    )
                })?;
                Ok((distances, SaturatedCosts::AbstractOperator(saturated)))
            }
            (AbstractionComponent::PatternDatabase(pdb), _) => {
                // Phase 5 will add regional PDB support; until then every PDB
                // saturation deliberately degrades to label granularity.
                let operator_costs = residual.operator_costs_for_label_cp();
                let (distances, saturated) = if let Some(state_id) = cap_state_id {
                    let cap_distances = pdb.build_goal_distances(&operator_costs).map_err(|error| {
                        EvaluationError::ComputationFailed(format!(
                            "failed to compute PDB PERIM cap table {component_id}: {error}"
                        ))
                    })?;
                    let h_cap = cap_distances.get(state_id).copied().ok_or_else(|| {
                        EvaluationError::InvalidState(format!(
                            "PDB component {component_id} state id {state_id} out of bounds for {} states",
                            cap_distances.len()
                        ))
                    })?;
                    pdb.build_cost_partitioned_distance_table_capped(&operator_costs, h_cap)
                } else {
                    pdb.build_cost_partitioned_distance_table(&operator_costs)
                }
                .map_err(|error| {
                    EvaluationError::ComputationFailed(format!(
                        "failed to compute PDB saturation {component_id}: {error}"
                    ))
                })?;
                Ok((distances, SaturatedCosts::Uniform(saturated)))
            }
        }
    }
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
                // As in standalone_envelope_value: an entry this table does not
                // have is an unknown lower bound, which is zero, not a proof
                // that the state is a dead end.
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
    standalone_lookup_tables: Vec<LookupTable>,
    stolen_costs_by_abstraction: Vec<f64>,
    rng: SmallRng,
    improvement_ended: bool,
    required_mask: Vec<bool>,
    offline_sample_ids: Vec<Vec<Option<usize>>>,
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
            standalone_lookup_tables: Vec::new(),
            stolen_costs_by_abstraction: Vec::new(),
            rng: SmallRng::seed_from_u64(seed),
            improvement_ended: false,
            required_mask: Vec::new(),
            offline_sample_ids: Vec::new(),
        }
    }
}

pub struct SaturatedCostPartitioningOnlineHeuristic<'task> {
    name: String,
    task: &'task dyn AbstractNumericTask,
    components: RefCell<Vec<AbstractionComponent<'task>>>,
    config: ScpCollectionConfig,
    debug_diagnostics: bool,
    original_operator_costs: Vec<f64>,
    state: RefCell<ScpOnlineState>,
    lookup_scratch: RefCell<DomainAbstractionLookupScratch>,
    component_ids_scratch: RefCell<Vec<Option<usize>>>,
    sampling_task: RefCell<Option<TaskRef<'task>>>,
}

impl<'task> SaturatedCostPartitioningOnlineHeuristic<'task> {
    pub fn new(
        name: Option<String>,
        abstractions: Vec<DomainAbstraction>,
        pdbs: Vec<PatternDatabase<'task>>,
        config: ScpCollectionConfig,
        task: &'task dyn AbstractNumericTask,
    ) -> Result<Self, EvaluationError> {
        Self::new_with_cartesian(name, abstractions, Vec::new(), pdbs, config, task)
    }

    pub fn new_with_cartesian(
        name: Option<String>,
        abstractions: Vec<DomainAbstraction>,
        cartesian_abstractions: Vec<CartesianAbstraction>,
        pdbs: Vec<PatternDatabase<'task>>,
        config: ScpCollectionConfig,
        task: &'task dyn AbstractNumericTask,
    ) -> Result<Self, EvaluationError> {
        Self::new_with_cartesian_and_sampling_task(
            name,
            abstractions,
            cartesian_abstractions,
            pdbs,
            config,
            task,
            None,
        )
    }

    fn new_with_cartesian_and_sampling_task(
        name: Option<String>,
        abstractions: Vec<DomainAbstraction>,
        cartesian_abstractions: Vec<CartesianAbstraction>,
        pdbs: Vec<PatternDatabase<'task>>,
        config: ScpCollectionConfig,
        task: &'task dyn AbstractNumericTask,
        sampling_task: Option<TaskRef<'task>>,
    ) -> Result<Self, EvaluationError> {
        let mut components =
            Vec::with_capacity(abstractions.len() + cartesian_abstractions.len() + pdbs.len());
        components.extend(
            abstractions
                .into_iter()
                .enumerate()
                .map(|(index, abstraction)| {
                    AbstractionComponent::domain(Some(format!("scp_online_{index}")), abstraction)
                }),
        );
        components.extend(cartesian_abstractions.into_iter().enumerate().map(
            |(index, abstraction)| {
                AbstractionComponent::cartesian(
                    Some(format!("scp_online_cartesian_{index}")),
                    abstraction,
                )
            },
        ));
        components.extend(pdbs.into_iter().map(AbstractionComponent::pattern_database));
        Self::new_from_components_and_sampling_task(
            name,
            components,
            config,
            task,
            sampling_task,
            false,
        )
    }

    fn new_from_components_and_sampling_task(
        name: Option<String>,
        components: Vec<AbstractionComponent<'task>>,
        config: ScpCollectionConfig,
        task: &'task dyn AbstractNumericTask,
        sampling_task: Option<TaskRef<'task>>,
        debug_diagnostics: bool,
    ) -> Result<Self, EvaluationError> {
        if components.is_empty() {
            return Err(EvaluationError::ComputationFailed(
                "SCP requires at least one abstraction component".to_string(),
            ));
        }
        if config.online && config.interval == 0 {
            return Err(EvaluationError::ComputationFailed(
                "online SCP interval must be greater than zero".to_string(),
            ));
        }
        if config.online && config.diversify {
            return Err(EvaluationError::ComputationFailed(
                "offline SCP diversification requires online=false".to_string(),
            ));
        }
        if config.order_generator == OrderGenerator::Diverse && !config.diversify {
            return Err(EvaluationError::ComputationFailed(
                "diverse SCP orders require diversify=true".to_string(),
            ));
        }
        if config.diversify && config.samples == 0 {
            return Err(EvaluationError::ComputationFailed(
                "offline SCP diversification requires samples > 0".to_string(),
            ));
        }
        if config.diversify && config.max_orders == 0 {
            return Err(EvaluationError::ComputationFailed(
                "offline SCP diversification requires max_orders > 0".to_string(),
            ));
        }
        if config.diversify
            && (config.initial_order_generation_max_time.is_nan()
                || config.initial_order_generation_max_time < 0.0)
        {
            return Err(EvaluationError::ComputationFailed(
                "offline SCP diversification requires initial_order_generation_max_time >= 0"
                    .to_string(),
            ));
        }
        if config.diversify && sampling_task.is_none() {
            return Err(EvaluationError::ComputationFailed(
                "offline SCP diversification requires an owned task reference for sampling"
                    .to_string(),
            ));
        }
        if config.max_size == 0 {
            return Err(EvaluationError::ComputationFailed(
                "SCP max_size must be greater than zero".to_string(),
            ));
        }
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
            .map(|op| metric_operator_cost_from_initial_values(task, op))
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
                            "scp_online debug: collection abstraction {component_id}: states={}, goal_facts={}",
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
                    "scp_online debug: {} component {component_id}: original_initial_h={initial_h}, states={}",
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
                "scp_online debug: collection max original-cost initial h before cost partitioning = {max_initial_h}"
            );
        }

        let surplus_costs =
            compute_all_surplus_costs(&original_costs, &saturated_costs_by_abstraction);
        let mut stolen_costs: Vec<f64> = saturated_costs_by_abstraction
            .iter()
            .map(|saturated| compute_costs_stolen_by_heuristic(saturated, &surplus_costs))
            .collect();
        if config.partitioning.uses_regions() {
            let regional_scores = compute_regional_conflict_scores(
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

        let mut st = ScpOnlineState::new(config.random_seed);
        st.h_values_by_abstraction = h_values;
        st.stolen_costs_by_abstraction = stolen_costs;
        st.required_mask = vec![false; components.len()];

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

    pub fn from_components(
        name: Option<String>,
        components: Vec<AbstractionComponent<'task>>,
        config: ScpCollectionConfig,
        task: &'task dyn AbstractNumericTask,
    ) -> Result<Self, EvaluationError> {
        Self::new_from_components_and_sampling_task(name, components, config, task, None, false)
    }

    pub(crate) fn from_components_with_debug(
        name: Option<String>,
        components: Vec<AbstractionComponent<'task>>,
        config: ScpCollectionConfig,
        task: &'task dyn AbstractNumericTask,
        debug_diagnostics: bool,
    ) -> Result<Self, EvaluationError> {
        Self::new_from_components_and_sampling_task(
            name,
            components,
            config,
            task,
            None,
            debug_diagnostics,
        )
    }

    pub fn from_components_with_sampling_task(
        name: Option<String>,
        components: Vec<AbstractionComponent<'task>>,
        config: ScpCollectionConfig,
        task: &'task dyn AbstractNumericTask,
        sampling_task: TaskRef<'task>,
    ) -> Result<Self, EvaluationError> {
        Self::new_from_components_and_sampling_task(
            name,
            components,
            config,
            task,
            Some(sampling_task),
            false,
        )
    }

    pub(crate) fn from_components_with_sampling_task_and_debug(
        name: Option<String>,
        components: Vec<AbstractionComponent<'task>>,
        config: ScpCollectionConfig,
        task: &'task dyn AbstractNumericTask,
        sampling_task: TaskRef<'task>,
        debug_diagnostics: bool,
    ) -> Result<Self, EvaluationError> {
        Self::new_from_components_and_sampling_task(
            name,
            components,
            config,
            task,
            Some(sampling_task),
            debug_diagnostics,
        )
    }

    fn add_saturation_steps<R, Build, Reduce, Log>(
        &self,
        cp: &mut CostPartitioningHeuristic,
        residual: &mut R,
        context: SaturationStepContext<'_>,
        mut build: Build,
        mut reduce: Reduce,
        mut log: Log,
    ) -> Result<bool, EvaluationError>
    where
        R: ?Sized,
        Build: FnMut(&R, Option<usize>) -> Result<(Vec<f64>, SaturatedCosts), EvaluationError>,
        Reduce: FnMut(&mut R, &SaturatedCosts) -> Result<bool, EvaluationError>,
        Log: FnMut(&str, &[f64], &SaturatedCosts),
    {
        let SaturationStepContext {
            component_id,
            current_state_id,
            abstract_state_ids,
            saturator,
            step_prefix,
        } = context;
        if saturator != Saturator::All && current_state_id.is_none() {
            return Err(EvaluationError::InvalidState(format!(
                "missing abstract state id for PERIM component {component_id}"
            )));
        }
        for (phase, cap_state_id) in saturator.cap_sequence(current_state_id) {
            let step = format!("{step_prefix} {phase}");
            let (distances, saturated) = build(residual, cap_state_id)?;
            log(&step, &distances, &saturated);
            if should_skip_zero_current_table(
                self.config.diversify,
                &step,
                component_id,
                &distances,
                abstract_state_ids,
            ) {
                return Ok(true);
            }
            if !reduce(residual, &saturated)? {
                return Ok(false);
            }
            cp.add_h_values(component_id, distances);
        }
        Ok(true)
    }

    fn add_label_component_step(
        &self,
        cp: &mut CostPartitioningHeuristic,
        remaining_costs: &mut [f64],
        context: ComponentStepContext<'_, '_>,
    ) -> Result<bool, EvaluationError> {
        let ComponentStepContext {
            component_id,
            component,
            task,
            abstract_state_ids,
            deadline,
            step_prefix,
            ..
        } = context;
        let current_state_id = abstract_state_ids.get(component_id).copied().flatten();
        let protocol = ComponentSaturation {
            component,
            task,
            combine_labels: self.config.combine_labels,
            cost_space: SaturationCostSpace::Label,
            current_state_id,
        };
        self.add_saturation_steps(
            cp,
            remaining_costs,
            SaturationStepContext {
                component_id,
                current_state_id,
                abstract_state_ids,
                saturator: self.config.saturator,
                step_prefix,
            },
            |costs, cap_state_id| {
                let residual = TransitionResidualCosts::from_operator_costs(costs);
                protocol.saturate(&residual, component_id, cap_state_id, deadline)
            },
            |costs, saturated| {
                let SaturatedCosts::Uniform(saturated) = saturated else {
                    unreachable!("label saturation must return uniform operator costs")
                };
                reduce_costs(costs, saturated)?;
                Ok(true)
            },
            |step, distances, saturated| {
                let SaturatedCosts::Uniform(saturated) = saturated else {
                    unreachable!("label saturation must return uniform operator costs")
                };
                log_label_table_summary(
                    step,
                    component_id,
                    distances,
                    saturated,
                    abstract_state_ids,
                );
            },
        )
    }

    fn add_transition_component_step(
        &self,
        cp: &mut CostPartitioningHeuristic,
        remaining_costs: &mut TransitionResidualCosts,
        context: ComponentStepContext<'_, '_>,
    ) -> Result<bool, EvaluationError> {
        let ComponentStepContext {
            component_id,
            component,
            task,
            abstract_state_ids,
            deadline,
            cost_space,
            saturator,
            step_prefix,
        } = context;
        let current_state_id = abstract_state_ids.get(component_id).copied().flatten();
        let protocol = ComponentSaturation {
            component,
            task,
            combine_labels: self.config.combine_labels,
            cost_space,
            current_state_id,
        };
        let operator_regions = match component {
            AbstractionComponent::Domain(heuristic) => {
                heuristic.abstraction().abstract_operator_regions.as_slice()
            }
            AbstractionComponent::Cartesian(heuristic) => {
                heuristic.abstraction().abstract_operator_regions.as_slice()
            }
            AbstractionComponent::PatternDatabase(_) => &[],
        };
        self.add_saturation_steps(
            cp,
            remaining_costs,
            SaturationStepContext {
                component_id,
                current_state_id,
                abstract_state_ids,
                saturator,
                step_prefix,
            },
            |residual, cap_state_id| {
                protocol.saturate(residual, component_id, cap_state_id, deadline)
            },
            |residual, saturated| match saturated {
                SaturatedCosts::Uniform(costs) => {
                    residual
                        .reduce_operator_costs_uniform(costs)
                        .map_err(|error| {
                            EvaluationError::ComputationFailed(format!(
                                "failed to reduce uniform residual costs for component {component_id}: {error:#}"
                            ))
                        })?;
                    Ok(true)
                }
                SaturatedCosts::AbstractOperator(costs) => Self::reduce_abstract_operator_costs(
                    residual,
                    component_id,
                    operator_regions,
                    costs,
                    deadline,
                    &format!(
                        "failed to reduce abstract-operator residual costs for component {component_id}"
                    ),
                ),
                SaturatedCosts::Regional(allocation) => Self::reduce_regional_allocation(
                    residual,
                    allocation,
                    deadline,
                    &format!(
                        "failed to reduce regional residual costs for component {component_id}"
                    ),
                ),
            },
            |step, distances, saturated| match saturated {
                SaturatedCosts::Uniform(costs) => log_transition_table_summary(
                    step,
                    component_id,
                    distances,
                    costs,
                    abstract_state_ids,
                ),
                SaturatedCosts::AbstractOperator(costs) => log_transition_table_summary(
                    step,
                    component_id,
                    distances,
                    &costs.operator_costs,
                    abstract_state_ids,
                ),
                SaturatedCosts::Regional(allocation) => log_transition_table_summary(
                    step,
                    component_id,
                    distances,
                    &regional_allocation_amounts(allocation),
                    abstract_state_ids,
                ),
            },
        )
    }

    fn compute_order_for_state(
        &self,
        task: &dyn AbstractNumericTask,
        state: &mut ScpOnlineState,
        abstract_state_ids: &[Option<usize>],
        components: &[AbstractionComponent<'_>],
        deadline: Option<Instant>,
    ) -> Result<Vec<usize>, EvaluationError> {
        let order = self.compute_state_dependent_order(
            task,
            state,
            abstract_state_ids,
            components,
            deadline,
        )?;
        if self.config.partitioning.uses_regions()
            && let Some(goal_cover_order) = self.cartesian_goal_cover_order(
                &order,
                components,
                &standalone_current_h_values(state, abstract_state_ids),
                true,
            )
        {
            if state.evaluated_states == 0 {
                info!(
                    "scp_online: using goal-cover regional-SCP order, first_components={:?}",
                    goal_cover_order.iter().take(24).collect::<Vec<_>>()
                );
            }
            return Ok(goal_cover_order);
        }
        Ok(order)
    }

    fn compute_state_dependent_order(
        &self,
        task: &dyn AbstractNumericTask,
        state: &mut ScpOnlineState,
        abstract_state_ids: &[Option<usize>],
        components: &[AbstractionComponent<'_>],
        deadline: Option<Instant>,
    ) -> Result<Vec<usize>, EvaluationError> {
        match self.config.order_generator {
            OrderGenerator::Greedy => Ok(Self::compute_greedy_order_for_state(
                state,
                abstract_state_ids,
                self.config.scoring_function,
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
                components,
                deadline,
            ),
            OrderGenerator::Diverse => Ok(Self::compute_greedy_order_for_state(
                state,
                abstract_state_ids,
                self.config.scoring_function,
            )),
        }
    }

    fn compute_diversification_orders(
        &self,
        task: &dyn AbstractNumericTask,
        state: &mut ScpOnlineState,
        abstract_state_ids: &[Option<usize>],
        components: &[AbstractionComponent<'_>],
        deadline: Option<Instant>,
    ) -> Result<Vec<Vec<usize>>, EvaluationError> {
        if self.config.order_generator != OrderGenerator::Diverse {
            return self
                .compute_state_dependent_order(
                    task,
                    state,
                    abstract_state_ids,
                    components,
                    deadline,
                )
                .map(|order| vec![order]);
        }

        let greedy = Self::compute_greedy_order_for_state(
            state,
            abstract_state_ids,
            self.config.scoring_function,
        );
        let mut random = (0..state.h_values_by_abstraction.len()).collect::<Vec<_>>();
        random.shuffle(&mut state.rng);
        Ok(deduplicate_orders(vec![greedy, random]))
    }

    fn cartesian_goal_cover_order(
        &self,
        base_order: &[usize],
        components: &[AbstractionComponent<'_>],
        standalone_current_h: &[f64],
        require_pure_cartesian_collection: bool,
    ) -> Option<Vec<usize>> {
        cartesian_goal_cover_order(
            base_order,
            components,
            standalone_current_h,
            require_pure_cartesian_collection,
            GoalCoverOrderVariant::default(),
        )
    }

    fn compact_cartesian_goal_cover_orders(
        &self,
        base_order: &[usize],
        components: &[AbstractionComponent<'_>],
        standalone_current_h: &[f64],
    ) -> Vec<(usize, Vec<usize>)> {
        let mut variants_by_goal = HashMap::<usize, usize>::new();
        for abstraction in components
            .iter()
            .filter_map(AbstractionComponent::as_cartesian)
        {
            if let Some(goal_id) = abstraction.metadata.collection_goal_id {
                *variants_by_goal.entry(goal_id).or_default() += 1;
            }
        }
        let progressive_roots = components
            .iter()
            .filter_map(AbstractionComponent::as_cartesian)
            .any(|abstraction| abstraction.metadata.progressive_refinement_root);
        let goal_count = variants_by_goal
            .values()
            .filter(|&&variant_count| progressive_roots || variant_count >= 2)
            .count();
        let max_variants = variants_by_goal.values().copied().max().unwrap_or(0);
        if goal_count == 0 || (!progressive_roots && max_variants < 2) {
            return Vec::new();
        }

        // Rotate the anchor goal before varying its construction variant. This
        // guarantees coverage of every goal when the 64-order cap is large
        // enough, including states where that goal is the last one remaining.
        let variants = compact_goal_cover_variants(goal_count, max_variants, progressive_roots);

        let mut seen = HashSet::new();
        variants
            .into_iter()
            .filter_map(|variant| {
                let order = cartesian_goal_cover_order(
                    base_order,
                    components,
                    standalone_current_h,
                    false,
                    variant,
                )?;
                let first_component = *order
                    .first()
                    .expect("compact goal-cover order must not be empty");
                let goal_id = components[first_component]
                    .as_cartesian()
                    .expect("compact goal-cover order must start with a Cartesian component")
                    .metadata
                    .collection_goal_id
                    .expect("compact goal-cover order must start with its anchor goal");
                seen.insert(order.clone()).then_some((goal_id, order))
            })
            .collect()
    }

    fn cartesian_specialist_goal_for_order(
        &self,
        order: &[usize],
        components: &[AbstractionComponent<'_>],
    ) -> Option<usize> {
        if order.len() != components.len()
            || !order.iter().copied().all(|id| {
                components
                    .get(id)
                    .and_then(AbstractionComponent::as_cartesian)
                    .is_some_and(|abstraction| abstraction.metadata.collection_goal_id.is_some())
            })
        {
            return None;
        }
        if !components[order[0]]
            .as_cartesian()?
            .metadata
            .progressive_refinement_root
        {
            return None;
        }
        order
            .first()
            .and_then(|&id| components[id].as_cartesian()?.metadata.collection_goal_id)
    }

    fn prefixed_cartesian_goal_cover_order(
        &self,
        base_order: &[usize],
        components: &[AbstractionComponent<'_>],
        standalone_current_h: &[f64],
    ) -> Option<Vec<usize>> {
        cartesian_goal_cover_order(
            base_order,
            components,
            standalone_current_h,
            false,
            GoalCoverOrderVariant {
                non_goal_prefix: true,
                ..Default::default()
            },
        )
    }

    fn compute_greedy_order_for_state(
        state: &mut ScpOnlineState,
        abstract_state_ids: &[Option<usize>],
        scoring_function: ScoringFunction,
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

        let scores: Vec<f64> = (0..total)
            .map(|abs_id| {
                let h = current_h[abs_id];
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

    fn compute_dynamic_greedy_order_for_state(
        &self,
        task: &dyn AbstractNumericTask,
        state: &mut ScpOnlineState,
        abstract_state_ids: &[Option<usize>],
        components: &[AbstractionComponent<'_>],
        deadline: Option<Instant>,
    ) -> Result<Vec<usize>, EvaluationError> {
        if self.config.partitioning.uses_regions() {
            return Err(EvaluationError::ComputationFailed(
                "dynamic_greedy_orders is only implemented for label SCP; regional SCP needs residual regional-cost scoring, not label-order scoring".to_string(),
            ));
        }

        let total = state.h_values_by_abstraction.len();
        let mut remaining_components: Vec<usize> = (0..total).collect();
        let mut remaining_costs = self.original_operator_costs.clone();
        let mut order = Vec::with_capacity(total);

        'selection: while !remaining_components.is_empty() {
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
                let component = components.get(pos).ok_or_else(|| {
                    EvaluationError::ComputationFailed(format!(
                        "dynamic order requested missing component {pos}"
                    ))
                })?;
                let result = match component {
                    AbstractionComponent::Domain(heuristic) => {
                        let abstraction = heuristic.abstraction();
                        let abstraction_task = abstraction.task_for_factory(task);
                        Self::compute_domain_cp_entry(
                            abstraction,
                            abstraction_task,
                            self.config.combine_labels,
                            &remaining_costs,
                            deadline,
                        )
                    }
                    AbstractionComponent::Cartesian(heuristic) => {
                        build_explicit_label_cost_partitioning_table(
                            &heuristic.abstraction().transition_system,
                            &remaining_costs,
                            None,
                            deadline,
                        )
                        .map_err(|error| {
                            Self::construction_error(
                                &format!(
                                    "failed to compute Cartesian dynamic-order table for component {pos}"
                                ),
                                error,
                            )
                        })
                    }
                    AbstractionComponent::PatternDatabase(pdb) => pdb
                        .build_cost_partitioned_distance_table(&remaining_costs)
                        .map_err(|error| {
                            EvaluationError::ComputationFailed(format!(
                                "failed to compute PDB dynamic-order table {pos}: {error}"
                            ))
                        }),
                };
                let (distances, saturated) = match result {
                    Ok(entry) => entry,
                    Err(error) if Self::is_online_deadline_error_eval(&error) => {
                        info!("scp_online: dynamic greedy order stopped at deadline");
                        break 'selection;
                    }
                    Err(error) => return Err(error),
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
        required_mask: Option<&[bool]>,
        ids: &mut Vec<Option<usize>>,
    ) -> Result<(), EvaluationError> {
        let components = self.components.borrow();
        if let Some(mask) = required_mask {
            assert_eq!(
                mask.len(),
                components.len(),
                "required component mask must match the abstraction collection"
            );
        }
        ids.clear();
        ids.resize(components.len(), None);

        let registry = eval_state.state_registry();
        let mut scratch = self.lookup_scratch.borrow_mut();
        eval_state.state().fill_state(registry, &mut scratch.prop);
        registry
            .fill_numeric_vars(eval_state.state(), &mut scratch.numeric)
            .map_err(|error| {
                EvaluationError::ComputationFailed(format!(
                    "failed to read numeric state: {error:?}"
                ))
            })?;
        scratch.required_domain_ids.clear();
        scratch.required_domain_ids.extend(
            components
                .iter()
                .enumerate()
                .filter(|(id, component)| {
                    required_mask.is_none_or(|mask| mask[*id])
                        && matches!(component, AbstractionComponent::Domain(_))
                })
                .map(|(id, _)| id),
        );
        scratch.comparisons.clear();
        if scratch.required_domain_ids.len() > 1 {
            let first_id = scratch.required_domain_ids[0];
            let AbstractionComponent::Domain(heuristic) = &components[first_id] else {
                unreachable!("required domain component id must reference a domain abstraction")
            };
            let DomainAbstractionLookupScratch {
                numeric,
                comparisons,
                ..
            } = &mut *scratch;
            heuristic.fill_comparison_values_from_projected_state_values(numeric, comparisons)?;
        }

        for (component_id, component) in components.iter().enumerate() {
            if required_mask.is_some_and(|mask| !mask[component_id]) {
                continue;
            }
            ids[component_id] = match component {
                AbstractionComponent::Domain(heuristic) => Some(
                    heuristic.compute_abstract_hash_from_projected_state_values_inner(
                        &scratch.prop,
                        &scratch.numeric,
                        Some(&scratch.comparisons),
                    )?,
                ),
                AbstractionComponent::Cartesian(heuristic) => Some(
                    heuristic
                        .abstraction()
                        .hierarchy
                        .map_state(&scratch.prop, &scratch.numeric)
                        .map_err(|error| EvaluationError::ComputationFailed(error.to_string()))?,
                ),
                AbstractionComponent::PatternDatabase(pdb) => pdb
                    .abstract_state_id_from_source_state_values(&scratch.prop, &scratch.numeric)
                    .map_err(EvaluationError::ComputationFailed)?,
            };
        }

        Ok(())
    }

    fn compute_max_h(state: &ScpOnlineState, ids: &[Option<usize>]) -> f64 {
        let partitioned = state
            .cp_heuristics
            .iter()
            .map(|cp| cp.compute_heuristic(ids))
            .fold(0.0, f64::max);
        let constructing_standalone = standalone_envelope_value(state, ids);
        let retained_standalone = state
            .standalone_lookup_tables
            .iter()
            .map(|table| {
                lookup_distance(
                    table.abstraction_id,
                    &table.distances,
                    table.unknown_value,
                    ids,
                )
            })
            .fold(0.0, f64::max);
        partitioned
            .max(constructing_standalone)
            .max(retained_standalone)
    }

    fn required_lookup_mask(state: &ScpOnlineState, component_count: usize) -> Vec<bool> {
        let mut ids: Vec<usize> = state
            .cp_heuristics
            .iter()
            .flat_map(|cp| cp.lookup_tables.iter().map(|table| table.abstraction_id))
            .collect();
        ids.extend(
            state
                .standalone_lookup_tables
                .iter()
                .map(|table| table.abstraction_id),
        );
        ids.sort_unstable();
        ids.dedup();
        let mut mask = vec![false; component_count];
        for id in ids {
            *mask
                .get_mut(id)
                .expect("retained lookup table must reference an existing component") = true;
        }
        mask
    }

    fn retain_standalone_envelope(state: &mut ScpOnlineState, component_count: usize) {
        assert!(
            state.standalone_lookup_tables.is_empty(),
            "standalone envelope must be retained exactly once"
        );
        state.standalone_lookup_tables = std::mem::take(&mut state.h_values_by_abstraction)
            .into_iter()
            .enumerate()
            .filter_map(|(abstraction_id, distances)| {
                distances
                    .iter()
                    .any(|value| *value > 0.0)
                    .then_some(LookupTable {
                        abstraction_id,
                        distances,
                        // See standalone_envelope_value: a miss is unknown,
                        // not unreachable, so it contributes nothing.
                        unknown_value: 0.0,
                    })
            })
            .collect();
        state.size_kb = state.size_kb.saturating_add(
            state
                .standalone_lookup_tables
                .iter()
                .map(|table| table.distances.len())
                .sum::<usize>()
                .saturating_mul(std::mem::size_of::<f64>())
                / 1024,
        );
        state.required_mask = Self::required_lookup_mask(state, component_count);
    }

    fn is_online_deadline_error(error: &anyhow::Error) -> bool {
        crate::resource_limits::is_deadline_exceeded(error)
    }

    fn is_online_deadline_error_eval(error: &EvaluationError) -> bool {
        matches!(error, EvaluationError::ConstructionDeadlineExceeded)
    }

    fn construction_error(context: &str, error: anyhow::Error) -> EvaluationError {
        if Self::is_online_deadline_error(&error) {
            EvaluationError::ConstructionDeadlineExceeded
        } else {
            EvaluationError::ComputationFailed(format!("{context}: {error:#}"))
        }
    }

    fn reduce_abstract_operator_costs(
        remaining_costs: &mut TransitionResidualCosts,
        abstraction_id: usize,
        operator_regions: &[AbstractOperatorRegions],
        tcf: &AbstractOperatorCostFunction,
        deadline: Option<Instant>,
        context: &str,
    ) -> Result<bool, EvaluationError> {
        match remaining_costs.reduce_by_abstract_operator_regions_with_deadline(
            abstraction_id,
            operator_regions,
            tcf,
            deadline,
        ) {
            Ok(()) => Ok(true),
            Err(error) if Self::is_online_deadline_error(&error) => Ok(false),
            Err(error) => Err(EvaluationError::ComputationFailed(format!(
                "{context}: {error:#}"
            ))),
        }
    }

    fn reduce_regional_allocation(
        remaining_costs: &mut TransitionResidualCosts,
        allocation: &RegionalCostAllocation,
        deadline: Option<Instant>,
        context: &str,
    ) -> Result<bool, EvaluationError> {
        match remaining_costs.reduce_by_regional_allocation_with_deadline(allocation, deadline) {
            Ok(()) => Ok(true),
            Err(error) if Self::is_online_deadline_error(&error) => Ok(false),
            Err(error) => Err(EvaluationError::ComputationFailed(format!(
                "{context}: {error:#}"
            ))),
        }
    }

    fn update_improvement_status(&self, state: &mut ScpOnlineState) {
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

    fn release_abstractions_if_finished(&self, state: &mut ScpOnlineState) {
        if !state.improve_heuristic && !state.improvement_ended {
            for component in self.components.borrow_mut().iter_mut() {
                component.discard_transition_data();
            }
            state.improvement_ended = true;
        }
    }

    fn should_build_cp(&self, state: &ScpOnlineState) -> bool {
        state.improve_heuristic
            && (state.evaluated_states == 0
                || (self.config.online
                    && state.evaluated_states.is_multiple_of(self.config.interval)))
    }

    fn maybe_build_cp(
        &self,
        task: &dyn AbstractNumericTask,
        state: &mut ScpOnlineState,
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
        let standalone_current_h = standalone_current_h_values(state, abstract_state_ids);
        let collection = PartitionedCollection {
            task,
            components: &components,
            abstract_state_ids,
            standalone_current_h: &standalone_current_h,
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
            "scp_online: building {mode} CP at evaluation {}, stored_cps={}, current_h={}, size={} KiB, order_len={}, saturator={}, elapsed={:.3}s",
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
        let mut candidates = if self.config.partitioning.uses_regions() {
            self.build_best_abstract_operator_cp_from_candidate_orders(
                collection,
                &mut order,
                deadline,
                initial_order_generation_max_time,
            )?
        } else {
            self.build_best_label_cp_from_candidate_orders(
                collection,
                &mut order,
                deadline,
                initial_order_generation_max_time,
            )?
        };

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
                info!("scp_online: {mode} CP attempt produced no lookup tables");
                return Ok(Vec::new());
            }
            return Ok(vec![best]);
        }
        if self.config.diversify {
            let best = candidates.partitions.swap_remove(candidates.best_index);
            candidates.partitions.retain(|cp| !cp.is_empty());
            if best.is_empty() && candidates.partitions.is_empty() {
                info!("scp_online: {mode} CP attempt produced no positive lookup tables");
                return Ok(Vec::new());
            }
            let mut initial_candidates = Vec::with_capacity(candidates.partitions.len() + 1);
            if !best.is_empty() {
                initial_candidates.push(best);
            }
            initial_candidates.extend(candidates.partitions);
            return self.build_offline_diversified_portfolio(
                collection,
                state,
                initial_candidates,
                deadline,
            );
        }
        candidates.partitions.retain(|cp| !cp.is_empty());
        if candidates.partitions.is_empty() {
            info!("scp_online: {mode} CP attempt produced no lookup tables");
            return Ok(Vec::new());
        }
        info!(
            "scp_online: retaining {} offline SCP order partitions",
            candidates.partitions.len()
        );
        Ok(candidates.partitions)
    }

    fn build_offline_diversified_portfolio(
        &self,
        collection: PartitionedCollection<'_, '_>,
        state: &mut ScpOnlineState,
        initial_candidates: Vec<CostPartitioningHeuristic>,
        table_deadline: Option<Instant>,
    ) -> Result<Vec<CostPartitioningHeuristic>, EvaluationError> {
        let PartitionedCollection {
            task,
            components,
            abstract_state_ids: initial_abstract_state_ids,
            ..
        } = collection;
        assert!(!self.config.online);
        assert!(self.config.diversify);
        assert!(!initial_candidates.is_empty());
        assert!(!initial_candidates[0].is_empty());

        let diversification_deadline = self
            .config
            .max_time
            .is_finite()
            .then(|| Instant::now() + Duration::from_secs_f64(self.config.max_time.max(0.0)));
        let deadline = earliest_deadline(table_deadline, diversification_deadline);
        let initial_h = initial_candidates[0]
            .compute_heuristic(initial_abstract_state_ids)
            .max(standalone_envelope_value(state, initial_abstract_state_ids));
        self.generate_offline_samples(state, initial_h, deadline)?;
        assert!(
            !state.offline_sample_ids.is_empty(),
            "offline diversification must retain at least the initial state sample"
        );

        let mut sample_best = state
            .offline_sample_ids
            .iter()
            .map(|ids| standalone_envelope_value(state, ids))
            .collect::<Vec<_>>();
        let mut portfolio = Vec::new();
        let standalone_size_kb = standalone_lookup_values_size_kb(&state.h_values_by_abstraction);
        if standalone_size_kb > self.config.max_size {
            return Err(EvaluationError::ComputationFailed(format!(
                "standalone abstraction envelope requires {standalone_size_kb} KiB, exceeding max_size={} KiB",
                self.config.max_size
            )));
        }
        let mut portfolio_size_kb = standalone_size_kb;
        let mut evaluated_orders = initial_candidates.len();
        let mandatory_indices =
            mandatory_goal_specialist_indices(&initial_candidates, &state.offline_sample_ids);
        let represented_goal_count = mandatory_indices
            .iter()
            .filter_map(|&index| initial_candidates[index].specialist_goal_id)
            .collect::<HashSet<_>>()
            .len();
        let specialist_count = mandatory_indices
            .iter()
            .filter(|&&index| initial_candidates[index].specialist_goal_id.is_some())
            .count();
        let mut initial_candidates = initial_candidates.into_iter().map(Some).collect::<Vec<_>>();
        let mut mandatory_partitions = Vec::new();
        for index in mandatory_indices {
            let candidate = initial_candidates[index]
                .take()
                .expect("mandatory SCP candidate indices must be unique");
            if !mandatory_partitions.contains(&candidate) {
                mandatory_partitions.push(candidate);
            }
        }
        if mandatory_partitions.len() > self.config.max_orders {
            return Err(EvaluationError::ComputationFailed(format!(
                "offline SCP requires {} orders to retain the global best and configured goal specialists, exceeding max_orders={}",
                mandatory_partitions.len(),
                self.config.max_orders,
            )));
        }
        for candidate in mandatory_partitions {
            retain_mandatory_partition(
                candidate,
                &state.offline_sample_ids,
                &mut sample_best,
                &mut portfolio,
                &mut portfolio_size_kb,
                self.config.max_size,
            )
            .map_err(EvaluationError::ComputationFailed)?;
        }
        info!(
            "scp_online: reserved {standalone_size_kb} KiB for the standalone envelope and retained {specialist_count} specialists across {represented_goal_count} goals plus the global best"
        );

        for candidate in initial_candidates.into_iter().flatten() {
            if portfolio.len() >= self.config.max_orders
                || portfolio_size_kb >= self.config.max_size
                || deadline.is_some_and(|end| Instant::now() >= end)
            {
                break;
            }
            retain_if_sample_improving(
                candidate,
                &state.offline_sample_ids,
                &mut sample_best,
                &mut portfolio,
                &mut portfolio_size_kb,
                self.config.max_size,
            );
        }
        assert!(
            !portfolio.is_empty(),
            "the global best SCP must be retained"
        );

        for sample_index in 1..state.offline_sample_ids.len() {
            if portfolio.len() >= self.config.max_orders
                || portfolio_size_kb >= self.config.max_size
                || deadline.is_some_and(|end| Instant::now() >= end)
            {
                break;
            }
            let sample_ids = state.offline_sample_ids[sample_index].clone();
            let standalone_h = standalone_current_h_values(state, &sample_ids);
            // The same collection, evaluated at this sample rather than at the
            // state the portfolio was seeded from.
            let sample_collection = PartitionedCollection {
                abstract_state_ids: &sample_ids,
                standalone_current_h: &standalone_h,
                ..collection
            };
            let orders = self.compute_diversification_orders(
                task,
                state,
                &sample_ids,
                components,
                deadline,
            )?;
            for mut order in orders {
                if portfolio.len() >= self.config.max_orders
                    || portfolio_size_kb >= self.config.max_size
                    || deadline.is_some_and(|end| Instant::now() >= end)
                {
                    break;
                }
                let candidate = if self.config.partitioning.uses_regions() {
                    self.build_abstract_operator_cp(
                        sample_collection,
                        &order,
                        deadline,
                        self.config.saturator,
                    )
                } else {
                    self.build_label_cp(sample_collection, &order, deadline)
                };
                let mut candidate = match candidate {
                    Ok(candidate) => candidate,
                    Err(error) if Self::is_online_deadline_error_eval(&error) => break,
                    Err(error) => return Err(error),
                };
                evaluated_orders += 1;

                if self.config.order_optimization_max_time > 0.0 {
                    let local_deadline =
                        optimization_deadline(self.config.order_optimization_max_time);
                    self.optimize_order_with_hill_climbing(
                        sample_collection,
                        &mut order,
                        &mut candidate,
                        earliest_deadline(deadline, local_deadline),
                    )?;
                }

                retain_if_sample_improving(
                    candidate,
                    &state.offline_sample_ids,
                    &mut sample_best,
                    &mut portfolio,
                    &mut portfolio_size_kb,
                    self.config.max_size,
                );
            }
        }

        let sample_count = state.offline_sample_ids.len();
        info!(
            "scp_online: offline diversification retained {} of {} evaluated partitions over {} samples ({} KiB)",
            portfolio.len(),
            evaluated_orders,
            sample_count,
            portfolio_size_kb,
        );
        state.offline_sample_ids.clear();
        state.offline_sample_ids.shrink_to_fit();
        Ok(portfolio)
    }

    fn generate_offline_samples(
        &self,
        state: &mut ScpOnlineState,
        initial_h: f64,
        deadline: Option<Instant>,
    ) -> Result<(), EvaluationError> {
        if !state.offline_sample_ids.is_empty() {
            return Ok(());
        }
        let sampling_task = self
            .sampling_task
            .borrow_mut()
            .take()
            .expect("offline diversification was validated to have an owned sampling task");
        let mut registry = StateRegistry::for_task(sampling_task.clone());
        let successor_generator = SuccessorTree::new(&*sampling_task);
        let initial_state = registry.get_initial_state();
        let average_cost = if self.original_operator_costs.is_empty() {
            0.0
        } else {
            self.original_operator_costs.iter().sum::<f64>()
                / self.original_operator_costs.len() as f64
        };
        let mut applicable = Vec::new();
        let mut propositional = Vec::new();
        let mut successor_numeric = Vec::new();
        let mut successor_cost = Vec::new();
        let mut expansion_context = ExpansionContext::default();
        let mut ids = Vec::new();

        self.map_sample_state(&initial_state, &registry, self.task, &mut ids)?;
        state.offline_sample_ids.push(ids.clone());

        while state.offline_sample_ids.len() < self.config.samples {
            if deadline.is_some_and(|end| Instant::now() >= end) {
                info!(
                    "scp_online: offline sampling deadline reached after {} samples",
                    state.offline_sample_ids.len()
                );
                break;
            }
            let walk_length = random_walk_length(initial_h, average_cost, &mut state.rng)?;
            let mut current = initial_state.clone();
            for _ in 0..walk_length {
                if deadline.is_some_and(|end| Instant::now() >= end) {
                    break;
                }
                current.fill_state(&registry, &mut propositional);
                applicable.clear();
                successor_generator.get_applicable_operators(&propositional, &mut applicable);
                let Some(&operator_id) = applicable.choose(&mut state.rng) else {
                    break;
                };
                registry
                    .build_expansion_context(&current, &mut expansion_context)
                    .map_err(|error| {
                        EvaluationError::ComputationFailed(format!(
                            "failed to build random-walk expansion context: {error:?}"
                        ))
                    })?;
                let operator = sampling_task
                    .get_operators()
                    .get(operator_id as usize)
                    .expect("successor generator returned an invalid operator id");
                let (successor, _) = registry
                    .apply_operator_in_context(
                        &current,
                        operator,
                        &expansion_context,
                        &mut successor_numeric,
                        &mut successor_cost,
                    )
                    .map_err(|error| {
                        EvaluationError::ComputationFailed(format!(
                            "failed to apply random-walk operator {}: {error:?}",
                            operator.name()
                        ))
                    })?;
                current = successor;
            }
            self.map_sample_state(&current, &registry, self.task, &mut ids)?;
            state.offline_sample_ids.push(ids.clone());
        }
        info!(
            "scp_online: generated {} offline random-walk samples",
            state.offline_sample_ids.len()
        );
        Ok(())
    }

    fn map_sample_state(
        &self,
        concrete_state: &ConcreteState,
        registry: &StateRegistry<'task>,
        task: &'task dyn AbstractNumericTask,
        ids: &mut Vec<Option<usize>>,
    ) -> Result<(), EvaluationError> {
        let eval_state = EvaluationState::new(concrete_state, task, registry);
        self.compute_abstract_state_ids_into(&eval_state, None, ids)?;
        Ok(())
    }

    fn build_best_label_cp_from_candidate_orders(
        &self,
        collection: PartitionedCollection<'_, '_>,
        incumbent_order: &mut Vec<usize>,
        baseline_deadline: Option<Instant>,
        optimization_max_time: f64,
    ) -> Result<CandidateCostPartitions, EvaluationError> {
        let PartitionedCollection {
            abstract_state_ids,
            standalone_current_h,
            ..
        } = collection;
        let mut best_order = incumbent_order.clone();
        let baseline = self.build_label_cp(collection, &best_order, baseline_deadline)?;
        let mut best_h = baseline.compute_heuristic(abstract_state_ids);
        let mut partitions = vec![baseline];
        let mut best_index = 0;
        let candidate_deadline = earliest_deadline(
            baseline_deadline,
            optimization_deadline(optimization_max_time),
        );

        for candidate in optimization_max_time
            .is_sign_positive()
            .then(|| self.candidate_label_orders(incumbent_order, standalone_current_h))
            .into_iter()
            .flatten()
        {
            if candidate_deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                break;
            }
            if candidate == best_order {
                continue;
            }
            let candidate_cp = match self.build_label_cp(collection, &candidate, candidate_deadline)
            {
                Ok(cp) => cp,
                Err(error) if Self::is_online_deadline_error_eval(&error) => {
                    info!(
                        "scp_online: label candidate order stopped while computing table (deadline)"
                    );
                    break;
                }
                Err(error) => return Err(error),
            };
            let candidate_h = candidate_cp.compute_heuristic(abstract_state_ids);
            let candidate_index = partitions.len();
            partitions.push(candidate_cp);
            if candidate_h > best_h
                || (candidate_h == best_h
                    && partitions[best_index].is_empty()
                    && !partitions[candidate_index].is_empty())
            {
                if self.debug_diagnostics {
                    info!("scp_online: label candidate order improved h {best_h} -> {candidate_h}");
                }
                best_h = candidate_h;
                best_order = candidate;
                best_index = candidate_index;
            }
        }

        *incumbent_order = best_order;
        Ok(CandidateCostPartitions {
            partitions,
            best_index,
        })
    }

    fn build_best_abstract_operator_cp_from_candidate_orders(
        &self,
        collection: PartitionedCollection<'_, '_>,
        incumbent_order: &mut Vec<usize>,
        baseline_deadline: Option<Instant>,
        optimization_max_time: f64,
    ) -> Result<CandidateCostPartitions, EvaluationError> {
        let PartitionedCollection {
            components,
            abstract_state_ids,
            standalone_current_h,
            ..
        } = collection;
        let mut best_order = incumbent_order.clone();
        let mut baseline = self.build_abstract_operator_cp(
            collection,
            &best_order,
            baseline_deadline,
            self.config.saturator,
        )?;
        baseline.specialist_goal_id =
            self.cartesian_specialist_goal_for_order(&best_order, components);
        let mut best_h = baseline.compute_heuristic(abstract_state_ids);
        let mut partitions = vec![baseline];
        let mut best_index = 0;
        let candidate_deadline = earliest_deadline(
            baseline_deadline,
            optimization_deadline(optimization_max_time),
        );

        for (specialist_goal_id, candidate) in optimization_max_time
            .is_sign_positive()
            .then(|| {
                self.candidate_abstract_operator_orders(
                    incumbent_order,
                    components,
                    standalone_current_h,
                )
            })
            .into_iter()
            .flatten()
        {
            if candidate_deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                break;
            }
            if candidate == best_order {
                continue;
            }
            let mut candidate_cp = match self.build_abstract_operator_cp(
                collection,
                &candidate,
                candidate_deadline,
                self.config.saturator,
            ) {
                Ok(cp) => cp,
                Err(error) if Self::is_online_deadline_error_eval(&error) => {
                    info!(
                        "scp_online: abstract-operator candidate order stopped while computing table (deadline)"
                    );
                    break;
                }
                Err(error) => return Err(error),
            };
            candidate_cp.specialist_goal_id = specialist_goal_id
                .or_else(|| self.cartesian_specialist_goal_for_order(&candidate, components));
            let candidate_h = candidate_cp.compute_heuristic(abstract_state_ids);
            let candidate_index = partitions.len();
            partitions.push(candidate_cp);
            if candidate_h > best_h
                || (candidate_h == best_h
                    && partitions[best_index].is_empty()
                    && !partitions[candidate_index].is_empty())
            {
                if self.debug_diagnostics {
                    info!("scp_online: candidate order improved h {best_h} -> {candidate_h}");
                }
                best_h = candidate_h;
                best_order = candidate;
                best_index = candidate_index;
            }
        }

        *incumbent_order = best_order;
        Ok(CandidateCostPartitions {
            partitions,
            best_index,
        })
    }

    fn candidate_abstract_operator_orders(
        &self,
        base_order: &[usize],
        components: &[AbstractionComponent<'_>],
        standalone_current_h: &[f64],
    ) -> Vec<(Option<usize>, Vec<usize>)> {
        let mut orders = Vec::new();
        orders.push((None, base_order.to_vec()));

        let mut declaration_order = base_order.to_vec();
        declaration_order.sort_unstable();
        orders.push((None, declaration_order));

        if let Some(prefixed_goal_cover_order) =
            self.prefixed_cartesian_goal_cover_order(base_order, components, standalone_current_h)
        {
            orders.push((None, prefixed_goal_cover_order));
        }

        orders.extend(
            self.compact_cartesian_goal_cover_orders(base_order, components, standalone_current_h)
                .into_iter()
                .map(|(goal_id, order)| (Some(goal_id), order)),
        );

        if let Some(goal_cover_order) =
            self.cartesian_goal_cover_order(base_order, components, standalone_current_h, false)
        {
            orders.push((None, goal_cover_order));
        }

        orders.push((
            None,
            max_heuristic_greedy_order(base_order, standalone_current_h),
        ));

        let mut by_collection = base_order.to_vec();
        by_collection.sort_by_key(|&id| abstraction_collection_iteration(components, id));
        orders.push((None, by_collection));

        let mut progression_first = base_order.to_vec();
        progression_first.sort_by(|&left, &right| {
            abstraction_is_target_centered(components, left)
                .cmp(&abstraction_is_target_centered(components, right))
                .then_with(|| {
                    standalone_current_h
                        .get(right)
                        .copied()
                        .unwrap_or(0.0)
                        .total_cmp(&standalone_current_h.get(left).copied().unwrap_or(0.0))
                })
                .then_with(|| left.cmp(&right))
        });
        orders.push((None, progression_first));

        let mut target_first = base_order.to_vec();
        target_first.sort_by(|&left, &right| {
            abstraction_is_target_centered(components, right)
                .cmp(&abstraction_is_target_centered(components, left))
                .then_with(|| {
                    standalone_current_h
                        .get(right)
                        .copied()
                        .unwrap_or(0.0)
                        .total_cmp(&standalone_current_h.get(left).copied().unwrap_or(0.0))
                })
                .then_with(|| left.cmp(&right))
        });
        orders.push((None, target_first));

        for seed_offset in 0..3 {
            let mut random_order = base_order.to_vec();
            random_order.shuffle(&mut SmallRng::seed_from_u64(
                self.config
                    .random_seed
                    .unwrap_or(0x5C9_0A11)
                    .wrapping_add(seed_offset),
            ));
            orders.push((None, random_order));
        }

        deduplicate_specialist_orders(orders)
    }

    fn candidate_label_orders(
        &self,
        base_order: &[usize],
        standalone_current_h: &[f64],
    ) -> Vec<Vec<usize>> {
        deduplicate_orders(vec![
            base_order.to_vec(),
            max_heuristic_greedy_order(base_order, standalone_current_h),
        ])
    }

    fn optimize_order_with_hill_climbing(
        &self,
        collection: PartitionedCollection<'_, '_>,
        incumbent_order: &mut [usize],
        incumbent_cp: &mut CostPartitioningHeuristic,
        optimization_deadline: Option<Instant>,
    ) -> Result<(), EvaluationError> {
        let abstract_state_ids = collection.abstract_state_ids;
        let mut incumbent_h = incumbent_cp.compute_heuristic(abstract_state_ids);
        if self.debug_diagnostics {
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
                    let neighbor_result = if self.config.partitioning.uses_regions() {
                        self.build_abstract_operator_cp(
                            collection,
                            incumbent_order,
                            optimization_deadline,
                            self.config.saturator,
                        )
                    } else {
                        self.build_label_cp(collection, incumbent_order, optimization_deadline)
                    };
                    let neighbor_cp = match neighbor_result {
                        Ok(cp) => cp,
                        Err(error) if Self::is_online_deadline_error_eval(&error) => {
                            incumbent_order.swap(i, j);
                            info!(
                                "scp_online: order optimization stopped while computing table (deadline)"
                            );
                            return Ok(());
                        }
                        Err(error) => {
                            incumbent_order.swap(i, j);
                            return Err(error);
                        }
                    };
                    let neighbor_h = neighbor_cp.compute_heuristic(abstract_state_ids);
                    if neighbor_h > incumbent_h {
                        if self.debug_diagnostics {
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

    fn build_abstract_operator_cp(
        &self,
        collection: PartitionedCollection<'_, '_>,
        order: &[usize],
        deadline: Option<Instant>,
        saturator: Saturator,
    ) -> Result<CostPartitioningHeuristic, EvaluationError> {
        let PartitionedCollection {
            task,
            components,
            abstract_state_ids,
            standalone_current_h,
            original_costs,
        } = collection;
        let mut cp = CostPartitioningHeuristic::default();
        let mut remaining_costs = TransitionResidualCosts::from_operator_costs(original_costs);

        for sweep in 0..=self.config.residual_sweeps {
            if sweep > 0 {
                info!(
                    "scp_online: starting regional residual abstraction sweep {sweep}/{}",
                    self.config.residual_sweeps
                );
            }
            for &component_id in order {
                if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                    break;
                }
                let component = components.get(component_id).ok_or_else(|| {
                    EvaluationError::InvalidState(format!(
                        "regional SCP order references missing component {component_id}"
                    ))
                })?;
                if let AbstractionComponent::Domain(heuristic) = component {
                    let abstraction = heuristic.abstraction();
                    debug!(
                        "scp_online: abstract-operator CP step abstraction {component_id}, abstract_states={}, standalone_h={}, metadata={}",
                        abstraction_state_count(abstraction),
                        standalone_current_h
                            .get(component_id)
                            .copied()
                            .unwrap_or(0.0),
                        abstraction_metadata_summary(abstraction),
                    );
                    log_abstract_operator_region_summary(
                        component_id,
                        &abstraction.abstract_operator_regions,
                    );
                    self.log_abstract_operator_label_diagnostic(
                        abstraction,
                        abstraction.task_for_factory(task),
                        component_id,
                        abstract_state_ids,
                        &remaining_costs,
                    )?;
                }

                let result = self.add_transition_component_step(
                    &mut cp,
                    &mut remaining_costs,
                    ComponentStepContext {
                        component_id,
                        component,
                        task,
                        abstract_state_ids,
                        deadline,
                        cost_space: SaturationCostSpace::Regional,
                        saturator,
                        step_prefix: "abstract-operator",
                    },
                );
                match result {
                    Ok(true) => log_transition_residual_summary(&remaining_costs),
                    Ok(false) => {
                        info!(
                            "scp_online: abstract-operator component {component_id} stopped while reducing residual costs (deadline)"
                        );
                        break;
                    }
                    Err(error) if Self::is_online_deadline_error_eval(&error) => {
                        info!(
                            "scp_online: abstract-operator component {component_id} stopped while computing table (deadline)"
                        );
                        break;
                    }
                    Err(error) => return Err(error),
                }
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
        if !self.debug_diagnostics || !enabled!(Level::INFO) {
            return Ok(());
        }
        let label_remaining_costs = remaining_costs.operator_costs_for_label_cp();
        let (label_distances, label_saturated) = Self::compute_domain_cp_entry(
            abstraction,
            abstraction_task,
            self.config.combine_labels,
            &label_remaining_costs,
            None,
        )?;
        let label_h = current_h_for_distances(abstraction_id, &label_distances, abstract_state_ids);
        let (positive_labels, total_label_saturated) = positive_cost_stats(&label_saturated);
        let stats = abstract_operator_region_stats(&abstraction.abstract_operator_regions);
        info!(
            "scp_online: abstract-operator label diagnostic abstraction {abstraction_id}: label_equivalent_h={label_h}, positive_saturated_labels={positive_labels}, total_label_saturated={total_label_saturated:.6}, operator_region_labels={}, bounded_operator_region_labels={}",
            stats.total_labels(),
            stats.bounded_labels(),
        );
        log_positive_label_operator_region_diagnostics(
            abstraction_id,
            abstraction_task,
            &abstraction.abstract_operator_regions,
            &label_saturated,
        );
        Ok(())
    }

    fn build_label_cp(
        &self,
        collection: PartitionedCollection<'_, '_>,
        order: &[usize],
        deadline: Option<Instant>,
    ) -> Result<CostPartitioningHeuristic, EvaluationError> {
        let PartitionedCollection {
            task,
            components,
            abstract_state_ids,
            original_costs,
            ..
        } = collection;
        let mut cp = CostPartitioningHeuristic::default();
        let mut remaining_costs = original_costs.to_vec();

        for &component_id in order {
            if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                break;
            }
            let component = components.get(component_id).ok_or_else(|| {
                EvaluationError::InvalidState(format!(
                    "label SCP order references missing component {component_id}"
                ))
            })?;
            match self.add_label_component_step(
                &mut cp,
                &mut remaining_costs,
                ComponentStepContext {
                    component_id,
                    component,
                    task,
                    abstract_state_ids,
                    deadline,
                    cost_space: SaturationCostSpace::Label,
                    saturator: self.config.saturator,
                    step_prefix: "label",
                },
            ) {
                Ok(true) => {}
                Ok(false) => unreachable!("label residual reduction has no deadline"),
                Err(error) if Self::is_online_deadline_error_eval(&error) => {
                    info!(
                        "scp_online: label component {component_id} stopped while computing table (deadline)"
                    );
                    break;
                }
                Err(error) => return Err(error),
            }
        }

        Ok(cp)
    }

    fn build_label_fill_scp(
        &self,
        collection: PartitionedCollection<'_, '_>,
        order: &[usize],
        deadline: Option<Instant>,
    ) -> Result<(CostPartitioningHeuristic, Vec<f64>), EvaluationError> {
        let PartitionedCollection {
            task,
            components,
            abstract_state_ids,
            original_costs,
            ..
        } = collection;
        let mut cp = CostPartitioningHeuristic::default();
        let mut remaining_costs = original_costs.to_vec();

        for &component_id in order {
            if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                break;
            }
            let component = components.get(component_id).ok_or_else(|| {
                EvaluationError::InvalidState(format!(
                    "fillSCP order references missing component {component_id}"
                ))
            })?;
            if matches!(component, AbstractionComponent::PatternDatabase(_)) {
                return Err(EvaluationError::InvalidState(format!(
                    "fillSCP order references unsupported PDB component {component_id}"
                )));
            }
            let completed = self.add_label_component_step(
                &mut cp,
                &mut remaining_costs,
                ComponentStepContext {
                    component_id,
                    component,
                    task,
                    abstract_state_ids,
                    deadline,
                    cost_space: SaturationCostSpace::Label,
                    saturator: self.config.saturator,
                    step_prefix: "fillSCP label",
                },
            )?;
            assert!(completed, "label residual reduction has no deadline");
        }

        Ok((cp, remaining_costs))
    }

    fn build_abstract_operator_fill_scp(
        &self,
        collection: PartitionedCollection<'_, '_>,
        order: &[usize],
        deadline: Option<Instant>,
        saturator: Saturator,
    ) -> Result<
        (
            CostPartitioningHeuristic,
            Vec<f64>,
            Vec<LmCutResidualOperatorCostPartition>,
        ),
        EvaluationError,
    > {
        let PartitionedCollection {
            task,
            components,
            abstract_state_ids,
            original_costs,
            ..
        } = collection;
        let mut cp = CostPartitioningHeuristic::default();
        let mut remaining_costs = TransitionResidualCosts::from_operator_costs(original_costs);

        for &component_id in order {
            if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                break;
            }
            let component = components.get(component_id).ok_or_else(|| {
                EvaluationError::InvalidState(format!(
                    "fillSCP abstract-operator order references missing component {component_id}"
                ))
            })?;
            if matches!(component, AbstractionComponent::PatternDatabase(_)) {
                return Err(EvaluationError::InvalidState(format!(
                    "fillSCP abstract-operator order references unsupported PDB component {component_id}"
                )));
            }
            if !self.add_transition_component_step(
                &mut cp,
                &mut remaining_costs,
                ComponentStepContext {
                    component_id,
                    component,
                    task,
                    abstract_state_ids,
                    deadline,
                    cost_space: SaturationCostSpace::AbstractOperator,
                    saturator,
                    step_prefix: "fillSCP abstract-operator",
                },
            )? {
                info!(
                    "fillSCP: abstract-operator component {component_id} stopped while reducing residual costs (deadline)"
                );
                break;
            }
            log_transition_residual_summary(&remaining_costs);
        }

        let residual_partitions = remaining_costs.operator_cost_partitions_for_lmcut(4, 4);
        Ok((
            cp,
            remaining_costs.operator_costs_for_label_cp(),
            residual_partitions,
        ))
    }

    fn retain_cp(
        state: &mut ScpOnlineState,
        cp: CostPartitioningHeuristic,
        abstract_state_ids: &[Option<usize>],
        max_h: &mut f64,
        retain_alternative: bool,
        max_size_kb: usize,
    ) {
        if state.cp_heuristics.contains(&cp) {
            info!(
                "scp_online: discarded duplicate CP with {} lookup tables",
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
                "scp_online: retained CP, current-state h {} -> {}, lookup_tables={}, components={:?}, size={} KiB, alternative={}",
                *max_h,
                new_h,
                cp.lookup_tables.len(),
                component_values,
                size_kb,
                !improves_current_state,
            );
            state.size_kb = state.size_kb.saturating_add(size_kb);
            state.cp_heuristics.push(cp);
            state.required_mask = Self::required_lookup_mask(state, abstract_state_ids.len());
            *max_h = (*max_h).max(new_h);
        } else if !fits_size_limit {
            info!(
                "scp_online: discarded CP because storing {} KiB would exceed max_size={} KiB (stored={} KiB)",
                size_kb, max_size_kb, state.size_kb
            );
        } else {
            info!(
                "scp_online: rejected CP, candidate_h={} did not improve current_h={}, lookup_tables={}",
                new_h,
                *max_h,
                cp.lookup_tables.len(),
            );
        }
    }

    fn compute_domain_cp_entry(
        abstraction: &DomainAbstraction,
        task: &dyn AbstractNumericTask,
        combine_labels: bool,
        remaining_costs: &[f64],
        deadline: Option<Instant>,
    ) -> Result<(Vec<f64>, Vec<f64>), EvaluationError> {
        let start = Instant::now();
        let (table, saturated) = abstraction
            .factory
            .build_cost_partitioned_distance_table(
                task,
                combine_labels,
                remaining_costs,
                DistanceTableOptions::default()
                    .with_goal_facts(&abstraction.distance_table.goal_facts)
                    .with_deadline(deadline),
            )
            .map_err(|error| Self::construction_error("failed to compute SCP table", error))?;
        debug!(
            "scp_online: label distance-table/CP construction finished in {:.3}s, states={}, saturated_costs={}",
            start.elapsed().as_secs_f64(),
            table.distances.len(),
            saturated.len()
        );
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
            .build_cost_partitioned_distance_table(
                task,
                combine_labels,
                remaining_costs,
                DistanceTableOptions::default()
                    .with_goal_facts(&abstraction.distance_table.goal_facts),
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
    components: &[AbstractionComponent<'_>],
    component_id: usize,
) -> usize {
    components
        .get(component_id)
        .and_then(AbstractionComponent::as_domain)
        .and_then(|abstraction| abstraction.metadata.collection_iteration)
        .unwrap_or(usize::MAX)
}

#[derive(Debug, Clone, Copy, Default)]
struct GoalCoverOrderVariant {
    anchor_goal_offset: usize,
    anchor_offset: usize,
    complementary_round: usize,
    representative_round: usize,
    non_goal_prefix: bool,
    compact: bool,
}

fn compact_goal_cover_variants(
    goal_count: usize,
    variants_per_goal: usize,
    guarantee_specialist_coverage: bool,
) -> Vec<GoalCoverOrderVariant> {
    assert!(goal_count > 0);
    assert!(variants_per_goal > 0);
    let pairwise_variant_count = goal_count
        .saturating_mul(variants_per_goal)
        .saturating_mul(variants_per_goal)
        .min(64);
    let specialist_coverage_count = if guarantee_specialist_coverage {
        goal_count.saturating_mul(variants_per_goal.min(4))
    } else {
        0
    };
    let variant_count = pairwise_variant_count.max(specialist_coverage_count);
    (0..variant_count)
        .map(|variant_index| {
            let anchor_goal_offset = variant_index % goal_count;
            let anchor_round = variant_index / goal_count;
            let anchor_offset = anchor_round % variants_per_goal;
            let representative_round = (anchor_round / variants_per_goal) % variants_per_goal;
            GoalCoverOrderVariant {
                anchor_goal_offset,
                anchor_offset,
                complementary_round: anchor_offset.wrapping_add(representative_round),
                representative_round,
                compact: true,
                ..Default::default()
            }
        })
        .collect()
}

fn cartesian_goal_cover_order(
    base_order: &[usize],
    components: &[AbstractionComponent<'_>],
    standalone_current_h: &[f64],
    require_pure_cartesian_collection: bool,
    variant: GoalCoverOrderVariant,
) -> Option<Vec<usize>> {
    let is_goal_cartesian = |component_id: usize| {
        components
            .get(component_id)
            .and_then(AbstractionComponent::as_cartesian)
            .is_some_and(|abstraction| abstraction.metadata.collection_goal_id.is_some())
    };
    let pure_cartesian_collection = base_order.iter().copied().all(is_goal_cartesian);
    if (require_pure_cartesian_collection || variant.compact) && !pure_cartesian_collection {
        return None;
    }

    let mut by_goal: HashMap<usize, Vec<usize>> = HashMap::new();
    for &component_id in base_order {
        let Some(abstraction) = components
            .get(component_id)
            .and_then(AbstractionComponent::as_cartesian)
        else {
            continue;
        };
        let Some(goal_id) = abstraction.metadata.collection_goal_id else {
            continue;
        };
        by_goal.entry(goal_id).or_default().push(component_id);
    }
    let progressive_roots = components
        .iter()
        .filter_map(AbstractionComponent::as_cartesian)
        .any(|abstraction| abstraction.metadata.progressive_refinement_root);
    if by_goal.is_empty()
        || (!progressive_roots && by_goal.values().all(|components| components.len() < 2))
    {
        return None;
    }

    let current_h = |component_id: usize| {
        standalone_current_h
            .get(component_id)
            .copied()
            .unwrap_or(0.0)
    };
    let abstraction = |component_id: usize| {
        components
            .get(component_id)
            .and_then(AbstractionComponent::as_cartesian)
            .expect("goal-cover order component must reference a Cartesian abstraction")
    };
    let goal_max_h = |components: &[usize]| {
        components
            .iter()
            .copied()
            .map(current_h)
            .fold(0.0, f64::max)
    };
    let mut sorted_goals = by_goal
        .iter()
        .filter(|(_, components)| progressive_roots || components.len() >= 2)
        .collect::<Vec<_>>();
    sorted_goals.sort_by(|(left_goal, left), (right_goal, right)| {
        goal_max_h(right)
            .total_cmp(&goal_max_h(left))
            .then_with(|| left_goal.cmp(right_goal))
    });
    let (&anchor_goal, anchor_components) =
        sorted_goals[variant.anchor_goal_offset % sorted_goals.len()];

    let compare_anchor = |&left: &usize, &right: &usize| {
        let left_abstraction = abstraction(left);
        let right_abstraction = abstraction(right);
        current_h(right)
            .total_cmp(&current_h(left))
            .then_with(|| {
                right_abstraction
                    .metadata
                    .split_selection_rank
                    .cmp(&left_abstraction.metadata.split_selection_rank)
            })
            .then_with(|| {
                (right_abstraction.metadata.refinement_direction
                    == CartesianRefinementDirection::Regression)
                    .cmp(
                        &(left_abstraction.metadata.refinement_direction
                            == CartesianRefinementDirection::Regression),
                    )
            })
            .then_with(|| left.cmp(&right))
    };
    let mut sorted_anchor_components = anchor_components.clone();
    sorted_anchor_components.sort_by(compare_anchor);
    let first_anchor = sorted_anchor_components
        .get(variant.anchor_offset % sorted_anchor_components.len())
        .copied()?;
    let first_metadata = &abstraction(first_anchor).metadata;
    let mut complementary_anchors = anchor_components
        .iter()
        .copied()
        .filter(|&component_id| component_id != first_anchor)
        .collect::<Vec<_>>();
    complementary_anchors.sort_by(|&left, &right| {
        let complement_score = |component_id: usize| {
            let metadata = &abstraction(component_id).metadata;
            (
                metadata.refinement_direction != first_metadata.refinement_direction,
                metadata.split_selection_rank == first_metadata.split_selection_rank,
            )
        };
        complement_score(right)
            .cmp(&complement_score(left))
            .then_with(|| current_h(right).total_cmp(&current_h(left)))
            .then_with(|| {
                abstraction(left)
                    .transition_system
                    .transitions
                    .len()
                    .cmp(&abstraction(right).transition_system.transitions.len())
            })
            .then_with(|| left.cmp(&right))
    });
    let complementary_anchor = (!complementary_anchors.is_empty())
        .then(|| complementary_anchors[variant.complementary_round % complementary_anchors.len()]);

    let mut order = Vec::with_capacity(base_order.len());
    let mut selected = HashSet::with_capacity(base_order.len());
    if variant.non_goal_prefix {
        for &component_id in base_order {
            if !is_goal_cartesian(component_id) {
                order.push(component_id);
                selected.insert(component_id);
            }
        }
    }
    order.push(first_anchor);
    selected.insert(first_anchor);
    if let Some(component_id) = complementary_anchor {
        order.push(component_id);
        selected.insert(component_id);
    }
    if progressive_roots {
        for component_id in sorted_anchor_components {
            if selected.insert(component_id) {
                order.push(component_id);
            }
        }
    }

    let mut other_goals = by_goal
        .iter()
        .filter(|(goal_id, _)| **goal_id != anchor_goal)
        .collect::<Vec<_>>();
    other_goals.sort_by(|(left_goal, left), (right_goal, right)| {
        goal_max_h(right)
            .total_cmp(&goal_max_h(left))
            .then_with(|| left_goal.cmp(right_goal))
    });
    for (representative_index, (_, components)) in other_goals.into_iter().enumerate() {
        let mut representatives = components.clone();
        representatives.sort_by(|&left, &right| {
            let variant_score = |component_id: usize| {
                let metadata = &abstraction(component_id).metadata;
                (
                    metadata.refinement_direction == first_metadata.refinement_direction,
                    metadata.split_selection_rank == first_metadata.split_selection_rank,
                )
            };
            variant_score(right)
                .cmp(&variant_score(left))
                .then_with(|| {
                    abstraction(left)
                        .transition_system
                        .transitions
                        .len()
                        .cmp(&abstraction(right).transition_system.transitions.len())
                })
                .then_with(|| current_h(right).total_cmp(&current_h(left)))
                .then_with(|| left.cmp(&right))
        });
        let representative_offset = variant
            .representative_round
            .wrapping_add(representative_index.wrapping_mul(variant.anchor_offset + 1))
            % representatives.len();
        let representative = representatives[representative_offset];
        order.push(representative);
        selected.insert(representative);
    }

    if variant.compact {
        debug_assert_eq!(
            order.len(),
            if progressive_roots {
                by_goal.len() + anchor_components.len() - 1
            } else {
                by_goal.len() + 1
            }
        );
        return Some(order);
    }

    let mut remaining = base_order
        .iter()
        .copied()
        .filter(|component_id| !selected.contains(component_id))
        .collect::<Vec<_>>();
    remaining.sort_by(
        |&left, &right| match (is_goal_cartesian(left), is_goal_cartesian(right)) {
            (true, true) => abstraction(left)
                .transition_system
                .transitions
                .len()
                .cmp(&abstraction(right).transition_system.transitions.len())
                .then_with(|| current_h(right).total_cmp(&current_h(left)))
                .then_with(|| left.cmp(&right)),
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            (false, false) => base_order
                .iter()
                .position(|&id| id == left)
                .cmp(&base_order.iter().position(|&id| id == right)),
        },
    );
    order.extend(remaining);
    debug_assert_eq!(order.len(), base_order.len());
    debug_assert_eq!(
        order.iter().copied().collect::<HashSet<_>>().len(),
        base_order.len()
    );
    Some(order)
}

fn max_heuristic_greedy_order(base_order: &[usize], standalone_current_h: &[f64]) -> Vec<usize> {
    let mut order = base_order.to_vec();
    order.sort_by(|&left, &right| {
        let left_h = standalone_current_h.get(left).copied().unwrap_or(0.0);
        let right_h = standalone_current_h.get(right).copied().unwrap_or(0.0);
        right_h.total_cmp(&left_h).then_with(|| left.cmp(&right))
    });
    order
}

fn optimization_deadline(max_time: f64) -> Option<Instant> {
    (max_time.is_finite() && max_time.is_sign_positive())
        .then(|| Instant::now() + Duration::from_secs_f64(max_time))
}

fn earliest_deadline(left: Option<Instant>, right: Option<Instant>) -> Option<Instant> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.min(right)),
        (Some(deadline), None) | (None, Some(deadline)) => Some(deadline),
        (None, None) => None,
    }
}

fn random_walk_length(
    initial_h: f64,
    average_operator_cost: f64,
    rng: &mut SmallRng,
) -> Result<usize, EvaluationError> {
    if !initial_h.is_finite() || initial_h < 0.0 {
        return Err(EvaluationError::ComputationFailed(format!(
            "offline SCP sampling requires a finite non-negative initial h, got {initial_h}"
        )));
    }
    if !average_operator_cost.is_finite() || average_operator_cost < 0.0 {
        return Err(EvaluationError::ComputationFailed(format!(
            "offline SCP sampling requires a finite non-negative average operator cost, got {average_operator_cost}"
        )));
    }
    let trials_f64 = if initial_h <= f64::EPSILON || average_operator_cost <= f64::EPSILON {
        10.0
    } else {
        4.0 * (initial_h / average_operator_cost).round()
    };
    if trials_f64 > usize::MAX as f64 {
        return Err(EvaluationError::ComputationFailed(format!(
            "offline SCP random-walk trial count does not fit usize: {trials_f64}"
        )));
    }
    let trials = trials_f64 as usize;
    Ok((0..trials).filter(|_| rng.gen_bool(0.5)).count())
}

fn retain_if_sample_improving(
    candidate: CostPartitioningHeuristic,
    sample_ids: &[Vec<Option<usize>>],
    sample_best: &mut [f64],
    portfolio: &mut Vec<CostPartitioningHeuristic>,
    portfolio_size_kb: &mut usize,
    max_size_kb: usize,
) -> bool {
    assert_eq!(sample_ids.len(), sample_best.len());
    if portfolio.contains(&candidate) {
        return false;
    }
    let candidate_size = candidate.estimate_size_in_kb();
    if !portfolio.is_empty() && portfolio_size_kb.saturating_add(candidate_size) > max_size_kb {
        return false;
    }
    let values = sample_ids
        .iter()
        .map(|ids| candidate.compute_heuristic(ids))
        .collect::<Vec<_>>();
    if !values
        .iter()
        .zip(sample_best.iter())
        .any(|(&value, &best)| value > best)
    {
        return false;
    }
    for (best, value) in sample_best.iter_mut().zip(values) {
        *best = (*best).max(value);
    }
    *portfolio_size_kb = portfolio_size_kb.saturating_add(candidate_size);
    portfolio.push(candidate);
    true
}

fn lookup_distance(
    abstraction_id: usize,
    distances: &[f64],
    unknown_value: f64,
    abstract_state_ids: &[Option<usize>],
) -> f64 {
    abstract_state_ids
        .get(abstraction_id)
        .copied()
        .flatten()
        .and_then(|state_id| distances.get(state_id))
        .copied()
        .unwrap_or(unknown_value)
}

fn standalone_envelope_value(state: &ScpOnlineState, abstract_state_ids: &[Option<usize>]) -> f64 {
    state
        .h_values_by_abstraction
        .iter()
        .enumerate()
        .map(|(abstraction_id, distances)| {
            // A miss contributes nothing, for every kind of abstraction.
            //
            // This used to be infinity for a prefix of component kinds, on the
            // reading that a domain abstraction covers every concrete state, so
            // a state it has no entry for cannot reach the goal. That reading is
            // wrong here: when the online path is not rebuilding a partitioning,
            // it computes abstract state ids only for abstractions selected by
            // `required_mask`, so every other one is legitimately absent.
            // Absent then meant infinite, the maximum swallowed every finite
            // estimate, and the search recorded a solvable state as a dead end.
            //
            // A distance that is present and infinite still means unreachable,
            // and still propagates, because that one was computed rather than
            // assumed.
            lookup_distance(abstraction_id, distances, 0.0, abstract_state_ids)
        })
        .fold(0.0, f64::max)
}

fn standalone_lookup_values_size_kb(tables: &[Vec<f64>]) -> usize {
    tables
        .iter()
        .filter(|distances| distances.iter().any(|value| *value > 0.0))
        .map(Vec::len)
        .sum::<usize>()
        .saturating_mul(std::mem::size_of::<f64>())
        / 1024
}

fn mandatory_goal_specialist_indices(
    candidates: &[CostPartitioningHeuristic],
    sample_ids: &[Vec<Option<usize>>],
) -> Vec<usize> {
    assert!(!candidates.is_empty());
    const SPECIALISTS_PER_GOAL: usize = 4;
    let mut candidates_by_goal = HashMap::<usize, Vec<(usize, f64)>>::new();
    for (index, candidate) in candidates.iter().enumerate() {
        let Some(goal_id) = candidate.specialist_goal_id else {
            continue;
        };
        let score = sample_ids
            .iter()
            .map(|ids| candidate.compute_heuristic(ids))
            .sum::<f64>();
        assert!(!score.is_nan(), "goal-specialist SCP score must not be NaN");
        candidates_by_goal
            .entry(goal_id)
            .or_default()
            .push((index, score));
    }

    let mut goals = candidates_by_goal.into_iter().collect::<Vec<_>>();
    goals.sort_by_key(|(goal_id, _)| *goal_id);
    let mut indices = vec![0];
    for (_, mut goal_candidates) in goals {
        goal_candidates.sort_by(|left, right| {
            (right.0 == 0)
                .cmp(&(left.0 == 0))
                .then_with(|| right.1.total_cmp(&left.1))
                .then_with(|| left.0.cmp(&right.0))
        });
        indices.extend(
            goal_candidates
                .into_iter()
                .take(SPECIALISTS_PER_GOAL)
                .map(|(index, _)| index)
                .filter(|&index| index != 0),
        );
    }
    indices
}

fn retain_mandatory_partition(
    candidate: CostPartitioningHeuristic,
    sample_ids: &[Vec<Option<usize>>],
    sample_best: &mut [f64],
    portfolio: &mut Vec<CostPartitioningHeuristic>,
    portfolio_size_kb: &mut usize,
    max_size_kb: usize,
) -> Result<(), String> {
    assert_eq!(sample_ids.len(), sample_best.len());
    let candidate_size = candidate.estimate_size_in_kb();
    let required_size = portfolio_size_kb.saturating_add(candidate_size);
    if required_size > max_size_kb {
        return Err(format!(
            "mandatory goal-specialist SCPs require {required_size} KiB, exceeding max_size={max_size_kb} KiB"
        ));
    }
    for (best, ids) in sample_best.iter_mut().zip(sample_ids) {
        *best = (*best).max(candidate.compute_heuristic(ids));
    }
    *portfolio_size_kb = required_size;
    portfolio.push(candidate);
    Ok(())
}

fn abstraction_is_target_centered(
    components: &[AbstractionComponent<'_>],
    component_id: usize,
) -> bool {
    components
        .get(component_id)
        .and_then(AbstractionComponent::as_domain)
        .and_then(|abstraction| abstraction.metadata.flaw_kind.as_deref())
        .is_some_and(|flaw_kind| flaw_kind == "target_centered")
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
    diversify: bool,
    step: &str,
    abstraction_id: usize,
    distances: &[f64],
    abstract_state_ids: &[Option<usize>],
) -> bool {
    if diversify {
        return false;
    }
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

fn regional_allocation_amounts(allocation: &RegionalCostAllocation) -> Vec<f64> {
    allocation
        .entries()
        .iter()
        .map(|entry| entry.amount)
        .collect()
}

impl Heuristic for SaturatedCostPartitioningOnlineHeuristic<'_> {
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
                Some(&state.required_mask),
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
            Self::retain_standalone_envelope(&mut state, abstract_state_ids.len());
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
                state.h_values_by_abstraction.is_empty(),
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

// ---------------------------------------------------------------------------
// Greedy order utilities
// ---------------------------------------------------------------------------

fn deduplicate_orders(orders: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    let mut seen = HashSet::with_capacity(orders.len());
    orders
        .into_iter()
        .filter(|order| seen.insert(order.clone()))
        .collect()
}

fn deduplicate_specialist_orders(
    orders: Vec<(Option<usize>, Vec<usize>)>,
) -> Vec<(Option<usize>, Vec<usize>)> {
    let mut index_by_order = HashMap::<Vec<usize>, usize>::with_capacity(orders.len());
    let mut unique = Vec::<(Option<usize>, Vec<usize>)>::with_capacity(orders.len());
    for (specialist_goal_id, order) in orders {
        if let Some(&index) = index_by_order.get(&order) {
            if unique[index].0.is_none() {
                unique[index].0 = specialist_goal_id;
            }
            continue;
        }
        index_by_order.insert(order.clone(), unique.len());
        unique.push((specialist_goal_id, order));
    }
    unique
}

fn compute_score(h: f64, stolen_costs: f64, scoring_function: ScoringFunction) -> f64 {
    match scoring_function {
        ScoringFunction::MaxHeuristic => h,
        ScoringFunction::MinStolenCosts => -stolen_costs,
        ScoringFunction::MaxHeuristicPerStolenCosts => h / stolen_costs.max(1.0),
    }
}

fn standalone_current_h_values(
    state: &ScpOnlineState,
    abstract_state_ids: &[Option<usize>],
) -> Vec<f64> {
    (0..state.h_values_by_abstraction.len())
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

fn compute_regional_conflict_scores(
    components: &[AbstractionComponent<'_>],
    saturated_by_component: &[Vec<f64>],
    operator_costs: &[f64],
) -> Result<Vec<Option<f64>>, EvaluationError> {
    let mut regions_by_component = Vec::with_capacity(components.len());
    for (component_id, component) in components.iter().enumerate() {
        let (operator_regions, expected_operator_regions) = match component {
            AbstractionComponent::Domain(heuristic) => {
                let abstraction = heuristic.abstraction();
                (
                    abstraction.abstract_operator_regions.as_slice(),
                    abstraction.abstract_operators.len(),
                )
            }
            AbstractionComponent::Cartesian(heuristic) => {
                let abstraction = heuristic.abstraction();
                (
                    abstraction.abstract_operator_regions.as_slice(),
                    abstraction.transition_system.transitions.len(),
                )
            }
            AbstractionComponent::PatternDatabase(_) => {
                regions_by_component.push(None);
                continue;
            }
        };
        if operator_regions.len() != expected_operator_regions {
            return Err(EvaluationError::ComputationFailed(format!(
                "region SCP component {component_id} has {} operator regions for {expected_operator_regions} retained abstract transitions/operators",
                operator_regions.len()
            )));
        }
        let mut by_operator = HashMap::<usize, Vec<&StateRegion>>::new();
        for operator_region in operator_regions {
            for label in &operator_region.labels {
                if label.concrete_op_id >= operator_costs.len() {
                    return Err(EvaluationError::ComputationFailed(format!(
                        "region SCP component {component_id} operator region references missing operator {}",
                        label.concrete_op_id
                    )));
                }
                by_operator
                    .entry(label.concrete_op_id)
                    .or_default()
                    .push(&label.source);
            }
        }
        regions_by_component.push(Some(by_operator));
    }

    let mut scores = regions_by_component
        .iter()
        .map(|regions| regions.as_ref().map(|_| 0.0))
        .collect::<Vec<_>>();
    for left in 0..components.len() {
        let Some(left_by_operator) = &regions_by_component[left] else {
            continue;
        };
        for right in left + 1..components.len() {
            let Some(right_by_operator) = &regions_by_component[right] else {
                continue;
            };
            let left_saturated = &saturated_by_component[left];
            let right_saturated = &saturated_by_component[right];
            for (&operator_id, left_regions) in left_by_operator {
                let Some(right_regions) = right_by_operator.get(&operator_id) else {
                    continue;
                };
                let left_amount = left_saturated
                    .get(operator_id)
                    .copied()
                    .unwrap_or(f64::NEG_INFINITY);
                let right_amount = right_saturated
                    .get(operator_id)
                    .copied()
                    .unwrap_or(f64::NEG_INFINITY);
                let base_cost = operator_costs[operator_id];
                if !left_amount.is_finite() || !right_amount.is_finite() || !base_cost.is_finite() {
                    continue;
                }
                let conflict = pair_regional_conflict(
                    left_regions,
                    left_amount,
                    right_regions,
                    right_amount,
                    base_cost,
                );
                if conflict > 1e-9 {
                    *scores[left]
                        .as_mut()
                        .expect("regional component must have a conflict score") += conflict;
                    *scores[right]
                        .as_mut()
                        .expect("regional component must have a conflict score") += conflict;
                }
            }
        }
    }
    Ok(scores)
}

fn pair_regional_conflict(
    left_regions: &[&StateRegion],
    left_amount: f64,
    right_regions: &[&StateRegion],
    right_amount: f64,
    base_cost: f64,
) -> f64 {
    let excess = (left_amount + right_amount - base_cost).max(0.0);
    if excess <= 1e-9 {
        return 0.0;
    }
    if left_regions
        .iter()
        .any(|left| right_regions.iter().any(|right| left.overlaps(right)))
    {
        excess
    } else {
        0.0
    }
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
            return Err(EvaluationError::ComputationFailed(format!(
                "saturated cost for operator {op_id} must be finite"
            )));
        }
        if *s < -1e-9 {
            return Err(EvaluationError::ComputationFailed(format!(
                "negative saturated cost for operator {op_id}: {s}"
            )));
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
